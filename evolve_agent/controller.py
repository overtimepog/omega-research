"""
Main controller for EvolveAgent
"""

import asyncio
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import traceback

from evolve_agent.config import Config, load_config
from evolve_agent.database import Program, ProgramDatabase
from evolve_agent.evaluator import Evaluator
from evolve_agent.llm.ensemble import LLMEnsemble
from evolve_agent.prompt.sampler import PromptSampler
from evolve_agent.reward_model import RewardModel
from evolve_agent.utils.code_utils import (
    apply_diff,
    compute_unified_diff,
    extract_code_language,
    extract_diffs,
    format_diff_summary,
    parse_evolve_blocks,
    parse_full_rewrite,
)
from evolve_agent.models.change_documentation import (
    ChangeDocumentation,
    CodeChange,
    MetricChange,
)
from evolve_agent.utils.format_utils import (
    format_metrics_safe,
    format_improvement_safe,
)

logger = logging.getLogger(__name__)

def _format_metrics(metrics: Dict[str, Any]) -> str:
    """Safely format metrics, handling both numeric and string values"""
    formatted_parts = []
    for name, value in metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            try:
                formatted_parts.append(f"{name}={value:.4f}")
            except (ValueError, TypeError):
                formatted_parts.append(f"{name}={value}")
        else:
            formatted_parts.append(f"{name}={value}")
    return ", ".join(formatted_parts)


def _format_improvement(improvement: Dict[str, Any]) -> str:
    """Safely format improvement metrics"""
    formatted_parts = []
    for name, diff in improvement.items():
        if isinstance(diff, (int, float)) and not isinstance(diff, bool):
            try:
                formatted_parts.append(f"{name}={diff:+.4f}")
            except (ValueError, TypeError):
                formatted_parts.append(f"{name}={diff}")
        else:
            formatted_parts.append(f"{name}={diff}")
    return ", ".join(formatted_parts)


class EvolveAgent:
    """
    Main controller for EvolveAgent

    Orchestrates the evolution process, coordinating between the prompt sampler,
    LLM ensemble, evaluator, and program database.

    Features:
    - Tracks the absolute best program across evolution steps
    - Ensures the best solution is not lost during the MAP-Elites process
    - Always includes the best program in the selection process for inspiration
    - Maintains detailed logs and metadata about improvements
    """

    def __init__(
        self,
        initial_program_path: str,
        initial_proposal_path: str,
        evaluation_file: str,
        config_path: Optional[str] = None,
        config: Optional[Config] = None,
        output_dir: Optional[str] = None,
    ):
        # Load configuration
        if config is not None:
            # Use provided Config object directly
            self.config = config
        else:
            # Load from file or use defaults
            self.config = load_config(config_path)

        # Set up output directory
        self.output_dir = output_dir or os.path.join(
            os.path.dirname(initial_program_path), "evolve_agent_output"
        )
        os.makedirs(self.output_dir, exist_ok=True)

        # Set up logging
        self._setup_logging()

        # Set random seed for reproducibility if specified
        if self.config.random_seed is not None:
            import random
            import numpy as np

            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)
            logger.info(f"Set random seed to {self.config.random_seed} for reproducibility")

        # Load initial program
        self.initial_program_path = initial_program_path
        self.initial_program_code = self._load_initial_program()
        self.language = extract_code_language(self.initial_program_code)

        self.initial_proposal_path = initial_proposal_path
        self.initial_proposal = self._load_initial_proposal()

        # Extract file extension from initial program
        self.file_extension = os.path.splitext(initial_program_path)[1]
        if not self.file_extension:
            # Default to .py if no extension found
            self.file_extension = ".py"
        else:
            # Make sure it starts with a dot
            if not self.file_extension.startswith("."):
                self.file_extension = f".{self.file_extension}"

        # Initialize components
        self.llm_ensemble = LLMEnsemble(self.config.llm.models)
        self.llm_evaluator_ensemble = LLMEnsemble(self.config.llm.evaluator_models)
        self.reward_model = RewardModel(self.config.rewardmodel)

        self.prompt_sampler = PromptSampler(self.config.prompt)
        self.evaluator_prompt_sampler = PromptSampler(self.config.prompt)
        self.evaluator_prompt_sampler.set_templates("evaluator_system_message")

        # Pass random seed to database if specified
        if self.config.random_seed is not None:
            self.config.database.random_seed = self.config.random_seed

        self.database = ProgramDatabase(self.config.database)

        self.evaluator = Evaluator(
            self.config.evaluator,
            evaluation_file,
            self.llm_evaluator_ensemble,
            self.evaluator_prompt_sampler,
            database=self.database,
        )

        logger.info(f"Initialized EvolveAgent with {initial_program_path} " f"and {evaluation_file}")

    def _setup_logging(self) -> None:
        """Set up logging with detailed file logs and clean console output"""
        log_dir = self.config.log_dir or os.path.join(self.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)

        # Set up root logger at DEBUG to capture everything
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

        # File handler - captures ALL details (DEBUG and above)
        log_file = os.path.join(log_dir, f"evolve_agent_{time.strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Capture everything in file
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        root_logger.addHandler(file_handler)

        # Console handler - show INFO and above, but API details are at DEBUG
        # This shows progress without verbose prompts and code dumps
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)  # Show INFO, WARNING, ERROR on console
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        root_logger.addHandler(console_handler)

        logger.info(f"Logging to {log_file} (file: DEBUG, console: INFO)")

    async def _generate_new_proposal(
        self,
        parent_proposal: List[str],
        parent_program: str,
        parent_metrics: Dict[str, Any],
        inspirations: List[Program],
        evolution_round: int,
    ) -> List[str]:
        """
        Generate a new research proposal based on parent proposal, program, and metrics
        
        Args:
            parent_proposal: The parent program's proposal
            parent_program: The parent program's code
            parent_metrics: The parent program's metrics
            inspirations: List of inspiration programs
            evolution_round: Current evolution round
            
        Returns:
            List of strings representing the new proposal
        """
        # Build a prompt for proposal generation
        proposal_prompt = self._build_proposal_prompt(
            parent_proposal=parent_proposal,
            parent_program=parent_program,
            parent_metrics=parent_metrics,
            inspirations=inspirations,
            evolution_round=evolution_round,
        )
        
        # Generate new proposal using LLM
        try:
            proposal_response = await self.llm_ensemble.generate_with_context(
                system_message=proposal_prompt["system"],
                messages=[{"role": "user", "content": proposal_prompt["user"]}],
            )
            
            # Parse the proposal response
            new_proposal = self._parse_proposal_response(proposal_response)
            
            logger.info(f"Generated new proposal for evolution round {evolution_round}")
            return new_proposal
            
        except Exception as e:
            logger.warning(f"Failed to generate new proposal: {e}")
            # Fallback to parent proposal with some modification
            return self._modify_parent_proposal(parent_proposal, parent_metrics)

    def _build_proposal_prompt(
        self,
        parent_proposal: List[str],
        parent_program: str,
        parent_metrics: Dict[str, Any],
        inspirations: List[Program],
        evolution_round: int,
    ) -> Dict[str, str]:
        """
        Build a prompt for generating a new research proposal
        
        Args:
            parent_proposal: The parent program's proposal
            parent_program: The parent program's code
            parent_metrics: The parent program's metrics
            inspirations: List of inspiration programs
            evolution_round: Current evolution round
            
        Returns:
            Dictionary with 'system' and 'user' keys
        """
        # Format parent proposal
        parent_proposal_str = "\n".join(parent_proposal) if parent_proposal else ""
        
        # Format metrics
        metrics_str = _format_metrics(parent_metrics)
        
        # Format inspirations
        inspirations_str = ""
        if inspirations:
            inspirations_str = "\n".join([
                f"Program {i+1}:\nProposal: {' '.join(prog.proposal if prog.proposal else [])}\nMetrics: {_format_metrics(prog.metrics)}"
                for i, prog in enumerate(inspirations[:3])
            ])
        
        system_message = """You are a research advisor tasked with evolving and improving research proposals. 
Your goal is to generate a new research proposal that builds upon the current proposal while addressing its limitations and incorporating insights from successful approaches.

Focus on:
1. Identifying weaknesses in the current approach based on performance metrics
2. Proposing novel improvements that could enhance performance
3. Learning from successful inspirations while maintaining originality
4. Ensuring the new proposal is technically sound and implementable"""

        user_message = f"""Based on the following information, generate an improved research proposal:

- Current Proposal:
{parent_proposal_str}

- Current Program:
```python
{parent_program}
```

- Current Metrics
{metrics_str}

Please generate a new research proposal that:
1. Addresses the limitations shown in the current metrics
2. Incorporates insights from successful approaches
3. Proposes specific technical improvements
4. Maintains clarity and technical rigor

Return the proposal as a clear, concise research abstract."""

        return {
            "system": system_message,
            "user": user_message
        }

    def _parse_proposal_response(self, response: str) -> List[str]:
        """
        Parse the LLM response to extract the new proposal
        
        Args:
            response: Raw LLM response
            
        Returns:
            List of strings representing the proposal
        """
        # Clean up the response
        proposal_text = response.strip()
        
        # Remove any markdown formatting
        proposal_text = re.sub(r'^```.*?\n', '', proposal_text, flags=re.MULTILINE)
        proposal_text = re.sub(r'\n```$', '', proposal_text, flags=re.MULTILINE)
           
        return [proposal_text]

    def _modify_parent_proposal(self, parent_proposal: List[str], parent_metrics: Dict[str, Any]) -> List[str]:
        """
        Fallback method to modify parent proposal when generation fails
        
        Args:
            parent_proposal: The parent proposal
            parent_metrics: The parent metrics
            
        Returns:
            Modified proposal
        """
        if not parent_proposal:
            return ["Enhanced research approach with improved methodology and performance optimization."]
        
        # Simple modification by adding improvement context
        modified_proposal = []
        for part in parent_proposal:
            modified_part = part
            if "improvement" not in part.lower():
                modified_part += " This approach has been enhanced based on performance analysis and optimization insights."
            modified_proposal.append(modified_part)
        
        return modified_proposal

    async def _attempt_bug_fix(
        self,
        buggy_code: str,
        error_metrics: Dict[str, Any],
        proposal: List[str],
        parent_code: str,
        original_program_id: str,
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]], bool]:
        """
        Attempt to fix a program that encountered an error during evaluation.

        Args:
            buggy_code: The code that produced an error
            error_metrics: Metrics dict containing error info (error_type, error_message, traceback)
            proposal: The research proposal for context
            parent_code: The parent program code (for reference)
            original_program_id: The ID of the buggy program

        Returns:
            Tuple of (fixed_code, fixed_metrics, success)
            - If successful: (code, metrics, True)
            - If all attempts fail: (None, None, False)
        """
        # Check if bug fixing is enabled
        if not self.config.evaluator.enable_bug_fixer:
            return (None, None, False)

        # Extract error information
        error_type = error_metrics.get("error_type", "Unknown")
        error_message = error_metrics.get("error_message", "No error message")
        error_traceback = error_metrics.get("traceback", "No traceback available")

        logger.info(f"Bug fixer activated for {error_type}: {error_message}")

        max_attempts = self.config.evaluator.max_fix_attempts
        current_code = buggy_code
        current_error_info = {
            "type": error_type,
            "message": error_message,
            "traceback": error_traceback,
        }

        for attempt in range(max_attempts):
            logger.info(f"Bug fix attempt {attempt + 1}/{max_attempts}")

            try:
                # Build bug fix prompt
                proposal_text = "\n".join(proposal) if proposal else "No proposal available"

                # Get bug fix template
                bug_fix_template = self.prompt_sampler.template_manager.get_template("bug_fix")

                # Format the prompt
                bug_fix_prompt = bug_fix_template.format(
                    proposal_text=proposal_text,
                    error_type=current_error_info["type"],
                    error_message=current_error_info["message"],
                    traceback=current_error_info["traceback"],
                    buggy_code=current_code,
                    language=self.language,
                )

                # Generate fix using LLM
                system_message = "You are an expert debugger specializing in fixing runtime errors in research code."
                llm_response = await self.llm_ensemble.generate_with_context(
                    system_message=system_message,
                    messages=[{"role": "user", "content": bug_fix_prompt}],
                )

                # Parse diffs from response
                diff_blocks = extract_diffs(llm_response)

                if not diff_blocks:
                    logger.warning(f"Bug fix attempt {attempt + 1}: No valid diffs found in LLM response")
                    continue

                # Apply diffs to create fixed code
                try:
                    fixed_code = apply_diff(current_code, llm_response)
                    logger.info(f"Bug fix attempt {attempt + 1}: Applied {len(diff_blocks)} diffs")
                except Exception as diff_error:
                    logger.warning(f"Bug fix attempt {attempt + 1}: Failed to apply diffs: {diff_error}")
                    continue

                # Evaluate the fixed code
                fix_attempt_id = f"{original_program_id}_fix_{attempt + 1}"
                fixed_metrics = await self.evaluator.evaluate_program(fixed_code, fix_attempt_id)

                # Check if the fix was successful (no error in metrics)
                if "error" not in fixed_metrics or fixed_metrics.get("error", 0) >= 0:
                    logger.info(f"✅ Bug fix successful on attempt {attempt + 1}!")
                    logger.info(f"Fixed metrics: {format_metrics_safe(fixed_metrics)}")
                    return (fixed_code, fixed_metrics, True)
                else:
                    # Still has an error, update error info for next attempt
                    logger.warning(f"Bug fix attempt {attempt + 1} still has errors")
                    current_code = fixed_code
                    current_error_info = {
                        "type": fixed_metrics.get("error_type", "Unknown"),
                        "message": fixed_metrics.get("error_message", "No error message"),
                        "traceback": fixed_metrics.get("traceback", "No traceback available"),
                    }

            except Exception as e:
                logger.warning(f"Bug fix attempt {attempt + 1} failed with exception: {e}")
                continue

        logger.warning(f"❌ All {max_attempts} bug fix attempts failed")
        return (None, None, False)

    def _load_initial_program(self) -> str:
        """Load the initial program from file"""
        with open(self.initial_program_path, "r") as f:
            return f.read()

    def _load_initial_proposal(self) -> List[str]:
        """Load the initial proposal from file"""
        with open(self.initial_proposal_path, "r") as f:
            proposal = f.read()
            return [proposal]

    async def run(
        self,
        iterations: Optional[int] = None,
        target_score: Optional[float] = None,
    ) -> Program:
        """
        Run the evolution process

        Args:
            iterations: Maximum number of iterations (uses config if None)
            target_score: Target score to reach (continues until reached if specified)

        Returns:
            Best program found
        """
        max_iterations = iterations or self.config.max_iterations

        # Define start_iteration before creating the initial program
        start_iteration = self.database.last_iteration

        # Only add initial program if starting fresh (not resuming from checkpoint)
        # Check if we're resuming AND no program matches initial code to avoid pollution
        should_add_initial = (
            start_iteration == 0
            and len(self.database.programs) == 0
            and not any(
                p.code == self.initial_program_code for p in self.database.programs.values()
            )
            and self.initial_proposal
        )

        if should_add_initial:
            logger.info("Adding initial program to database")
            initial_program_id = str(uuid.uuid4())

            # Evaluate the initial program
            initial_metrics = await self.evaluator.evaluate_program(
                self.initial_program_code, initial_program_id
            )

            initial_idea_reward = await self.reward_model.score_research_proposal(self.initial_proposal)

            initial_program = Program(
                id=initial_program_id,
                code=self.initial_program_code,
                language=self.language,
                metrics=initial_metrics,
                proposal=self.initial_proposal,
                idea_reward=initial_idea_reward,
                iteration_found=start_iteration,
            )

            self.database.add(initial_program)
        else:
            logger.info(
                f"Skipping initial program addition (resuming from iteration {start_iteration} with {len(self.database.programs)} existing programs)"
            )

        # Main evolution loop
        total_iterations = start_iteration + max_iterations

        logger.info(
            f"Starting evolution from iteration {start_iteration} for {max_iterations} iterations (total: {total_iterations})"
        )

        # Island-based evolution variables
        programs_per_island = max(
            1, max_iterations // (self.config.database.num_islands * 10)
        )  # Dynamic allocation
        current_island_counter = 0

        logger.info(f"Using island-based evolution with {self.config.database.num_islands} islands")
        self.database.log_island_status()

        for i in range(start_iteration, total_iterations):
            iteration_start = time.time()

            # Manage island evolution - switch islands periodically
            if i > start_iteration and current_island_counter >= programs_per_island:
                self.database.next_island()
                current_island_counter = 0
                logger.debug(f"Switched to island {self.database.current_island}")

            current_island_counter += 1

            # Sample parent and inspirations from current island
            parent, inspirations = self.database.sample()

            # Get artifacts for the parent program if available
            parent_artifacts = self.database.get_artifacts(parent.id)

            # Step 1: Generate new proposal using parent's proposal, program, and metrics
            new_proposal = await self._generate_new_proposal(
                parent_proposal=parent.proposal,
                parent_program=parent.code,
                parent_metrics=parent.metrics,
                inspirations=inspirations,
                evolution_round=i
            )

            # Generate concise summary for console display
            proposal_summary = await self.reward_model.summarize_proposal(new_proposal)

            # Step 2: Score the new proposal using RewardModel
            new_proposal_results = await self.reward_model.score_research_proposal(new_proposal)
            new_proposal_score = new_proposal_results[0].get('score', -1.0)

            # Step 3: Check if proposal score meets threshold
            if new_proposal_score < self.config.rewardmodel.proposal_score_threshold:
                logger.info(f"Iteration {i+1}: Proposal score {new_proposal_score:.4f} below threshold "
                           f"{self.config.rewardmodel.proposal_score_threshold:.4f}, skipping program generation")
                logger.info(f"  Proposal: {proposal_summary}")
                continue

            logger.info(f"Iteration {i+1}: Score {new_proposal_score:.1f}/10 | {proposal_summary}")
            

            # Step 4: Build prompt for program generation using all information
            prompt = self.prompt_sampler.build_prompt(
                current_program=parent.code,
                parent_program=parent.code,
                program_metrics=parent.metrics,
                previous_programs=[p.to_dict() for p in self.database.get_top_programs(3)],
                top_programs=[p.to_dict() for p in inspirations],
                language=self.language,
                evolution_round=i,
                allow_full_rewrite=self.config.allow_full_rewrites,
                program_artifacts=parent_artifacts if parent_artifacts else None,
                current_proposal=new_proposal,
                parent_proposal=parent.proposal,
                proposal_score=new_proposal_score,
            )

            # Generate code modification with retry logic (Research: LLM code generation best practices 2025)
            # Implements exponential backoff and feedback-driven prompt refinement on retries
            child_code = None
            changes_summary = None
            max_retries = self.config.max_diff_generation_retries

            try:
                for retry_attempt in range(max_retries):
                    # Generate with potentially enhanced prompt on retries
                    current_prompt = prompt.copy()
                    if retry_attempt > 0:
                        # Enhance prompt with explicit feedback about the failure
                        current_prompt["user"] += (
                            f"\n\nIMPORTANT: Previous attempt {retry_attempt} failed to generate valid "
                            f"{'diff blocks' if self.config.diff_based_evolution else 'code'}. "
                            f"Please ensure your response contains properly formatted "
                            f"{'<<<<<<< SEARCH / ======= / >>>>>>> REPLACE blocks' if self.config.diff_based_evolution else 'code blocks'}."
                        )

                    llm_response = await self.llm_ensemble.generate_with_context(
                        system_message=current_prompt["system"],
                        messages=[{"role": "user", "content": current_prompt["user"]}],
                    )

                    # Parse the response
                    if self.config.diff_based_evolution:
                        diff_blocks = extract_diffs(llm_response)
                        if not diff_blocks:
                            if retry_attempt < max_retries - 1:
                                # Exponential backoff: 0.5s, 1s, 2s
                                delay = 0.5 * (2 ** retry_attempt)
                                logger.warning(
                                    f"Iteration {i+1}: No valid diffs found (attempt {retry_attempt+1}/{max_retries}), "
                                    f"retrying in {delay}s..."
                                )
                                await asyncio.sleep(delay)
                                continue
                            else:
                                logger.warning(
                                    f"Iteration {i+1}: No valid diffs found after {max_retries} attempts, "
                                    f"skipping iteration"
                                )
                                break

                        # Success! Apply the diffs
                        child_code = apply_diff(parent.code, llm_response)
                        changes_summary = format_diff_summary(diff_blocks)
                        logger.info(f"Diff is applied successfully{' (after ' + str(retry_attempt) + ' retries)' if retry_attempt > 0 else ''}!")
                        break
                    else:
                        # Parse full rewrite
                        new_code = parse_full_rewrite(llm_response, self.language)

                        if not new_code:
                            if retry_attempt < max_retries - 1:
                                delay = 0.5 * (2 ** retry_attempt)
                                logger.warning(
                                    f"Iteration {i+1}: No valid code found (attempt {retry_attempt+1}/{max_retries}), "
                                    f"retrying in {delay}s..."
                                )
                                await asyncio.sleep(delay)
                                continue
                            else:
                                logger.warning(
                                    f"Iteration {i+1}: No valid code found after {max_retries} attempts, "
                                    f"skipping iteration"
                                )
                                break

                        child_code = new_code
                        changes_summary = "Full rewrite"
                        break

                # If we failed all retries, skip this iteration
                if child_code is None:
                    continue

                # Check code length
                if len(child_code) > self.config.max_code_length:
                    logger.warning(
                        f"Iteration {i+1}: Generated code exceeds maximum length "
                        f"({len(child_code)} > {self.config.max_code_length})"
                    )
                    continue

                # Evaluate the child program
                child_id = str(uuid.uuid4())
                child_metrics = await self.evaluator.evaluate_program(child_code, child_id)

                # Handle artifacts if they exist
                artifacts = self.evaluator.get_pending_artifacts(child_id)

                # Defense-in-depth: Validate metrics for NaN/Inf values (Research: PyTorch gradient monitoring 2025)
                # This catches any NaN/Inf that might slip through the evaluator validation
                import math
                if "error" not in child_metrics:  # Only check if not already an error
                    for metric_name, metric_value in child_metrics.items():
                        if isinstance(metric_value, (int, float)) and (math.isnan(metric_value) or math.isinf(metric_value)):
                            logger.warning(f"Iteration {i+1}: Detected invalid {metric_name}={metric_value}, converting to error format")
                            # Convert to error format to trigger bug fixing
                            child_metrics = {
                                "error": -1.0,
                                "error_type": "InvalidMetricValue",
                                "error_message": f"Metric '{metric_name}' has invalid value: {metric_value}",
                                "traceback": f"NaN or Inf detected in {metric_name} after evaluation (defense-in-depth check)",
                                "failure_stage": "evaluation"
                            }
                            break

                # Check if program has an error and attempt to fix it
                has_error = "error" in child_metrics and child_metrics.get("error", 0) < 0
                if has_error:
                    logger.warning(f"Iteration {i+1}: Program evaluation failed with {child_metrics.get('error_type', 'Unknown')}")

                    # Attempt to fix the bug
                    fixed_code, fixed_metrics, fix_success = await self._attempt_bug_fix(
                        buggy_code=child_code,
                        error_metrics=child_metrics,
                        proposal=new_proposal,
                        parent_code=parent.code,
                        original_program_id=child_id,
                    )

                    if fix_success:
                        # Use the fixed version
                        child_code = fixed_code
                        child_metrics = fixed_metrics
                        changes_summary += " (auto-fixed)"
                        logger.info(f"Iteration {i+1}: Bug automatically fixed, proceeding with fixed version")
                    else:
                        # All fix attempts failed, skip this iteration
                        logger.warning(f"Iteration {i+1}: Unable to fix bug, skipping program (not added to database)")
                        continue

                # Create a child program with the new proposal
                child_program = Program(
                    id=child_id,
                    code=child_code,
                    language=self.language,
                    parent_id=parent.id,
                    generation=parent.generation + 1,
                    metrics=child_metrics,
                    proposal=new_proposal,
                    idea_reward=new_proposal_score,
                    metadata={
                        "changes": changes_summary,
                        "parent_metrics": parent.metrics,
                    },
                )

                # Add to database (will be added to current island)
                self.database.add(child_program, iteration=i + 1)

                # Log prompts
                self.database.log_prompt(
                    template_key=(
                        "full_rewrite_user" if self.config.allow_full_rewrites else "diff_user"
                    ),
                    program_id=child_id,
                    prompt=prompt,
                    responses=[llm_response],
                )

                # Store artifacts if they exist
                if artifacts:
                    self.database.store_artifacts(child_id, artifacts)

                # Log prompts
                self.database.log_prompt(
                    template_key=(
                        "full_rewrite_user" if self.config.allow_full_rewrites else "diff_user"
                    ),
                    program_id=child_id,
                    prompt=prompt,
                    responses=[llm_response],
                )

                # Increment generation for current island
                self.database.increment_island_generation()

                # Check if migration should occur
                if self.database.should_migrate():
                    logger.info(f"Performing migration at iteration {i+1}")
                    self.database.migrate_programs()
                    self.database.log_island_status()

                # Log progress
                iteration_time = time.time() - iteration_start
                self._log_iteration(i, parent, child_program, iteration_time)

                # Specifically check if this is the new best program
                if self.database.best_program_id == child_program.id:
                    logger.info(f"🌟 New best solution found at iteration {i+1}: {child_program.id}")
                    logger.info(f"Metrics: {format_metrics_safe(child_program.metrics)}")
                    # Auto-save the new best solution
                    await self._save_best_solution_incremental(child_program)

                # Save checkpoint
                if (i + 1) % self.config.checkpoint_interval == 0:
                    self._save_checkpoint(i + 1)
                    # Also log island status at checkpoints
                    logger.info(f"Island status at checkpoint {i+1}:")
                    self.database.log_island_status()

                # Check if target score reached
                if target_score is not None:
                    # Only consider numeric metrics for target score calculation
                    numeric_metrics = [
                        v
                        for v in child_metrics.values()
                        if isinstance(v, (int, float)) and not isinstance(v, bool)
                    ]
                    if numeric_metrics:
                        avg_score = sum(numeric_metrics) / len(numeric_metrics)
                        if avg_score >= target_score:
                            logger.info(
                                f"Target score {target_score} reached after {i+1} iterations"
                            )
                            break

            except Exception as e:
                logger.exception(f"Error in iteration {i+1}: {str(e)}")
                continue

        # Get the best program using our tracking mechanism
        best_program = None
        if self.database.best_program_id:
            best_program = self.database.get(self.database.best_program_id)
            logger.info(f"Using tracked best program: {self.database.best_program_id}")

        # Fallback to calculating best program if tracked program not found
        if best_program is None:
            best_program = self.database.get_best_program()
            logger.info("Using calculated best program (tracked program not found)")

        # Check if there's a better program by combined_score that wasn't tracked
        if "combined_score" in best_program.metrics:
            best_by_combined = self.database.get_best_program(metric="combined_score")
            if (
                best_by_combined
                and best_by_combined.id != best_program.id
                and "combined_score" in best_by_combined.metrics
            ):
                # If the combined_score of this program is significantly better, use it instead
                if (
                    best_by_combined.metrics["combined_score"]
                    > best_program.metrics["combined_score"] + 0.02
                ):
                    logger.warning(
                        f"Found program with better combined_score: {best_by_combined.id}"
                    )
                    logger.warning(
                        f"Score difference: {best_program.metrics['combined_score']:.4f} vs {best_by_combined.metrics['combined_score']:.4f}"
                    )
                    best_program = best_by_combined

        if best_program:
            logger.info(
                f"Evolution complete. Best program has metrics: "
                f"{format_metrics_safe(best_program.metrics)}"
            )

            # Save the best program (using our tracked best program)
            self._save_best_program(best_program)

            return best_program
        else:
            logger.warning("No valid programs found during evolution")
            # Return None if no programs found instead of undefined initial_program
            return None

    def _log_iteration(
        self,
        iteration: int,
        parent: Program,
        child: Program,
        elapsed_time: float,
    ) -> None:
        """
        Log iteration progress

        Args:
            iteration: Iteration number
            parent: Parent program
            child: Child program
            elapsed_time: Elapsed time in seconds
        """
        # Calculate improvement using safe formatting
        improvement_str = format_improvement_safe(parent.metrics, child.metrics)

        logger.info(
            f"Iteration {iteration+1}: Child {child.id} from parent {parent.id} "
            f"in {elapsed_time:.2f}s. Metrics: "
            f"{format_metrics_safe(child.metrics)} "
            f"(Δ: {improvement_str})"
        )

    def _save_checkpoint(self, iteration: int) -> None:
        """
        Save a checkpoint

        Args:
            iteration: Current iteration number
        """
        checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Create specific checkpoint directory
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{iteration}")
        os.makedirs(checkpoint_path, exist_ok=True)

        # Save the database
        self.database.save(checkpoint_path, iteration)

        # Save the best program found so far
        best_program = None
        if self.database.best_program_id:
            best_program = self.database.get(self.database.best_program_id)
        else:
            best_program = self.database.get_best_program()

        if best_program:
            # Save the best program at this checkpoint
            best_program_path = os.path.join(checkpoint_path, f"best_program{self.file_extension}")
            with open(best_program_path, "w") as f:
                f.write(best_program.code)

            # Save metrics
            best_program_info_path = os.path.join(checkpoint_path, "best_program_info.json")
            with open(best_program_info_path, "w") as f:
                import json

                json.dump(
                    {
                        "id": best_program.id,
                        "generation": best_program.generation,
                        "iteration": best_program.iteration_found,
                        "current_iteration": iteration,
                        "metrics": best_program.metrics,
                        "language": best_program.language,
                        "timestamp": best_program.timestamp,
                        "saved_at": time.time(),
                    },
                    f,
                    indent=2,
                )

            logger.info(
                f"Saved best program at checkpoint {iteration} with metrics: "
                f"{format_metrics_safe(best_program.metrics)}"
            )

        logger.info(f"Saved checkpoint at iteration {iteration} to {checkpoint_path}")

    def _save_best_program(self, program: Optional[Program] = None) -> None:
        """
        Save the best program

        Args:
            program: Best program (if None, uses the tracked best program)
        """
        # If no program is provided, use the tracked best program from the database
        if program is None:
            if self.database.best_program_id:
                program = self.database.get(self.database.best_program_id)
            else:
                # Fallback to calculating best program if no tracked best program
                program = self.database.get_best_program()

        if not program:
            logger.warning("No best program found to save")
            return

        best_dir = os.path.join(self.output_dir, "best")
        os.makedirs(best_dir, exist_ok=True)

        # Use the extension from the initial program file
        filename = f"best_program{self.file_extension}"
        code_path = os.path.join(best_dir, filename)

        with open(code_path, "w") as f:
            f.write(program.code)

        # Save complete program info including metrics
        info_path = os.path.join(best_dir, "best_program_info.json")
        with open(info_path, "w") as f:
            import json

            json.dump(
                {
                    "id": program.id,
                    "generation": program.generation,
                    "iteration": program.iteration_found,
                    "timestamp": program.timestamp,
                    "parent_id": program.parent_id,
                    "metrics": program.metrics,
                    "language": program.language,
                    "saved_at": time.time(),
                },
                f,
                indent=2,
            )

        logger.info(f"Saved best program to {code_path} with program info to {info_path}")

    async def _generate_changes_documentation(
        self, child_program: Program, parent_program: Optional[Program]
    ) -> Optional[str]:
        """
        Generate documentation explaining code changes using LLM

        Args:
            child_program: The improved program
            parent_program: The parent program (if available)

        Returns:
            Markdown documentation string or None if generation fails
        """
        import json

        if parent_program is None:
            logger.info("No parent program available for changes documentation")
            return None

        logger.info(
            f"Generating changes documentation (child: {child_program.id}, parent: {parent_program.id})"
        )

        try:
            # Compute unified diff
            unified_diff = compute_unified_diff(parent_program.code, child_program.code, context_lines=5)

            if not unified_diff.strip():
                logger.info("No code changes detected between parent and child")
                return None

            # Format proposal
            proposal_text = "\n".join(child_program.proposal) if child_program.proposal else "No specific proposal available"

            # Format metrics
            parent_metrics_str = _format_metrics(parent_program.metrics) if parent_program.metrics else "No metrics"
            child_metrics_str = _format_metrics(child_program.metrics) if child_program.metrics else "No metrics"

            # Build prompt
            system_message = self.prompt_sampler.template_manager.get_template("changes_doc_system")
            user_message = self.prompt_sampler.template_manager.get_template("changes_doc_user").format(
                language=self.language,
                parent_code=parent_program.code,
                child_code=child_program.code,
                unified_diff=unified_diff,
                proposal=proposal_text,
                parent_metrics=parent_metrics_str,
                child_metrics=child_metrics_str,
            )

            # Call LLM with retry logic
            max_retries = getattr(self.config, "changes_doc_max_retries", 3)

            for retry_attempt in range(max_retries):
                try:
                    logger.debug(f"Calling LLM for changes documentation (attempt {retry_attempt + 1}/{max_retries})")

                    llm_response = await self.llm_ensemble.generate_with_context(
                        system_message=system_message,
                        messages=[{"role": "user", "content": user_message}],
                    )

                    # Parse JSON response
                    # Remove markdown code blocks if present
                    cleaned_response = llm_response.strip()
                    if cleaned_response.startswith("```"):
                        # Extract content between ```json and ```
                        json_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", cleaned_response, re.DOTALL)
                        if json_match:
                            cleaned_response = json_match.group(1)

                    response_data = json.loads(cleaned_response)

                    # Build ChangeDocumentation object
                    changes = []
                    for change_data in response_data.get("changes", []):
                        changes.append(
                            CodeChange(
                                title=change_data.get("title", ""),
                                location=change_data.get("location", ""),
                                old_code=change_data.get("old_code", ""),
                                new_code=change_data.get("new_code", ""),
                                reason=change_data.get("reason", ""),
                                impact=change_data.get("impact", ""),
                            )
                        )

                    # Build metric changes
                    metric_changes = []
                    for metric_name in set(parent_program.metrics.keys()) | set(child_program.metrics.keys()):
                        parent_val = parent_program.metrics.get(metric_name, 0.0)
                        child_val = child_program.metrics.get(metric_name, 0.0)

                        # Only include numeric metrics
                        if isinstance(parent_val, (int, float)) and isinstance(child_val, (int, float)):
                            metric_changes.append(
                                MetricChange(
                                    name=metric_name,
                                    before=float(parent_val),
                                    after=float(child_val),
                                )
                            )

                    doc = ChangeDocumentation(
                        summary=response_data.get("summary", ""),
                        changes=changes,
                        metric_changes=metric_changes,
                        overall_impact=response_data.get("overall_impact", ""),
                    )

                    # Convert to markdown
                    markdown_doc = doc.to_markdown()

                    logger.info(
                        f"Successfully generated changes documentation ({len(changes)} changes, {len(metric_changes)} metrics)"
                    )

                    return markdown_doc

                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Failed to parse JSON response (attempt {retry_attempt + 1}/{max_retries}): {e}"
                    )
                    if retry_attempt < max_retries - 1:
                        delay = 0.5 * (2 ** retry_attempt)
                        logger.debug(f"Retrying in {delay}s...")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"Failed to parse JSON response after {max_retries} attempts")
                        # Fallback: create basic documentation from diff
                        return self._create_fallback_documentation(
                            parent_program, child_program, unified_diff
                        )

                except Exception as e:
                    logger.warning(
                        f"Error generating changes documentation (attempt {retry_attempt + 1}/{max_retries}): {e}"
                    )
                    if retry_attempt < max_retries - 1:
                        delay = 0.5 * (2 ** retry_attempt)
                        logger.debug(f"Retrying in {delay}s...")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"Failed to generate changes documentation after {max_retries} attempts: {e}"
                        )
                        logger.debug(traceback.format_exc())
                        return None

        except Exception as e:
            logger.error(f"Unexpected error in _generate_changes_documentation: {e}")
            logger.debug(traceback.format_exc())
            return None

        return None

    def _create_fallback_documentation(
        self, parent_program: Program, child_program: Program, unified_diff: str
    ) -> str:
        """
        Create basic fallback documentation when LLM generation fails

        Args:
            parent_program: The parent program
            child_program: The child program
            unified_diff: The unified diff

        Returns:
            Basic markdown documentation
        """
        md = "# Code Evolution Changes\n\n"
        md += "## Summary\n\n"
        md += "The code was modified to improve performance metrics.\n\n"

        # Metrics table
        if parent_program.metrics and child_program.metrics:
            md += "## Metrics Improvement\n\n"
            md += "| Metric | Before | After | Change |\n"
            md += "|--------|--------|-------|--------|\n"

            for metric_name in set(parent_program.metrics.keys()) | set(child_program.metrics.keys()):
                parent_val = parent_program.metrics.get(metric_name, 0.0)
                child_val = child_program.metrics.get(metric_name, 0.0)

                if isinstance(parent_val, (int, float)) and isinstance(child_val, (int, float)):
                    change_pct = ((child_val - parent_val) / abs(parent_val)) * 100 if parent_val != 0 else 0.0
                    md += f"| {metric_name} | {parent_val:.6f} | {child_val:.6f} | {change_pct:+.1f}% |\n"

            md += "\n"

        # Diff
        md += "## Code Changes\n\n"
        md += "```diff\n"
        md += unified_diff
        md += "\n```\n"

        return md

    async def _save_best_solution_incremental(self, program: Program) -> None:
        """
        Save the best solution to a configurable directory, replacing any previous best solution.
        Called whenever a new best solution is found during evolution.

        Args:
            program: The new best program to save
        """
        # Determine the directory from config or use default
        if self.config.best_solution_dir:
            best_solution_dir = self.config.best_solution_dir
        else:
            best_solution_dir = os.path.join(self.output_dir, "best_solution")

        os.makedirs(best_solution_dir, exist_ok=True)

        # Delete any existing best solution files in the directory
        # This ensures only the current best solution exists
        import glob
        existing_files = glob.glob(os.path.join(best_solution_dir, "best_solution*"))
        for file_path in existing_files:
            try:
                os.remove(file_path)
                logger.debug(f"Removed old best solution file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove old file {file_path}: {e}")

        # Save the code file
        code_filename = f"best_solution{self.file_extension}"
        code_path = os.path.join(best_solution_dir, code_filename)
        with open(code_path, "w") as f:
            f.write(program.code)

        # Save the proposal if available
        if program.proposal:
            proposal_path = os.path.join(best_solution_dir, "best_solution_proposal.txt")
            with open(proposal_path, "w") as f:
                # proposal is a List[str], join them with newlines
                if isinstance(program.proposal, list):
                    f.write("\n".join(program.proposal))
                else:
                    f.write(str(program.proposal))

        # Save metadata
        info_path = os.path.join(best_solution_dir, "best_solution_info.json")
        with open(info_path, "w") as f:
            import json

            json.dump(
                {
                    "id": program.id,
                    "generation": program.generation,
                    "iteration": program.iteration_found,
                    "timestamp": program.timestamp,
                    "parent_id": program.parent_id,
                    "metrics": program.metrics,
                    "language": program.language,
                    "saved_at": time.time(),
                },
                f,
                indent=2,
            )

        logger.info(
            f"💾 Auto-saved best solution to {best_solution_dir} "
            f"(Score: {format_metrics_safe(program.metrics)})"
        )

        # Generate and save changes documentation if enabled
        generate_changes_doc = getattr(self.config, "generate_changes_doc", True)
        if generate_changes_doc and program.parent_id:
            parent_program = self.database.get(program.parent_id)

            if parent_program:
                changes_md = await self._generate_changes_documentation(program, parent_program)

                if changes_md:
                    changes_path = os.path.join(best_solution_dir, "best_solution_changes.md")
                    with open(changes_path, "w", encoding="utf-8") as f:
                        f.write(changes_md)
                    logger.info(f"📝 Saved changes documentation to {changes_path}")
                else:
                    logger.debug("Changes documentation generation returned None, skipping save")
            else:
                logger.debug(f"Parent program {program.parent_id} not found in database")
        elif not generate_changes_doc:
            logger.debug("Changes documentation generation is disabled in config")
        else:
            logger.debug("No parent program available for changes documentation")

if __name__=='__main__':
    # Initialize the system
    evolve_agent = EvolveAgent(
        initial_program_path="/data/zhuotaodeng/yzj/alpha-research/results/initial_program.py",
        evaluation_file="/data/zhuotaodeng/yzj/alpha-research/results/evaluator.py",
        config_path="/data/zhuotaodeng/yzj/alpha-research/configs/default_config.yaml"
    )

    print(evolve_agent)
    # # Run the evolution
    # best_program = await evolve.run(iterations=1000)
    # print(f"Best program metrics:")
    # for name, value in best_program.metrics.items():
    #     print(f"  {name}: {value:.4f}")