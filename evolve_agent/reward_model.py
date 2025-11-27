"""
Reward Model for scoring research proposals using OpenRouter API.

This module provides LLM-based evaluation of research proposals, scoring them
on clarity, novelty, technical rigor, and potential impact using the latest
best practices for LLM-as-a-judge approaches (2025).
"""

import re
import os
import json
import asyncio
import logging
from typing import List, Dict, Optional, Any
from pathlib import Path

from openai import AsyncOpenAI, APIError, APITimeoutError, RateLimitError

from evolve_agent.config import RewardModelConfig

logger = logging.getLogger(__name__)


class RewardModel:
    """
    Score research proposals using OpenRouter API with LLM-as-a-judge approach.

    Implements best practices from 2025 research:
    - Lower temperature (0.2-0.3) for consistent evaluations
    - Exponential backoff for retries
    - Score-based reward modeling with boxed format
    - Structured error handling and logging
    """

    BOX_PATTERN = r"\boxed{}"

    SYSTEM_PROMPT = """You are an expert reviewer tasked with evaluating the quality of a research proposal.
Your evaluations must be consistent, objective, and based on clear criteria."""

    SCORING_PROMPT_TEMPLATE = """Carefully evaluate the following research proposal and assign a score from 1 to 10.

Evaluation Criteria:
- Clarity: Is the proposal well-written and easy to understand?
- Novelty: Does it introduce new ideas or approaches?
- Technical Rigor: Is the methodology sound and well-justified?
- Potential Impact: Could this research make a significant contribution?

Scoring Guidelines:
- Scores 1-3: Poor quality, major flaws
- Scores 4-5: Below average, significant issues
- Score 6: Slightly above borderline, acceptable
- Scores 7-8: Good quality, solid contribution
- Scores 9-10: Excellent, exceptional contribution

You MUST respond with valid JSON in this exact format:
{{
  "score": <integer 1-10>,
  "explanation": "<your detailed evaluation reasoning>"
}}

Research Proposal:
{proposal}
"""

    # JSON schema for structured score output (2025 best practice)
    SCORE_JSON_SCHEMA = {
        "name": "research_proposal_score",
        "schema": {
            "type": "object",
            "properties": {
                "score": {
                    "type": "integer",
                    "description": "Numerical score from 1 to 10",
                    "minimum": 1,
                    "maximum": 10
                },
                "explanation": {
                    "type": "string",
                    "description": "Detailed reasoning for the score"
                }
            },
            "required": ["score", "explanation"],
            "additionalProperties": False
        },
        "strict": True
    }

    # Summarization prompt for concise proposal summaries
    SUMMARY_PROMPT_TEMPLATE = """Summarize the following research proposal in 1-2 concise sentences.
Focus on the key contribution and approach. Be clear and specific.

Research Proposal:
{proposal}

Provide ONLY the summary, nothing else."""

    FAILURE_ANALYSIS_PROMPT_TEMPLATE = """Analyze why this program modification failed to meet performance expectations.

**Proposal Summary:**
{proposal_summary}

**Parent Code:**
```python
{parent_code}
```

**Modified Code (FAILED):**
```python
{child_code}
```

**Parent Performance:**
{parent_metrics}

**Child Performance (BELOW THRESHOLD):**
{child_metrics}

**Performance Ratio:** {performance_ratio:.1%} of best (threshold: {threshold:.1%})

Analyze the code changes and provide a concise 1-2 sentence technical explanation:
- What specific code change caused the regression?
- What aspect of the implementation was flawed?

Be specific about the CODE, not just metrics. Focus on ROOT CAUSE.

You MUST respond with valid JSON in this exact format:
{{
  "failure_reason": "<your 1-2 sentence technical explanation referencing the code>"
}}"""

    # JSON schema for summary output
    SUMMARY_JSON_SCHEMA = {
        "name": "proposal_summary",
        "schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Concise 1-2 sentence summary of the research proposal"
                }
            },
            "required": ["summary"],
            "additionalProperties": False
        },
        "strict": True
    }

    # JSON schema for failure analysis output
    FAILURE_ANALYSIS_JSON_SCHEMA = {
        "name": "failure_analysis",
        "schema": {
            "type": "object",
            "properties": {
                "failure_reason": {
                    "type": "string",
                    "description": "Concise 1-2 sentence technical explanation of why the program failed"
                }
            },
            "required": ["failure_reason"],
            "additionalProperties": False
        },
        "strict": True
    }

    def __init__(self, config: RewardModelConfig):
        """
        Initialize the RewardModel with OpenRouter API.

        Args:
            config (RewardModelConfig): Configuration with api_key, base_url, and parameters.

        Raises:
            ValueError: If API key or base URL is missing.
        """
        self.config = config

        if not self.config.api_key or not self.config.base_url:
            raise ValueError("API key and base URL are required for OpenRouter API.")

        # Initialize AsyncOpenAI client with timeout and retry configuration
        self.client = AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=60.0,  # 60 second timeout
            max_retries=0,  # We handle retries manually for better control
        )

        # Ensure output directory exists
        if self.config.jsonl_file:
            os.makedirs(os.path.dirname(self.config.jsonl_file) or ".", exist_ok=True)

        logger.info(f"Initialized RewardModel with OpenRouter API: {self.config.base_url}")
        logger.info(f"Model: {self.config.model_name}, Temperature: {self.config.temperature}")

    def parse_score_from_json(self, text: str) -> tuple[float, str]:
        """
        Parse score from JSON structured output (2025 best practice).

        Attempts to parse the response as JSON with 'score' and 'explanation' fields.
        This is the primary parsing method, with regex as fallback.

        Args:
            text (str): Model output containing JSON.

        Returns:
            tuple[float, str]: (score, explanation) where score is -1.0 if parsing fails.
        """
        try:
            # Try to parse as JSON
            data = json.loads(text.strip())

            if isinstance(data, dict) and "score" in data:
                score = float(data["score"])
                explanation = data.get("explanation", text)

                # Validate score range
                if 1 <= score <= 10:
                    logger.debug(f"Successfully parsed JSON score: {score}")
                    return score, explanation
                else:
                    logger.warning(f"JSON score {score} outside valid range [1, 10]")
                    return -1.0, text
            else:
                logger.debug("JSON missing 'score' field")
                return -1.0, text

        except json.JSONDecodeError as e:
            logger.debug(f"Failed to parse JSON: {e}")
            return -1.0, text
        except (ValueError, TypeError) as e:
            logger.debug(f"Invalid score value in JSON: {e}")
            return -1.0, text

    def parse_score_from_text(self, text: str) -> float:
        """
        Parse the numerical score from the model's output text (FALLBACK METHOD).

        Looks for score in \\boxed{X.X} format. This is a fallback for models
        that don't support JSON mode. Primary parsing is via parse_score_from_json().

        Args:
            text (str): Model output containing the score.

        Returns:
            float: Parsed score between 1 and 10, or -1.0 if invalid.
        """
        # Fixed regex pattern: \d+ requires at least one digit before optional decimal
        # This matches: \boxed{7}, \boxed{7.5}, \boxed{10}, etc.
        match = re.search(r'\\boxed\{(\d+\.?\d*)\}', text)
        if match:
            try:
                score = float(match.group(1))
                if 1 <= score <= 10:
                    logger.debug(f"Successfully parsed regex score: {score}")
                    return score
                logger.warning(f"Score {score} outside valid range [1, 10]")
            except ValueError:
                logger.warning(f"Could not parse score from: {match.group(1)}")

        logger.debug(f"No valid score found in text: {text[:200]}...")
        return -1.0

    async def _score_single_proposal(
        self,
        proposal: str,
        title: str = "",
        max_retries: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Score a single research proposal with retry logic and exponential backoff.

        Implements best practices from 2025 research:
        - Exponential backoff for retries
        - Specific error handling for rate limits and timeouts
        - Detailed logging for debugging

        Args:
            proposal (str): The research proposal text to score.
            title (str): Optional title for logging purposes.
            max_retries (int, optional): Override config max_retries.

        Returns:
            Dict containing score, evaluation text, and metadata.
        """
        max_retries = max_retries or self.config.max_retries
        retry_count = 0
        last_error = None

        # Prepare the prompt
        prompt = self.SCORING_PROMPT_TEMPLATE.format(proposal=proposal)
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        while retry_count <= max_retries:
            try:
                logger.debug(f"Scoring attempt {retry_count + 1}/{max_retries + 1} for: {title or 'proposal'}")

                # Use JSON schema for structured output (2025 best practice)
                response = await self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=messages,
                    temperature=self.config.temperature,  # Lower temperature for consistency
                    max_tokens=self.config.max_tokens,
                    top_p=self.config.top_p,
                    response_format={
                        "type": "json_schema",
                        "json_schema": self.SCORE_JSON_SCHEMA
                    }
                )

                output_text = response.choices[0].message.content.strip()

                # Try JSON parsing first (primary method)
                score, explanation = self.parse_score_from_json(output_text)

                # If JSON parsing fails, fall back to regex (for models without JSON support)
                if score == -1.0:
                    logger.debug("JSON parsing failed, trying regex fallback...")
                    score = self.parse_score_from_text(output_text)
                    explanation = output_text

                if score != -1.0:
                    # Valid score obtained
                    logger.info(f"Successfully scored '{title}': {score}/10")
                    return {
                        "title": title,
                        "score": score,
                        "evaluation": explanation,
                        "proposal": proposal,
                        "retries": retry_count,
                        "success": True
                    }

                # Invalid score format, retry
                logger.warning(f"Invalid score format in response for '{title}', retrying...")
                logger.debug(f"Response was: {output_text[:200]}")
                retry_count += 1

            except RateLimitError as e:
                logger.warning(f"Rate limit hit for '{title}': {e}")
                last_error = e
                retry_count += 1

            except APITimeoutError as e:
                logger.warning(f"Timeout for '{title}': {e}")
                last_error = e
                retry_count += 1

            except APIError as e:
                logger.error(f"API error for '{title}': {e}")
                last_error = e
                retry_count += 1

            except Exception as e:
                logger.error(f"Unexpected error for '{title}': {e}")
                last_error = e
                retry_count += 1

            # Exponential backoff before retry (best practice from 2025 research)
            if retry_count <= max_retries:
                backoff_time = self.config.retry_delay * (2 ** (retry_count - 1))
                logger.debug(f"Waiting {backoff_time}s before retry...")
                await asyncio.sleep(backoff_time)

        # All retries exhausted
        logger.error(f"Failed to score '{title}' after {max_retries + 1} attempts. Last error: {last_error}")
        return {
            "title": title,
            "score": -1.0,
            "evaluation": f"Error: {last_error}",
            "proposal": proposal,
            "retries": retry_count,
            "success": False
        }

    async def score_research_proposal(self, data: List[Any]) -> List[Dict]:
        """
        Score one or more research proposals.

        Main entry point for reward model evaluation. Accepts either:
        - List of strings (proposals)
        - List of dicts with 'title', 'abstract', and optional 'gt_score'

        Args:
            data (List[Any]): Proposals to score.

        Returns:
            List[Dict]: Results with scores and evaluations.
        """
        # Normalize input format
        if not data:
            return []

        if isinstance(data[0], str):
            # Convert string list to dict format
            data = [{"title": "", "abstract": proposal, "gt_score": 0} for proposal in data]

        logger.info(f"Scoring {len(data)} research proposals...")

        results = []
        for idx, item in enumerate(data, 1):
            title = item.get("title", f"Proposal {idx}")
            proposal = item.get("abstract", "")

            result = await self._score_single_proposal(
                proposal=proposal,
                title=title
            )

            # Add ground truth score if available
            result["gt_score"] = item.get("gt_score", 0)
            results.append(result)

            # Optional: Write to JSONL file for tracking
            if self.config.jsonl_file and result["success"]:
                await self._write_result_to_jsonl(result)

        success_count = sum(1 for r in results if r["success"])
        logger.info(f"Completed scoring: {success_count}/{len(data)} successful")

        return results

    async def summarize_proposal(self, proposal: List[str]) -> str:
        """
        Generate a concise summary of a research proposal for console display.

        Uses LLM to create a 1-2 sentence summary focusing on key contribution
        and approach. Uses JSON mode for reliable parsing.

        Args:
            proposal (List[str]): Full research proposal text as list of strings.

        Returns:
            str: Concise summary (1-2 sentences), or first 100 chars of proposal if summarization fails.
        """
        # Skip if summarization is disabled
        if not getattr(self.config, 'enable_summarization', True):
            proposal_text = "\n".join(proposal) if isinstance(proposal, list) else str(proposal)
            return proposal_text[:100] + "..." if len(proposal_text) > 100 else proposal_text

        # Format proposal
        proposal_text = "\n".join(proposal) if isinstance(proposal, list) else str(proposal)

        # Skip if proposal is already short
        if len(proposal_text) <= 150:
            return proposal_text

        # Prepare the prompt
        prompt = self.SUMMARY_PROMPT_TEMPLATE.format(proposal=proposal_text)
        messages = [
            {"role": "system", "content": "You are a concise technical summarizer."},
            {"role": "user", "content": prompt}
        ]

        # Get summary tokens limit from config or use default
        summary_max_tokens = getattr(self.config, 'summary_max_tokens', 150)

        # Try with JSON schema first (for models that support it)
        try:
            response = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=0.7,  # Higher temperature for more natural summaries
                max_tokens=summary_max_tokens,
                top_p=0.95,
                response_format={
                    "type": "json_schema",
                    "json_schema": self.SUMMARY_JSON_SCHEMA
                }
            )

            output_text = response.choices[0].message.content.strip()

            # Parse JSON response
            try:
                data = json.loads(output_text)
                if "summary" in data:
                    summary = data["summary"].strip()
                    logger.debug(f"Generated proposal summary: {summary[:50]}...")
                    return summary
            except json.JSONDecodeError:
                logger.debug("Failed to parse summary JSON, using raw output")
                return output_text[:200] if len(output_text) > 200 else output_text

        except Exception as e:
            # If JSON schema not supported, try without it
            if "structured outputs not support" in str(e) or "400" in str(e):
                logger.debug("Model doesn't support JSON schema, trying plain mode")
                try:
                    response = await self.client.chat.completions.create(
                        model=self.config.model_name,
                        messages=messages,
                        temperature=0.7,
                        max_tokens=summary_max_tokens,
                        top_p=0.95
                    )

                    output_text = response.choices[0].message.content
                    if output_text:
                        output_text = output_text.strip()
                    logger.debug(f"Raw response: {repr(output_text)}")

                    if not output_text:
                        logger.debug("Empty response from model, using fallback")
                        return proposal_text[:100] + "..." if len(proposal_text) > 100 else proposal_text

                    logger.debug(f"Generated proposal summary (plain mode): {output_text[:50]}...")

                    # Return raw output, limiting length
                    return output_text[:250] if len(output_text) > 250 else output_text

                except Exception as e2:
                    logger.debug(f"Plain mode also failed: {e2}")
            else:
                logger.debug(f"Failed to generate summary: {e}")

        # Fallback to truncated proposal
        return proposal_text[:100] + "..." if len(proposal_text) > 100 else proposal_text

    async def explain_failure(
        self,
        proposal_summary: str,
        parent_code: str,
        child_code: str,
        parent_metrics: Dict[str, float],
        child_metrics: Dict[str, float],
        performance_ratio: float,
        threshold: float
    ) -> str:
        """
        Generate a concise explanation of why a program failed using LLM code analysis.

        Uses the reward model to analyze the actual code changes and performance
        regression to provide specific technical insights about what went wrong.

        Args:
            proposal_summary: Concise summary of the proposal that was implemented
            parent_code: Parent program's source code
            child_code: Child program's source code (failed)
            parent_metrics: Parent program's performance metrics
            child_metrics: Child program's performance metrics
            performance_ratio: Child score / best score (e.g., 0.82 = 82%)
            threshold: Minimum acceptable ratio (e.g., 0.85 = 85%)

        Returns:
            str: Concise 1-2 sentence code-specific explanation, or generic message if analysis fails
        """
        # Format metrics for readability
        parent_metrics_str = "\n".join([f"- {k}: {v:.4f}" if isinstance(v, float) else f"- {k}: {v}"
                                       for k, v in parent_metrics.items()])
        child_metrics_str = "\n".join([f"- {k}: {v:.4f}" if isinstance(v, float) else f"- {k}: {v}"
                                      for k, v in child_metrics.items()])

        # Truncate code if too long (keep first 100 lines)
        parent_code_lines = parent_code.split("\n")[:100]
        child_code_lines = child_code.split("\n")[:100]
        parent_code_truncated = "\n".join(parent_code_lines)
        child_code_truncated = "\n".join(child_code_lines)

        if len(parent_code.split("\n")) > 100:
            parent_code_truncated += "\n# ... (truncated)"
        if len(child_code.split("\n")) > 100:
            child_code_truncated += "\n# ... (truncated)"

        # Prepare the prompt with code
        prompt = self.FAILURE_ANALYSIS_PROMPT_TEMPLATE.format(
            proposal_summary=proposal_summary,
            parent_code=parent_code_truncated,
            child_code=child_code_truncated,
            parent_metrics=parent_metrics_str,
            child_metrics=child_metrics_str,
            performance_ratio=performance_ratio,
            threshold=threshold
        )

        messages = [
            {"role": "system", "content": "You are an expert code reviewer analyzing performance regressions."},
            {"role": "user", "content": prompt}
        ]

        # Try with JSON schema first (for models that support it)
        try:
            response = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=0.5,  # Moderate temperature for analytical reasoning
                max_tokens=200,  # Short explanation
                top_p=0.95,
                response_format={
                    "type": "json_schema",
                    "json_schema": self.FAILURE_ANALYSIS_JSON_SCHEMA
                }
            )

            output_text = response.choices[0].message.content.strip()

            # Parse JSON response
            try:
                data = json.loads(output_text)
                if "failure_reason" in data:
                    reason = data["failure_reason"].strip()
                    logger.debug(f"Generated failure analysis: {reason[:80]}...")
                    return reason
            except json.JSONDecodeError:
                logger.debug("Failed to parse failure analysis JSON, using raw output")
                return output_text[:250] if len(output_text) > 250 else output_text

        except Exception as e:
            # If JSON schema not supported, try without it
            if "structured outputs not support" in str(e) or "400" in str(e):
                logger.debug("Model doesn't support JSON schema for failure analysis, trying plain mode")
                try:
                    response = await self.client.chat.completions.create(
                        model=self.config.model_name,
                        messages=messages,
                        temperature=0.5,
                        max_tokens=200,
                        top_p=0.95
                    )

                    output_text = response.choices[0].message.content
                    if output_text:
                        output_text = output_text.strip()
                        logger.debug(f"Generated failure analysis (plain mode): {output_text[:80]}...")
                        return output_text[:250] if len(output_text) > 250 else output_text

                except Exception as e2:
                    logger.warning(f"Failed to generate failure analysis (plain mode): {e2}")
            else:
                logger.warning(f"Failed to generate failure analysis: {e}")

        # Fallback to generic reason
        return f"Performance dropped to {performance_ratio:.1%} of parent, below {threshold:.1%} threshold"

    async def _write_result_to_jsonl(self, result: Dict) -> None:
        """
        Write a scoring result to JSONL file for tracking.

        Args:
            result (Dict): Result dictionary to write.
        """
        try:
            # Use synchronous write for simplicity (can be upgraded to aiofiles if needed)
            with open(self.config.jsonl_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.warning(f"Failed to write result to JSONL: {e}")


if __name__ == "__main__":
    async def test_reward_model():
        """Test the reward model with sample proposals."""
        # Sample data for testing
        sample_data = [
            {
                "title": "A Novel Approach to Quantum Computing",
                "abstract": "This proposal introduces a new quantum algorithm that enhances computational efficiency by leveraging entangled states in a scalable architecture. The approach is validated through simulations showing a 20% improvement over existing methods.",
                "gt_score": 7.5
            },
            {
                "title": "AI-Driven Climate Modeling",
                "abstract": "We propose an AI-based framework for improving climate predictions using deep learning to integrate heterogeneous environmental data. Preliminary results demonstrate enhanced accuracy in long-term forecasts.",
                "gt_score": 8.0
            }
        ]

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Test with OpenRouter API
        try:
            import os
            from dotenv import load_dotenv

            load_dotenv()
            api_key = os.getenv("OPENROUTER_API_KEY")

            if not api_key:
                print("ERROR: OPENROUTER_API_KEY not found in environment")
                return

            config = RewardModelConfig(
                model_type="api",
                model_name="qwen/qwen-2.5-7b-instruct",  # Fast and cost-effective
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                temperature=0.3,  # Lower temperature for consistency
                max_tokens=1000,
                jsonl_file="reward_results.jsonl",
                max_retries=3,
                retry_delay=2
            )

            model = RewardModel(config)
            results = await model.score_research_proposal(sample_data)

            print("\n" + "=" * 80)
            print("Reward Model Results:")
            print("=" * 80)
            for result in results:
                print(f"\nTitle: {result['title']}")
                print(f"Score: {result['score']}/10 (Ground Truth: {result['gt_score']})")
                print(f"Success: {result['success']}, Retries: {result['retries']}")
                print(f"Evaluation: {result['evaluation'][:200]}...")
            print("=" * 80)

        except Exception as e:
            print(f"Test failed: {e}")
            import traceback
            traceback.print_exc()

    # Run the test
    asyncio.run(test_reward_model())
