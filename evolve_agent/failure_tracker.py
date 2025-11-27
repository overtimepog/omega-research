"""
Failure Tracker for Toxic Trait Tracking System

This module implements negative selection pressure for evolutionary code optimization
by tracking underperforming programs ("toxic traits") and excluding them from breeding.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

from evolve_agent.config import ToxicTraitConfig
from evolve_agent.database import Program

logger = logging.getLogger(__name__)


class FailureTracker:
    """
    Tracks failed programs (toxic traits) and maintains failure history.

    The FailureTracker marks programs that perform significantly worse than the
    CURRENT BEST as "toxic" and excludes them from parent/inspiration sampling,
    reducing wasted compute on poor solution spaces.

    Attributes:
        config: ToxicTraitConfig instance with tracking parameters
        benchmark_name: Name of the benchmark being optimized
        failure_file_path: Path to JSON file storing failure history
        toxic_programs: In-memory set of toxic program IDs for O(1) lookup
        failures: List of failure records with metadata
    """

    def __init__(
        self,
        config: ToxicTraitConfig,
        db_path: str,
        benchmark_name: str
    ):
        """
        Initialize FailureTracker.

        Args:
            config: ToxicTraitConfig instance with threshold and settings
            db_path: Base path for database storage
            benchmark_name: Name of benchmark (used in filename)
        """
        self.config = config
        self.benchmark_name = benchmark_name

        # In-memory set for O(1) toxic program lookup
        self.toxic_programs: Set[str] = set()

        # List of failure records with metadata
        self.failures: List[Dict] = []

        # Set up failure file path
        base_path = Path(db_path)
        failures_dir = base_path / "failures"
        failures_dir.mkdir(parents=True, exist_ok=True)

        self.failure_file_path = failures_dir / f"{benchmark_name}_failures.json"

        # Load existing failures from disk
        self.load()

    def add_failure(
        self,
        child_program: Program,
        parent_program: Program,
        proposal_summary: str,
        failure_reason: str
    ) -> None:
        """
        Record a program as toxic and persist to disk.

        Args:
            child_program: The underperforming child program
            parent_program: The parent program used for comparison
            proposal_summary: Concise summary of the proposal from reward model
            failure_reason: LLM-generated explanation of why the program failed
        """
        # Calculate performance ratio
        comparison_metric = self.config.comparison_metric
        parent_score = parent_program.metrics.get(comparison_metric, 0.0)
        child_score = child_program.metrics.get(comparison_metric, 0.0)

        # Handle division by zero
        if parent_score > 0:
            performance_ratio = child_score / parent_score
        else:
            performance_ratio = 0.0

        # Build failure record (includes code for analysis)
        failure_record = {
            "program_id": child_program.id,
            "parent_id": parent_program.id,
            "timestamp": time.time(),
            "iteration": getattr(child_program, 'iteration_found', 0),
            "proposal_summary": proposal_summary,
            "failure_reason": failure_reason,
            "child_code": child_program.code,  # Store failing code for analysis
            "parent_code": parent_program.code,  # Store parent code for comparison
            "parent_metrics": parent_program.metrics,
            "child_metrics": child_program.metrics,
            "performance_ratio": performance_ratio,
            "threshold": self.config.threshold,
            "comparison_metric": comparison_metric
        }

        # Add to in-memory structures
        self.toxic_programs.add(child_program.id)
        self.failures.append(failure_record)

        # Persist to disk immediately
        self.save()

        # Log the toxic marking with failure analysis
        logger.warning(
            f"Marked program {child_program.id} as toxic: {failure_reason} "
            f"[ratio={performance_ratio:.3f} < threshold={self.config.threshold:.3f}]"
        )

    def is_toxic(self, program_id: str) -> bool:
        """
        Check if a program is marked as toxic.

        Args:
            program_id: Program ID to check

        Returns:
            True if program is toxic, False otherwise

        Note:
            This is an O(1) lookup using in-memory set.
        """
        return program_id in self.toxic_programs

    def get_failure_history(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Get recent failure records for LLM prompts.

        Args:
            limit: Maximum number of failures to return. Defaults to
                   config.max_failures_in_prompt if not specified.

        Returns:
            List of failure records, most recent first
        """
        if limit is None:
            limit = self.config.max_failures_in_prompt

        # Return most recent failures (sorted by timestamp descending)
        sorted_failures = sorted(
            self.failures,
            key=lambda f: f.get("timestamp", 0),
            reverse=True
        )

        return sorted_failures[:limit]

    def should_mark_toxic(
        self,
        child_program: Program,
        best_program: Program
    ) -> bool:
        """
        Determine if a child program should be marked as toxic.

        Compares child performance to the CURRENT BEST program using the configured
        metric and threshold. This ensures the baseline keeps rising as better
        solutions are found (dynamic threshold).

        Args:
            child_program: Child program to evaluate
            best_program: Current best program in population (dynamic baseline)

        Returns:
            True if child should be marked toxic, False otherwise
        """
        comparison_metric = self.config.comparison_metric

        # Check if metrics exist (not just defaulting to 0.0)
        if comparison_metric not in best_program.metrics or comparison_metric not in child_program.metrics:
            logger.debug(
                f"Skipping toxic check for {child_program.id}: "
                f"missing comparison metric '{comparison_metric}'"
            )
            return False

        # Get best and child scores (compare against BEST, not parent)
        best_score = best_program.metrics.get(comparison_metric, 0.0)
        child_score = child_program.metrics.get(comparison_metric, 0.0)

        # Handle missing metrics gracefully
        if not isinstance(best_score, (int, float)) or not isinstance(child_score, (int, float)):
            logger.debug(
                f"Skipping toxic check for {child_program.id}: "
                f"non-numeric metrics (best={best_score}, child={child_score})"
            )
            return False

        # Handle division by zero (no baseline yet)
        if best_score == 0:
            logger.debug(
                f"Skipping toxic check for {child_program.id}: best score is zero (no baseline)"
            )
            return False

        # Calculate performance ratio against CURRENT BEST (dynamic baseline)
        performance_ratio = child_score / best_score

        # Check if below threshold
        is_toxic = performance_ratio < self.config.threshold

        if is_toxic:
            logger.debug(
                f"Program {child_program.id} marked toxic: "
                f"{child_score:.4f} < {self.config.threshold:.1%} * {best_score:.4f} "
                f"(ratio={performance_ratio:.3f})"
            )

        return is_toxic

    def save(self) -> None:
        """
        Save failure records to JSON file with atomic write.

        Uses temp file + rename pattern for crash safety.
        """
        try:
            # Write to temp file first
            temp_path = self.failure_file_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(self.failures, f, indent=2)

            # Atomic rename
            temp_path.replace(self.failure_file_path)

            logger.debug(f"Saved {len(self.failures)} failures to {self.failure_file_path}")

        except Exception as e:
            logger.warning(f"Failed to save failures: {e}")

    def load(self) -> None:
        """
        Load failure records from JSON file.

        Gracefully handles missing or corrupt files.
        """
        if not self.failure_file_path.exists():
            logger.debug(f"No existing failure file at {self.failure_file_path}")
            return

        try:
            with open(self.failure_file_path, 'r') as f:
                self.failures = json.load(f)

            # Rebuild toxic programs set from failures
            self.toxic_programs = {f["program_id"] for f in self.failures}

            logger.info(
                f"Loaded {len(self.failures)} failures from {self.failure_file_path}"
            )

        except json.JSONDecodeError as e:
            logger.warning(
                f"Corrupt failure file at {self.failure_file_path}: {e}. "
                f"Starting with empty state."
            )
            self.failures = []
            self.toxic_programs = set()

        except Exception as e:
            logger.warning(f"Failed to load failures: {e}. Starting with empty state.")
            self.failures = []
            self.toxic_programs = set()
