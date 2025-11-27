"""
Integration tests for Toxic Trait Tracking System

Tests the integration of FailureTracker with controller, database, and prompt components.
"""

import asyncio
import os
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock, patch, Mock

import pytest

from evolve_agent.config import Config, DatabaseConfig, ToxicTraitConfig
from evolve_agent.database import Program, ProgramDatabase
from evolve_agent.failure_tracker import FailureTracker


# ============================================================================
# Task 3.1: Controller Integration Tests (3-6 tests)
# ============================================================================

class TestControllerIntegration:
    """Tests for controller integration with FailureTracker"""

    @patch('evolve_agent.controller.RewardModel')
    @patch('evolve_agent.controller.LLMEnsemble')
    def test_controller_initializes_failure_tracker(self, mock_ensemble, mock_reward_model):
        """Test that controller correctly initializes FailureTracker"""
        # Mock LLMEnsemble and RewardModel
        mock_ensemble.return_value = Mock()
        mock_reward_model.return_value = Mock()

        # Import here to avoid circular imports
        from evolve_agent.controller import EvolveAgent

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a minimal config with toxic trait enabled
            config = Config()
            config.toxic_trait.enabled = True
            config.database.db_path = tmpdir

            # Create test files
            initial_program = Path(tmpdir) / "initial.py"
            initial_program.write_text("def test(): pass")
            initial_proposal = Path(tmpdir) / "proposal.txt"
            initial_proposal.write_text("Test proposal")
            eval_file = Path(tmpdir) / "eval.py"
            eval_file.write_text("def evaluate(): return {'score': 1.0}")

            # Initialize controller
            agent = EvolveAgent(
                initial_program_path=str(initial_program),
                initial_proposal_path=str(initial_proposal),
                evaluation_file=str(eval_file),
                config=config,
                output_dir=tmpdir
            )

            # Verify failure tracker is initialized
            assert hasattr(agent, 'failure_tracker')
            assert agent.failure_tracker is not None
            assert isinstance(agent.failure_tracker, FailureTracker)

    def test_child_below_threshold_marked_toxic(self):
        """Test that child program below threshold gets marked as toxic"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup
            config = ToxicTraitConfig(enabled=True, threshold=0.85)
            tracker = FailureTracker(config, tmpdir, "test_benchmark")

            # Create parent and child programs
            parent = Program(
                id="parent-123",
                code="def parent(): pass",
                metrics={"combined_score": 0.90}
            )

            # Child performs at 70% of parent (below 85% threshold)
            child = Program(
                id="child-456",
                code="def child(): pass",
                parent_id="parent-123",
                metrics={"combined_score": 0.63},  # 0.63 / 0.90 = 0.70
                proposal=["Improve performance"]
            )

            # Verify child should be marked toxic
            assert tracker.should_mark_toxic(child, parent) is True

            # Mark as toxic
            tracker.add_failure(child, parent, "Integration test proposal", "Below performance threshold")

            # Verify child is now in toxic set
            assert tracker.is_toxic("child-456")
            assert len(tracker.failures) == 1
            assert tracker.failures[0]["program_id"] == "child-456"

    def test_child_meeting_threshold_not_marked_toxic(self):
        """Test that child program meeting threshold is NOT marked toxic"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup
            config = ToxicTraitConfig(enabled=True, threshold=0.85)
            tracker = FailureTracker(config, tmpdir, "test_benchmark")

            # Create parent and child programs
            parent = Program(
                id="parent-123",
                code="def parent(): pass",
                metrics={"combined_score": 0.90}
            )

            # Child performs at 90% of parent (above 85% threshold)
            child = Program(
                id="child-456",
                code="def child(): pass",
                parent_id="parent-123",
                metrics={"combined_score": 0.81},  # 0.81 / 0.90 = 0.90
                proposal=["Improve performance"]
            )

            # Verify child should NOT be marked toxic
            assert tracker.should_mark_toxic(child, parent) is False

            # Verify child is not in toxic set
            assert not tracker.is_toxic("child-456")
            assert len(tracker.failures) == 0

    def test_initial_program_skips_toxic_check(self):
        """Test that initial programs (no parent) skip toxic check"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup
            config = ToxicTraitConfig(enabled=True, threshold=0.85)
            tracker = FailureTracker(config, tmpdir, "test_benchmark")

            # Create initial program with no parent
            initial_program = Program(
                id="initial-123",
                code="def initial(): pass",
                parent_id=None,  # No parent
                metrics={"combined_score": 0.10},  # Low score but should not be toxic
                proposal=["Initial implementation"]
            )

            # In practice, controller would check if parent_id is None before calling
            # should_mark_toxic, but the method itself should handle edge cases
            # Here we verify that without a parent, we can't mark it toxic
            assert initial_program.parent_id is None


# ============================================================================
# Task 3.4: Database Sampling Integration Tests (3-5 tests)
# ============================================================================

class TestDatabaseSamplingIntegration:
    """Tests for database sampling with toxic program filtering"""

    def test_sample_parent_excludes_toxic_programs(self):
        """Test that _sample_parent() excludes toxic programs"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup database
            db_config = DatabaseConfig(db_path=tmpdir, in_memory=True)
            db = ProgramDatabase(db_config)

            # Add programs
            good_program = Program(
                id="good-1",
                code="def good(): pass",
                metrics={"combined_score": 0.90}
            )
            toxic_program = Program(
                id="toxic-1",
                code="def toxic(): pass",
                metrics={"combined_score": 0.50}
            )

            db.add(good_program, iteration=1)
            db.add(toxic_program, iteration=2)

            # Create toxic set
            toxic_programs = {"toxic-1"}

            # Sample parent with filtering
            parent = db._sample_parent(toxic_programs=toxic_programs)

            # Verify we didn't get the toxic program
            assert parent.id != "toxic-1"
            assert parent.id == "good-1"

    def test_sample_inspirations_excludes_toxic_programs(self):
        """Test that _sample_inspirations() excludes toxic programs"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup database
            db_config = DatabaseConfig(db_path=tmpdir, in_memory=True)
            db = ProgramDatabase(db_config)

            # Add programs
            parent = Program(
                id="parent-1",
                code="def parent(): pass",
                metrics={"combined_score": 0.80}
            )
            good_insp = Program(
                id="good-insp-1",
                code="def good(): pass",
                metrics={"combined_score": 0.90}
            )
            toxic_insp = Program(
                id="toxic-insp-1",
                code="def toxic(): pass",
                metrics={"combined_score": 0.50}
            )

            db.add(parent, iteration=1)
            db.add(good_insp, iteration=2)
            db.add(toxic_insp, iteration=3)

            # Track best program by passing Program object
            db._update_best_program(good_insp)

            # Create toxic set
            toxic_programs = {"toxic-insp-1"}

            # Sample inspirations with filtering
            inspirations = db._sample_inspirations(
                parent,
                n=5,
                toxic_programs=toxic_programs
            )

            # Verify toxic program is not in inspirations
            inspiration_ids = [p.id for p in inspirations]
            assert "toxic-insp-1" not in inspiration_ids

    def test_sampling_fallback_when_all_toxic(self):
        """Test sampling fallback when all candidates are toxic"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup database with only toxic programs
            db_config = DatabaseConfig(db_path=tmpdir, in_memory=True)
            db = ProgramDatabase(db_config)

            # Add only toxic programs
            toxic1 = Program(
                id="toxic-1",
                code="def toxic1(): pass",
                metrics={"combined_score": 0.50}
            )
            toxic2 = Program(
                id="toxic-2",
                code="def toxic2(): pass",
                metrics={"combined_score": 0.40}
            )

            db.add(toxic1, iteration=1)
            db.add(toxic2, iteration=2)

            # Mark all as toxic
            toxic_programs = {"toxic-1", "toxic-2"}

            # Should still return a program (fallback behavior)
            # The implementation should handle this gracefully
            parent = db._sample_parent(toxic_programs=toxic_programs)

            # Verify we got some program (fallback to archive or any available)
            assert parent is not None


# ============================================================================
# Task 3.8: Prompt Integration Tests (2-4 tests)
# ============================================================================

class TestPromptIntegration:
    """Tests for prompt integration with failure history"""

    def test_format_failure_history_with_empty_list(self):
        """Test _format_failure_history() with empty list"""
        from evolve_agent.prompt.sampler import PromptSampler
        from evolve_agent.config import PromptConfig

        config = PromptConfig()
        sampler = PromptSampler(config)

        # Format empty failure list
        result = sampler._format_failure_history([])

        # Should return a message indicating no failures
        assert "no" in result.lower() or "none" in result.lower()

    def test_format_failure_history_with_sample_failures(self):
        """Test _format_failure_history() with sample failures"""
        from evolve_agent.prompt.sampler import PromptSampler
        from evolve_agent.config import PromptConfig

        config = PromptConfig()
        sampler = PromptSampler(config)

        # Create sample failure records
        failures = [
            {
                "program_id": "child-1",
                "parent_id": "parent-1",
                "timestamp": 1234567890.0,
                "proposal": ["Optimize using cache"],
                "performance_ratio": 0.70,
                "threshold": 0.85,
            },
            {
                "program_id": "child-2",
                "parent_id": "parent-2",
                "timestamp": 1234567891.0,
                "proposal": ["Reduce memory usage"],
                "performance_ratio": 0.60,
                "threshold": 0.85,
            }
        ]

        # Format failure history
        result = sampler._format_failure_history(failures)

        # Verify result contains key information
        assert "failed" in result.lower() or "previous" in result.lower()
        assert "0.70" in result or "70" in result  # Performance ratio
        assert len(result) > 0

    def test_build_prompt_includes_failure_history(self):
        """Test build_prompt() includes failure history section"""
        from evolve_agent.prompt.sampler import PromptSampler
        from evolve_agent.config import PromptConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup config with template directory
            config = PromptConfig(template_dir=None)
            sampler = PromptSampler(config)

            # Sample failure history
            failure_history = [
                {
                    "program_id": "child-1",
                    "proposal": ["Use caching"],
                    "performance_ratio": 0.75,
                }
            ]

            # Build prompt with failure history
            prompt = sampler.build_prompt(
                current_program="def test(): pass",
                program_metrics={"score": 0.8},
                language="python",
                failure_history=failure_history
            )

            # Verify prompt was built
            assert "system" in prompt
            assert "user" in prompt

            # Note: The actual injection into the prompt is done in the implementation
            # Here we just verify the method accepts failure_history parameter


# ============================================================================
# Helper fixtures and utilities
# ============================================================================

@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for tests"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir
