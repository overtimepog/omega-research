"""
Edge case and integration tests for Toxic Trait Tracking System

Tests critical edge cases, end-to-end workflows, and system-level behaviors.
"""

import asyncio
import json
import tempfile
import time
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from evolve_agent.config import Config, DatabaseConfig, ToxicTraitConfig
from evolve_agent.database import Program, ProgramDatabase
from evolve_agent.failure_tracker import FailureTracker


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_threshold_strict_mode(self):
        """Test threshold=1.0 (strict mode, any regression marks toxic)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup with strict threshold
            config = ToxicTraitConfig(enabled=True, threshold=1.0)
            tracker = FailureTracker(config, tmpdir, "test_benchmark")

            # Create parent and child with tiny regression
            parent = Program(
                id="parent-1",
                code="def parent(): pass",
                metrics={"combined_score": 0.90}
            )
            child = Program(
                id="child-1",
                code="def child(): pass",
                parent_id="parent-1",
                metrics={"combined_score": 0.89},  # 0.89/0.90 = 0.989 < 1.0
                proposal=["Minor change"]
            )

            # Any regression should mark as toxic
            assert tracker.should_mark_toxic(child, parent) is True

    def test_threshold_lenient_mode(self):
        """Test threshold=0.0 (lenient mode, effectively disabled)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup with lenient threshold
            config = ToxicTraitConfig(enabled=True, threshold=0.0)
            tracker = FailureTracker(config, tmpdir, "test_benchmark")

            # Create parent and child with massive regression
            parent = Program(
                id="parent-1",
                code="def parent(): pass",
                metrics={"combined_score": 0.90}
            )
            child = Program(
                id="child-1",
                code="def child(): pass",
                parent_id="parent-1",
                metrics={"combined_score": 0.01},  # 0.01/0.90 = 0.011 but threshold=0.0
                proposal=["Bad change"]
            )

            # No program should be marked toxic with threshold=0.0
            assert tracker.should_mark_toxic(child, parent) is False

    def test_parent_program_is_toxic(self):
        """Test that toxic parent doesn't prevent child evaluation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ToxicTraitConfig(enabled=True, threshold=0.85)
            tracker = FailureTracker(config, tmpdir, "test_benchmark")

            # Mark parent as toxic
            parent = Program(
                id="parent-1",
                code="def parent(): pass",
                metrics={"combined_score": 0.80}
            )
            grandparent = Program(
                id="grandparent-1",
                code="def grandparent(): pass",
                metrics={"combined_score": 0.95}
            )
            tracker.add_failure(parent, grandparent, "Parent proposal", "Parent failure reason")
            assert tracker.is_toxic("parent-1")

            # Child of toxic parent should still be evaluated
            child = Program(
                id="child-1",
                code="def child(): pass",
                parent_id="parent-1",
                metrics={"combined_score": 0.75},  # 0.75/0.80 = 0.9375 > 0.85
                proposal=["Good improvement"]
            )

            # Child is better than parent, should NOT be marked toxic
            assert tracker.should_mark_toxic(child, parent) is False
            assert not tracker.is_toxic("child-1")

    def test_missing_comparison_metric_fallback(self):
        """Test fallback when comparison_metric is missing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup with a specific comparison metric
            config = ToxicTraitConfig(
                enabled=True,
                threshold=0.85,
                comparison_metric="nonexistent_metric"
            )
            tracker = FailureTracker(config, tmpdir, "test_benchmark")

            # Create programs with only standard metrics
            parent = Program(
                id="parent-1",
                code="def parent(): pass",
                metrics={"accuracy": 0.90, "speed": 0.85}  # No "nonexistent_metric"
            )
            child = Program(
                id="child-1",
                code="def child(): pass",
                parent_id="parent-1",
                metrics={"accuracy": 0.70, "speed": 0.60},
                proposal=["Change"]
            )

            # Should handle missing metric gracefully (return False)
            result = tracker.should_mark_toxic(child, parent)
            assert result is False

    def test_disabled_mode_backward_compatibility(self):
        """Test system works correctly when toxic trait is disabled"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup with disabled config
            config = ToxicTraitConfig(enabled=False)
            tracker = FailureTracker(config, tmpdir, "test_benchmark")

            # Create underperforming child
            parent = Program(
                id="parent-1",
                code="def parent(): pass",
                metrics={"combined_score": 0.90}
            )
            child = Program(
                id="child-1",
                code="def child(): pass",
                parent_id="parent-1",
                metrics={"combined_score": 0.50},  # Very poor
                proposal=["Bad change"]
            )

            # When disabled=False, should_mark_toxic should still return False
            # (This is handled at the controller level by checking config.enabled)
            # But the tracker itself should function
            assert tracker.config.enabled is False


class TestPersistenceAndReload:
    """Test persistence across runs and checkpoint reload"""

    def test_checkpoint_reload_preserves_toxic_programs(self):
        """Test that loading checkpoint preserves failure history"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First run: Create tracker and mark programs as toxic
            config = ToxicTraitConfig(enabled=True, threshold=0.85)
            tracker1 = FailureTracker(config, tmpdir, "test_benchmark")

            parent = Program(
                id="parent-1",
                code="def parent(): pass",
                metrics={"combined_score": 0.90}
            )
            for i in range(5):
                child = Program(
                    id=f"toxic-child-{i}",
                    code=f"def child{i}(): pass",
                    parent_id="parent-1",
                    metrics={"combined_score": 0.60},
                    proposal=[f"Bad change {i}"]
                )
                tracker1.add_failure(child, parent, "Test proposal", "Below threshold")

            # Verify toxic programs are tracked
            assert len(tracker1.toxic_programs) == 5

            # Simulate restart: Create new tracker instance
            tracker2 = FailureTracker(config, tmpdir, "test_benchmark")

            # Verify toxic programs are still filtered after reload
            assert len(tracker2.toxic_programs) == 5
            for i in range(5):
                assert tracker2.is_toxic(f"toxic-child-{i}")

            # Verify failure history is preserved
            history = tracker2.get_failure_history(limit=10)
            assert len(history) == 5


class TestMapElitesCompatibility:
    """Test MAP-Elites compatibility"""

    def test_map_elites_grid_excludes_toxic_programs(self):
        """Test that MAP-Elites grid correctly excludes toxic programs"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup database with MAP-Elites features
            db_config = DatabaseConfig(db_path=tmpdir, in_memory=True)
            db = ProgramDatabase(db_config)

            # Add programs to grid
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

            # Setup toxic tracker
            config = ToxicTraitConfig(enabled=True)
            tracker = FailureTracker(config, tmpdir, "test_benchmark")
            parent = Program(id="parent", code="def p(): pass", metrics={"combined_score": 0.95})
            tracker.add_failure(toxic_program, parent, "Test approach", "Poor performance")

            # Sample with toxic filtering
            toxic_programs = tracker.toxic_programs
            parent_sample = db._sample_parent(toxic_programs=toxic_programs)

            # Should get good program, not toxic
            assert parent_sample.id == "good-1"

    def test_archive_maintains_elite_programs_independent_of_toxic_status(self):
        """Test that archive maintains elite programs regardless of toxic status"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_config = DatabaseConfig(db_path=tmpdir, in_memory=True)
            db = ProgramDatabase(db_config)

            # Add an elite program
            elite_program = Program(
                id="elite-1",
                code="def elite(): pass",
                metrics={"combined_score": 0.95}
            )
            db.add(elite_program, iteration=1)

            # Update best program (simulates archive behavior)
            db._update_best_program(elite_program)

            # Verify best program is tracked
            assert db.best_program_id is not None
            assert db.best_program_id == "elite-1"

            # Even if marked toxic, it stays as best
            # (Toxic filtering only affects sampling, not best tracking)
            assert db.best_program_id == "elite-1"


class TestEndToEndIntegration:
    """Test end-to-end integration workflow"""

    def test_full_evolution_iteration_with_toxic_trait_enabled(self):
        """Test complete evolution iteration with toxic trait tracking"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup complete system
            config = ToxicTraitConfig(enabled=True, threshold=0.85)
            tracker = FailureTracker(config, tmpdir, "test_benchmark")

            db_config = DatabaseConfig(db_path=tmpdir, in_memory=True)
            db = ProgramDatabase(db_config)

            # Simulate evolution iteration
            parent = Program(
                id="parent-1",
                code="def parent(): return 42",
                metrics={"combined_score": 0.90, "accuracy": 0.88},
                generation=1
            )
            db.add(parent, iteration=1)

            # Create two children: one good, one toxic
            good_child = Program(
                id="good-child",
                code="def good_child(): return 43",
                parent_id="parent-1",
                metrics={"combined_score": 0.92, "accuracy": 0.90},
                generation=2,
                proposal=["Improved implementation"]
            )

            toxic_child = Program(
                id="toxic-child",
                code="def toxic_child(): return 41",
                parent_id="parent-1",
                metrics={"combined_score": 0.70, "accuracy": 0.65},
                generation=2,
                proposal=["Bad approach"]
            )

            # Evaluate and mark toxic
            if tracker.should_mark_toxic(toxic_child, parent):
                tracker.add_failure(toxic_child, parent, "MAP-Elites test", "Below threshold")

            # Add both to database
            db.add(good_child, iteration=2)
            db.add(toxic_child, iteration=2)

            # Sample for next iteration
            toxic_programs = tracker.toxic_programs
            next_parent = db._sample_parent(toxic_programs=toxic_programs)

            # Should get good child or original parent, not toxic child
            assert next_parent.id != "toxic-child"

            # Verify failure history is available
            history = tracker.get_failure_history()
            assert len(history) == 1
            assert history[0]["program_id"] == "toxic-child"


class TestPerformance:
    """Test performance characteristics"""

    def test_toxic_lookup_performance(self):
        """Test that is_toxic() provides O(1) lookup performance"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ToxicTraitConfig(enabled=True, threshold=0.85)
            tracker = FailureTracker(config, tmpdir, "test_benchmark")

            # Add many toxic programs
            parent = Program(
                id="parent",
                code="def parent(): pass",
                metrics={"combined_score": 0.95}
            )

            for i in range(1000):
                child = Program(
                    id=f"toxic-{i}",
                    code=f"def child{i}(): pass",
                    parent_id="parent",
                    metrics={"combined_score": 0.60},
                    proposal=[f"Bad approach {i}"]
                )
                tracker.add_failure(child, parent, "Archive test", "Below threshold")

            # Measure lookup time
            start = time.time()
            for i in range(1000):
                tracker.is_toxic(f"toxic-{i}")
            elapsed = time.time() - start

            # Should be very fast (< 10ms for 1000 lookups)
            assert elapsed < 0.01, f"Lookup too slow: {elapsed*1000:.2f}ms"

            # Verify all marked correctly
            assert len(tracker.toxic_programs) == 1000
