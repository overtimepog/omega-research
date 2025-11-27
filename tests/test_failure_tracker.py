"""
Tests for FailureTracker class
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from evolve_agent.config import ToxicTraitConfig
from evolve_agent.database import Program
from evolve_agent.failure_tracker import FailureTracker


class TestFailureTracker:
    """Test suite for FailureTracker class"""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def config(self):
        """Create a default ToxicTraitConfig for testing"""
        return ToxicTraitConfig(
            enabled=True,
            threshold=0.85,
            comparison_metric="combined_score",
            max_failures_in_prompt=10
        )

    @pytest.fixture
    def parent_program(self):
        """Create a parent program with good metrics"""
        return Program(
            id="parent_123",
            code="def solution(): return 42",
            metrics={"combined_score": 0.90, "accuracy": 0.85},
            generation=1
        )

    @pytest.fixture
    def child_program_below_threshold(self, parent_program):
        """Create a child program below performance threshold"""
        return Program(
            id="child_456",
            code="def solution(): return 41",
            parent_id=parent_program.id,
            metrics={"combined_score": 0.70, "accuracy": 0.65},  # 70/90 = 0.778 < 0.85
            generation=2
        )

    @pytest.fixture
    def child_program_above_threshold(self, parent_program):
        """Create a child program above performance threshold"""
        return Program(
            id="child_789",
            code="def solution(): return 43",
            parent_id=parent_program.id,
            metrics={"combined_score": 0.88, "accuracy": 0.87},  # 88/90 = 0.978 > 0.85
            generation=2
        )

    def test_initialization_and_file_path_setup(self, temp_dir, config):
        """Test FailureTracker initialization and file path setup"""
        tracker = FailureTracker(config, temp_dir, "test_benchmark")

        # Verify initialization
        assert tracker.config == config
        assert tracker.benchmark_name == "test_benchmark"
        assert isinstance(tracker.toxic_programs, set)
        assert len(tracker.toxic_programs) == 0
        assert isinstance(tracker.failures, list)
        assert len(tracker.failures) == 0

        # Verify file path setup
        expected_path = Path(temp_dir) / "failures" / "test_benchmark_failures.json"
        assert tracker.failure_file_path == expected_path

        # Verify failures directory was created
        assert expected_path.parent.exists()

    def test_add_failure_and_json_serialization(
        self, temp_dir, config, parent_program, child_program_below_threshold
    ):
        """Test add_failure() method and JSON serialization"""
        tracker = FailureTracker(config, temp_dir, "test_benchmark")

        # Add a failure
        tracker.add_failure(
            child_program_below_threshold,
            parent_program,
            "Optimize attention mechanism",
            "Introduced inefficient memory access pattern"
        )

        # Verify in-memory state
        assert child_program_below_threshold.id in tracker.toxic_programs
        assert len(tracker.failures) == 1

        # Verify failure record structure
        failure = tracker.failures[0]
        assert failure["program_id"] == child_program_below_threshold.id
        assert failure["parent_id"] == parent_program.id
        assert failure["proposal_summary"] == "Optimize attention mechanism"
        assert failure["failure_reason"] == "Introduced inefficient memory access pattern"
        assert failure["threshold"] == 0.85
        assert failure["comparison_metric"] == "combined_score"
        assert "timestamp" in failure
        assert "performance_ratio" in failure
        assert failure["parent_metrics"] == parent_program.metrics
        assert failure["child_metrics"] == child_program_below_threshold.metrics

        # Verify JSON file was created and is valid
        assert tracker.failure_file_path.exists()
        with open(tracker.failure_file_path, 'r') as f:
            saved_data = json.load(f)
            assert len(saved_data) == 1
            assert saved_data[0]["program_id"] == child_program_below_threshold.id

    def test_is_toxic_lookup(self, temp_dir, config, parent_program, child_program_below_threshold):
        """Test is_toxic() method with in-memory set lookup"""
        tracker = FailureTracker(config, temp_dir, "test_benchmark")

        # Initially, program should not be toxic
        assert not tracker.is_toxic(child_program_below_threshold.id)

        # Add failure
        tracker.add_failure(
            child_program_below_threshold,
            parent_program,
            "Apply gradient clipping",
            "Clipping threshold too aggressive"
        )

        # Now program should be toxic
        assert tracker.is_toxic(child_program_below_threshold.id)

        # Other programs should not be toxic
        assert not tracker.is_toxic("some_other_program")
        assert not tracker.is_toxic(parent_program.id)

    def test_get_failure_history_with_limit(
        self, temp_dir, config, parent_program
    ):
        """Test get_failure_history() method with limit parameter"""
        tracker = FailureTracker(config, temp_dir, "test_benchmark")

        # Add multiple failures
        for i in range(15):
            child = Program(
                id=f"child_{i}",
                code=f"def solution(): return {i}",
                parent_id=parent_program.id,
                metrics={"combined_score": 0.60 + i * 0.01},
                generation=2
            )
            tracker.add_failure(child, parent_program, f"Proposal {i}", f"Reason for failure {i}")

        # Test getting limited history
        history = tracker.get_failure_history(limit=5)
        assert len(history) == 5

        # Verify most recent failures are returned (last 5 added)
        for i, failure in enumerate(history):
            expected_id = f"child_{14-i}"  # Most recent first
            assert failure["program_id"] == expected_id

        # Test getting all with default limit
        full_history = tracker.get_failure_history()
        assert len(full_history) == 10  # max_failures_in_prompt from config

        # Test getting more than available
        all_history = tracker.get_failure_history(limit=100)
        assert len(all_history) == 15

    def test_save_load_persistence_cycle(
        self, temp_dir, config, parent_program, child_program_below_threshold
    ):
        """Test save/load persistence cycle"""
        # Create tracker and add failures
        tracker1 = FailureTracker(config, temp_dir, "test_benchmark")
        tracker1.add_failure(
            child_program_below_threshold,
            parent_program,
            "Test proposal summary",
            "Test failure reason"
        )

        # Create new tracker instance (should load from file)
        tracker2 = FailureTracker(config, temp_dir, "test_benchmark")

        # Verify data was restored
        assert len(tracker2.failures) == 1
        assert child_program_below_threshold.id in tracker2.toxic_programs
        assert tracker2.is_toxic(child_program_below_threshold.id)

        # Verify failure record matches
        assert tracker2.failures[0]["program_id"] == child_program_below_threshold.id
        assert tracker2.failures[0]["parent_id"] == parent_program.id

    def test_graceful_handling_of_missing_file(self, temp_dir, config):
        """Test graceful handling when failure file doesn't exist"""
        # Create tracker with non-existent file
        tracker = FailureTracker(config, temp_dir, "nonexistent_benchmark")

        # Should initialize with empty state
        assert len(tracker.failures) == 0
        assert len(tracker.toxic_programs) == 0

    def test_graceful_handling_of_corrupt_json(self, temp_dir, config):
        """Test graceful handling of corrupt JSON files"""
        # Create corrupt JSON file
        failures_dir = Path(temp_dir) / "failures"
        failures_dir.mkdir(exist_ok=True)
        corrupt_file = failures_dir / "corrupt_benchmark_failures.json"
        with open(corrupt_file, 'w') as f:
            f.write("{ invalid json content ][")

        # Create tracker (should handle corrupt file gracefully)
        tracker = FailureTracker(config, temp_dir, "corrupt_benchmark")

        # Should initialize with empty state despite corrupt file
        assert len(tracker.failures) == 0
        assert len(tracker.toxic_programs) == 0

    def test_should_mark_toxic_comparison(
        self, temp_dir, config, parent_program,
        child_program_below_threshold, child_program_above_threshold
    ):
        """Test should_mark_toxic() helper method for performance comparison"""
        tracker = FailureTracker(config, temp_dir, "test_benchmark")

        # Child below threshold should be marked toxic
        assert tracker.should_mark_toxic(child_program_below_threshold, parent_program)

        # Child above threshold should not be marked toxic
        assert not tracker.should_mark_toxic(child_program_above_threshold, parent_program)

    def test_should_mark_toxic_with_missing_metrics(
        self, temp_dir, config, parent_program
    ):
        """Test should_mark_toxic() with missing or invalid metrics"""
        tracker = FailureTracker(config, temp_dir, "test_benchmark")

        # Child with missing combined_score
        child_no_score = Program(
            id="child_no_score",
            code="def solution(): return 0",
            parent_id=parent_program.id,
            metrics={"accuracy": 0.5},  # missing combined_score
            generation=2
        )

        # Should handle missing metrics gracefully (return False)
        assert not tracker.should_mark_toxic(child_no_score, parent_program)

        # Parent with zero score (division by zero case)
        parent_zero = Program(
            id="parent_zero",
            code="def solution(): return 0",
            metrics={"combined_score": 0.0},
            generation=1
        )
        child_of_zero = Program(
            id="child_of_zero",
            code="def solution(): return 1",
            parent_id=parent_zero.id,
            metrics={"combined_score": 0.5},
            generation=2
        )

        # Should handle division by zero gracefully (return False)
        assert not tracker.should_mark_toxic(child_of_zero, parent_zero)
