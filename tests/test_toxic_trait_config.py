"""
Tests for ToxicTraitConfig configuration
"""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from evolve_agent.config import Config, ToxicTraitConfig


class TestToxicTraitConfig:
    """Test suite for ToxicTraitConfig dataclass"""

    def test_default_configuration_values(self):
        """Test that ToxicTraitConfig has correct default values"""
        config = ToxicTraitConfig()

        assert config.enabled is True
        assert config.threshold == 0.85
        assert config.comparison_metric == "combined_score"
        assert config.failure_history_path is None
        assert config.max_failures_in_prompt == 10

    def test_configuration_loading_from_yaml(self):
        """Test loading ToxicTraitConfig from YAML configuration"""
        # Create a temporary YAML config file
        config_data = {
            "toxic_trait": {
                "enabled": False,
                "threshold": 0.90,
                "comparison_metric": "normalized_average",
                "failure_history_path": "/custom/path/failures",
                "max_failures_in_prompt": 15
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            # Load config from YAML
            config = Config.from_yaml(temp_path)

            # Verify toxic_trait config was loaded correctly
            assert config.toxic_trait.enabled is False
            assert config.toxic_trait.threshold == 0.90
            assert config.toxic_trait.comparison_metric == "normalized_average"
            assert config.toxic_trait.failure_history_path == "/custom/path/failures"
            assert config.toxic_trait.max_failures_in_prompt == 15
        finally:
            # Clean up temp file
            os.unlink(temp_path)

    def test_environment_variable_expansion(self):
        """Test environment variable expansion in failure_history_path"""
        # Set a test environment variable
        os.environ['TEST_FAILURES_DIR'] = '/tmp/test_failures'

        config_data = {
            "toxic_trait": {
                "failure_history_path": "${TEST_FAILURES_DIR}/benchmark_failures"
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            # Load config from YAML
            config = Config.from_yaml(temp_path)

            # Verify environment variable was expanded
            assert config.toxic_trait.failure_history_path == "/tmp/test_failures/benchmark_failures"
        finally:
            # Clean up
            os.unlink(temp_path)
            del os.environ['TEST_FAILURES_DIR']

    def test_toxic_trait_integration_in_main_config(self):
        """Test that ToxicTraitConfig integrates properly into main Config class"""
        config = Config()

        # Verify toxic_trait field exists and has correct default
        assert hasattr(config, 'toxic_trait')
        assert isinstance(config.toxic_trait, ToxicTraitConfig)
        assert config.toxic_trait.enabled is True
        assert config.toxic_trait.threshold == 0.85
