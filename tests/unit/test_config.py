"""Tests for configuration management"""

import pytest
import tempfile
import os
from pathlib import Path
import yaml

from kepler.core.config import KeplerConfig, SplunkConfig, GCPConfig, TrainingConfig


class TestKeplerConfig:
    """Test configuration loading and validation"""
    
    def test_create_template(self):
        """Test that configuration template is created correctly"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_kepler.yml"
            
            KeplerConfig.create_template("test_project", str(config_path))
            
            assert config_path.exists()
            
            # Verify content
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            assert config_data["project_name"] == "test_project"
            assert "splunk" in config_data
            assert "gcp" in config_data
            assert "training" in config_data
            assert "deployment" in config_data
    
    def test_splunk_config_validation(self):
        """Test Splunk configuration validation"""
        # Valid config
        valid_config = {
            "host": "https://localhost:8089",
            "token": "test_token"
        }
        splunk_config = SplunkConfig(**valid_config)
        assert splunk_config.host == "https://localhost:8089"
        assert splunk_config.hec_url == "https://localhost:8088/services/collector/event"
        
        # Invalid host (no protocol)
        with pytest.raises(ValueError, match="Host must start with http"):
            SplunkConfig(host="localhost:8089", token="test_token")
    
    def test_training_config_validation(self):
        """Test training configuration validation"""
        # Valid config
        valid_config = TrainingConfig()
        assert valid_config.default_algorithm == "random_forest"
        assert valid_config.test_size == 0.2
        
        # Invalid algorithm
        with pytest.raises(ValueError, match="Algorithm must be one of"):
            TrainingConfig(default_algorithm="invalid_algorithm")
        
        # Invalid test_size
        with pytest.raises(ValueError, match="test_size must be between"):
            TrainingConfig(test_size=0.9)
    
    def test_environment_variable_substitution(self):
        """Test environment variable substitution"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_kepler.yml"
            
            # Create config with env vars
            config_data = {
                "project_name": "test_project",
                "splunk": {
                    "host": "https://localhost:8089",
                    "token": "${TEST_SPLUNK_TOKEN}"
                },
                "gcp": {
                    "project_id": "${TEST_GCP_PROJECT}"
                }
            }
            
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)
            
            # Set environment variables
            os.environ["TEST_SPLUNK_TOKEN"] = "test_token_value"
            os.environ["TEST_GCP_PROJECT"] = "test_project_id"
            
            try:
                config = KeplerConfig.from_file(str(config_path))
                assert config.splunk.token == "test_token_value"
                assert config.gcp.project_id == "test_project_id"
            finally:
                # Clean up environment variables
                del os.environ["TEST_SPLUNK_TOKEN"]
                del os.environ["TEST_GCP_PROJECT"]
    
    def test_missing_environment_variable(self):
        """Test error handling for missing environment variables"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_kepler.yml"
            
            config_data = {
                "project_name": "test_project",
                "splunk": {
                    "host": "https://localhost:8089",
                    "token": "${MISSING_TOKEN}"
                },
                "gcp": {
                    "project_id": "test_project"
                }
            }
            
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)
            
            with pytest.raises(ValueError, match="Environment variable 'MISSING_TOKEN' not found"):
                KeplerConfig.from_file(str(config_path))