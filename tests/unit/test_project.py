"""Tests for project management"""

import pytest
import tempfile
from pathlib import Path

from kepler.core.project import KeplerProject


class TestKeplerProject:
    """Test project management functionality"""
    
    def test_project_initialization(self):
        """Test project initialization creates correct structure"""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)
            project = KeplerProject(str(project_path))
            
            success = project.initialize("test_project")
            assert success
            
            # Check directory structure
            assert (project_path / "data" / "raw").exists()
            assert (project_path / "data" / "processed").exists()
            assert (project_path / "models").exists()
            assert (project_path / "notebooks").exists()
            assert (project_path / "scripts").exists()
            assert (project_path / "logs").exists()
            
            # Check files
            assert (project_path / "kepler.yml").exists()
            assert (project_path / "README.md").exists()
            assert (project_path / ".env.template").exists()
            assert (project_path / ".gitignore").exists()
            
            # Check gitkeep files
            assert (project_path / "data" / "raw" / ".gitkeep").exists()
            assert (project_path / "models" / ".gitkeep").exists()
    
    def test_is_kepler_project(self):
        """Test project detection"""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)
            project = KeplerProject(str(project_path))
            
            # Before initialization
            assert not project.is_kepler_project
            
            # After initialization
            project.initialize("test_project")
            assert project.is_kepler_project
    
    def test_validate_project(self):
        """Test project validation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)
            project = KeplerProject(str(project_path))
            
            # Should fail before initialization
            assert not project.validate_project()
            
            # Should pass after initialization
            project.initialize("test_project")
            
            # Create environment variables for testing
            import os
            os.environ["SPLUNK_TOKEN"] = "test_token"
            os.environ["SPLUNK_HEC_TOKEN"] = "test_hec_token"
            os.environ["GCP_PROJECT_ID"] = "test_project_id"
            
            try:
                assert project.validate_project()
            finally:
                # Clean up environment variables
                del os.environ["SPLUNK_TOKEN"]
                del os.environ["SPLUNK_HEC_TOKEN"]
                del os.environ["GCP_PROJECT_ID"]
    
    def test_get_config(self):
        """Test getting project configuration"""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)
            project = KeplerProject(str(project_path))
            
            # Should return None before initialization
            assert project.get_config() is None
            
            # Initialize and test config
            project.initialize("test_project")
            
            # Set up environment variables
            import os
            os.environ["SPLUNK_TOKEN"] = "test_token"
            os.environ["SPLUNK_HEC_TOKEN"] = "test_hec_token"
            os.environ["GCP_PROJECT_ID"] = "test_project_id"
            
            try:
                config = project.get_config()
                assert config is not None
                assert config.project_name == "test_project"
                assert config.splunk.token == "test_token"
                assert config.gcp.project_id == "test_project_id"
            finally:
                # Clean up environment variables
                del os.environ["SPLUNK_TOKEN"]
                del os.environ["SPLUNK_HEC_TOKEN"]
                del os.environ["GCP_PROJECT_ID"]
    
    def test_force_overwrite(self):
        """Test force overwrite functionality"""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)
            project = KeplerProject(str(project_path))
            
            # Initialize first time
            success = project.initialize("original_project")
            assert success
            
            # Check original content
            with open(project_path / "README.md", 'r') as f:
                original_content = f.read()
            assert "original_project" in original_content
            
            # Initialize again with force=True
            success = project.initialize("new_project", force=True)
            assert success
            
            # Check updated content
            with open(project_path / "README.md", 'r') as f:
                new_content = f.read()
            assert "new_project" in new_content
    
    def test_readme_creation(self):
        """Test README.md content"""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)
            project = KeplerProject(str(project_path))
            
            project.initialize("my_test_project")
            
            readme_path = project_path / "README.md"
            assert readme_path.exists()
            
            with open(readme_path, 'r') as f:
                content = f.read()
            
            assert "my_test_project" in content
            assert "kepler extract" in content
            assert "kepler train" in content
            assert "kepler deploy" in content
    
    def test_env_template_creation(self):
        """Test .env.template content"""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)
            project = KeplerProject(str(project_path))
            
            project.initialize("test_project")
            
            env_path = project_path / ".env.template"
            assert env_path.exists()
            
            with open(env_path, 'r') as f:
                content = f.read()
            
            assert "SPLUNK_TOKEN" in content
            assert "SPLUNK_HEC_TOKEN" in content
            assert "GCP_PROJECT_ID" in content