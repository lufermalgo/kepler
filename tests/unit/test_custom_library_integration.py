"""
Unit tests for custom library integration system
Tests Task 1.7: Custom library integration (local, GitHub, private repos)
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import subprocess

from kepler.core.library_manager import LibraryManager, LibrarySource, LibrarySpec
from kepler.utils.exceptions import LibraryManagementError


class TestCustomLibraryIntegration:
    """Test custom library integration capabilities"""
    
    @pytest.fixture
    def temp_project(self):
        """Create temporary project directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def library_manager(self, temp_project):
        """Create LibraryManager instance for testing"""
        return LibraryManager(project_path=str(temp_project))
    
    def test_ssh_authentication_setup(self, library_manager):
        """Test SSH authentication setup for private repos"""
        with patch('os.path.expanduser') as mock_expanduser, \
             patch('pathlib.Path.exists') as mock_exists, \
             patch('os.chmod') as mock_chmod, \
             patch.object(library_manager, '_test_ssh_connection') as mock_test:
            
            mock_expanduser.return_value = "/home/user/.ssh/id_rsa"
            mock_exists.return_value = True
            mock_test.return_value = True
            
            result = library_manager.setup_ssh_authentication()
            
            assert result is True
            mock_chmod.assert_called_once_with(Path("/home/user/.ssh/id_rsa"), 0o600)
            mock_test.assert_called_once()
    
    def test_ssh_authentication_missing_key(self, library_manager):
        """Test SSH setup when key file doesn't exist"""
        with patch('os.path.expanduser') as mock_expanduser, \
             patch('pathlib.Path.exists') as mock_exists:
            
            mock_expanduser.return_value = "/home/user/.ssh/id_rsa"
            mock_exists.return_value = False
            
            result = library_manager.setup_ssh_authentication()
            
            assert result is False
    
    @patch('subprocess.run')
    def test_ssh_connection_test(self, mock_subprocess, library_manager):
        """Test SSH connection validation"""
        # Mock successful SSH test
        mock_result = Mock()
        mock_result.returncode = 1  # SSH test typically returns 1 with success message
        mock_result.stderr = "Hi username! You've successfully authenticated"
        mock_subprocess.return_value = mock_result
        
        result = library_manager._test_ssh_connection()
        
        assert result is True
        assert mock_subprocess.call_count > 0
    
    def test_github_installation_https(self, library_manager):
        """Test GitHub library installation via HTTPS"""
        with patch.object(library_manager, 'install_library') as mock_install:
            mock_install.return_value = True
            
            result = library_manager.install_from_github("huggingface/transformers", tag="v4.21.0")
            
            assert result is True
            mock_install.assert_called_once()
            
            # Verify LibrarySpec creation
            call_args = mock_install.call_args[0][0]
            assert call_args.name == "transformers"
            assert call_args.source == LibrarySource.GITHUB
            assert "v4.21.0" in call_args.url
    
    def test_github_installation_ssh(self, library_manager):
        """Test GitHub library installation via SSH"""
        with patch.object(library_manager, 'install_library') as mock_install:
            mock_install.return_value = True
            
            result = library_manager.install_from_github("git@github.com:company/private-lib.git")
            
            assert result is True
            mock_install.assert_called_once()
            
            call_args = mock_install.call_args[0][0]
            assert call_args.name == "private-lib"
            assert call_args.source == LibrarySource.GITHUB
    
    def test_github_installation_with_subdirectory(self, library_manager):
        """Test GitHub installation from subdirectory"""
        with patch.object(library_manager, 'install_library') as mock_install:
            mock_install.return_value = True
            
            result = library_manager.install_from_github(
                "monorepo/project", 
                subdirectory="ml-package"
            )
            
            assert result is True
            call_args = mock_install.call_args[0][0]
            assert "#subdirectory=ml-package" in call_args.url
    
    def test_repo_name_extraction(self, library_manager):
        """Test repository name extraction from various URL formats"""
        test_cases = [
            ("https://github.com/user/repo.git", "repo"),
            ("git@github.com:user/repo.git", "repo"),
            ("user/repo", "repo"),
            ("https://github.com/org/my-library", "my-library"),
            ("git@gitlab.com:group/subgroup/project.git", "project")
        ]
        
        for repo_url, expected_name in test_cases:
            result = library_manager._extract_repo_name(repo_url)
            assert result == expected_name, f"Failed for {repo_url}: got {result}, expected {expected_name}"
    
    def test_local_library_installation(self, library_manager, temp_project):
        """Test local library installation"""
        # Create mock local library
        lib_path = temp_project / "custom-libs" / "test-lib"
        lib_path.mkdir(parents=True)
        (lib_path / "setup.py").write_text("from setuptools import setup; setup(name='test-lib')")
        
        with patch('subprocess.run') as mock_subprocess:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_subprocess.return_value = mock_result
            
            result = library_manager.install_local_library(str(lib_path), editable=True)
            
            assert result is True
            mock_subprocess.assert_called_once()
            
            # Verify editable install command
            call_args = mock_subprocess.call_args[0][0]
            assert "-e" in call_args
            # Check that the path is in the command (handle symlink resolution)
            assert any(str(lib_path) in arg or lib_path.name in arg for arg in call_args)
    
    def test_local_library_missing_setup(self, library_manager, temp_project):
        """Test local library installation when setup.py is missing"""
        lib_path = temp_project / "custom-libs" / "invalid-lib"
        lib_path.mkdir(parents=True)
        # No setup.py created
        
        result = library_manager.install_local_library(str(lib_path))
        
        assert result is False
    
    def test_wheel_installation(self, library_manager, temp_project):
        """Test wheel file installation"""
        wheel_path = temp_project / "test-package-1.0.0-py3-none-any.whl"
        wheel_path.write_text("fake wheel content")  # Create fake wheel file
        
        with patch('subprocess.run') as mock_subprocess:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_subprocess.return_value = mock_result
            
            result = library_manager.install_from_wheel(str(wheel_path))
            
            assert result is True
            mock_subprocess.assert_called_once()
    
    def test_wheel_installation_invalid_file(self, library_manager, temp_project):
        """Test wheel installation with invalid file"""
        invalid_file = temp_project / "not-a-wheel.txt"
        invalid_file.write_text("not a wheel")
        
        result = library_manager.install_from_wheel(str(invalid_file))
        
        assert result is False
    
    def test_custom_library_template_creation(self, library_manager, temp_project):
        """Test creation of custom library template"""
        library_name = "my-custom-algorithm"
        author = "Ana Rodriguez"
        
        lib_path = library_manager.create_custom_library_template(library_name, author)
        
        lib_path = Path(lib_path)
        assert lib_path.exists()
        assert (lib_path / "setup.py").exists()
        assert (lib_path / "my_custom_algorithm" / "__init__.py").exists()
        assert (lib_path / "README.md").exists()
        
        # Verify content
        setup_content = (lib_path / "setup.py").read_text()
        assert library_name in setup_content
        assert author in setup_content
        
        init_content = (lib_path / "my_custom_algorithm" / "__init__.py").read_text()
        assert author in init_content
        assert "hello_kepler" in init_content
    
    @patch('subprocess.run')
    def test_clone_and_install_repo(self, mock_subprocess, library_manager, temp_project):
        """Test repository cloning and installation"""
        # Mock successful git clone
        mock_result = Mock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result
        
        with patch.object(library_manager, 'install_local_library') as mock_install:
            mock_install.return_value = True
            
            result = library_manager.clone_and_install_repo(
                "https://github.com/research/experimental-ai.git",
                branch="main"
            )
            
            assert result is True
            
            # Verify git clone command
            clone_call = mock_subprocess.call_args[0][0]
            assert "git" in clone_call
            assert "clone" in clone_call
            assert "-b" in clone_call
            assert "main" in clone_call
            
            # Verify local install was called
            mock_install.assert_called_once()
    
    def test_private_repo_config_ssh(self, library_manager):
        """Test private repository configuration with SSH"""
        repo_url = "git@github.com:company/private-lib.git"
        
        config = library_manager.create_private_repo_config(repo_url, auth_method="ssh")
        
        assert config["repo_url"] == repo_url
        assert config["auth_method"] == "ssh"
        assert config["install_url"].startswith("git+ssh://")
        assert config["requirements"].startswith("git+ssh://")
    
    def test_private_repo_config_https_token(self, library_manager):
        """Test private repository configuration with HTTPS token"""
        repo_url = "https://github.com/company/private-lib.git"
        token = "ghp_xxxxxxxxxxxx"
        
        config = library_manager.create_private_repo_config(
            repo_url, 
            auth_method="token", 
            token=token
        )
        
        assert config["repo_url"] == repo_url
        assert config["auth_method"] == "token"
        assert token in config["install_url"]
        assert "git+" in config["install_url"]
    
    def test_custom_library_validation_success(self, library_manager):
        """Test successful custom library validation"""
        with patch('pkg_resources.get_distribution') as mock_get_dist, \
             patch('importlib.import_module') as mock_import:
            
            # Mock successful distribution
            mock_dist = Mock()
            mock_dist.project_name = "custom-lib"
            mock_dist.version = "0.1.0"
            mock_dist.location = "/project/custom-libs/custom-lib"
            mock_dist.requires.return_value = []
            mock_get_dist.return_value = mock_dist
            
            # Mock successful import
            mock_import.return_value = Mock()
            
            validation = library_manager.validate_custom_library("custom-lib")
            
            assert validation["installed"] is True
            assert validation["importable"] is True
            assert validation["version"] == "0.1.0"
            assert validation["editable"] is True  # Not in site-packages
            assert validation["source"] == "local_editable"
            assert len(validation["errors"]) == 0
    
    def test_custom_library_validation_import_error(self, library_manager):
        """Test custom library validation with import error"""
        with patch('pkg_resources.get_distribution') as mock_get_dist, \
             patch('importlib.import_module') as mock_import:
            
            # Mock successful distribution but failed import
            mock_dist = Mock()
            mock_dist.project_name = "broken-lib"
            mock_dist.version = "0.1.0"
            mock_dist.location = "/project/custom-libs/broken-lib"
            mock_dist.requires.return_value = []
            mock_get_dist.return_value = mock_dist
            
            # Mock import error
            mock_import.side_effect = ImportError("Missing dependency")
            
            validation = library_manager.validate_custom_library("broken-lib")
            
            assert validation["installed"] is True
            assert validation["importable"] is False
            assert len(validation["errors"]) > 0
            assert "Import failed" in validation["errors"][0]


class TestCustomLibrarySDKAPI:
    """Test SDK API for custom library integration"""
    
    @patch('kepler.libs.LibraryManager')
    def test_setup_ssh_api(self, mock_manager_class):
        """Test kp.libs.setup_ssh() API"""
        from kepler.libs import setup_ssh
        
        mock_manager = Mock()
        mock_manager.setup_ssh_authentication.return_value = True
        mock_manager_class.return_value = mock_manager
        
        result = setup_ssh("/custom/ssh/key", test_connection=False)
        
        assert result is True
        mock_manager.setup_ssh_authentication.assert_called_once_with("/custom/ssh/key", False)
    
    @patch('kepler.libs.LibraryManager')
    def test_install_github_api(self, mock_manager_class):
        """Test kp.libs.install_github() API"""
        from kepler.libs import install_github
        
        mock_manager = Mock()
        mock_manager.install_from_github.return_value = True
        mock_manager_class.return_value = mock_manager
        
        result = install_github("research/experimental-ai", tag="v0.1.0")
        
        assert result is True
        mock_manager.install_from_github.assert_called_once_with(
            "research/experimental-ai", None, "v0.1.0", None, None
        )
    
    @patch('kepler.libs.LibraryManager')
    def test_install_local_api(self, mock_manager_class):
        """Test kp.libs.install_local() API"""
        from kepler.libs import install_local
        
        mock_manager = Mock()
        mock_manager.install_local_library.return_value = True
        mock_manager_class.return_value = mock_manager
        
        result = install_local("./custom-libs/my-lib", editable=False)
        
        assert result is True
        mock_manager.install_local_library.assert_called_once_with("./custom-libs/my-lib", False)
    
    @patch('kepler.libs.LibraryManager')
    def test_create_custom_lib_api(self, mock_manager_class):
        """Test kp.libs.create_custom_lib() API"""
        from kepler.libs import create_custom_lib
        
        mock_manager = Mock()
        mock_manager.create_custom_library_template.return_value = "/path/to/lib"
        mock_manager_class.return_value = mock_manager
        
        result = create_custom_lib("industrial-ml", "Ana Rodriguez")
        
        assert result == "/path/to/lib"
        mock_manager.create_custom_library_template.assert_called_once_with("industrial-ml", "Ana Rodriguez")
    
    @patch('kepler.libs.LibraryManager')
    def test_validate_custom_api(self, mock_manager_class):
        """Test kp.libs.validate_custom() API"""
        from kepler.libs import validate_custom
        
        mock_manager = Mock()
        expected_validation = {
            "library_name": "test-lib",
            "installed": True,
            "importable": True,
            "version": "0.1.0",
            "errors": []
        }
        mock_manager.validate_custom_library.return_value = expected_validation
        mock_manager_class.return_value = mock_manager
        
        result = validate_custom("test-lib")
        
        assert result == expected_validation
        mock_manager.validate_custom_library.assert_called_once_with("test-lib")
    
    @patch('kepler.libs.LibraryManager')
    def test_configure_private_repo_api(self, mock_manager_class):
        """Test kp.libs.configure_private_repo() API"""
        from kepler.libs import configure_private_repo
        
        mock_manager = Mock()
        expected_config = {
            "repo_url": "git@github.com:company/private.git",
            "auth_method": "ssh",
            "install_url": "git+ssh://git@github.com:company/private.git"
        }
        mock_manager.create_private_repo_config.return_value = expected_config
        mock_manager_class.return_value = mock_manager
        
        result = configure_private_repo("git@github.com:company/private.git", auth_method="ssh")
        
        assert result == expected_config
        mock_manager.create_private_repo_config.assert_called_once_with(
            "git@github.com:company/private.git", "ssh", None, None
        )


class TestRealWorldScenarios:
    """Test real-world custom library integration scenarios"""
    
    @pytest.fixture
    def library_manager(self):
        return LibraryManager()
    
    def test_experimental_research_library_scenario(self, library_manager):
        """Test scenario: Ana finds experimental library on GitHub"""
        with patch.object(library_manager, 'install_library') as mock_install:
            mock_install.return_value = True
            
            # Ana installs experimental library from research paper
            result = library_manager.install_from_github(
                "research-lab/novel-algorithm", 
                commit="abc123",  # Specific commit from paper
                subdirectory="python-package"  # Library is in subdirectory
            )
            
            assert result is True
            
            # Verify correct URL construction
            call_args = mock_install.call_args[0][0]
            assert "abc123" in call_args.url
            assert "subdirectory=python-package" in call_args.url
    
    def test_corporate_private_library_scenario(self, library_manager):
        """Test scenario: Ana uses company's private AI library"""
        repo_url = "git@internal-gitlab.com:ai-team/industrial-models.git"
        
        # Configure private repo access
        config = library_manager.create_private_repo_config(repo_url, auth_method="ssh")
        
        assert config["auth_method"] == "ssh"
        assert "git+ssh://" in config["install_url"]
        
        # Install private library
        with patch.object(library_manager, 'install_library') as mock_install:
            mock_install.return_value = True
            
            result = library_manager.install_from_github(repo_url)
            assert result is True
    
    def test_custom_development_workflow(self, library_manager, tmp_path):
        """Test scenario: Ana develops custom library for her project"""
        # Create custom library template
        with patch.object(library_manager, 'project_path', tmp_path):
            lib_path = library_manager.create_custom_library_template(
                "predictive-maintenance-utils", 
                "Ana Rodriguez"
            )
            
            lib_path = Path(lib_path)
            assert lib_path.exists()
            
            # Verify template structure
            assert (lib_path / "setup.py").exists()
            assert (lib_path / "predictive_maintenance_utils" / "__init__.py").exists()
            
            # Verify content includes author
            init_content = (lib_path / "predictive_maintenance_utils" / "__init__.py").read_text()
            assert "Ana Rodriguez" in init_content
            
            # Install in editable mode
            with patch('subprocess.run') as mock_subprocess:
                mock_result = Mock()
                mock_result.returncode = 0
                mock_subprocess.return_value = mock_result
                
                result = library_manager.install_local_library(str(lib_path), editable=True)
                assert result is True
    
    def test_fork_modification_scenario(self, library_manager):
        """Test scenario: Ana uses modified fork of popular library"""
        fork_url = "https://github.com/ana-rodriguez/transformers-industrial.git"
        
        with patch.object(library_manager, 'install_library') as mock_install:
            mock_install.return_value = True
            
            # Install from personal fork with industrial modifications
            result = library_manager.install_from_github(
                fork_url, 
                branch="industrial-optimizations"
            )
            
            assert result is True
            
            call_args = mock_install.call_args[0][0]
            assert call_args.name == "transformers-industrial"
            assert "industrial-optimizations" in call_args.url


class TestPRDCompliance:
    """Test compliance with PRD requirements"""
    
    def test_prd_requirement_unlimited_python_support(self):
        """
        Test PRD Requirement: "El sistema DEBE soportar importación de cualquier librería Python estándar"
        """
        from kepler.libs import install_github, install_local, install_wheel
        
        # All custom library installation methods should be available
        assert callable(install_github)
        assert callable(install_local)
        assert callable(install_wheel)
    
    def test_prd_requirement_github_support(self):
        """
        Test PRD Requirement: Support for GitHub repositories
        """
        from kepler.libs import install_github, clone_repo
        
        # GitHub-specific methods should be available
        assert callable(install_github)
        assert callable(clone_repo)
    
    def test_prd_requirement_private_repo_support(self):
        """
        Test PRD Requirement: Support for private repositories
        """
        from kepler.libs import configure_private_repo, setup_ssh
        
        # Private repository methods should be available
        assert callable(configure_private_repo)
        assert callable(setup_ssh)
    
    def test_prd_requirement_local_development_support(self):
        """
        Test PRD Requirement: Support for local custom library development
        """
        from kepler.libs import create_custom_lib, install_local, validate_custom
        
        # Local development methods should be available
        assert callable(create_custom_lib)
        assert callable(install_local)
        assert callable(validate_custom)
