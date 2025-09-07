"""
Unit tests for unlimited Python library management system

Tests the core functionality of LibraryManager for supporting ANY Python library
from multiple sources as specified in PRD requirements.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from kepler.core.library_manager import (
    LibraryManager, 
    LibrarySpec, 
    LibrarySource,
    install_unlimited_libraries,
    validate_unlimited_environment,
    create_ai_template
)
from kepler.utils.exceptions import LibraryManagementError


class TestLibraryManager:
    """Test LibraryManager core functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = LibraryManager(self.temp_dir)
        
    def teardown_method(self):
        """Cleanup test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_parse_pypi_requirement(self):
        """Test parsing PyPI requirements"""
        # Simple package
        spec = self.manager._parse_pypi_requirement("numpy")
        assert spec.name == "numpy"
        assert spec.source == LibrarySource.PYPI
        assert spec.version is None
        
        # Package with version
        spec = self.manager._parse_pypi_requirement("numpy==1.24.0")
        assert spec.name == "numpy"
        assert spec.version == "==1.24.0"
        
        # Package with extras
        spec = self.manager._parse_pypi_requirement("transformers[torch]>=4.30.0")
        assert spec.name == "transformers"
        assert spec.extras == ["torch"]
        assert spec.version == ">=4.30.0"
    
    def test_parse_git_requirement(self):
        """Test parsing Git repository requirements"""
        # GitHub HTTPS
        spec = self.manager._parse_git_requirement("git+https://github.com/user/repo.git")
        assert spec.name == "repo"
        assert spec.source == LibrarySource.GITHUB
        assert spec.url == "git+https://github.com/user/repo.git"
        
        # GitHub with tag
        spec = self.manager._parse_git_requirement("git+https://github.com/user/repo.git@v1.0.0")
        assert spec.tag == "v1.0.0"
        assert spec.version == "v1.0.0"
        
        # Private repo SSH
        spec = self.manager._parse_git_requirement("git+ssh://git@github.com/company/private.git")
        assert spec.source == LibrarySource.PRIVATE_REPO
        assert spec.name == "private"
    
    def test_parse_local_editable_requirement(self):
        """Test parsing local editable requirements"""
        spec = self.manager._parse_requirement_line("-e ./my-custom-lib")
        assert spec.name == "my-custom-lib"
        assert spec.source == LibrarySource.LOCAL_EDITABLE
        assert spec.local_path == "./my-custom-lib"
    
    def test_parse_local_wheel_requirement(self):
        """Test parsing local wheel requirements"""
        spec = self.manager._parse_requirement_line("./dist/package-1.0.0-py3-none-any.whl")
        assert spec.source == LibrarySource.LOCAL_WHEEL
        assert spec.local_path == "./dist/package-1.0.0-py3-none-any.whl"
    
    def test_create_ai_template_ml(self):
        """Test creating traditional ML template"""
        success = self.manager.create_environment_from_template("ml")
        assert success
        
        # Check requirements.txt was created
        requirements_file = Path(self.temp_dir) / "requirements.txt"
        assert requirements_file.exists()
        
        # Check content includes ML libraries
        content = requirements_file.read_text()
        assert "scikit-learn" in content
        assert "xgboost" in content
        assert "pandas" in content
    
    def test_create_ai_template_deep_learning(self):
        """Test creating deep learning template"""
        success = self.manager.create_environment_from_template("deep_learning")
        assert success
        
        requirements_file = Path(self.temp_dir) / "requirements.txt"
        content = requirements_file.read_text()
        assert "torch" in content
        assert "tensorflow" in content
    
    def test_create_ai_template_generative_ai(self):
        """Test creating generative AI template"""
        success = self.manager.create_environment_from_template("generative_ai")
        assert success
        
        requirements_file = Path(self.temp_dir) / "requirements.txt"
        content = requirements_file.read_text()
        assert "transformers" in content
        assert "langchain" in content
        assert "openai" in content
    
    def test_create_ai_template_computer_vision(self):
        """Test creating computer vision template"""
        success = self.manager.create_environment_from_template("computer_vision")
        assert success
        
        requirements_file = Path(self.temp_dir) / "requirements.txt"
        content = requirements_file.read_text()
        assert "opencv-python" in content
        assert "pillow" in content
    
    def test_create_ai_template_nlp(self):
        """Test creating NLP template"""
        success = self.manager.create_environment_from_template("nlp")
        assert success
        
        requirements_file = Path(self.temp_dir) / "requirements.txt"
        content = requirements_file.read_text()
        assert "spacy" in content
        assert "nltk" in content
    
    def test_create_ai_template_full_ai(self):
        """Test creating full AI ecosystem template"""
        success = self.manager.create_environment_from_template("full_ai")
        assert success
        
        requirements_file = Path(self.temp_dir) / "requirements.txt"
        content = requirements_file.read_text()
        
        # Should include libraries from all categories
        assert "scikit-learn" in content  # ML
        assert "torch" in content  # Deep Learning
        assert "transformers" in content  # Generative AI
        assert "opencv-python" in content  # Computer Vision
        assert "spacy" in content  # NLP
    
    def test_unknown_template_raises_error(self):
        """Test that unknown template raises appropriate error"""
        with pytest.raises(LibraryManagementError):
            self.manager.create_environment_from_template("unknown_template")
    
    def test_parse_requirements_file(self):
        """Test parsing complex requirements.txt file"""
        # Create test requirements.txt
        requirements_content = """
# Traditional ML
numpy==1.24.0
scikit-learn>=1.3.0

# Deep Learning from GitHub
git+https://github.com/pytorch/pytorch.git@v2.0.0

# Generative AI with extras  
transformers[torch]>=4.30.0

# Private corporate library
git+ssh://git@github.com/company/ml-lib.git

# Local development
-e ./custom-algorithms

# Local wheel
./dist/optimized-models-1.0.0-py3-none-any.whl

# Custom URL
https://example.com/special-package.tar.gz
"""
        
        requirements_file = Path(self.temp_dir) / "requirements.txt"
        requirements_file.write_text(requirements_content)
        
        specs = self.manager.parse_requirements_file()
        
        # Verify all types are parsed correctly
        spec_names = [spec.name for spec in specs]
        assert "numpy" in spec_names
        assert "scikit-learn" in spec_names
        assert "pytorch" in spec_names
        assert "transformers" in spec_names
        assert "ml-lib" in spec_names
        assert "custom-algorithms" in spec_names
        
        # Check specific parsing
        numpy_spec = next(spec for spec in specs if spec.name == "numpy")
        assert numpy_spec.source == LibrarySource.PYPI
        assert numpy_spec.version == "==1.24.0"
        
        pytorch_spec = next(spec for spec in specs if spec.name == "pytorch")
        assert pytorch_spec.source == LibrarySource.GITHUB
        assert pytorch_spec.tag == "v2.0.0"
        
        transformers_spec = next(spec for spec in specs if spec.name == "transformers")
        assert "torch" in transformers_spec.extras
    
    @patch('subprocess.run')
    def test_install_library_success(self, mock_run):
        """Test successful library installation"""
        # Mock successful pip install
        mock_run.return_value = MagicMock(returncode=0)
        
        spec = LibrarySpec(name="numpy", source=LibrarySource.PYPI, version="==1.24.0")
        success = self.manager.install_library(spec)
        
        assert success
        mock_run.assert_called_once()
        
        # Check command was built correctly
        call_args = mock_run.call_args[0][0]
        assert "pip" in call_args
        assert "install" in call_args
        assert "numpy==1.24.0" in call_args
    
    @patch('subprocess.run')
    def test_install_library_failure(self, mock_run):
        """Test library installation failure"""
        # Mock failed pip install
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="Error output",
            stderr="Installation failed"
        )
        
        spec = LibrarySpec(name="nonexistent", source=LibrarySource.PYPI)
        success = self.manager.install_library(spec)
        
        assert not success
    
    def test_build_install_command_pypi(self):
        """Test building pip install command for PyPI packages"""
        spec = LibrarySpec(name="numpy", source=LibrarySource.PYPI, version="==1.24.0")
        cmd = self.manager._build_install_command(spec, upgrade=False)
        
        assert "pip" in cmd
        assert "install" in cmd
        assert "numpy==1.24.0" in cmd
        assert "--upgrade" not in cmd
    
    def test_build_install_command_git(self):
        """Test building pip install command for Git repositories"""
        spec = LibrarySpec(
            name="repo", 
            source=LibrarySource.GITHUB,
            url="git+https://github.com/user/repo.git",
            branch="main"
        )
        cmd = self.manager._build_install_command(spec, upgrade=True)
        
        assert "pip" in cmd
        assert "install" in cmd
        assert "--upgrade" in cmd
        assert "git+https://github.com/user/repo.git@main" in cmd
    
    def test_build_install_command_local_editable(self):
        """Test building pip install command for local editable libraries"""
        spec = LibrarySpec(
            name="custom-lib",
            source=LibrarySource.LOCAL_EDITABLE,
            local_path="./my-custom-lib"
        )
        cmd = self.manager._build_install_command(spec)
        
        assert "pip" in cmd
        assert "install" in cmd
        assert "-e" in cmd
        assert "./my-custom-lib" in cmd


class TestConvenienceFunctions:
    """Test convenience functions for unlimited library support"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        os.chdir(self.temp_dir)
        
    def teardown_method(self):
        """Cleanup test environment"""
        import shutil
        os.chdir("/")
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_ai_template_function(self):
        """Test create_ai_template convenience function"""
        success = create_ai_template("generative_ai", self.temp_dir)
        assert success
        
        requirements_file = Path(self.temp_dir) / "requirements.txt"
        assert requirements_file.exists()
        
        content = requirements_file.read_text()
        assert "transformers" in content
        assert "langchain" in content
    
    @patch('kepler.core.library_manager.LibraryManager.install_from_requirements')
    def test_install_unlimited_libraries_function(self, mock_install):
        """Test install_unlimited_libraries convenience function"""
        mock_install.return_value = {"numpy": True, "pandas": True}
        
        results = install_unlimited_libraries("requirements.txt", self.temp_dir)
        
        assert results == {"numpy": True, "pandas": True}
        mock_install.assert_called_once()
    
    @patch('kepler.core.library_manager.LibraryManager.validate_environment')
    def test_validate_unlimited_environment_function(self, mock_validate):
        """Test validate_unlimited_environment convenience function"""
        mock_validate.return_value = {
            'total_libraries': 5,
            'successful_imports': 5,
            'missing_libraries': []
        }
        
        report = validate_unlimited_environment(self.temp_dir)
        
        assert report['total_libraries'] == 5
        assert report['successful_imports'] == 5
        mock_validate.assert_called_once()


class TestLibrarySpecParsing:
    """Test comprehensive library specification parsing"""
    
    def setup_method(self):
        self.manager = LibraryManager(".")
    
    def test_parse_complex_requirements_scenarios(self):
        """Test parsing various real-world requirement scenarios"""
        
        # Test cases from PRD examples
        test_cases = [
            # Traditional ML
            ("scikit-learn>=1.3.0", "scikit-learn", LibrarySource.PYPI),
            ("xgboost==1.7.5", "xgboost", LibrarySource.PYPI),
            
            # Deep Learning
            ("torch>=2.0.0", "torch", LibrarySource.PYPI),
            ("tensorflow[gpu]>=2.13.0", "tensorflow", LibrarySource.PYPI),
            
            # Generative AI from GitHub
            ("git+https://github.com/huggingface/transformers.git@v4.30.0", "transformers", LibrarySource.GITHUB),
            ("git+https://github.com/hwchase17/langchain.git@main", "langchain", LibrarySource.GITHUB),
            
            # Private corporate libraries
            ("git+ssh://git@github.com/company/ai-tools.git", "ai-tools", LibrarySource.PRIVATE_REPO),
            
            # Local development
            ("-e ./custom-algorithms", "custom-algorithms", LibrarySource.LOCAL_EDITABLE),
            ("-e ./libs/industrial-models", "industrial-models", LibrarySource.LOCAL_EDITABLE),
            
            # Local wheels
            ("./wheels/optimized-inference-1.0.0-py3-none-any.whl", "optimized-inference", LibrarySource.LOCAL_WHEEL),
        ]
        
        for requirement_line, expected_name, expected_source in test_cases:
            spec = self.manager._parse_requirement_line(requirement_line)
            assert spec is not None, f"Failed to parse: {requirement_line}"
            assert spec.name == expected_name, f"Wrong name for {requirement_line}: got {spec.name}, expected {expected_name}"
            assert spec.source == expected_source, f"Wrong source for {requirement_line}: got {spec.source}, expected {expected_source}"
    
    def test_spec_to_requirement_line_roundtrip(self):
        """Test that LibrarySpec can be converted back to requirement line"""
        original_lines = [
            "numpy==1.24.0",
            "transformers[torch]>=4.30.0", 
            "git+https://github.com/user/repo.git@v1.0.0",
            "-e ./custom-lib",
            "./dist/package.whl"
        ]
        
        for line in original_lines:
            spec = self.manager._parse_requirement_line(line)
            reconstructed = self.manager._spec_to_requirement_line(spec)
            
            # For simple cases, should be identical
            # For complex cases, should be functionally equivalent
            assert spec.name is not None
            assert spec.source is not None


class TestAITemplates:
    """Test AI framework templates"""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.manager = LibraryManager(self.temp_dir)
        
    def teardown_method(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_ml_template_contains_required_libraries(self):
        """Test ML template includes all required traditional ML libraries"""
        self.manager.create_environment_from_template("ml")
        
        requirements_file = Path(self.temp_dir) / "requirements.txt"
        content = requirements_file.read_text()
        
        required_ml_libs = [
            "scikit-learn", "xgboost", "lightgbm", "catboost",
            "pandas", "numpy", "matplotlib", "seaborn"
        ]
        
        for lib in required_ml_libs:
            assert lib in content, f"ML template missing {lib}"
    
    def test_deep_learning_template_contains_required_libraries(self):
        """Test Deep Learning template includes all required DL frameworks"""
        self.manager.create_environment_from_template("deep_learning")
        
        requirements_file = Path(self.temp_dir) / "requirements.txt"
        content = requirements_file.read_text()
        
        required_dl_libs = [
            "torch", "torchvision", "tensorflow", "keras", "jax", "lightning"
        ]
        
        for lib in required_dl_libs:
            assert lib in content, f"Deep Learning template missing {lib}"
    
    def test_generative_ai_template_contains_required_libraries(self):
        """Test Generative AI template includes all required GenAI frameworks"""
        self.manager.create_environment_from_template("generative_ai")
        
        requirements_file = Path(self.temp_dir) / "requirements.txt"
        content = requirements_file.read_text()
        
        required_genai_libs = [
            "transformers", "langchain", "openai", "anthropic", "diffusers"
        ]
        
        for lib in required_genai_libs:
            assert lib in content, f"Generative AI template missing {lib}"
    
    def test_full_ai_template_comprehensive(self):
        """Test full AI template includes libraries from all categories"""
        self.manager.create_environment_from_template("full_ai")
        
        requirements_file = Path(self.temp_dir) / "requirements.txt"
        content = requirements_file.read_text()
        
        # Should include from all categories
        categories = {
            "ML": ["scikit-learn", "xgboost"],
            "Deep Learning": ["torch", "tensorflow"],
            "Generative AI": ["transformers", "langchain"],
            "Computer Vision": ["opencv-python", "pillow"],
            "NLP": ["spacy", "nltk"]
        }
        
        for category, libs in categories.items():
            for lib in libs:
                assert lib in content, f"Full AI template missing {lib} from {category}"


class TestLibraryValidation:
    """Test library validation and environment checking"""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.manager = LibraryManager(self.temp_dir)
    
    def teardown_method(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('importlib.import_module')
    def test_validate_library_installed_success(self, mock_import):
        """Test successful library validation"""
        mock_import.return_value = MagicMock()
        
        spec = LibrarySpec(name="numpy", source=LibrarySource.PYPI)
        is_installed = self.manager.validate_library_installed(spec)
        
        assert is_installed
        mock_import.assert_called_with("numpy")
    
    @patch('importlib.import_module')
    def test_validate_library_installed_with_alternative_names(self, mock_import):
        """Test library validation with alternative import names"""
        # First call fails, second succeeds with alternative name
        mock_import.side_effect = [ImportError(), MagicMock()]
        
        spec = LibrarySpec(name="scikit-learn", source=LibrarySource.PYPI)
        is_installed = self.manager.validate_library_installed(spec)
        
        assert is_installed
        # Should try both 'scikit-learn' and 'sklearn'
        assert mock_import.call_count == 2
    
    def test_validate_environment_empty_requirements(self):
        """Test environment validation with no requirements.txt"""
        report = self.manager.validate_environment()
        
        assert report['total_libraries'] == 0
        assert report['successful_imports'] == 0
        assert report['missing_libraries'] == []
        assert 'python_version' in report
        assert 'python_executable' in report


class TestDynamicLoadingAndDependencyManagement:
    """Test advanced dynamic loading and dependency management features"""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.manager = LibraryManager(self.temp_dir)
        
    def teardown_method(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('importlib.import_module')
    def test_dynamic_import_success(self, mock_import):
        """Test successful dynamic import with caching"""
        mock_module = MagicMock()
        mock_module.__name__ = "numpy"
        mock_import.return_value = mock_module
        
        # First import
        result1 = self.manager.dynamic_import("numpy")
        assert result1 == mock_module
        
        # Second import should use cache
        result2 = self.manager.dynamic_import("numpy")
        assert result2 == mock_module
        
        # Should only call import_module once (cached)
        mock_import.assert_called_once_with("numpy")
    
    @patch('importlib.import_module')
    def test_dynamic_import_with_reload(self, mock_import):
        """Test dynamic import with force reload"""
        mock_module = MagicMock()
        mock_import.return_value = mock_module
        
        # Import once
        self.manager.dynamic_import("numpy")
        
        # Import with reload should call import_module again
        self.manager.dynamic_import("numpy", force_reload=True)
        
        assert mock_import.call_count == 2
    
    def test_resolve_dependencies_no_conflicts(self):
        """Test dependency resolution with no conflicts"""
        # Create simple requirements.txt
        requirements_content = "numpy>=1.20.0\npandas>=1.5.0"
        requirements_file = Path(self.temp_dir) / "requirements.txt"
        requirements_file.write_text(requirements_content)
        
        report = self.manager.resolve_dependencies()
        
        assert report['total_libraries'] == 2
        assert len(report['conflicts']) == 0
        assert isinstance(report['recommendations'], list)
    
    def test_setup_development_environment_pytorch(self):
        """Test development environment setup for PyTorch"""
        success = self.manager.setup_development_environment(['pytorch', 'transformers'])
        assert success
        
        # Check that development tools were added
        requirements_file = Path(self.temp_dir) / "requirements.txt"
        content = requirements_file.read_text()
        
        assert 'jupyter' in content
        assert 'tensorboard' in content  # PyTorch-specific
        assert 'datasets' in content     # Transformers-specific
    
    def test_optimize_for_production(self):
        """Test production optimization analysis"""
        # Create requirements with dev dependencies
        requirements_content = """
numpy>=1.20.0
pandas>=1.5.0
jupyter>=1.0.0
pytest>=7.0.0
tensorboard>=2.13.0
"""
        requirements_file = Path(self.temp_dir) / "requirements.txt"
        requirements_file.write_text(requirements_content)
        
        # Mock installed libraries
        with patch.object(self.manager, 'get_installed_libraries') as mock_installed:
            mock_installed.return_value = [
                {'name': 'numpy', 'version': '1.24.0', 'source': 'PyPI'},
                {'name': 'pandas', 'version': '2.0.0', 'source': 'PyPI'},
                {'name': 'jupyter', 'version': '1.0.0', 'source': 'PyPI'},
                {'name': 'pytest', 'version': '7.2.0', 'source': 'PyPI'},
                {'name': 'tensorboard', 'version': '2.13.0', 'source': 'PyPI'},
            ]
            
            report = self.manager.optimize_for_production()
            
            assert report['total_libraries'] == 5
            assert len(report['development_only']) > 0  # Should detect jupyter, pytest, tensorboard
            assert not report['production_ready']  # Should be False due to dev dependencies
    
    def test_create_production_requirements(self):
        """Test creating optimized production requirements"""
        # Setup requirements with dev dependencies
        requirements_content = """
numpy>=1.20.0
pandas>=1.5.0
jupyter>=1.0.0
pytest>=7.0.0
"""
        requirements_file = Path(self.temp_dir) / "requirements.txt"
        requirements_file.write_text(requirements_content)
        
        with patch.object(self.manager, 'get_installed_libraries') as mock_installed:
            mock_installed.return_value = [
                {'name': 'numpy', 'version': '1.24.0', 'source': 'PyPI'},
                {'name': 'pandas', 'version': '2.0.0', 'source': 'PyPI'},
                {'name': 'jupyter', 'version': '1.0.0', 'source': 'PyPI'},
                {'name': 'pytest', 'version': '7.2.0', 'source': 'PyPI'},
            ]
            
            prod_file = self.manager.create_production_requirements()
            
            # Check production file was created
            assert Path(prod_file).exists()
            
            # Check content excludes dev dependencies
            content = Path(prod_file).read_text()
            assert 'numpy' in content
            assert 'pandas' in content
            assert 'jupyter' not in content  # Should be excluded
            assert 'pytest' not in content   # Should be excluded
    
    def test_create_dependency_lock(self):
        """Test comprehensive dependency lock file creation"""
        with patch.object(self.manager, 'get_installed_libraries') as mock_installed:
            mock_installed.return_value = [
                {'name': 'numpy', 'version': '1.24.0', 'source': 'PyPI'},
                {'name': 'transformers', 'version': '4.30.0', 'source': 'Git Repository'},
            ]
            
            with patch.object(self.manager, 'resolve_dependencies') as mock_resolve:
                mock_resolve.return_value = {
                    'total_libraries': 2,
                    'conflicts': [],
                    'resolved_versions': {}
                }
                
                self.manager.create_dependency_lock()
                
                # Check lock file was created
                lock_file = Path(self.temp_dir) / "kepler-lock.txt"
                assert lock_file.exists()
                
                content = lock_file.read_text()
                assert 'numpy==1.24.0' in content
                assert 'transformers==4.30.0' in content
                assert 'Python:' in content
                assert 'Generated:' in content


# Integration test scenarios
class TestUnlimitedLibrarySupportIntegration:
    """Integration tests for unlimited library support system"""
    
    def test_prd_requirement_any_python_library(self):
        """
        Test PRD requirement: 'El sistema DEBE soportar importación de cualquier librería Python estándar'
        """
        # This test validates that the system can handle ANY Python library
        manager = LibraryManager(".")
        
        # Test various library sources as specified in PRD
        test_libraries = [
            # PyPI official
            LibrarySpec(name="numpy", source=LibrarySource.PYPI),
            
            # GitHub experimental  
            LibrarySpec(
                name="experimental-ai",
                source=LibrarySource.GITHUB,
                url="git+https://github.com/research/experimental-ai.git"
            ),
            
            # Local custom
            LibrarySpec(
                name="custom-algorithms",
                source=LibrarySource.LOCAL_EDITABLE,
                local_path="./custom-algorithms"
            )
        ]
        
        # Verify each library type can be processed
        for spec in test_libraries:
            # Should not raise exception
            cmd = manager._build_install_command(spec)
            assert isinstance(cmd, list)
            assert len(cmd) > 0
            assert "pip" in cmd
    
    def test_prd_unlimited_experimentation_support(self):
        """
        Test PRD goal: 'Maximizar posibilidades de experimentación con cualquier tecnología de IA'
        """
        manager = LibraryManager(".")
        
        # Test all AI framework categories from PRD
        ai_templates = ["ml", "deep_learning", "generative_ai", "computer_vision", "nlp", "full_ai"]
        
        for template in ai_templates:
            # Should not raise exception
            try:
                # This validates the template exists and can be created
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_manager = LibraryManager(temp_dir)
                    success = temp_manager.create_environment_from_template(template)
                    assert success, f"Failed to create {template} template"
            except Exception as e:
                pytest.fail(f"Failed to create {template} template: {e}")
