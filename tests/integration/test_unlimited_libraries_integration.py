"""
Integration tests for unlimited Python library support

Tests real-world scenarios of installing and using ANY Python library
as specified in PRD requirements.
"""

import pytest
import tempfile
import os
from pathlib import Path
import subprocess
import sys

from kepler.core.library_manager import LibraryManager, LibrarySource, LibrarySpec
import kepler.libs as kp_libs
from kepler.utils.exceptions import LibraryManagementError


class TestUnlimitedLibraryIntegration:
    """Test unlimited library support in realistic scenarios"""
    
    def setup_method(self):
        """Setup isolated test environment"""
        self.original_dir = os.getcwd()
        self.temp_dir = tempfile.mkdtemp()
        os.chdir(self.temp_dir)
        
        # Create test project structure
        self.project_dir = Path(self.temp_dir) / "test-ai-project"
        self.project_dir.mkdir()
        os.chdir(self.project_dir)
        
    def teardown_method(self):
        """Cleanup test environment"""
        os.chdir(self.original_dir)
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_prd_requirement_any_python_library_sources(self):
        """
        Test PRD requirement: Support for ANY Python library from multiple sources
        
        Validates all library sources mentioned in PRD:
        - PyPI oficial (sklearn, transformers, pytorch)
        - GitHub experimental (research repos, alpha versions)
        - Corporate private (internal company libraries)
        - Custom developments (your own algorithms)
        - Forked versions (modified existing libraries)
        """
        manager = LibraryManager(".")
        
        # Test PyPI libraries (should work)
        pypi_spec = LibrarySpec(name="requests", source=LibrarySource.PYPI, version=">=2.25.0")
        cmd = manager._build_install_command(pypi_spec)
        assert "pip" in cmd
        assert "install" in cmd
        assert "requests>=2.25.0" in cmd
        
        # Test GitHub libraries (should work)
        github_spec = LibrarySpec(
            name="test-repo", 
            source=LibrarySource.GITHUB,
            url="git+https://github.com/user/test-repo.git",
            tag="v1.0.0"
        )
        cmd = manager._build_install_command(github_spec)
        assert "git+https://github.com/user/test-repo.git@v1.0.0" in cmd
        
        # Test private repositories (should work)
        private_spec = LibrarySpec(
            name="private-lib",
            source=LibrarySource.PRIVATE_REPO,
            url="git+ssh://git@github.com/company/private-lib.git"
        )
        cmd = manager._build_install_command(private_spec)
        assert "git+ssh://git@github.com/company/private-lib.git" in cmd
        
        # Test local editable (should work)
        local_spec = LibrarySpec(
            name="custom-algo",
            source=LibrarySource.LOCAL_EDITABLE,
            local_path="./custom-algorithms"
        )
        cmd = manager._build_install_command(local_spec)
        assert "-e" in cmd
        assert "./custom-algorithms" in cmd
    
    def test_ai_framework_templates_comprehensive(self):
        """
        Test that AI framework templates include all required categories
        
        Validates PRD examples:
        - ML: sklearn, xgboost, lightgbm, catboost
        - Deep Learning: torch, tensorflow, keras, jax
        - Generative AI: transformers, langchain, openai, anthropic
        - Computer Vision: opencv, pillow, torchvision
        - NLP: spacy, nltk, transformers
        """
        manager = LibraryManager(".")
        
        # Test each template category
        templates_to_test = {
            'ml': ['scikit-learn', 'xgboost', 'pandas'],
            'deep_learning': ['torch', 'tensorflow', 'keras'],
            'generative_ai': ['transformers', 'langchain', 'openai'],
            'computer_vision': ['opencv-python', 'pillow'],
            'nlp': ['spacy', 'nltk', 'transformers']
        }
        
        for template_name, expected_libs in templates_to_test.items():
            # Create template
            success = manager.create_environment_from_template(template_name)
            assert success, f"Failed to create {template_name} template"
            
            # Verify content
            requirements_file = Path("requirements.txt")
            content = requirements_file.read_text()
            
            for lib in expected_libs:
                assert lib in content, f"{template_name} template missing {lib}"
            
            # Clean up for next test
            requirements_file.unlink()
    
    def test_sdk_api_integration(self):
        """Test SDK API integration for unlimited library support"""
        
        # Test template creation via SDK
        success = kp_libs.template("ml")
        assert success
        
        # Verify requirements.txt was created
        requirements_file = Path("requirements.txt")
        assert requirements_file.exists()
        
        content = requirements_file.read_text()
        assert "scikit-learn" in content
        assert "xgboost" in content
        
        # Test validation via SDK
        report = kp_libs.validate()
        assert 'total_libraries' in report
        assert 'missing_libraries' in report
        assert isinstance(report['missing_libraries'], list)
        
        # Test library info
        frameworks = kp_libs.get_ai_framework_info()
        assert 'machine_learning' in frameworks
        assert 'deep_learning' in frameworks
        assert 'generative_ai' in frameworks
        
        # Clean up
        requirements_file.unlink()
    
    def test_complex_requirements_parsing_realistic(self):
        """Test parsing of complex real-world requirements.txt"""
        
        # Create realistic requirements.txt with mixed sources
        complex_requirements = """
# === TRADITIONAL ML ===
scikit-learn>=1.3.0
xgboost==1.7.5
lightgbm>=3.3.0

# === DEEP LEARNING ===  
torch>=2.0.0
tensorflow[gpu]>=2.13.0

# === GENERATIVE AI FROM GITHUB ===
git+https://github.com/huggingface/transformers.git@v4.30.0
git+https://github.com/hwchase17/langchain.git@main

# === EXPERIMENTAL RESEARCH ===
git+https://github.com/research-lab/novel-algorithm.git@v0.1.0-alpha

# === CORPORATE PRIVATE ===
git+ssh://git@github.com/company/ai-toolkit.git@v2.1.0

# === LOCAL DEVELOPMENT ===
-e ./custom-algorithms
-e ./libs/industrial-models

# === LOCAL WHEELS ===
./wheels/optimized-inference-1.0.0-py3-none-any.whl
./dist/custom-model-toolkit-0.5.tar.gz

# === CUSTOM URL ===
https://download.example.com/special-ai-package.tar.gz
"""
        
        requirements_file = Path("requirements.txt")
        requirements_file.write_text(complex_requirements)
        
        manager = LibraryManager(".")
        specs = manager.parse_requirements_file()
        
        # Verify all types were parsed correctly
        sources_found = set(spec.source for spec in specs)
        expected_sources = {
            LibrarySource.PYPI,
            LibrarySource.GITHUB,
            LibrarySource.PRIVATE_REPO,
            LibrarySource.LOCAL_EDITABLE,
            LibrarySource.LOCAL_WHEEL,
            LibrarySource.LOCAL_TARBALL,
            LibrarySource.CUSTOM_URL
        }
        
        assert sources_found == expected_sources, f"Missing sources: {expected_sources - sources_found}"
        
        # Verify specific parsing
        spec_names = [spec.name for spec in specs]
        expected_names = [
            'scikit-learn', 'xgboost', 'lightgbm',  # Traditional ML
            'torch', 'tensorflow',  # Deep Learning
            'transformers', 'langchain',  # Generative AI from GitHub
            'novel-algorithm',  # Experimental
            'ai-toolkit',  # Corporate
            'custom-algorithms', 'industrial-models',  # Local editable
            'optimized-inference', 'custom-model-toolkit',  # Local files
            'special-ai-package'  # Custom URL
        ]
        
        for expected_name in expected_names:
            assert expected_name in spec_names, f"Missing parsed library: {expected_name}"
        
        # Clean up
        requirements_file.unlink()
    
    def test_development_vs_production_workflow(self):
        """Test complete development to production workflow"""
        
        # 1. Create development environment
        success = kp_libs.template("generative_ai")
        assert success
        
        # 2. Add development tools
        manager = LibraryManager(".")
        dev_success = manager.setup_development_environment(['transformers', 'langchain'])
        assert dev_success
        
        # Verify development tools were added
        requirements_content = Path("requirements.txt").read_text()
        assert 'jupyter' in requirements_content
        assert 'datasets' in requirements_content  # Transformers-specific
        
        # 3. Analyze for production optimization
        optimization_report = kp_libs.optimize_for_production()
        assert 'development_only' in optimization_report
        assert 'production_ready' in optimization_report
        
        # Should detect development-only libraries
        dev_lib_names = [item['name'] for item in optimization_report['development_only']]
        
        # 4. Create production requirements
        prod_file = manager.create_production_requirements()
        assert Path(prod_file).exists()
        
        prod_content = Path(prod_file).read_text()
        # Production file should exclude development tools
        assert 'transformers' in prod_content  # Core AI library should remain
        
        # Clean up
        Path("requirements.txt").unlink()
        Path(prod_file).unlink()
    
    def test_dynamic_import_with_sdk(self):
        """Test dynamic import functionality via SDK"""
        
        # Test importing standard library (should work)
        json_module = kp_libs.import_module("json")
        assert json_module is not None
        assert hasattr(json_module, 'loads')
        
        # Test importing with alternative names
        try:
            # This should work even if sklearn is installed as scikit-learn
            sklearn_module = kp_libs.import_module("sklearn")
            assert sklearn_module is not None
        except LibraryManagementError:
            # Expected if sklearn not installed - that's fine
            pass
    
    def test_framework_info_and_recommendations(self):
        """Test AI framework information and recommendation system"""
        
        # Get framework info
        frameworks = kp_libs.get_ai_framework_info()
        
        # Verify all AI categories are present
        expected_categories = [
            'machine_learning', 'deep_learning', 'generative_ai', 
            'computer_vision', 'nlp'
        ]
        
        for category in expected_categories:
            assert category in frameworks
            assert 'description' in frameworks[category]
            assert 'examples' in frameworks[category]
            assert 'use_cases' in frameworks[category]
            assert 'template' in frameworks[category]
        
        # Test recommendations for specific use cases
        vision_recs = kp_libs.get_framework_recommendations("industrial image analysis")
        assert 'primary' in vision_recs
        assert 'supporting' in vision_recs
        
        # Should recommend vision libraries for image analysis
        all_recs = vision_recs['primary'] + vision_recs['supporting'] + vision_recs['optional']
        assert any('opencv' in lib for lib in all_recs)
        
        nlp_recs = kp_libs.get_framework_recommendations("text sentiment analysis")
        all_nlp_recs = nlp_recs['primary'] + nlp_recs['supporting'] + nlp_recs['optional']
        assert any('transform' in lib for lib in all_nlp_recs) or any('spacy' in lib for lib in all_nlp_recs)


class TestLibraryCompatibilityAndValidation:
    """Test library compatibility checking and validation"""
    
    def setup_method(self):
        self.original_dir = os.getcwd()
        self.temp_dir = tempfile.mkdtemp()
        os.chdir(self.temp_dir)
        
    def teardown_method(self):
        os.chdir(self.original_dir)
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_compatibility_checking(self):
        """Test library compatibility analysis"""
        
        # Test compatible libraries
        compat = kp_libs.check_compatibility(['numpy', 'pandas', 'matplotlib'])
        assert compat['compatible']
        assert len(compat['conflicts']) == 0
        
        # Test potentially conflicting libraries
        compat_gpu = kp_libs.check_compatibility(['tensorflow', 'torch'])
        # Should detect potential GPU resource conflicts
        assert len(compat_gpu['warnings']) > 0
    
    def test_library_source_validation(self):
        """Test that all library source types can be processed"""
        manager = LibraryManager(".")
        
        # Test all source types from PRD
        test_sources = [
            ("numpy>=1.24.0", LibrarySource.PYPI),
            ("git+https://github.com/user/repo.git@main", LibrarySource.GITHUB),
            ("git+ssh://git@github.com/company/private.git", LibrarySource.PRIVATE_REPO),
            ("-e ./custom-lib", LibrarySource.LOCAL_EDITABLE),
            ("./dist/package.whl", LibrarySource.LOCAL_WHEEL),
            ("./dist/package.tar.gz", LibrarySource.LOCAL_TARBALL),
            ("https://example.com/package.tar.gz", LibrarySource.CUSTOM_URL)
        ]
        
        for requirement, expected_source in test_sources:
            spec = manager._parse_requirement_line(requirement)
            assert spec is not None, f"Failed to parse: {requirement}"
            assert spec.source == expected_source, f"Wrong source for {requirement}"
            
            # Verify install command can be built
            cmd = manager._build_install_command(spec)
            assert isinstance(cmd, list)
            assert len(cmd) > 0
            assert "pip" in cmd


class TestRealWorldAIFrameworkScenarios:
    """Test real-world AI framework usage scenarios"""
    
    def setup_method(self):
        self.original_dir = os.getcwd()
        self.temp_dir = tempfile.mkdtemp()
        os.chdir(self.temp_dir)
        
    def teardown_method(self):
        os.chdir(self.original_dir)
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_generative_ai_project_workflow(self):
        """Test complete Generative AI project setup workflow"""
        
        # 1. Create Generative AI template
        success = kp_libs.template("generative_ai")
        assert success
        
        # 2. Verify template includes all required GenAI libraries
        requirements_file = Path("requirements.txt")
        content = requirements_file.read_text()
        
        required_genai_libs = [
            'transformers',  # Hugging Face transformers
            'langchain',     # LangChain for AI agents
            'openai',        # OpenAI API
            'anthropic',     # Claude API
            'diffusers'      # Stable Diffusion
        ]
        
        for lib in required_genai_libs:
            assert lib in content, f"Generative AI template missing {lib}"
        
        # 3. Test validation (libraries not installed, but parsing works)
        report = kp_libs.validate()
        assert report['total_libraries'] > 0
        assert 'missing_libraries' in report
        
        # 4. Test dependency analysis
        deps = kp_libs.analyze_dependencies()
        assert 'total_libraries' in deps
        assert 'conflicts' in deps
        
    def test_computer_vision_project_workflow(self):
        """Test complete Computer Vision project setup workflow"""
        
        # 1. Create Computer Vision template
        success = kp_libs.template("computer_vision")
        assert success
        
        # 2. Verify CV-specific libraries
        content = Path("requirements.txt").read_text()
        
        required_cv_libs = [
            'opencv-python',   # OpenCV for image processing
            'pillow',          # PIL for image manipulation
            'torchvision',     # PyTorch vision
            'albumentations'   # Image augmentations
        ]
        
        for lib in required_cv_libs:
            assert lib in content, f"Computer Vision template missing {lib}"
    
    def test_mixed_ai_framework_project(self):
        """Test project using multiple AI framework categories"""
        
        # 1. Start with full AI template
        success = kp_libs.template("full_ai")
        assert success
        
        # 2. Verify includes libraries from all categories
        content = Path("requirements.txt").read_text()
        
        # Should include from all AI categories
        category_representatives = {
            'ML': 'scikit-learn',
            'Deep Learning': 'torch',
            'Generative AI': 'transformers',
            'Computer Vision': 'opencv-python',
            'NLP': 'spacy'
        }
        
        for category, lib in category_representatives.items():
            assert lib in content, f"Full AI template missing {lib} from {category}"
        
        # 3. Test adding custom experimental library
        experimental_lib = "git+https://github.com/research/experimental-ai.git@v0.1.0-alpha"
        
        # Should not raise exception
        manager = LibraryManager(".")
        spec = manager._parse_git_requirement(experimental_lib)
        assert spec.source == LibrarySource.GITHUB
        assert spec.tag == "v0.1.0-alpha"
    
    def test_local_custom_library_integration(self):
        """Test integration with local custom libraries"""
        
        # 1. Create mock local library structure
        custom_lib_dir = Path("custom-algorithms")
        custom_lib_dir.mkdir()
        
        # Create basic setup.py
        setup_py_content = '''
from setuptools import setup, find_packages

setup(
    name="custom-algorithms",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["numpy", "scikit-learn"]
)
'''
        (custom_lib_dir / "setup.py").write_text(setup_py_content)
        
        # Create package structure
        pkg_dir = custom_lib_dir / "custom_algorithms"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text('__version__ = "0.1.0"')
        
        # 2. Test local editable installation parsing
        manager = LibraryManager(".")
        spec = manager._parse_requirement_line("-e ./custom-algorithms")
        
        assert spec.source == LibrarySource.LOCAL_EDITABLE
        assert spec.local_path == "./custom-algorithms"
        assert spec.name == "custom-algorithms"
        
        # 3. Test install command building
        cmd = manager._build_install_command(spec)
        assert "-e" in cmd
        assert "./custom-algorithms" in cmd
    
    def test_prd_unlimited_experimentation_validation(self):
        """
        Validate PRD goal: 'Maximizar posibilidades de experimentación con cualquier tecnología de IA'
        """
        
        # Test that system can handle the full spectrum of AI technologies
        ai_technologies = [
            # Traditional ML
            ("scikit-learn>=1.3.0", "Traditional Machine Learning"),
            
            # Deep Learning
            ("torch>=2.0.0", "Deep Learning"),
            ("tensorflow>=2.13.0", "Deep Learning"),
            
            # Generative AI
            ("transformers>=4.30.0", "Generative AI - LLMs"),
            ("langchain>=0.0.200", "Generative AI - Agents"),
            ("openai>=0.27.0", "Generative AI - API"),
            ("diffusers>=0.18.0", "Generative AI - Image Generation"),
            
            # Computer Vision
            ("opencv-python>=4.8.0", "Computer Vision"),
            ("pillow>=10.0.0", "Image Processing"),
            
            # NLP
            ("spacy>=3.6.0", "Natural Language Processing"),
            ("nltk>=3.8.0", "Text Analysis"),
            
            # Specialized
            ("prophet>=1.1.0", "Time Series Forecasting"),
            ("gymnasium>=0.28.0", "Reinforcement Learning"),
            ("networkx>=3.1.0", "Graph Analysis"),
            
            # Experimental sources
            ("git+https://github.com/research/experimental-ai.git", "Experimental Research"),
            ("git+ssh://git@github.com/company/proprietary.git", "Corporate Private"),
            ("-e ./custom-algorithm", "Local Development")
        ]
        
        manager = LibraryManager(".")
        
        for library_spec, technology_type in ai_technologies:
            try:
                spec = manager._parse_requirement_line(library_spec)
                assert spec is not None, f"Cannot parse {technology_type}: {library_spec}"
                
                # Verify install command can be built
                cmd = manager._build_install_command(spec)
                assert isinstance(cmd, list), f"Cannot build install command for {technology_type}"
                assert len(cmd) > 0, f"Empty install command for {technology_type}"
                
            except Exception as e:
                pytest.fail(f"Failed to handle {technology_type} ({library_spec}): {e}")
        
        # Success means Kepler can handle ANY AI technology as required by PRD
