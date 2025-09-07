"""
Kepler Libs Module - Unlimited Python Library Support API

Simple SDK interface for managing any Python library from any source.
Implements the PRD philosophy: "Si está en Python, Kepler lo soporta"

Usage:
    import kepler as kp
    
    # Create AI framework templates
    kp.libs.template("generative_ai")
    kp.libs.template("deep_learning") 
    
    # Install any library
    kp.libs.install("transformers")
    kp.libs.install("git+https://github.com/research/experimental-ai.git")
    
    # Validate environment
    kp.libs.validate()
    
    # Dynamic imports
    torch = kp.libs.import_module("torch")
"""

from typing import Dict, List, Optional, Any, Union
from kepler.core.library_manager import (
    LibraryManager, 
    LibrarySpec, 
    LibrarySource,
    create_ai_template,
    install_unlimited_libraries,
    validate_unlimited_environment
)
from kepler.utils.logging import get_logger
from kepler.utils.exceptions import LibraryManagementError


# Global library manager instance
_lib_manager = None


def _get_manager() -> LibraryManager:
    """Get or create global LibraryManager instance"""
    global _lib_manager
    if _lib_manager is None:
        _lib_manager = LibraryManager(".")
    return _lib_manager


def template(ai_type: str) -> bool:
    """
    Create requirements.txt template for specific AI framework type
    
    Args:
        ai_type: Type of AI framework template
                'ml' - Traditional ML (sklearn, xgboost, lightgbm)
                'deep_learning' - Deep Learning (pytorch, tensorflow, keras)
                'generative_ai' - Generative AI (transformers, langchain, openai)
                'computer_vision' - Computer Vision (opencv, pillow, torchvision)
                'nlp' - Natural Language Processing (spacy, nltk, transformers)
                'full_ai' - Complete AI ecosystem (all above combined)
    
    Returns:
        True if template created successfully
        
    Example:
        import kepler as kp
        
        # Create Generative AI template
        kp.libs.template("generative_ai")
        
        # Install all libraries from template
        kp.libs.install_all()
    """
    logger = get_logger(__name__)
    logger.info(f"Creating {ai_type} template...")
    
    try:
        manager = _get_manager()
        return manager.create_environment_from_template(ai_type)
    except Exception as e:
        raise LibraryManagementError(f"Failed to create template: {e}", suggestion=f"Check template name: {ai_type}")


def install(library: Union[str, List[str]], upgrade: bool = False) -> Union[bool, Dict[str, bool]]:
    """
    Install any Python library from any source
    
    Supports:
    - PyPI packages: "transformers", "torch>=2.0.0"
    - GitHub repos: "git+https://github.com/user/repo.git"
    - Private repos: "git+ssh://git@github.com/company/repo.git"
    - Local libraries: "-e ./custom-lib"
    - Wheel files: "./dist/package.whl"
    
    Args:
        library: Library specification(s) to install
        upgrade: Whether to upgrade if already installed
        
    Returns:
        Installation success status (bool for single, dict for multiple)
        
    Examples:
        import kepler as kp
        
        # Install from PyPI
        kp.libs.install("transformers>=4.30.0")
        
        # Install experimental from GitHub
        kp.libs.install("git+https://github.com/research/novel-ai.git@v0.1.0-alpha")
        
        # Install multiple
        kp.libs.install(["torch", "transformers", "langchain"])
        
        # Install custom local library
        kp.libs.install("-e ./my-custom-algorithm")
    """
    logger = get_logger(__name__)
    manager = _get_manager()
    
    if isinstance(library, list):
        # Install multiple libraries
        results = {}
        for lib in library:
            results[lib] = install(lib, upgrade=upgrade)
        return results
    
    # Install single library
    logger.info(f"Installing library: {library}")
    
    try:
        # Parse library specification
        if library.startswith('git+'):
            spec = manager._parse_git_requirement(library)
        elif library.startswith('-e '):
            spec = manager._parse_requirement_line(library)
        elif library.startswith('./') or library.startswith('../'):
            spec = manager._parse_requirement_line(library)
        else:
            # Assume PyPI package
            spec = manager._parse_pypi_requirement(library)
        
        return manager.install_library(spec, upgrade=upgrade)
        
    except Exception as e:
        raise LibraryManagementError(f"Failed to install {library}: {e}")


def install_all(requirements_file: str = "requirements.txt") -> Dict[str, bool]:
    """
    Install all libraries from requirements.txt
    
    Args:
        requirements_file: Path to requirements file
        
    Returns:
        Dict mapping library names to installation success
        
    Example:
        import kepler as kp
        
        # Install all libraries from current project
        results = kp.libs.install_all()
        
        # Check results
        successful = sum(results.values())
        total = len(results)
        print(f"Installed {successful}/{total} libraries")
    """
    logger = get_logger(__name__)
    logger.info("Installing all libraries from requirements.txt...")
    
    manager = _get_manager()
    return manager.install_from_requirements(requirements_file)


def validate() -> Dict[str, Any]:
    """
    Validate current Python library environment
    
    Returns:
        Comprehensive validation report including:
        - Total libraries vs successfully imported
        - Missing libraries list
        - Python version and environment info
        - Individual library status
        
    Example:
        import kepler as kp
        
        report = kp.libs.validate()
        print(f"Environment health: {report['successful_imports']}/{report['total_libraries']}")
        
        if report['missing_libraries']:
            print(f"Missing: {', '.join(report['missing_libraries'])}")
    """
    logger = get_logger(__name__)
    logger.info("Validating library environment...")
    
    manager = _get_manager()
    return manager.validate_environment()


def list_installed() -> List[Dict[str, Any]]:
    """
    Get list of all installed Python libraries with metadata
    
    Returns:
        List of installed libraries with name, version, source, location
        
    Example:
        import kepler as kp
        
        libraries = kp.libs.list_installed()
        for lib in libraries:
            print(f"{lib['name']} {lib['version']} ({lib['source']})")
    """
    manager = _get_manager()
    return manager.get_installed_libraries()


def import_module(module_name: str, force_reload: bool = False) -> Any:
    """
    Dynamically import any Python module with caching
    
    Provides intelligent importing with:
    - Caching for performance
    - Alternative name resolution
    - Force reload for development
    - Clear error messages with suggestions
    
    Args:
        module_name: Name of module to import
        force_reload: Force reload even if cached
        
    Returns:
        Imported module object
        
    Example:
        import kepler as kp
        
        # Dynamic import with caching
        torch = kp.libs.import_module("torch")
        transformers = kp.libs.import_module("transformers")
        
        # Force reload for development
        torch = kp.libs.import_module("torch", force_reload=True)
    """
    manager = _get_manager()
    return manager.dynamic_import(module_name, force_reload=force_reload)


def analyze_dependencies() -> Dict[str, Any]:
    """
    Analyze dependency graph and detect conflicts
    
    Returns:
        Comprehensive dependency analysis including:
        - Total libraries and resolved versions
        - Conflict detection and resolution recommendations
        - Dependency tree analysis
        
    Example:
        import kepler as kp
        
        analysis = kp.libs.analyze_dependencies()
        
        if analysis['conflicts']:
            print(f"Conflicts detected: {len(analysis['conflicts'])}")
            for conflict in analysis['conflicts']:
                print(f"  {conflict['library']}: {conflict['requested']} vs {conflict['installed']}")
        else:
            print("No dependency conflicts detected")
    """
    manager = _get_manager()
    return manager.resolve_dependencies()


def create_lock_file() -> str:
    """
    Create dependency lock file with exact versions
    
    Returns:
        Path to created lock file
        
    Example:
        import kepler as kp
        
        lock_file = kp.libs.create_lock_file()
        print(f"Lock file created: {lock_file}")
    """
    manager = _get_manager()
    manager.create_dependency_lock()
    return str(manager.kepler_lock_file)


def optimize_for_production() -> Dict[str, Any]:
    """
    Analyze environment for production deployment optimization
    
    Returns:
        Production optimization report with:
        - Development-only libraries to remove
        - Size optimization suggestions
        - Alternative library recommendations
        - Production readiness assessment
        
    Example:
        import kepler as kp
        
        report = kp.libs.optimize_for_production()
        
        if not report['production_ready']:
            print("Environment needs optimization:")
            for item in report['development_only']:
                print(f"  Remove: {item['name']} ({item['reason']})")
    """
    manager = _get_manager()
    return manager.optimize_for_production()


def setup_dev_environment(ai_frameworks: List[str] = None) -> bool:
    """
    Setup optimized development environment for AI frameworks
    
    Args:
        ai_frameworks: AI frameworks to optimize for
                      (e.g., ['pytorch', 'transformers', 'langchain'])
    
    Returns:
        True if setup successful
        
    Example:
        import kepler as kp
        
        # Setup for Generative AI development
        kp.libs.setup_dev_environment(['transformers', 'langchain', 'openai'])
        
        # Setup for Computer Vision
        kp.libs.setup_dev_environment(['torch', 'opencv', 'pillow'])
    """
    manager = _get_manager()
    return manager.setup_development_environment(ai_frameworks)


# Convenience functions that match CLI functionality
def install_from_github(repo_url: str, branch: str = None, tag: str = None) -> bool:
    """
    Install library directly from GitHub repository
    
    Args:
        repo_url: GitHub repository URL (https://github.com/user/repo)
        branch: Specific branch to install from
        tag: Specific tag to install from
        
    Returns:
        True if installation successful
        
    Example:
        import kepler as kp
        
        # Install from main branch
        kp.libs.install_from_github("https://github.com/huggingface/transformers")
        
        # Install specific version
        kp.libs.install_from_github("https://github.com/pytorch/pytorch", tag="v2.0.0")
    """
    git_url = f"git+{repo_url}.git"
    if branch:
        git_url += f"@{branch}"
    elif tag:
        git_url += f"@{tag}"
    
    return install(git_url)


def install_from_private_repo(repo_url: str, ssh_key: str = None) -> bool:
    """
    Install library from private repository
    
    Args:
        repo_url: Private repository URL
        ssh_key: Path to SSH key (optional, uses default if None)
        
    Returns:
        True if installation successful
        
    Example:
        import kepler as kp
        
        # Install from private corporate repo
        kp.libs.install_from_private_repo("git@github.com:company/ai-toolkit.git")
    """
    if not repo_url.startswith('git+ssh://'):
        if repo_url.startswith('git@'):
            repo_url = f"git+ssh://{repo_url}"
        else:
            repo_url = f"git+ssh://git@{repo_url}"
    
    return install(repo_url)


def install_local_library(path: str, editable: bool = True) -> bool:
    """
    Install library from local development directory
    
    Args:
        path: Path to local library directory
        editable: Install in editable mode (recommended for development)
        
    Returns:
        True if installation successful
        
    Example:
        import kepler as kp
        
        # Install custom algorithm in development
        kp.libs.install_local_library("./my-custom-algorithm")
        
        # Install without editable mode
        kp.libs.install_local_library("./stable-library", editable=False)
    """
    if editable:
        spec = f"-e {path}"
    else:
        spec = path
    
    return install(spec)


# Integration with existing Kepler ecosystem
def get_ai_framework_info() -> Dict[str, Dict[str, Any]]:
    """
    Get information about supported AI frameworks
    
    Returns:
        Comprehensive information about all AI framework categories
        supported by Kepler, including example libraries and use cases
        
    Example:
        import kepler as kp
        
        frameworks = kp.libs.get_ai_framework_info()
        
        for category, info in frameworks.items():
            print(f"{category}: {info['description']}")
            print(f"  Examples: {', '.join(info['examples'])}")
    """
    return {
        'machine_learning': {
            'description': 'Traditional ML algorithms and statistical learning',
            'examples': ['scikit-learn', 'xgboost', 'lightgbm', 'catboost'],
            'use_cases': ['Classification', 'Regression', 'Clustering', 'Dimensionality Reduction'],
            'template': 'ml'
        },
        'deep_learning': {
            'description': 'Neural networks and deep learning frameworks',
            'examples': ['pytorch', 'tensorflow', 'keras', 'jax', 'lightning'],
            'use_cases': ['Neural Networks', 'CNN', 'RNN', 'LSTM', 'Transformers'],
            'template': 'deep_learning'
        },
        'generative_ai': {
            'description': 'Generative AI, LLMs, and AI agents',
            'examples': ['transformers', 'langchain', 'openai', 'anthropic', 'diffusers'],
            'use_cases': ['Text Generation', 'Image Generation', 'AI Agents', 'Chatbots', 'Fine-tuning'],
            'template': 'generative_ai'
        },
        'computer_vision': {
            'description': 'Image processing and computer vision',
            'examples': ['opencv-python', 'pillow', 'torchvision', 'albumentations'],
            'use_cases': ['Image Classification', 'Object Detection', 'Image Segmentation', 'OCR'],
            'template': 'computer_vision'
        },
        'nlp': {
            'description': 'Natural language processing and text analysis',
            'examples': ['spacy', 'nltk', 'transformers', 'datasets', 'tokenizers'],
            'use_cases': ['Text Classification', 'Sentiment Analysis', 'NER', 'Language Translation'],
            'template': 'nlp'
        }
    }


# Advanced functionality
def check_compatibility(libraries: List[str]) -> Dict[str, Any]:
    """
    Check compatibility between multiple libraries
    
    Args:
        libraries: List of library names to check compatibility
        
    Returns:
        Compatibility analysis report
        
    Example:
        import kepler as kp
        
        # Check if PyTorch and TensorFlow can coexist
        compat = kp.libs.check_compatibility(['torch', 'tensorflow'])
        
        if compat['compatible']:
            print("Libraries are compatible")
        else:
            print("Potential conflicts detected")
    """
    manager = _get_manager()
    
    # Simple compatibility check - in production would be more sophisticated
    compatibility_report = {
        'libraries': libraries,
        'compatible': True,
        'warnings': [],
        'conflicts': []
    }
    
    # Check for known incompatibilities
    known_conflicts = [
        (['tensorflow', 'torch'], 'Both frameworks may compete for GPU resources'),
        (['opencv-python', 'opencv-contrib-python'], 'Use only one OpenCV variant'),
    ]
    
    for conflict_libs, warning in known_conflicts:
        if all(lib in libraries for lib in conflict_libs):
            compatibility_report['warnings'].append({
                'libraries': conflict_libs,
                'warning': warning
            })
    
    return compatibility_report


def get_framework_recommendations(use_case: str) -> Dict[str, List[str]]:
    """
    Get AI framework recommendations for specific use cases
    
    Args:
        use_case: Description of the use case
        
    Returns:
        Recommended libraries by category
        
    Example:
        import kepler as kp
        
        # Get recommendations for image analysis
        recs = kp.libs.get_framework_recommendations("industrial image analysis")
        
        for category, libs in recs.items():
            print(f"{category}: {', '.join(libs)}")
    """
    # Simplified recommendation system - in production would use ML/NLP to analyze use case
    use_case_lower = use_case.lower()
    
    recommendations = {
        'primary': [],
        'supporting': [],
        'optional': []
    }
    
    # Image/Vision related
    if any(word in use_case_lower for word in ['image', 'vision', 'camera', 'photo', 'visual']):
        recommendations['primary'].extend(['opencv-python', 'pillow'])
        recommendations['supporting'].extend(['torch', 'torchvision'])
        recommendations['optional'].extend(['albumentations', 'scikit-image'])
    
    # Text/NLP related
    if any(word in use_case_lower for word in ['text', 'language', 'nlp', 'chat', 'sentiment']):
        recommendations['primary'].extend(['transformers', 'spacy'])
        recommendations['supporting'].extend(['datasets', 'tokenizers'])
        recommendations['optional'].extend(['langchain', 'nltk'])
    
    # Time series/Predictive
    if any(word in use_case_lower for word in ['predict', 'forecast', 'time', 'series', 'trend']):
        recommendations['primary'].extend(['scikit-learn', 'pandas'])
        recommendations['supporting'].extend(['xgboost', 'lightgbm'])
        recommendations['optional'].extend(['prophet', 'statsmodels'])
    
    # Generative AI
    if any(word in use_case_lower for word in ['generate', 'create', 'llm', 'gpt', 'ai agent']):
        recommendations['primary'].extend(['transformers', 'langchain'])
        recommendations['supporting'].extend(['openai', 'anthropic'])
        recommendations['optional'].extend(['diffusers', 'accelerate'])
    
    # Remove duplicates while preserving order
    for category in recommendations:
        recommendations[category] = list(dict.fromkeys(recommendations[category]))
    
    return recommendations
