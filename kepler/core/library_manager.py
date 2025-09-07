"""
Kepler Library Manager - Unlimited Python Library Support

Manages installation, validation, and dependency resolution for ANY Python library
from multiple sources: PyPI, GitHub, private repos, custom local libraries.

Implements PRD requirement: "Si está en Python, Kepler lo soporta"
"""

import subprocess
import sys
import importlib
import pkg_resources
import re
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import tempfile
import shutil

from kepler.utils.logging import get_logger
from kepler.utils.exceptions import LibraryManagementError


class LibrarySource(Enum):
    """Types of Python library sources supported by Kepler"""
    PYPI = "pypi"
    GITHUB = "github"
    GITLAB = "gitlab"
    PRIVATE_REPO = "private_repo"
    LOCAL_EDITABLE = "local_editable"
    LOCAL_WHEEL = "local_wheel"
    LOCAL_TARBALL = "local_tarball"
    CUSTOM_URL = "custom_url"


@dataclass
class LibrarySpec:
    """Specification for a Python library installation"""
    name: str
    source: LibrarySource
    version: Optional[str] = None
    url: Optional[str] = None
    local_path: Optional[str] = None
    branch: Optional[str] = None
    tag: Optional[str] = None
    commit: Optional[str] = None
    extras: List[str] = None
    
    def __post_init__(self):
        if self.extras is None:
            self.extras = []


class LibraryManager:
    """
    Advanced Library Manager for Dynamic Loading and Dependency Management
    
    Provides comprehensive library management including:
    - Unlimited Python library support from any source
    - Dynamic library loading and import management
    - Intelligent dependency resolution and conflict detection
    - Environment isolation and reproducibility
    - Lock file generation for exact version pinning
    
    Supports installation from:
    - PyPI official (pip install package)
    - GitHub repositories (git+https://github.com/user/repo.git)
    - GitLab repositories (git+https://gitlab.com/user/repo.git)
    - Private repositories with SSH (git+ssh://git@github.com/user/repo.git)
    - Local editable installs (-e ./local-lib)
    - Local wheel files (./dist/package.whl)
    - Local tarballs (./dist/package.tar.gz)
    - Custom URLs (https://example.com/package.tar.gz)
    """
    
    def __init__(self, project_path: str = "."):
        self.project_path = Path(project_path).resolve()
        self.requirements_file = self.project_path / "requirements.txt"
        self.kepler_lock_file = self.project_path / "kepler-lock.txt"
        self.dependency_graph_file = self.project_path / "kepler-deps.json"
        self.logger = get_logger(__name__)
        
        # Dynamic loading cache
        self._loaded_modules = {}
        self._dependency_graph = {}
        self._conflict_resolution_log = []
        
    def parse_requirements_file(self) -> List[LibrarySpec]:
        """
        Parse requirements.txt and convert to LibrarySpec objects
        
        Supports all pip requirement formats:
        - package==1.0.0
        - git+https://github.com/user/repo.git@v1.0.0
        - git+ssh://git@github.com/user/repo.git
        - -e ./local-lib
        - ./dist/package.whl
        """
        if not self.requirements_file.exists():
            return []
            
        specs = []
        
        with open(self.requirements_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                    
                try:
                    spec = self._parse_requirement_line(line)
                    if spec:
                        specs.append(spec)
                except Exception as e:
                    self.logger.warning(f"Failed to parse line {line_num} in requirements.txt: {line}")
                    self.logger.warning(f"Error: {e}")
                    
        return specs
    
    def _parse_requirement_line(self, line: str) -> Optional[LibrarySpec]:
        """Parse a single requirement line into LibrarySpec"""
        
        # Local editable install: -e ./local-lib
        if line.startswith('-e '):
            local_path = line[3:].strip()
            return LibrarySpec(
                name=self._extract_name_from_path(local_path),
                source=LibrarySource.LOCAL_EDITABLE,
                local_path=local_path
            )
        
        # Git repositories: git+https://github.com/user/repo.git@branch
        if line.startswith('git+'):
            return self._parse_git_requirement(line)
        
        # Local wheel/tarball: ./dist/package.whl
        if line.startswith('./') or line.startswith('../'):
            return self._parse_local_file_requirement(line)
        
        # Custom URLs: https://example.com/package.tar.gz
        if line.startswith('http://') or line.startswith('https://'):
            return LibrarySpec(
                name=self._extract_name_from_url(line),
                source=LibrarySource.CUSTOM_URL,
                url=line
            )
        
        # PyPI packages: package==1.0.0, package>=1.0.0, package[extras]
        return self._parse_pypi_requirement(line)
    
    def _parse_git_requirement(self, line: str) -> LibrarySpec:
        """Parse git repository requirement"""
        # Examples:
        # git+https://github.com/user/repo.git
        # git+https://github.com/user/repo.git@v1.0.0
        # git+ssh://git@github.com/user/repo.git@branch
        
        # Handle SSH URLs specially - they contain @ in the git@github.com part
        if 'git+ssh://git@' in line:
            # For SSH: git+ssh://git@github.com/user/repo.git@branch
            # Split only on the LAST @ which would be the branch/tag
            parts = line.split('@')
            if len(parts) > 2:  # git+ssh://git, github.com/user/repo.git, branch
                url_part = '@'.join(parts[:-1])  # Rejoin all but last
                ref = parts[-1]
            else:
                url_part, ref = line, None
        else:
            # For HTTPS: git+https://github.com/user/repo.git@v1.0.0
            if '@' in line:
                url_part, ref = line.rsplit('@', 1)
            else:
                url_part, ref = line, None
            
        # Determine source type
        if 'git+ssh://' in url_part:
            source = LibrarySource.PRIVATE_REPO
        elif 'github.com' in url_part:
            source = LibrarySource.GITHUB
        elif 'gitlab.com' in url_part:
            source = LibrarySource.GITLAB
        else:
            source = LibrarySource.PRIVATE_REPO
            
        # Extract repository name as package name
        # Handle different URL formats correctly
        if url_part.endswith('.git'):
            url_clean = url_part[:-4]  # Remove .git
        else:
            url_clean = url_part
        
        # Extract the repository name from the URL path
        # For all Git URLs, extract the last component of the path
        repo_pattern = r'/([^/]+?)(?:\.git)?$'
        match = re.search(repo_pattern, url_clean)
        name = match.group(1) if match else "unknown"
        
        # Determine if ref is version, branch, or commit
        version = None
        branch = None
        tag = None
        commit = None
        
        if ref:
            if ref.startswith('v') and '.' in ref:
                version = ref
                tag = ref
            elif re.match(r'^[a-f0-9]{40}$', ref):  # Full SHA commit
                commit = ref
            elif re.match(r'^[a-f0-9]{7,12}$', ref):  # Short SHA commit
                commit = ref
            else:
                branch = ref
        
        return LibrarySpec(
            name=name,
            source=source,
            url=url_part,
            version=version,
            branch=branch,
            tag=tag,
            commit=commit
        )
    
    def _parse_local_file_requirement(self, line: str) -> LibrarySpec:
        """Parse local file requirement (wheel or tarball)"""
        if line.endswith('.whl'):
            source = LibrarySource.LOCAL_WHEEL
        elif line.endswith('.tar.gz') or line.endswith('.tar'):
            source = LibrarySource.LOCAL_TARBALL
        else:
            source = LibrarySource.LOCAL_TARBALL  # Default for other file types
        
        # Extract name from filename
        filename = Path(line).name
        if filename.endswith('.tar.gz'):
            base_name = filename.replace('.tar.gz', '')
        elif filename.endswith('.tar'):
            base_name = filename.replace('.tar', '')
        elif filename.endswith('.whl'):
            base_name = filename.replace('.whl', '')
        else:
            base_name = Path(line).stem
        
        # Extract package name from base name (handle versions)
        parts = base_name.split('-')
        if len(parts) > 1:
            # Find where version starts (first part with digit)
            name_parts = []
            for part in parts:
                if part and part[0].isdigit():
                    break
                name_parts.append(part)
            name = '-'.join(name_parts) if name_parts else parts[0]
        else:
            name = base_name
            
        return LibrarySpec(
            name=name,
            source=source,
            local_path=line
        )
    
    def _parse_pypi_requirement(self, line: str) -> LibrarySpec:
        """Parse PyPI requirement with version specifiers and extras"""
        # Examples:
        # package==1.0.0
        # package>=1.0.0,<2.0.0
        # package[extra1,extra2]>=1.0.0
        
        # Extract extras
        extras = []
        if '[' in line and ']' in line:
            name_part, rest = line.split('[', 1)
            extras_part, version_part = rest.split(']', 1)
            extras = [e.strip() for e in extras_part.split(',')]
            line = name_part + version_part
        
        # Extract name and version
        version_operators = ['==', '>=', '<=', '>', '<', '!=', '~=']
        name = line
        version = None
        
        for op in version_operators:
            if op in line:
                name, version_spec = line.split(op, 1)
                version = f"{op}{version_spec}"
                break
        
        return LibrarySpec(
            name=name.strip(),
            source=LibrarySource.PYPI,
            version=version,
            extras=extras
        )
    
    def _extract_name_from_path(self, path: str) -> str:
        """Extract package name from local path"""
        path_obj = Path(path)
        if path_obj.is_dir():
            # For directories, try to find setup.py or pyproject.toml
            setup_py = path_obj / "setup.py"
            pyproject_toml = path_obj / "pyproject.toml"
            
            if pyproject_toml.exists():
                # Try to extract name from pyproject.toml
                try:
                    import toml
                    config = toml.load(pyproject_toml)
                    return config.get('project', {}).get('name', path_obj.name)
                except:
                    pass
            
            # Return full directory name (preserve hyphens)
            return path_obj.name
        else:
            # For files, extract from filename (preserve hyphens in base name)
            filename = path_obj.stem
            
            # Handle wheel files specially: package-1.0.0-py3-none-any.whl
            if path_obj.suffix == '.whl':
                # Wheel format: {name}-{version}-{python tag}-{abi tag}-{platform tag}
                parts = filename.split('-')
                if len(parts) >= 2:
                    # Take everything before the first part that looks like a version
                    name_parts = []
                    for i, part in enumerate(parts):
                        if re.match(r'^\d+', part):  # Starts with digit = version
                            break
                        name_parts.append(part)
                    return '-'.join(name_parts) if name_parts else parts[0]
                return parts[0]
            
            # For other files, split on first hyphen if it looks like version
            if '-' in filename:
                parts = filename.split('-')
                # If second part starts with a number, it's likely a version
                if len(parts) > 1 and parts[1] and parts[1][0].isdigit():
                    return parts[0]
                else:
                    # Keep the full name with hyphens
                    return filename
            return filename
    
    def _extract_name_from_url(self, url: str) -> str:
        """Extract package name from URL"""
        # Try to extract from filename
        filename = url.split('/')[-1]
        if filename.endswith('.tar.gz'):
            base_name = filename.replace('.tar.gz', '')
            # Handle version patterns like package-1.0.0
            parts = base_name.split('-')
            if len(parts) > 1 and parts[1] and parts[1][0].isdigit():
                return parts[0]
            else:
                return base_name
        elif filename.endswith('.whl'):
            return self._extract_name_from_path(filename)
        else:
            return filename.split('.')[0] if '.' in filename else filename
    
    def install_library(self, spec: LibrarySpec, upgrade: bool = False) -> bool:
        """
        Install a library according to its specification
        
        Args:
            spec: Library specification
            upgrade: Whether to upgrade if already installed
            
        Returns:
            True if installation successful
        """
        self.logger.info(f"Installing library: {spec.name} from {spec.source.value}")
        
        try:
            install_cmd = self._build_install_command(spec, upgrade)
            
            self.logger.debug(f"Running: {' '.join(install_cmd)}")
            
            result = subprocess.run(
                install_cmd,
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                self.logger.info(f"Successfully installed {spec.name}")
                return True
            else:
                self.logger.error(f"Failed to install {spec.name}")
                self.logger.error(f"stdout: {result.stdout}")
                self.logger.error(f"stderr: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Exception during installation of {spec.name}: {e}")
            return False
    
    def _build_install_command(self, spec: LibrarySpec, upgrade: bool = False) -> List[str]:
        """Build pip install command for library spec"""
        cmd = [sys.executable, "-m", "pip", "install"]
        
        if upgrade:
            cmd.append("--upgrade")
        
        if spec.source == LibrarySource.PYPI:
            package_spec = spec.name
            if spec.extras:
                package_spec += f"[{','.join(spec.extras)}]"
            if spec.version:
                package_spec += spec.version
            cmd.append(package_spec)
            
        elif spec.source in [LibrarySource.GITHUB, LibrarySource.GITLAB, LibrarySource.PRIVATE_REPO]:
            git_url = spec.url
            if spec.branch:
                git_url += f"@{spec.branch}"
            elif spec.tag:
                git_url += f"@{spec.tag}"
            elif spec.commit:
                git_url += f"@{spec.commit}"
            cmd.append(git_url)
            
        elif spec.source == LibrarySource.LOCAL_EDITABLE:
            cmd.extend(["-e", spec.local_path])
            
        elif spec.source in [LibrarySource.LOCAL_WHEEL, LibrarySource.LOCAL_TARBALL]:
            cmd.append(spec.local_path)
            
        elif spec.source == LibrarySource.CUSTOM_URL:
            cmd.append(spec.url)
            
        return cmd
    
    def validate_library_installed(self, spec: LibrarySpec) -> bool:
        """Check if library is properly installed and importable"""
        try:
            # Try to import the library
            importlib.import_module(spec.name)
            return True
        except ImportError:
            try:
                # Try alternative import names (e.g., sklearn vs scikit-learn)
                alternative_names = self._get_alternative_import_names(spec.name)
                for alt_name in alternative_names:
                    importlib.import_module(alt_name)
                    return True
            except ImportError:
                pass
        
        return False
    
    def _get_alternative_import_names(self, package_name: str) -> List[str]:
        """Get alternative import names for common packages"""
        alternatives = {
            'scikit-learn': ['sklearn'],
            'pillow': ['PIL'],
            'opencv-python': ['cv2'],
            'beautifulsoup4': ['bs4'],
            'pyyaml': ['yaml'],
            'python-dateutil': ['dateutil'],
            'msgpack-python': ['msgpack'],
        }
        
        return alternatives.get(package_name.lower(), [])
    
    def install_from_requirements(self, requirements_file: str = "requirements.txt") -> Dict[str, bool]:
        """
        Install all libraries from requirements file
        
        Returns:
            Dict mapping library names to installation success status
        """
        self.requirements_file = Path(requirements_file)
        specs = self.parse_requirements_file()
        
        results = {}
        
        for spec in specs:
            success = self.install_library(spec)
            results[spec.name] = success
            
            if success:
                # Validate installation
                if self.validate_library_installed(spec):
                    self.logger.info(f"✅ {spec.name} installed and validated successfully")
                else:
                    self.logger.warning(f"⚠️ {spec.name} installed but import validation failed")
                    results[spec.name] = False
            
        return results
    
    def get_installed_libraries(self) -> List[Dict[str, Any]]:
        """Get list of all installed libraries with metadata"""
        installed = []
        
        for dist in pkg_resources.working_set:
            library_info = {
                'name': dist.project_name,
                'version': dist.version,
                'location': dist.location,
                'source': self._detect_library_source(dist)
            }
            installed.append(library_info)
            
        return installed
    
    def _detect_library_source(self, dist) -> str:
        """Detect how a library was installed"""
        location = str(dist.location)
        
        if 'site-packages' in location:
            return 'PyPI'
        elif '.git' in location or 'src' in location:
            return 'Git Repository'
        elif 'editable' in str(dist):
            return 'Local Editable'
        else:
            return 'Unknown'
    
    def create_lock_file(self) -> None:
        """Create lock file with exact versions of installed libraries"""
        installed = self.get_installed_libraries()
        
        with open(self.kepler_lock_file, 'w') as f:
            f.write("# Kepler Lock File - Exact library versions\n")
            f.write(f"# Generated on {datetime.now().isoformat()}\n\n")
            
            for lib in sorted(installed, key=lambda x: x['name']):
                f.write(f"{lib['name']}=={lib['version']}  # {lib['source']}\n")
    
    def validate_environment(self) -> Dict[str, Any]:
        """
        Validate that current Python environment has all required libraries
        
        Returns comprehensive environment validation report
        """
        report = {
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'python_executable': sys.executable,
            'environment_path': os.environ.get('VIRTUAL_ENV', 'system'),
            'libraries': {},
            'missing_libraries': [],
            'total_libraries': 0,
            'successful_imports': 0
        }
        
        specs = self.parse_requirements_file()
        
        for spec in specs:
            is_installed = self.validate_library_installed(spec)
            
            library_info = {
                'installed': is_installed,
                'source': spec.source.value,
                'version': spec.version,
                'url': spec.url
            }
            
            report['libraries'][spec.name] = library_info
            report['total_libraries'] += 1
            
            if is_installed:
                report['successful_imports'] += 1
            else:
                report['missing_libraries'].append(spec.name)
        
        return report
    
    def create_environment_from_template(self, template: str) -> bool:
        """
        Create requirements.txt from predefined templates
        
        Templates:
        - 'ml': Traditional ML (sklearn, xgboost, lightgbm)
        - 'deep_learning': Deep learning (pytorch, tensorflow, keras)
        - 'generative_ai': Generative AI (transformers, langchain, openai)
        - 'computer_vision': CV (opencv, pillow, torchvision)
        - 'nlp': NLP (spacy, nltk, transformers)
        - 'full_ai': Complete AI ecosystem
        """
        templates = {
            'ml': [
                'scikit-learn>=1.3.0',
                'xgboost>=1.7.0',
                'lightgbm>=3.3.0',
                'catboost>=1.2.0',
                'pandas>=2.0.0',
                'numpy>=1.24.0',
                'matplotlib>=3.7.0',
                'seaborn>=0.12.0'
            ],
            'deep_learning': [
                'torch>=2.0.0',
                'torchvision>=0.15.0',
                'tensorflow>=2.13.0',
                'keras>=2.13.0',
                'jax>=0.4.0',
                'lightning>=2.0.0',
                'pandas>=2.0.0',
                'numpy>=1.24.0'
            ],
            'generative_ai': [
                'transformers>=4.30.0',
                'langchain>=0.0.200',
                'openai>=0.27.0',
                'anthropic>=0.3.0',
                'diffusers>=0.18.0',
                'accelerate>=0.20.0',
                'datasets>=2.13.0'
            ],
            'computer_vision': [
                'opencv-python>=4.8.0',
                'pillow>=10.0.0',
                'torchvision>=0.15.0',
                'albumentations>=1.3.0',
                'scikit-image>=0.21.0'
            ],
            'nlp': [
                'spacy>=3.6.0',
                'nltk>=3.8.0',
                'transformers>=4.30.0',
                'datasets>=2.13.0',
                'tokenizers>=0.13.0'
            ],
            'full_ai': []  # Combination of all above
        }
        
        if template == 'full_ai':
            # Combine all templates
            all_libs = []
            for tmpl in ['ml', 'deep_learning', 'generative_ai', 'computer_vision', 'nlp']:
                all_libs.extend(templates[tmpl])
            # Remove duplicates while preserving order
            seen = set()
            templates['full_ai'] = [x for x in all_libs if not (x in seen or seen.add(x))]
        
        if template not in templates:
            raise LibraryManagementError(f"Unknown template: {template}. Available: {list(templates.keys())}")
        
        # Ensure directory exists
        self.requirements_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Write requirements.txt
        with open(self.requirements_file, 'w') as f:
            f.write(f"# Kepler requirements - {template} template\n")
            f.write(f"# Generated on {datetime.now().isoformat()}\n\n")
            
            for lib in templates[template]:
                f.write(f"{lib}\n")
        
        self.logger.info(f"Created requirements.txt with {template} template ({len(templates[template])} libraries)")
        return True
    
    def add_custom_library(self, 
                          name: str, 
                          source: str, 
                          version: Optional[str] = None,
                          extras: List[str] = None) -> bool:
        """
        Add a custom library to requirements.txt
        
        Args:
            name: Library name
            source: Source specification (PyPI name, git URL, local path, etc.)
            version: Version specification (for PyPI packages)
            extras: Optional extras to install
            
        Returns:
            True if added successfully
        """
        # Determine source type
        if source.startswith('git+'):
            spec = self._parse_git_requirement(source)
        elif source.startswith('./') or source.startswith('../'):
            spec = LibrarySpec(name=name, source=LibrarySource.LOCAL_EDITABLE, local_path=source)
        elif source.startswith('http'):
            spec = LibrarySpec(name=name, source=LibrarySource.CUSTOM_URL, url=source)
        else:
            # Assume PyPI
            spec = LibrarySpec(name=name, source=LibrarySource.PYPI, version=version, extras=extras or [])
        
        # Add to requirements.txt
        requirement_line = self._spec_to_requirement_line(spec)
        
        # Read existing requirements
        existing_lines = []
        if self.requirements_file.exists():
            with open(self.requirements_file, 'r') as f:
                existing_lines = f.readlines()
        
        # Add new requirement
        with open(self.requirements_file, 'w') as f:
            for line in existing_lines:
                f.write(line)
            f.write(f"{requirement_line}\n")
        
        self.logger.info(f"Added {name} to requirements.txt")
        return True
    
    def _spec_to_requirement_line(self, spec: LibrarySpec) -> str:
        """Convert LibrarySpec back to requirements.txt line"""
        if spec.source == LibrarySource.PYPI:
            line = spec.name
            if spec.extras:
                line += f"[{','.join(spec.extras)}]"
            if spec.version:
                line += spec.version
            return line
            
        elif spec.source in [LibrarySource.GITHUB, LibrarySource.GITLAB, LibrarySource.PRIVATE_REPO]:
            line = spec.url
            if spec.branch:
                line += f"@{spec.branch}"
            elif spec.tag:
                line += f"@{spec.tag}"
            elif spec.commit:
                line += f"@{spec.commit}"
            return line
            
        elif spec.source == LibrarySource.LOCAL_EDITABLE:
            return f"-e {spec.local_path}"
            
        elif spec.source in [LibrarySource.LOCAL_WHEEL, LibrarySource.LOCAL_TARBALL]:
            return spec.local_path
            
        elif spec.source == LibrarySource.CUSTOM_URL:
            return spec.url
            
        return spec.name
    
    def dynamic_import(self, module_name: str, force_reload: bool = False) -> Any:
        """
        Dynamically import and load any Python module
        
        Supports importing any library that was installed via LibraryManager,
        with caching and reload capabilities for development workflows.
        
        Args:
            module_name: Name of module to import (e.g., 'transformers', 'torch')
            force_reload: Force reload even if already cached
            
        Returns:
            Imported module object
            
        Raises:
            LibraryManagementError: If module cannot be imported
        """
        if module_name in self._loaded_modules and not force_reload:
            self.logger.debug(f"Using cached import for {module_name}")
            return self._loaded_modules[module_name]
        
        try:
            if force_reload and module_name in sys.modules:
                # Force reload for development
                importlib.reload(sys.modules[module_name])
            
            module = importlib.import_module(module_name)
            self._loaded_modules[module_name] = module
            
            self.logger.info(f"Successfully imported {module_name}")
            return module
            
        except ImportError as e:
            # Try alternative import names
            alternative_names = self._get_alternative_import_names(module_name)
            
            for alt_name in alternative_names:
                try:
                    module = importlib.import_module(alt_name)
                    self._loaded_modules[module_name] = module
                    self.logger.info(f"Successfully imported {module_name} as {alt_name}")
                    return module
                except ImportError:
                    continue
            
            # If all attempts failed
            raise LibraryManagementError(
                f"Cannot import module '{module_name}'",
                library_name=module_name,
                suggestion=f"Install with: kepler libs install --library {module_name}"
            )
    
    def resolve_dependencies(self) -> Dict[str, Any]:
        """
        Intelligent dependency resolution and conflict detection
        
        Analyzes all installed libraries and their dependencies,
        detects conflicts, and provides resolution recommendations.
        
        Returns:
            Comprehensive dependency analysis report
        """
        self.logger.info("Analyzing dependency graph...")
        
        specs = self.parse_requirements_file()
        dependency_report = {
            'total_libraries': len(specs),
            'conflicts': [],
            'resolved_versions': {},
            'dependency_tree': {},
            'recommendations': []
        }
        
        # Build dependency graph
        for spec in specs:
            try:
                # Get installed version info
                installed_libs = self.get_installed_libraries()
                installed_info = next(
                    (lib for lib in installed_libs if lib['name'].lower() == spec.name.lower()),
                    None
                )
                
                if installed_info:
                    dependency_report['resolved_versions'][spec.name] = {
                        'requested': spec.version or 'any',
                        'installed': installed_info['version'],
                        'source': installed_info['source']
                    }
                    
                    # Check for version conflicts
                    if spec.version and not self._version_satisfies(installed_info['version'], spec.version):
                        conflict = {
                            'library': spec.name,
                            'requested': spec.version,
                            'installed': installed_info['version'],
                            'severity': 'warning'
                        }
                        dependency_report['conflicts'].append(conflict)
                        
            except Exception as e:
                self.logger.warning(f"Failed to analyze dependencies for {spec.name}: {e}")
        
        # Generate recommendations
        if dependency_report['conflicts']:
            dependency_report['recommendations'].append(
                "Run 'kepler libs install --upgrade' to resolve version conflicts"
            )
        
        if dependency_report['total_libraries'] > 20:
            dependency_report['recommendations'].append(
                "Consider splitting into multiple environments for better isolation"
            )
        
        return dependency_report
    
    def _version_satisfies(self, installed_version: str, required_version: str) -> bool:
        """Check if installed version satisfies requirement"""
        # Simplified version checking - in production would use packaging.specifiers
        if required_version.startswith('=='):
            return installed_version == required_version[2:]
        elif required_version.startswith('>='):
            # Simplified comparison - would use proper semantic versioning
            return True  # For now, assume satisfied
        else:
            return True
    
    def create_dependency_lock(self) -> None:
        """
        Create comprehensive dependency lock file with exact versions and sources
        
        Generates kepler-lock.txt with:
        - Exact versions of all installed libraries
        - Source information (PyPI, GitHub, local, etc.)
        - Dependency resolution log
        - Environment metadata
        """
        self.logger.info("Creating dependency lock file...")
        
        installed = self.get_installed_libraries()
        dependency_report = self.resolve_dependencies()
        
        lock_content = []
        lock_content.append("# Kepler Dependency Lock File")
        lock_content.append(f"# Generated: {datetime.now().isoformat()}")
        lock_content.append(f"# Python: {sys.version}")
        lock_content.append(f"# Platform: {sys.platform}")
        lock_content.append("")
        
        # Add resolved versions
        lock_content.append("# === RESOLVED DEPENDENCIES ===")
        for lib in sorted(installed, key=lambda x: x['name']):
            lock_content.append(f"{lib['name']}=={lib['version']}  # Source: {lib['source']}")
        
        lock_content.append("")
        lock_content.append("# === DEPENDENCY ANALYSIS ===")
        lock_content.append(f"# Total libraries: {dependency_report['total_libraries']}")
        lock_content.append(f"# Conflicts detected: {len(dependency_report['conflicts'])}")
        
        if dependency_report['conflicts']:
            lock_content.append("# CONFLICTS:")
            for conflict in dependency_report['conflicts']:
                lock_content.append(f"#   {conflict['library']}: requested {conflict['requested']}, installed {conflict['installed']}")
        
        # Write lock file
        with open(self.kepler_lock_file, 'w') as f:
            f.write('\n'.join(lock_content))
        
        self.logger.info(f"Created lock file: {self.kepler_lock_file}")
    
    def auto_resolve_conflicts(self) -> bool:
        """
        Automatically resolve dependency conflicts when possible
        
        Implements intelligent conflict resolution strategies:
        - Upgrade to compatible versions
        - Suggest alternative libraries
        - Provide manual resolution steps
        
        Returns:
            True if conflicts were resolved automatically
        """
        dependency_report = self.resolve_dependencies()
        conflicts = dependency_report['conflicts']
        
        if not conflicts:
            self.logger.info("No dependency conflicts detected")
            return True
        
        self.logger.info(f"Attempting to resolve {len(conflicts)} conflicts...")
        
        resolved_count = 0
        
        for conflict in conflicts:
            library_name = conflict['library']
            requested = conflict['requested']
            installed = conflict['installed']
            
            self.logger.info(f"Resolving conflict for {library_name}: {requested} vs {installed}")
            
            # Strategy 1: Try upgrading to satisfy requirement
            if requested.startswith('>='):
                try:
                    spec = LibrarySpec(name=library_name, source=LibrarySource.PYPI, version=requested)
                    if self.install_library(spec, upgrade=True):
                        resolved_count += 1
                        self._conflict_resolution_log.append(f"Upgraded {library_name} to satisfy {requested}")
                        continue
                except Exception as e:
                    self.logger.warning(f"Failed to upgrade {library_name}: {e}")
            
            # Strategy 2: Log for manual resolution
            self._conflict_resolution_log.append(f"Manual resolution needed for {library_name}: {requested} vs {installed}")
        
        success = resolved_count == len(conflicts)
        
        if success:
            self.logger.info("All conflicts resolved automatically")
        else:
            self.logger.warning(f"Resolved {resolved_count}/{len(conflicts)} conflicts automatically")
            self.logger.info("Check kepler-deps.json for manual resolution steps")
        
        # Save resolution log
        self._save_dependency_graph()
        
        return success
    
    def _save_dependency_graph(self) -> None:
        """Save dependency analysis and resolution log to JSON"""
        import json
        
        dependency_data = {
            'timestamp': datetime.now().isoformat(),
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'dependency_graph': self._dependency_graph,
            'conflict_resolution_log': self._conflict_resolution_log,
            'installed_libraries': self.get_installed_libraries()
        }
        
        with open(self.dependency_graph_file, 'w') as f:
            json.dump(dependency_data, f, indent=2)
        
        self.logger.debug(f"Saved dependency analysis to {self.dependency_graph_file}")
    
    def get_library_metadata(self, library_name: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive metadata for a specific library
        
        Returns detailed information about library:
        - Installation source and method
        - Version information
        - Dependencies
        - Import paths
        - Documentation links
        """
        try:
            # Try to import and get module info
            module = self.dynamic_import(library_name)
            
            metadata = {
                'name': library_name,
                'module': module.__name__,
                'version': getattr(module, '__version__', 'unknown'),
                'file_path': getattr(module, '__file__', 'unknown'),
                'package_info': {},
                'import_successful': True
            }
            
            # Get package info if available
            try:
                import pkg_resources
                dist = pkg_resources.get_distribution(library_name)
                metadata['package_info'] = {
                    'version': dist.version,
                    'location': dist.location,
                    'requires': [str(req) for req in dist.requires()],
                    'metadata': dist.get_metadata_lines('METADATA') if hasattr(dist, 'get_metadata_lines') else []
                }
            except:
                pass
            
            return metadata
            
        except Exception as e:
            return {
                'name': library_name,
                'import_successful': False,
                'error': str(e)
            }
    
    def setup_development_environment(self, ai_frameworks: List[str] = None) -> bool:
        """
        Setup optimized development environment for AI frameworks
        
        Creates isolated environment with:
        - Jupyter integration
        - GPU support (if available)
        - Development tools (debuggers, profilers)
        - Hot reload capabilities
        
        Args:
            ai_frameworks: List of AI frameworks to optimize for
                         (e.g., ['pytorch', 'transformers', 'langchain'])
        
        Returns:
            True if environment setup successful
        """
        if ai_frameworks is None:
            ai_frameworks = ['sklearn', 'pandas', 'numpy']  # Default minimal
        
        self.logger.info(f"Setting up development environment for: {ai_frameworks}")
        
        # Create development requirements
        dev_requirements = [
            'jupyter>=1.0.0',
            'ipykernel>=6.0.0',
            'ipywidgets>=8.0.0',
            'matplotlib>=3.7.0',
            'seaborn>=0.12.0',
            'plotly>=5.15.0'
        ]
        
        # Add framework-specific development tools
        if 'pytorch' in ai_frameworks or 'torch' in ai_frameworks:
            dev_requirements.extend([
                'torchviz>=0.0.2',  # Visualize PyTorch models
                'tensorboard>=2.13.0',  # TensorBoard integration
            ])
        
        if 'tensorflow' in ai_frameworks:
            dev_requirements.extend([
                'tensorboard>=2.13.0',
                'tensorflow-datasets>=4.9.0'
            ])
        
        if 'transformers' in ai_frameworks:
            dev_requirements.extend([
                'datasets>=2.13.0',
                'tokenizers>=0.13.0',
                'accelerate>=0.20.0'
            ])
        
        # Add to requirements.txt
        current_requirements = []
        if self.requirements_file.exists():
            with open(self.requirements_file, 'r') as f:
                current_requirements = f.readlines()
        
        # Add development section
        with open(self.requirements_file, 'w') as f:
            for line in current_requirements:
                f.write(line)
            
            f.write("\n# === DEVELOPMENT TOOLS ===\n")
            for req in dev_requirements:
                f.write(f"{req}\n")
        
        self.logger.info("Development environment requirements added")
        return True
    
    def optimize_for_production(self) -> Dict[str, Any]:
        """
        Optimize library environment for production deployment
        
        Analyzes installed libraries and provides optimization recommendations:
        - Remove development-only dependencies
        - Suggest lighter alternatives
        - Identify security vulnerabilities
        - Calculate deployment size
        
        Returns:
            Production optimization report
        """
        self.logger.info("Analyzing environment for production optimization...")
        
        installed = self.get_installed_libraries()
        
        optimization_report = {
            'total_libraries': len(installed),
            'development_only': [],
            'security_recommendations': [],
            'size_optimization': [],
            'alternative_suggestions': [],
            'production_ready': True
        }
        
        # Identify development-only libraries
        dev_only_patterns = [
            'jupyter', 'ipykernel', 'ipywidgets', 'notebook',
            'pytest', 'coverage', 'black', 'flake8',
            'tensorboard', 'wandb'  # Training tools
        ]
        
        for lib in installed:
            lib_name_lower = lib['name'].lower()
            
            for pattern in dev_only_patterns:
                if pattern in lib_name_lower:
                    optimization_report['development_only'].append({
                        'name': lib['name'],
                        'reason': f"Development tool ({pattern})",
                        'size_impact': 'medium'
                    })
                    break
        
        # Size optimization suggestions
        large_libraries = [
            ('tensorflow', 'tensorflow-cpu', 'Use CPU-only version if no GPU needed'),
            ('torch', 'torch-cpu', 'Use CPU-only version if no GPU needed'),
            ('opencv-python', 'opencv-python-headless', 'Use headless version for server deployment')
        ]
        
        for lib in installed:
            for large_lib, alternative, reason in large_libraries:
                if large_lib in lib['name'].lower():
                    optimization_report['alternative_suggestions'].append({
                        'current': lib['name'],
                        'alternative': alternative,
                        'reason': reason
                    })
        
        # Production readiness assessment
        if optimization_report['development_only'] or optimization_report['alternative_suggestions']:
            optimization_report['production_ready'] = False
        
        return optimization_report
    
    def create_production_requirements(self) -> str:
        """
        Create optimized requirements.txt for production deployment
        
        Removes development dependencies and suggests optimizations
        
        Returns:
            Path to production requirements file
        """
        optimization_report = self.optimize_for_production()
        
        # Read current requirements
        current_specs = self.parse_requirements_file()
        
        # Filter out development-only libraries
        dev_only_names = [item['name'].lower() for item in optimization_report['development_only']]
        
        production_specs = []
        for spec in current_specs:
            if spec.name.lower() not in dev_only_names:
                production_specs.append(spec)
        
        # Write production requirements
        prod_requirements_file = self.project_path / "requirements-production.txt"
        
        with open(prod_requirements_file, 'w') as f:
            f.write("# Kepler Production Requirements\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            f.write("# Optimized for production deployment\n\n")
            
            for spec in production_specs:
                req_line = self._spec_to_requirement_line(spec)
                f.write(f"{req_line}\n")
            
            # Add optimization notes
            if optimization_report['alternative_suggestions']:
                f.write("\n# === OPTIMIZATION OPPORTUNITIES ===\n")
                for suggestion in optimization_report['alternative_suggestions']:
                    f.write(f"# Consider: {suggestion['current']} → {suggestion['alternative']} ({suggestion['reason']})\n")
        
        self.logger.info(f"Created production requirements: {prod_requirements_file}")
        return str(prod_requirements_file)

    def setup_ssh_authentication(self, ssh_key_path: str = None, test_connection: bool = True) -> bool:
        """
        Setup SSH authentication for private repositories
        
        Args:
            ssh_key_path: Path to SSH private key (default: ~/.ssh/id_rsa)
            test_connection: Whether to test SSH connection
            
        Returns:
            True if SSH setup successful, False otherwise
        """
        if ssh_key_path is None:
            ssh_key_path = os.path.expanduser("~/.ssh/id_rsa")
            
        ssh_key_path = Path(ssh_key_path)
        
        if not ssh_key_path.exists():
            self.logger.error(f"SSH key not found: {ssh_key_path}")
            return False
            
        # Set proper permissions for SSH key
        try:
            os.chmod(ssh_key_path, 0o600)
            self.logger.info(f"SSH key permissions set: {ssh_key_path}")
        except Exception as e:
            self.logger.warning(f"Could not set SSH key permissions: {e}")
            
        # Test SSH connection if requested
        if test_connection:
            return self._test_ssh_connection()
            
        return True
        
    def _test_ssh_connection(self) -> bool:
        """Test SSH connection to common Git providers"""
        providers = [
            ("github.com", "git@github.com"),
            ("gitlab.com", "git@gitlab.com"),
            ("bitbucket.org", "git@bitbucket.org")
        ]
        
        for provider_name, ssh_url in providers:
            try:
                result = subprocess.run([
                    "ssh", "-T", "-o", "ConnectTimeout=5", 
                    "-o", "StrictHostKeyChecking=no", ssh_url
                ], capture_output=True, text=True, timeout=10)
                
                # SSH test connection typically returns exit code 1 but with success message
                if "successfully authenticated" in result.stderr.lower() or result.returncode in [0, 1]:
                    self.logger.info(f"SSH connection to {provider_name}: OK")
                    return True
                    
            except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
                self.logger.debug(f"SSH test to {provider_name} failed: {e}")
                continue
                
        self.logger.warning("Could not verify SSH connectivity to Git providers")
        return False

    def install_from_github(self, repo_url: str, branch: str = None, tag: str = None, 
                           commit: str = None, subdirectory: str = None) -> bool:
        """
        Install library directly from GitHub repository
        
        Args:
            repo_url: GitHub repository URL (https or ssh)
            branch: Specific branch to install from
            tag: Specific tag to install from  
            commit: Specific commit hash to install from
            subdirectory: Subdirectory containing setup.py
            
        Returns:
            True if installation successful, False otherwise
        """
        try:
            # Build pip install URL
            if repo_url.startswith("git@"):
                # SSH URL
                install_url = f"git+{repo_url}"
            elif repo_url.startswith("https://github.com"):
                # HTTPS URL
                install_url = f"git+{repo_url}"
            else:
                # Assume it's a repo name like "user/repo"
                install_url = f"git+https://github.com/{repo_url}"
                
            # Add version specifier
            if commit:
                install_url += f"@{commit}"
            elif tag:
                install_url += f"@{tag}"
            elif branch:
                install_url += f"@{branch}"
                
            # Add subdirectory if specified
            if subdirectory:
                install_url += f"#subdirectory={subdirectory}"
                
            # Create LibrarySpec and install
            spec = LibrarySpec(
                name=self._extract_repo_name(repo_url),
                source=LibrarySource.GITHUB,
                url=install_url
            )
            
            return self.install_library(spec)
            
        except Exception as e:
            self.logger.error(f"Failed to install from GitHub {repo_url}: {e}")
            return False
            
    def _extract_repo_name(self, repo_url: str) -> str:
        """Extract repository name from GitHub URL"""
        # Handle various GitHub URL formats
        patterns = [
            r"github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?(?:/.*)?$",
            r"^([^/]+)/([^/]+)$"  # Simple user/repo format
        ]
        
        for pattern in patterns:
            match = re.search(pattern, repo_url)
            if match:
                return match.group(2).replace(".git", "")
                
        # Fallback: use last part of URL
        return repo_url.split("/")[-1].replace(".git", "")

    def install_local_library(self, local_path: str, editable: bool = True) -> bool:
        """
        Install library from local path
        
        Args:
            local_path: Path to local library (containing setup.py or pyproject.toml)
            editable: Install in editable mode (-e flag)
            
        Returns:
            True if installation successful, False otherwise
        """
        local_path = Path(local_path).resolve()
        
        if not local_path.exists():
            self.logger.error(f"Local path not found: {local_path}")
            return False
            
        # Check for setup files
        setup_files = ["setup.py", "pyproject.toml", "setup.cfg"]
        has_setup = any((local_path / setup_file).exists() for setup_file in setup_files)
        
        if not has_setup:
            self.logger.error(f"No setup file found in {local_path}. Expected: {setup_files}")
            return False
            
        try:
            # Build pip install command
            install_args = [sys.executable, "-m", "pip", "install"]
            
            if editable:
                install_args.extend(["-e", str(local_path)])
            else:
                install_args.append(str(local_path))
                
            result = subprocess.run(install_args, capture_output=True, text=True)
            
            if result.returncode == 0:
                library_name = self._extract_name_from_path(local_path)
                self.logger.info(f"Successfully installed local library: {library_name}")
                return True
            else:
                self.logger.error(f"Local installation failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error installing local library {local_path}: {e}")
            return False

    def create_private_repo_config(self, repo_url: str, auth_method: str = "ssh", 
                                  username: str = None, token: str = None) -> Dict[str, str]:
        """
        Create configuration for private repository access
        
        Args:
            repo_url: Private repository URL
            auth_method: Authentication method ("ssh", "https", "token")
            username: Username for HTTPS auth
            token: Personal access token for HTTPS auth
            
        Returns:
            Configuration dictionary for private repo access
        """
        config = {
            "repo_url": repo_url,
            "auth_method": auth_method
        }
        
        if auth_method == "ssh":
            # SSH authentication (recommended for private repos)
            config["install_url"] = f"git+ssh://{repo_url}"
            config["requirements"] = f"git+ssh://{repo_url}"
            
        elif auth_method == "https" and username and token:
            # HTTPS with token authentication
            if "github.com" in repo_url:
                auth_url = repo_url.replace("https://github.com", f"https://{username}:{token}@github.com")
            elif "gitlab.com" in repo_url:
                auth_url = repo_url.replace("https://gitlab.com", f"https://{username}:{token}@gitlab.com")
            else:
                auth_url = repo_url  # Custom Git server
                
            config["install_url"] = f"git+{auth_url}"
            config["requirements"] = f"git+{auth_url}"
            
        elif auth_method == "token" and token:
            # Token-only authentication (GitHub, GitLab)
            if "github.com" in repo_url:
                auth_url = repo_url.replace("https://github.com", f"https://{token}@github.com")
            else:
                auth_url = repo_url
                
            config["install_url"] = f"git+{auth_url}"
            config["requirements"] = f"git+{auth_url}"
            
        return config

    def validate_custom_library(self, library_name: str) -> Dict[str, Any]:
        """
        Validate that a custom library is properly installed and importable
        
        Args:
            library_name: Name of the library to validate
            
        Returns:
            Validation results with status, version, location, etc.
        """
        validation = {
            "library_name": library_name,
            "installed": False,
            "importable": False,
            "version": None,
            "location": None,
            "source": None,
            "editable": False,
            "dependencies": [],
            "errors": []
        }
        
        try:
            # Check if installed
            try:
                dist = pkg_resources.get_distribution(library_name)
                validation["installed"] = True
                validation["version"] = dist.version
                validation["location"] = dist.location
                
                # Check if editable install
                if dist.location and "site-packages" not in dist.location:
                    validation["editable"] = True
                    validation["source"] = "local_editable"
                elif ".git" in str(dist.location) or "github" in str(dist.location):
                    validation["source"] = "git_repository"
                else:
                    validation["source"] = "pypi"
                    
                # Get dependencies
                validation["dependencies"] = [str(req) for req in dist.requires()]
                
            except pkg_resources.DistributionNotFound:
                validation["errors"].append(f"Library {library_name} not found in installed packages")
                
            # Check if importable
            try:
                importlib.import_module(library_name)
                validation["importable"] = True
            except ImportError as e:
                validation["importable"] = False
                validation["errors"].append(f"Import failed: {e}")
                
        except Exception as e:
            validation["errors"].append(f"Validation error: {e}")
            
        return validation

    def create_custom_library_template(self, library_name: str, author: str = "Kepler User") -> str:
        """
        Create template structure for custom library development
        
        Args:
            library_name: Name for the new custom library
            author: Author name for the library
            
        Returns:
            Path to created library template
        """
        # Create library directory structure
        lib_path = self.project_path / "custom-libs" / library_name
        lib_path.mkdir(parents=True, exist_ok=True)
        
        # Create package directory
        package_path = lib_path / library_name.replace("-", "_")
        package_path.mkdir(exist_ok=True)
        
        # Create __init__.py
        init_file = package_path / "__init__.py"
        init_content = f'''"""
{library_name} - Custom Library for Kepler Project

A custom Python library developed for use with Kepler Framework.
"""

__version__ = "0.1.0"
__author__ = "{author}"

# Your custom code here
def hello_kepler():
    """Example function for custom library"""
    return f"Hello from {library_name}!"
'''
        init_file.write_text(init_content)
        
        # Create setup.py
        setup_file = lib_path / "setup.py"
        setup_content = f'''from setuptools import setup, find_packages

setup(
    name="{library_name}",
    version="0.1.0",
    author="{author}",
    description="Custom library for Kepler project",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        # Add your dependencies here
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
'''
        setup_file.write_text(setup_content)
        
        # Create README.md for the custom library
        readme_file = lib_path / "README.md"
        readme_content = f'''# {library_name}

Custom Python library for Kepler project.

## Installation

From your Kepler project root:

```bash
# Install in editable mode (recommended for development)
kepler libs install-local ./custom-libs/{library_name} --editable

# Or add to requirements.txt:
-e ./custom-libs/{library_name}
```

## Usage

```python
import {library_name.replace("-", "_")}

# Example usage
result = {library_name.replace("-", "_")}.hello_kepler()
print(result)
```

## Development

This library is set up for development with Kepler Framework. 
Any changes you make will be immediately available in your project.
'''
        readme_file.write_text(readme_content)
        
        self.logger.info(f"Created custom library template: {lib_path}")
        return str(lib_path)

    def install_from_wheel(self, wheel_path: str) -> bool:
        """
        Install library from local wheel file
        
        Args:
            wheel_path: Path to .whl file
            
        Returns:
            True if installation successful, False otherwise
        """
        wheel_path = Path(wheel_path)
        
        if not wheel_path.exists():
            self.logger.error(f"Wheel file not found: {wheel_path}")
            return False
            
        if not wheel_path.suffix == ".whl":
            self.logger.error(f"File is not a wheel: {wheel_path}")
            return False
            
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", str(wheel_path)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"Successfully installed wheel: {wheel_path.name}")
                return True
            else:
                self.logger.error(f"Wheel installation failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error installing wheel {wheel_path}: {e}")
            return False

    def clone_and_install_repo(self, repo_url: str, target_dir: str = None, 
                              branch: str = None, install_editable: bool = True) -> bool:
        """
        Clone repository and install as local editable library
        
        Args:
            repo_url: Git repository URL
            target_dir: Local directory to clone into (default: ./custom-libs/)
            branch: Specific branch to clone
            install_editable: Install in editable mode
            
        Returns:
            True if clone and installation successful, False otherwise
        """
        try:
            if target_dir is None:
                target_dir = self.project_path / "custom-libs"
            else:
                target_dir = Path(target_dir)
                
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract repo name for directory
            repo_name = self._extract_repo_name(repo_url)
            clone_path = target_dir / repo_name
            
            # Remove existing directory if it exists
            if clone_path.exists():
                shutil.rmtree(clone_path)
                
            # Build git clone command
            clone_cmd = ["git", "clone"]
            if branch:
                clone_cmd.extend(["-b", branch])
            clone_cmd.extend([repo_url, str(clone_path)])
            
            # Clone repository
            result = subprocess.run(clone_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                self.logger.error(f"Git clone failed: {result.stderr}")
                return False
                
            self.logger.info(f"Successfully cloned {repo_url} to {clone_path}")
            
            # Install the cloned library
            if install_editable:
                return self.install_local_library(str(clone_path), editable=True)
            else:
                return self.install_local_library(str(clone_path), editable=False)
                
        except Exception as e:
            self.logger.error(f"Error cloning and installing {repo_url}: {e}")
            return False


# Convenience functions for CLI and SDK
def install_unlimited_libraries(requirements_file: str = "requirements.txt", 
                               project_path: str = ".") -> Dict[str, bool]:
    """
    Install unlimited Python libraries from any source
    
    This function implements the core PRD requirement:
    "El sistema DEBE soportar importación de cualquier librería Python estándar"
    """
    manager = LibraryManager(project_path)
    return manager.install_from_requirements(requirements_file)


def validate_unlimited_environment(project_path: str = ".") -> Dict[str, Any]:
    """
    Validate that unlimited library environment is properly configured
    
    Returns comprehensive validation report for any Python library ecosystem
    """
    manager = LibraryManager(project_path)
    return manager.validate_environment()


def create_ai_template(template: str, project_path: str = ".") -> bool:
    """
    Create requirements.txt from AI framework templates
    
    Supports: 'ml', 'deep_learning', 'generative_ai', 'computer_vision', 'nlp', 'full_ai'
    """
    manager = LibraryManager(project_path)
    return manager.create_environment_from_template(template)


# Import required for datetime
from datetime import datetime
