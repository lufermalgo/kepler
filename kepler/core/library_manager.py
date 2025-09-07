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
    Manages unlimited Python library support for Kepler projects
    
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
        self.logger = get_logger(__name__)
        
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
        else:
            source = LibrarySource.LOCAL_TARBALL
            
        name = self._extract_name_from_path(line)
        
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
            return filename.replace('.tar.gz', '').split('-')[0]
        elif filename.endswith('.whl'):
            return filename.split('-')[0]
        else:
            return "unknown"
    
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
