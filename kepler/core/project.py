"""
Project management for Kepler framework

Handles project initialization, structure creation, and management.
"""

import os
import shutil
from pathlib import Path
from typing import Optional
from rich.console import Console
from kepler.core.config import KeplerConfig


class KeplerProject:
    """Manages Kepler project lifecycle"""
    
    def __init__(self, project_path: str = "."):
        self.project_path = Path(project_path).resolve()
        self.console = Console()
        
    @property
    def config_path(self) -> Path:
        """Path to the kepler.yml configuration file"""
        return self.project_path / "kepler.yml"
        
    @property
    def is_kepler_project(self) -> bool:
        """Check if current directory is a Kepler project"""
        return self.config_path.exists()
    
    def initialize(self, project_name: str, force: bool = False) -> bool:
        """
        Initialize a new Kepler project
        
        Args:
            project_name: Name of the project
            force: Overwrite existing files if they exist
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create project directory structure
            self._create_project_structure()
            
            # Create kepler.yml configuration template
            if not self.config_path.exists() or force:
                KeplerConfig.create_template(project_name, str(self.config_path))
                self.console.print(f"✅ Created configuration file: {self.config_path}")
            else:
                self.console.print(f"⚠️  Configuration file already exists: {self.config_path}")
            
            # Create README.md
            readme_path = self.project_path / "README.md"
            if not readme_path.exists() or force:
                self._create_readme(project_name, readme_path)
                self.console.print(f"✅ Created README: {readme_path}")
            
            # Create .env template
            env_path = self.project_path / ".env.template"
            if not env_path.exists() or force:
                self._create_env_template(env_path)
                self.console.print(f"✅ Created environment template: {env_path}")
            
            # Create .gitignore if it doesn't exist
            gitignore_path = self.project_path / ".gitignore"
            if not gitignore_path.exists():
                self._create_gitignore(gitignore_path)
                self.console.print(f"✅ Created .gitignore: {gitignore_path}")
            
            self.console.print(f"\n🎉 Kepler project '{project_name}' initialized successfully!")
            self._print_next_steps()
            
            return True
            
        except Exception as e:
            self.console.print(f"❌ Error initializing project: {e}", style="red")
            return False
    
    def _create_project_structure(self):
        """Create the standard Kepler project directory structure"""
        directories = [
            "data/raw",
            "data/processed", 
            "models",
            "notebooks",
            "scripts",
            "logs"
        ]
        
        for directory in directories:
            dir_path = self.project_path / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            
            # Create .gitkeep files to preserve empty directories
            gitkeep_path = dir_path / ".gitkeep"
            if not gitkeep_path.exists():
                gitkeep_path.touch()
    
    def _create_readme(self, project_name: str, readme_path: Path):
        """Create a basic README.md file"""
        readme_content = f"""# {project_name}

Kepler ML project for industrial data analysis.

## Quick Start

1. **Configure your environment:**
   ```bash
   cp .env.template .env
   # Edit .env with your actual credentials
   ```

2. **Validate prerequisites:**
   ```bash
   kepler validate
   ```

3. **Extract data from Splunk:**
   ```bash
   kepler extract "index=your_index | head 1000"
   ```

4. **Train a model:**
   ```bash
   kepler train data.csv --target your_target_column
   ```

5. **Deploy to Cloud Run:**
   ```bash
   kepler deploy model.pkl
   ```

## Project Structure

```
{project_name}/
├── kepler.yml          # Configuration file
├── data/
│   ├── raw/           # Raw data from Splunk
│   └── processed/     # Processed data for training
├── models/            # Trained model files
├── notebooks/         # Jupyter notebooks for exploration
├── scripts/           # Custom scripts
└── logs/              # Application logs
```

## Configuration

Edit `kepler.yml` to configure:
- Splunk connection details
- GCP project settings
- ML training parameters
- Deployment options

## Environment Variables

Copy `.env.template` to `.env` and configure:
- `SPLUNK_TOKEN`: Your Splunk authentication token
- `SPLUNK_HEC_TOKEN`: HTTP Event Collector token
- `GCP_PROJECT_ID`: Your Google Cloud project ID

## Next Steps

1. Configure your Splunk and GCP credentials
2. Run `kepler validate` to check prerequisites
3. Start with data extraction: `kepler extract --help`

For more information, see the [Kepler documentation](https://github.com/company/kepler-framework).
"""
        
        with open(readme_path, 'w') as f:
            f.write(readme_content)
    
    def _create_env_template(self, env_path: Path):
        """Create .env template file"""
        env_content = """# Kepler Environment Variables Template
# Copy this file to .env and fill in your actual values

# Splunk Configuration
SPLUNK_TOKEN=your_splunk_authentication_token_here
SPLUNK_HEC_TOKEN=your_splunk_hec_token_here

# Google Cloud Platform
GCP_PROJECT_ID=your_gcp_project_id_here

# Optional: Override default settings
# SPLUNK_HOST=https://your-splunk-server:8089
# SPLUNK_HEC_URL=https://your-splunk-server:8088/services/collector
# GCP_REGION=us-central1

# Advanced Splunk Configuration
# SPLUNK_VERIFY_SSL=true
# SPLUNK_TIMEOUT=30
# SPLUNK_METRICS_INDEX=kepler_metrics
"""
        
        with open(env_path, 'w') as f:
            f.write(env_content)
    
    def _create_gitignore(self, gitignore_path: Path):
        """Create .gitignore file specific to Kepler projects"""
        gitignore_content = """# Kepler specific
.env
kepler.yml
*.pkl
*.joblib
models/
data/
logs/
.kepler/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.venv/
venv/
ENV/
env/

# Jupyter Notebook
.ipynb_checkpoints

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# GCP credentials
service-account-key.json
gcp-credentials.json
"""
        
        with open(gitignore_path, 'w') as f:
            f.write(gitignore_content)
    
    def _print_next_steps(self):
        """Print helpful next steps after project initialization"""
        from rich.panel import Panel
        
        next_steps = """
[bold]Next Steps:[/bold]

1. [yellow]Configure credentials:[/yellow]
   • Copy .env.template to .env
   • Add your Splunk and GCP credentials
   
2. [yellow]Validate setup:[/yellow]
   • Run: kepler validate
   
3. [yellow]Start using Kepler:[/yellow]
   • Extract data: kepler extract "your SPL query"
   • Train model: kepler train data.csv
   • Deploy model: kepler deploy model.pkl

[dim]Need help? Run any command with --help for details.[/dim]
        """
        
        panel = Panel(next_steps, title="🚀 Ready to Go!", border_style="green")
        self.console.print(panel)
    
    def validate_project(self) -> bool:
        """Validate that the current directory is a valid Kepler project"""
        if not self.is_kepler_project:
            self.console.print("❌ Not a Kepler project. Run 'kepler init <project-name>' first.", style="red")
            return False
        
        try:
            # Try to load configuration
            config = KeplerConfig.from_file(str(self.config_path))
            self.console.print("✅ Project configuration is valid")
            return True
        except Exception as e:
            self.console.print(f"❌ Invalid project configuration: {e}", style="red")
            return False
    
    def get_config(self) -> Optional[KeplerConfig]:
        """Get the project configuration if valid"""
        if not self.validate_project():
            return None
        
        try:
            return KeplerConfig.from_file(str(self.config_path))
        except Exception:
            return None