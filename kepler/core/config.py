"""
Configuration management for Kepler framework

Handles kepler.yml configuration file and environment variables.
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv
import subprocess


class SplunkConfig(BaseModel):
    """Splunk connection configuration"""
    host: str = Field(..., description="Splunk server URL (e.g., https://splunk.company.com:8089)")
    token: str = Field(..., description="Splunk authentication token")
    hec_token: Optional[str] = Field(None, description="HTTP Event Collector token")
    hec_url: Optional[str] = Field(None, description="HEC endpoint URL")
    metrics_index: str = Field("kepler_metrics", description="Default metrics index")
    timeout: int = Field(30, description="Request timeout in seconds")
    verify_ssl: bool = Field(True, description="Verify SSL certificates")
    
    @validator('host')
    def validate_host(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('Host must start with http:// or https://')
        return v
    
    @validator('hec_url', pre=True, always=True)
    def set_hec_url(cls, v, values):
        if v is None and 'host' in values:
            # Auto-generate HEC URL from host
            host = values['host'].rstrip('/')
            # Replace port 8089 with 8088 for HEC
            if ':8089' in host:
                hec_host = host.replace(':8089', ':8088')
            else:
                hec_host = host
            return f"{hec_host}/services/collector/event"
        return v


class GCPConfig(BaseModel):
    """Google Cloud Platform configuration"""
    project_id: str = Field(..., description="GCP project ID")
    region: str = Field("us-central1", description="Default GCP region")
    credentials_path: Optional[str] = Field(None, description="Path to service account JSON")
    
    @validator('credentials_path', pre=True, always=True)
    def set_credentials_path(cls, v):
        if v is None:
            # Try common locations
            candidates = [
                "service-account-key.json",
                "gcp-credentials.json", 
                os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
            ]
            for candidate in candidates:
                if Path(candidate).exists():
                    return candidate
        return v


class TrainingConfig(BaseModel):
    """ML training configuration"""
    default_algorithm: str = Field("random_forest", description="Default ML algorithm")
    test_size: float = Field(0.2, description="Train/test split ratio")
    random_state: int = Field(42, description="Random seed for reproducibility")
    
    @validator('default_algorithm')
    def validate_algorithm(cls, v):
        allowed = ["random_forest", "linear_regression", "xgboost"]
        if v not in allowed:
            raise ValueError(f'Algorithm must be one of: {allowed}')
        return v
    
    @validator('test_size')
    def validate_test_size(cls, v):
        if not 0.1 <= v <= 0.5:
            raise ValueError('test_size must be between 0.1 and 0.5')
        return v


class DeploymentConfig(BaseModel):
    """Deployment configuration"""
    service_name: str = Field("kepler-model-api", description="Cloud Run service name")
    port: int = Field(8080, description="Container port")
    cpu: str = Field("1", description="CPU allocation")
    memory: str = Field("2Gi", description="Memory allocation")
    max_instances: int = Field(10, description="Maximum instances")
    
    @validator('service_name')
    def validate_service_name(cls, v):
        # Cloud Run service names must be lowercase alphanumeric + hyphens
        if not v.replace('-', '').isalnum() or not v.islower():
            raise ValueError('Service name must be lowercase alphanumeric with hyphens only')
        return v


class KeplerConfig(BaseModel):
    """Main Kepler configuration"""
    project_name: str = Field(..., description="Project name")
    splunk: SplunkConfig
    gcp: GCPConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    deployment: DeploymentConfig = Field(default_factory=DeploymentConfig)
    
    @classmethod
    def from_file(cls, config_path: str = "kepler.yml") -> "KeplerConfig":
        """Load configuration from YAML file with environment variable substitution"""
        # Load environment variables
        load_dotenv()
        
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Read and parse YAML
        with open(config_file, 'r') as f:
            raw_config = yaml.safe_load(f)
        
        # Substitute environment variables
        config_data = cls._substitute_env_vars(raw_config)
        
        return cls(**config_data)
    
    @staticmethod
    def _substitute_env_vars(obj: Any) -> Any:
        """Recursively substitute ${VAR} with environment variables"""
        if isinstance(obj, dict):
            return {k: KeplerConfig._substitute_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [KeplerConfig._substitute_env_vars(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
            var_name = obj[2:-1]
            env_value = os.getenv(var_name)
            if env_value is None:
                raise ValueError(f"Environment variable '{var_name}' not found")
            return env_value
        else:
            return obj
    
    def to_file(self, config_path: str = "kepler.yml") -> None:
        """Save configuration to YAML file"""
        config_dict = self.dict()
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    @classmethod
    def create_template(cls, project_name: str, config_path: str = "kepler.yml") -> None:
        """Create a template kepler.yml file"""
        template = {
            "project_name": project_name,
            "splunk": {
                "host": "https://localhost:8089",
                "token": "${SPLUNK_TOKEN}",
                "hec_token": "${SPLUNK_HEC_TOKEN}",
                "metrics_index": "kepler_metrics"
            },
            "gcp": {
                "project_id": "${GCP_PROJECT_ID}",
                "region": "us-central1"
            },
            "training": {
                "default_algorithm": "random_forest",
                "test_size": 0.2,
                "random_state": 42
            },
            "deployment": {
                "service_name": f"{project_name.lower().replace('_', '-')}-api",
                "port": 8080,
                "cpu": "1",
                "memory": "2Gi"
            }
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(template, f, default_flow_style=False, indent=2)


# Prerequisite validation functions
def validate_prerequisites() -> Dict[str, bool]:
    """Validate that all prerequisites are available"""
    results = {}
    
    # Check Python version
    import sys
    results['python_version'] = sys.version_info >= (3, 8)
    
    # Check required packages
    required_packages = ['pandas', 'numpy', 'sklearn', 'xgboost', 'splunklib', 'google.cloud']
    for package in required_packages:
        try:
            __import__(package)
            results[f'package_{package}'] = True
        except ImportError:
            results[f'package_{package}'] = False
    
    # Check environment variables (if config exists)
    if Path("kepler.yml").exists():
        try:
            config = KeplerConfig.from_file()
            results['config_valid'] = True
            
            # Test Splunk connection (basic)
            import requests
            try:
                response = requests.get(f"{config.splunk.host}/services/auth/login", 
                                      timeout=5, verify=config.splunk.verify_ssl)
                results['splunk_reachable'] = response.status_code in [200, 401]  # 401 is ok, means server is up
            except:
                results['splunk_reachable'] = False
                
        except Exception as e:
            results['config_valid'] = False
            results['config_error'] = str(e)
    else:
        results['config_exists'] = False
    
    return results


def print_prerequisites_report():
    """Print a human-readable prerequisites report"""
    from rich.console import Console
    from rich.table import Table
    
    console = Console()
    results = validate_prerequisites()
    
    table = Table(title="Kepler Prerequisites Check")
    table.add_column("Component", style="cyan", no_wrap=True)
    table.add_column("Status", style="magenta")
    table.add_column("Details", style="green")
    
    for key, value in results.items():
        if key.endswith('_error'):
            continue
            
        status = "✅ OK" if value else "❌ FAIL"
        details = ""
        
        if key == 'python_version':
            import sys
            details = f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        elif key.startswith('package_'):
            package = key.replace('package_', '')
            details = f"Package: {package}"
        elif key == 'config_valid':
            if not value and 'config_error' in results:
                details = f"Error: {results['config_error']}"
        elif key == 'splunk_reachable':
            details = "Splunk server connection test"
        
        table.add_row(key, status, details)
    
    console.print(table)
    
    # Summary
    total_checks = len([k for k in results.keys() if not k.endswith('_error')])
    passed_checks = len([v for k, v in results.items() if not k.endswith('_error') and v])
    
    if passed_checks == total_checks:
        console.print("\n✅ All prerequisites met! You're ready to use Kepler.", style="bold green")
    else:
        console.print(f"\n⚠️  {total_checks - passed_checks} prerequisites failed. Please fix them before using Kepler.", style="bold yellow")


def detect_gcp_credentials() -> Dict[str, Optional[str]]:
    """
    Detecta automáticamente las credenciales de GCP disponibles en el sistema.
    
    Returns:
        Dict con información de credenciales encontradas
    """
    credentials_info = {
        'project_id': None,
        'credentials_path': None,
        'auth_method': None,
        'service_account_email': None,
        'gcloud_installed': False
    }
    
    # 1. Verificar si gcloud está instalado
    try:
        result = subprocess.run(['gcloud', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            credentials_info['gcloud_installed'] = True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        credentials_info['gcloud_installed'] = False
    
    # 2. Detectar Application Default Credentials (ADC)
    adc_path = os.path.expanduser('~/.config/gcloud/application_default_credentials.json')
    if os.path.exists(adc_path):
        try:
            with open(adc_path, 'r') as f:
                adc_data = json.load(f)
                credentials_info['credentials_path'] = adc_path
                credentials_info['auth_method'] = 'Application Default Credentials (ADC)'
                if 'client_email' in adc_data:
                    credentials_info['service_account_email'] = adc_data['client_email']
        except (json.JSONDecodeError, IOError):
            pass
    
    # 3. Verificar variable de entorno GOOGLE_APPLICATION_CREDENTIALS
    google_creds_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    if google_creds_path and os.path.exists(google_creds_path):
        try:
            with open(google_creds_path, 'r') as f:
                creds_data = json.load(f)
                credentials_info['credentials_path'] = google_creds_path
                credentials_info['auth_method'] = 'Service Account Key File'
                if 'client_email' in creds_data:
                    credentials_info['service_account_email'] = creds_data['client_email']
                if 'project_id' in creds_data:
                    credentials_info['project_id'] = creds_data['project_id']
        except (json.JSONDecodeError, IOError):
            pass
    
    # 4. Obtener project ID desde gcloud si está disponible
    if credentials_info['gcloud_installed'] and not credentials_info['project_id']:
        try:
            result = subprocess.run(['gcloud', 'config', 'get-value', 'project'],
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                project_id = result.stdout.strip()
                if project_id != '(unset)':
                    credentials_info['project_id'] = project_id
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    
    # 5. Verificar variable de entorno GCP_PROJECT_ID o GOOGLE_CLOUD_PROJECT
    env_project = os.environ.get('GCP_PROJECT_ID') or os.environ.get('GOOGLE_CLOUD_PROJECT')
    if env_project and not credentials_info['project_id']:
        credentials_info['project_id'] = env_project
    
    return credentials_info


def suggest_gcp_configuration(credentials_info: Dict[str, Optional[str]]) -> Dict[str, str]:
    """
    Sugiere configuración de GCP basada en las credenciales detectadas.
    
    Args:
        credentials_info: Info de credenciales detectadas
        
    Returns:
        Dict con sugerencias de configuración
    """
    suggestions = {}
    
    if credentials_info['project_id']:
        suggestions['project_id'] = credentials_info['project_id']
    else:
        suggestions['project_id'] = "${GCP_PROJECT_ID}"
    
    # Sugerir región por defecto
    suggestions['region'] = "us-central1"
    suggestions['compute_zone'] = "us-central1-a"
    
    # Configuración de Cloud Run
    suggestions['cloud_run'] = {
        'region': suggestions['region'],
        'service_name': 'kepler-model-api',
        'memory': '2Gi',
        'cpu': '1',
        'max_instances': 10
    }
    
    return suggestions


def print_gcp_detection_report(credentials_info: Dict[str, Optional[str]]):
    """Imprime un reporte detallado de las credenciales GCP detectadas."""
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    
    console = Console()
    
    # Panel principal
    if credentials_info['auth_method']:
        title = "✅ Credenciales GCP Detectadas"
        style = "green"
    else:
        title = "⚠️  Credenciales GCP No Detectadas"
        style = "yellow"
    
    # Tabla de detalles
    table = Table(title="Estado de Credenciales GCP")
    table.add_column("Elemento", style="cyan")
    table.add_column("Estado", style="magenta")
    table.add_column("Valor/Detalle", style="white")
    
    # gcloud CLI
    gcloud_status = "✅ Instalado" if credentials_info['gcloud_installed'] else "❌ No instalado"
    table.add_row("gcloud CLI", gcloud_status, "")
    
    # Método de autenticación
    if credentials_info['auth_method']:
        table.add_row("Método Auth", "✅ Detectado", credentials_info['auth_method'])
    else:
        table.add_row("Método Auth", "❌ No detectado", "ADC o Service Account requerido")
    
    # Project ID
    if credentials_info['project_id']:
        table.add_row("Project ID", "✅ Configurado", credentials_info['project_id'])
    else:
        table.add_row("Project ID", "⚠️  No detectado", "Requerirá configuración manual")
    
    # Service Account
    if credentials_info['service_account_email']:
        table.add_row("Service Account", "✅ Detectado", credentials_info['service_account_email'])
    else:
        table.add_row("Service Account", "ℹ️  N/A", "User credentials o no disponible")
    
    # Archivo de credenciales
    if credentials_info['credentials_path']:
        table.add_row("Archivo Creds", "✅ Encontrado", credentials_info['credentials_path'])
    
    console.print(Panel(table, title=title, border_style=style))
    
    # Sugerencias
    if not credentials_info['auth_method']:
        console.print("\n🔧 [bold yellow]Pasos para configurar GCP:[/bold yellow]")
        console.print("1. Instalar gcloud CLI: https://cloud.google.com/sdk/docs/install")
        console.print("2. Autenticarse: `gcloud auth application-default login`")
        console.print("3. Configurar proyecto: `gcloud config set project YOUR_PROJECT_ID`")
        console.print("4. Ejecutar: `kepler validate` nuevamente")
    
    elif not credentials_info['project_id']:
        console.print("\n🔧 [bold yellow]Configurar Project ID:[/bold yellow]")
        console.print("Ejecutar: `gcloud config set project YOUR_PROJECT_ID`")
        console.print("O configurar: `export GCP_PROJECT_ID=your-project-id`")


def load_config(config_file: str = "kepler.yml") -> KeplerConfig:
    """
    Load Kepler configuration automatically.
    
    This function provides a simple way to load configuration for data scientists,
    handling all the complexity of finding files and loading environment variables.
    
    Args:
        config_file: Path to configuration file (default: kepler.yml)
        
    Returns:
        KeplerConfig: Loaded configuration object
        
    Raises:
        ValueError: If configuration cannot be loaded
    """
    
    # Load environment variables from .env file if it exists
    env_files = ['.env', '../.env', '../../.env']
    for env_file in env_files:
        if os.path.exists(env_file):
            load_dotenv(env_file)
            break
    
    # Find configuration file
    config_paths = [
        config_file,
        f"../{config_file}",
        f"../../{config_file}",
        f"./{config_file}"
    ]
    
    config_path = None
    for path in config_paths:
        if os.path.exists(path):
            config_path = path
            break
    
    if not config_path:
        raise ValueError(f"❌ Configuration file '{config_file}' not found in expected locations")
    
    # Load YAML configuration
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
    except Exception as e:
        raise ValueError(f"❌ Error reading configuration file: {e}")
    
    # Merge with environment variables
    if 'splunk' in config_data:
        # Override with environment variables if present
        if os.getenv('SPLUNK_TOKEN'):
            config_data['splunk']['token'] = os.getenv('SPLUNK_TOKEN')
        if os.getenv('SPLUNK_HEC_TOKEN'):
            config_data['splunk']['hec_token'] = os.getenv('SPLUNK_HEC_TOKEN')
    
    if 'gcp' in config_data:
        if os.getenv('GCP_PROJECT_ID'):
            config_data['gcp']['project_id'] = os.getenv('GCP_PROJECT_ID')
    
    # Create configuration object
    try:
        return KeplerConfig(**config_data)
    except Exception as e:
        raise ValueError(f"❌ Invalid configuration: {e}")