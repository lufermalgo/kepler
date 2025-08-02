"""
Global configuration management for Kepler framework

Implements secure configuration handling with global config files
outside project directories to protect sensitive credentials.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

from kepler.utils.logging import get_logger
from kepler.utils.exceptions import ConfigurationError


@dataclass
class GlobalSplunkConfig:
    """Global Splunk configuration stored securely"""
    host: str = "https://localhost:8089"
    token: Optional[str] = None
    hec_token: Optional[str] = None
    hec_url: Optional[str] = None  # Auto-derived from host if not provided
    verify_ssl: bool = True
    timeout: int = 30
    metrics_index: str = "kepler_metrics"
    events_index: str = "kepler_events"


@dataclass 
class GlobalGCPConfig:
    """Global GCP configuration stored securely"""
    project_id: Optional[str] = None
    region: str = "us-central1"
    service_account_file: Optional[str] = None
    compute_zone: str = "us-central1-a"


@dataclass
class GlobalMLflowConfig:
    """Global MLflow configuration"""
    tracking_uri: Optional[str] = None
    registry_uri: Optional[str] = None
    artifact_location: Optional[str] = None


@dataclass
class GlobalKeplerConfig:
    """Complete global configuration for Kepler framework"""
    splunk: GlobalSplunkConfig = field(default_factory=GlobalSplunkConfig)
    gcp: GlobalGCPConfig = field(default_factory=GlobalGCPConfig)
    mlflow: GlobalMLflowConfig = field(default_factory=GlobalMLflowConfig)
    
    # Security settings
    log_sensitive_data: bool = False
    max_retry_attempts: int = 3
    connection_timeout: int = 30


class GlobalConfigManager:
    """
    Manages global Kepler configuration stored securely outside project directories
    
    Configuration hierarchy:
    1. Environment variables (highest priority)
    2. Global config file (~/.kepler/config.yml) 
    3. Project config file (project/kepler.yml) - non-sensitive only
    4. Defaults (lowest priority)
    """
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.GlobalConfigManager")
        self._global_config_dir = Path.home() / ".kepler"
        self._global_config_file = self._global_config_dir / "config.yml"
        self._config_cache = None
        
        # Load environment variables
        load_dotenv()
    
    def get_global_config_path(self) -> Path:
        """Get the path to the global configuration file"""
        return self._global_config_file
    
    def ensure_global_config_dir(self) -> None:
        """Ensure the global configuration directory exists with proper permissions"""
        try:
            self._global_config_dir.mkdir(mode=0o700, exist_ok=True)
            self.logger.debug(f"Global config directory: {self._global_config_dir}")
            
            # Set restrictive permissions if directory already existed
            os.chmod(self._global_config_dir, 0o700)
            
        except Exception as e:
            raise ConfigurationError(
                f"Failed to create global config directory: {e}",
                config_file=str(self._global_config_dir),
                suggestion="Check file system permissions"
            )
    
    def create_global_config_template(self, force: bool = False) -> bool:
        """
        Create a template global configuration file
        
        Args:
            force: Overwrite existing file if it exists
            
        Returns:
            True if template was created, False if already exists
        """
        self.ensure_global_config_dir()
        
        if self._global_config_file.exists() and not force:
            self.logger.info(f"Global config file already exists: {self._global_config_file}")
            return False
        
        template_config = {
            "# Kepler Global Configuration": None,
            "# This file contains sensitive credentials and should be protected": None,
            "# File permissions: 600 (owner read/write only)": None,
            "": None,
            "splunk": {
                "host": "https://your-splunk-server:8089",
                "token": "your_splunk_auth_token_here",
                "hec_token": "your_splunk_hec_token_here", 
                "verify_ssl": True,
                "timeout": 30,
                "metrics_index": "kepler_metrics",
                "events_index": "kepler_events"
            },
            "gcp": {
                "project_id": "your-gcp-project-id",
                "region": "us-central1",
                "service_account_file": "/path/to/your/service-account.json",
                "compute_zone": "us-central1-a"
            },
            "mlflow": {
                "tracking_uri": "https://your-mlflow-server",
                "registry_uri": None,
                "artifact_location": "gs://your-mlflow-artifacts"
            },
            "security": {
                "log_sensitive_data": False,
                "max_retry_attempts": 3,
                "connection_timeout": 30
            }
        }
        
        try:
            # Write the template file
            with open(self._global_config_file, 'w') as f:
                # Write comments and configuration
                for key, value in template_config.items():
                    if key.startswith("#") or key == "":
                        f.write(f"{key}\n")
                    else:
                        yaml.dump({key: value}, f, default_flow_style=False)
            
            # Set restrictive permissions
            os.chmod(self._global_config_file, 0o600)
            
            self.logger.info(f"Created global config template: {self._global_config_file}")
            return True
            
        except Exception as e:
            raise ConfigurationError(
                f"Failed to create global config template: {e}",
                config_file=str(self._global_config_file),
                suggestion="Check file system permissions and disk space"
            )
    
    def load_global_config(self) -> GlobalKeplerConfig:
        """
        Load global configuration with security validation
        
        Returns:
            GlobalKeplerConfig instance with merged configuration
        """
        if self._config_cache is not None:
            return self._config_cache
        
        # Start with defaults
        config = GlobalKeplerConfig()
        
        # Load from global config file if it exists
        if self._global_config_file.exists():
            try:
                self._validate_file_permissions()
                file_config = self._load_config_file()
                config = self._merge_config(config, file_config)
                self.logger.debug("Loaded global configuration file")
                
            except Exception as e:
                self.logger.warning(f"Failed to load global config file: {e}")
        
        # Override with environment variables
        config = self._apply_env_overrides(config)
        
        # Validate and apply security rules
        self._validate_security_config(config)
        
        # Cache the configuration
        self._config_cache = config
        
        return config
    
    def _validate_file_permissions(self) -> None:
        """Validate that the global config file has secure permissions"""
        try:
            file_stat = self._global_config_file.stat()
            file_mode = file_stat.st_mode & 0o777
            
            if file_mode != 0o600:
                self.logger.warning(
                    f"Global config file has insecure permissions: {oct(file_mode)}. "
                    f"Should be 600. Attempting to fix..."
                )
                os.chmod(self._global_config_file, 0o600)
                
        except Exception as e:
            raise ConfigurationError(
                f"Failed to validate config file permissions: {e}",
                config_file=str(self._global_config_file),
                suggestion="Ensure file has 600 permissions (owner read/write only)"
            )
    
    def _load_config_file(self) -> Dict[str, Any]:
        """Load and parse the global configuration file"""
        try:
            with open(self._global_config_file, 'r') as f:
                raw_config = yaml.safe_load(f) or {}
            
            # Filter out comment lines and empty keys
            return {k: v for k, v in raw_config.items() 
                   if not k.startswith("#") and k != ""}
                   
        except Exception as e:
            raise ConfigurationError(
                f"Failed to parse global config file: {e}",
                config_file=str(self._global_config_file),
                suggestion="Check YAML syntax and file format"
            )
    
    def _merge_config(self, base_config: GlobalKeplerConfig, file_config: Dict[str, Any]) -> GlobalKeplerConfig:
        """Merge file configuration into base configuration"""
        try:
            # Merge Splunk config
            if "splunk" in file_config:
                splunk_data = file_config["splunk"]
                for key, value in splunk_data.items():
                    if hasattr(base_config.splunk, key) and value is not None:
                        setattr(base_config.splunk, key, value)
            
            # Merge GCP config
            if "gcp" in file_config:
                gcp_data = file_config["gcp"]
                for key, value in gcp_data.items():
                    if hasattr(base_config.gcp, key) and value is not None:
                        setattr(base_config.gcp, key, value)
            
            # Merge MLflow config
            if "mlflow" in file_config:
                mlflow_data = file_config["mlflow"]
                for key, value in mlflow_data.items():
                    if hasattr(base_config.mlflow, key) and value is not None:
                        setattr(base_config.mlflow, key, value)
            
            # Merge security settings
            if "security" in file_config:
                security_data = file_config["security"]
                for key, value in security_data.items():
                    if hasattr(base_config, key) and value is not None:
                        setattr(base_config, key, value)
            
            return base_config
            
        except Exception as e:
            raise ConfigurationError(
                f"Failed to merge configuration: {e}",
                suggestion="Check configuration file format and field names"
            )
    
    def _apply_env_overrides(self, config: GlobalKeplerConfig) -> GlobalKeplerConfig:
        """Apply environment variable overrides"""
        # Splunk overrides
        if os.getenv("SPLUNK_HOST"):
            config.splunk.host = os.getenv("SPLUNK_HOST")
        if os.getenv("SPLUNK_TOKEN"):
            config.splunk.token = os.getenv("SPLUNK_TOKEN")
        if os.getenv("SPLUNK_HEC_TOKEN"):
            config.splunk.hec_token = os.getenv("SPLUNK_HEC_TOKEN")
        if os.getenv("SPLUNK_HEC_URL"):
            config.splunk.hec_url = os.getenv("SPLUNK_HEC_URL")
        if os.getenv("SPLUNK_VERIFY_SSL"):
            config.splunk.verify_ssl = os.getenv("SPLUNK_VERIFY_SSL").lower() in ["true", "1", "yes"]
        
        # GCP overrides
        if os.getenv("GCP_PROJECT_ID"):
            config.gcp.project_id = os.getenv("GCP_PROJECT_ID")
        if os.getenv("GCP_REGION"):
            config.gcp.region = os.getenv("GCP_REGION")
        if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            config.gcp.service_account_file = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        
        # MLflow overrides
        if os.getenv("MLFLOW_TRACKING_URI"):
            config.mlflow.tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        
        return config
    
    def _validate_security_config(self, config: GlobalKeplerConfig) -> None:
        """Validate security aspects of configuration"""
        # Auto-derive HEC URL if not provided
        if not config.splunk.hec_url and config.splunk.host:
            base_url = config.splunk.host.rstrip('/')
            # Replace standard Splunk port with HEC port
            if ":8089" in base_url:
                config.splunk.hec_url = base_url.replace(":8089", ":8088")
            else:
                # Assume standard HEC port
                config.splunk.hec_url = f"{base_url}:8088"
        
        # Validate that we have minimum required credentials
        missing_creds = []
        if not config.splunk.token:
            missing_creds.append("splunk.token")
        if not config.splunk.hec_token:
            missing_creds.append("splunk.hec_token")
        
        if missing_creds:
            self.logger.warning(f"Missing credentials: {missing_creds}")
            self.logger.info("Run 'kepler config init' to create configuration template")
    
    def validate_credentials_available(self) -> Dict[str, bool]:
        """
        Validate which credentials are available
        
        Returns:
            Dictionary with validation results for each credential type
        """
        config = self.load_global_config()
        
        return {
            "splunk_token": bool(config.splunk.token),
            "splunk_hec_token": bool(config.splunk.hec_token),
            "gcp_project_id": bool(config.gcp.project_id),
            "gcp_service_account": bool(config.gcp.service_account_file and 
                                     Path(config.gcp.service_account_file or "").exists()),
            "mlflow_tracking": bool(config.mlflow.tracking_uri)
        }
    
    def clear_cache(self) -> None:
        """Clear the configuration cache (useful for testing)"""
        self._config_cache = None


# Global instance
_global_config_manager = None

def get_global_config_manager() -> GlobalConfigManager:
    """Get the global configuration manager instance"""
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = GlobalConfigManager()
    return _global_config_manager

def get_global_config() -> GlobalKeplerConfig:
    """Get the global configuration"""
    return get_global_config_manager().load_global_config()