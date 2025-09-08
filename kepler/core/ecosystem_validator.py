"""
Kepler Ecosystem Validator - Task 7.2 Implementation
Complete ecosystem validation with actionable error messages

Validates connectivity, authentication, and configuration for:
- Splunk Enterprise (REST API + HEC)
- Google Cloud Platform (Cloud Run, Artifact Registry, etc.)
- MLOps tools (MLflow, DVC) 
- Development environment (Python, libraries, etc.)

Philosophy: "Validate everything before you start working"
"""

import os
import subprocess
import json
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from kepler.utils.logging import get_logger
from kepler.utils.exceptions import ValidationError, KeplerError
from kepler.core.library_manager import LibraryManager


class ValidationLevel(Enum):
    """Validation severity levels"""
    CRITICAL = "critical"    # Blocks all functionality
    WARNING = "warning"      # Reduces functionality
    INFO = "info"           # Informational only
    SUCCESS = "success"     # All good


class ValidationCategory(Enum):
    """Categories of validation checks"""
    PREREQUISITES = "prerequisites"    # Python, basic tools
    AUTHENTICATION = "authentication"  # Credentials, tokens
    CONNECTIVITY = "connectivity"      # Network, API access
    CONFIGURATION = "configuration"    # Config files, settings
    PERMISSIONS = "permissions"        # Access rights, roles
    FUNCTIONALITY = "functionality"    # End-to-end workflows


@dataclass
class ValidationResult:
    """Result of a single validation check"""
    check_name: str
    category: ValidationCategory
    level: ValidationLevel
    success: bool
    message: str
    details: Optional[str] = None
    hint: Optional[str] = None
    auto_fix_available: bool = False
    auto_fix_command: Optional[str] = None
    documentation_url: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class EcosystemValidationReport:
    """Complete ecosystem validation report"""
    overall_status: ValidationLevel
    total_checks: int
    successful_checks: int
    failed_checks: int
    warning_checks: int
    validation_time: float
    results: List[ValidationResult]
    summary: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage"""
        if self.total_checks == 0:
            return 0.0
        return (self.successful_checks / self.total_checks) * 100


class EcosystemValidator:
    """
    Main ecosystem validator orchestrating all platform validations
    
    Validates complete Kepler ecosystem including:
    - Development prerequisites (Python, libraries)
    - Splunk connectivity and permissions
    - GCP authentication and services
    - MLOps tools availability
    - End-to-end workflow capability
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.lib_manager = LibraryManager(".")
        self.results: List[ValidationResult] = []
        
        # Platform validators
        self.splunk_validator = SplunkValidator()
        self.gcp_validator = GCPValidator()
        self.prerequisites_validator = PrerequisitesValidator()
        
    def validate_complete_ecosystem(self, 
                                   include_optional: bool = True,
                                   auto_fix: bool = False) -> EcosystemValidationReport:
        """
        Validate complete Kepler ecosystem
        
        Args:
            include_optional: Include optional components (MLflow, DVC)
            auto_fix: Attempt automatic fixes for common issues
            
        Returns:
            Complete validation report with actionable recommendations
        """
        self.logger.info("Starting complete ecosystem validation")
        start_time = time.time()
        
        self.results = []
        
        # Step 1: Prerequisites validation
        self.logger.info("Step 1: Validating prerequisites...")
        prereq_results = self.prerequisites_validator.validate_all()
        self.results.extend(prereq_results)
        
        # Step 2: Splunk validation
        self.logger.info("Step 2: Validating Splunk connectivity...")
        splunk_results = self.splunk_validator.validate_all()
        self.results.extend(splunk_results)
        
        # Step 3: GCP validation
        self.logger.info("Step 3: Validating GCP services...")
        gcp_results = self.gcp_validator.validate_all()
        self.results.extend(gcp_results)
        
        # Step 4: MLOps tools validation (optional)
        if include_optional:
            self.logger.info("Step 4: Validating MLOps tools...")
            mlops_results = self._validate_mlops_tools()
            self.results.extend(mlops_results)
        
        # Step 5: End-to-end workflow validation
        self.logger.info("Step 5: Validating end-to-end workflows...")
        e2e_results = self._validate_e2e_workflows()
        self.results.extend(e2e_results)
        
        # Generate report
        validation_time = time.time() - start_time
        report = self._generate_validation_report(validation_time)
        
        # Apply auto-fixes if requested
        if auto_fix:
            self._apply_auto_fixes(report)
        
        self.logger.info(f"Ecosystem validation completed in {validation_time:.2f}s")
        return report
    
    def _validate_mlops_tools(self) -> List[ValidationResult]:
        """Validate MLOps tools availability"""
        results = []
        
        # MLflow validation
        try:
            import mlflow
            mlflow.get_tracking_uri()
            
            results.append(ValidationResult(
                check_name="MLflow availability",
                category=ValidationCategory.FUNCTIONALITY,
                level=ValidationLevel.SUCCESS,
                success=True,
                message="MLflow is available and configured",
                hint="Use kp.versioning.* for experiment tracking"
            ))
        except Exception as e:
            results.append(ValidationResult(
                check_name="MLflow availability",
                category=ValidationCategory.FUNCTIONALITY,
                level=ValidationLevel.WARNING,
                success=False,
                message="MLflow not available",
                details=str(e),
                hint="Install with: pip install mlflow",
                auto_fix_available=True,
                auto_fix_command="pip install mlflow"
            ))
        
        # DVC validation
        try:
            result = subprocess.run(
                ["dvc", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                results.append(ValidationResult(
                    check_name="DVC availability",
                    category=ValidationCategory.FUNCTIONALITY,
                    level=ValidationLevel.SUCCESS,
                    success=True,
                    message="DVC is available",
                    hint="Use kp.versioning.* for data versioning"
                ))
            else:
                raise Exception("DVC command failed")
                
        except Exception as e:
            results.append(ValidationResult(
                check_name="DVC availability", 
                category=ValidationCategory.FUNCTIONALITY,
                level=ValidationLevel.WARNING,
                success=False,
                message="DVC not available",
                details=str(e),
                hint="Install with: pip install dvc",
                auto_fix_available=True,
                auto_fix_command="pip install dvc"
            ))
        
        return results
    
    def _validate_e2e_workflows(self) -> List[ValidationResult]:
        """Validate end-to-end workflow capabilities"""
        results = []
        
        # Test data extraction capability
        try:
            # Check if we can import data extraction
            from kepler.data import from_splunk
            
            results.append(ValidationResult(
                check_name="Data extraction workflow",
                category=ValidationCategory.FUNCTIONALITY,
                level=ValidationLevel.SUCCESS,
                success=True,
                message="Data extraction API available",
                hint="Use kp.data.from_splunk() to extract data"
            ))
        except Exception as e:
            results.append(ValidationResult(
                check_name="Data extraction workflow",
                category=ValidationCategory.FUNCTIONALITY,
                level=ValidationLevel.CRITICAL,
                success=False,
                message="Data extraction not available",
                details=str(e),
                hint="Check Kepler installation and imports"
            ))
        
        # Test training capability
        try:
            from kepler.train_unified import train
            
            results.append(ValidationResult(
                check_name="Model training workflow",
                category=ValidationCategory.FUNCTIONALITY,
                level=ValidationLevel.SUCCESS,
                success=True,
                message="Unified training API available",
                hint="Use kp.train_unified.train() for any AI framework"
            ))
        except Exception as e:
            results.append(ValidationResult(
                check_name="Model training workflow",
                category=ValidationCategory.FUNCTIONALITY,
                level=ValidationLevel.CRITICAL,
                success=False,
                message="Training API not available",
                details=str(e),
                hint="Check Kepler installation and dependencies"
            ))
        
        # Test deployment capability
        try:
            from kepler.deploy import to_cloud_run
            
            results.append(ValidationResult(
                check_name="Model deployment workflow",
                category=ValidationCategory.FUNCTIONALITY,
                level=ValidationLevel.SUCCESS,
                success=True,
                message="Deployment API available",
                hint="Use kp.deploy.to_cloud_run() to deploy models"
            ))
        except Exception as e:
            results.append(ValidationResult(
                check_name="Model deployment workflow",
                category=ValidationCategory.FUNCTIONALITY,
                level=ValidationLevel.CRITICAL,
                success=False,
                message="Deployment API not available",
                details=str(e),
                hint="Check Kepler installation and GCP setup"
            ))
        
        return results
    
    def _generate_validation_report(self, validation_time: float) -> EcosystemValidationReport:
        """Generate comprehensive validation report"""
        
        # Calculate statistics
        total_checks = len(self.results)
        successful_checks = len([r for r in self.results if r.success])
        failed_checks = len([r for r in self.results if not r.success and r.level == ValidationLevel.CRITICAL])
        warning_checks = len([r for r in self.results if not r.success and r.level == ValidationLevel.WARNING])
        
        # Determine overall status
        if failed_checks > 0:
            overall_status = ValidationLevel.CRITICAL
        elif warning_checks > 0:
            overall_status = ValidationLevel.WARNING
        else:
            overall_status = ValidationLevel.SUCCESS
        
        # Generate summary by category
        summary = {}
        for category in ValidationCategory:
            category_results = [r for r in self.results if r.category == category]
            if category_results:
                category_success = len([r for r in category_results if r.success])
                summary[category.value] = {
                    "total": len(category_results),
                    "successful": category_success,
                    "success_rate": (category_success / len(category_results)) * 100
                }
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        
        return EcosystemValidationReport(
            overall_status=overall_status,
            total_checks=total_checks,
            successful_checks=successful_checks,
            failed_checks=failed_checks,
            warning_checks=warning_checks,
            validation_time=validation_time,
            results=self.results,
            summary=summary,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on validation results"""
        recommendations = []
        
        # Critical issues first
        critical_issues = [r for r in self.results if r.level == ValidationLevel.CRITICAL and not r.success]
        if critical_issues:
            recommendations.append("🚨 CRITICAL: Fix these issues before using Kepler:")
            for issue in critical_issues:
                if issue.hint:
                    recommendations.append(f"   • {issue.check_name}: {issue.hint}")
        
        # Warnings
        warning_issues = [r for r in self.results if r.level == ValidationLevel.WARNING and not r.success]
        if warning_issues:
            recommendations.append("⚠️ WARNINGS: Consider fixing these for better experience:")
            for issue in warning_issues:
                if issue.hint:
                    recommendations.append(f"   • {issue.check_name}: {issue.hint}")
        
        # Auto-fixes available
        auto_fixable = [r for r in self.results if r.auto_fix_available and not r.success]
        if auto_fixable:
            recommendations.append("🔧 AUTO-FIXES: Run these commands to fix issues automatically:")
            for issue in auto_fixable:
                if issue.auto_fix_command:
                    recommendations.append(f"   • {issue.auto_fix_command}")
        
        # Success recommendations
        successful_platforms = []
        if any(r.success and "splunk" in r.check_name.lower() for r in self.results):
            successful_platforms.append("Splunk")
        if any(r.success and "gcp" in r.check_name.lower() for r in self.results):
            successful_platforms.append("GCP")
        
        if successful_platforms:
            platforms_str = " + ".join(successful_platforms)
            recommendations.append(f"✅ READY: You can start using Kepler with {platforms_str}")
            recommendations.append("   • Try: kepler extract 'search index=main | head 10'")
            recommendations.append("   • Try: kp.data.from_splunk('search index=main', time_range='-1h')")
        
        return recommendations
    
    def _apply_auto_fixes(self, report: EcosystemValidationReport) -> None:
        """Apply automatic fixes for common issues"""
        auto_fixable = [r for r in report.results if r.auto_fix_available and not r.success]
        
        if not auto_fixable:
            return
        
        self.logger.info(f"Applying {len(auto_fixable)} automatic fixes...")
        
        for fix in auto_fixable:
            try:
                self.logger.info(f"Applying fix: {fix.auto_fix_command}")
                
                # Execute auto-fix command
                result = subprocess.run(
                    fix.auto_fix_command.split(),
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minutes max per fix
                )
                
                if result.returncode == 0:
                    self.logger.info(f"✅ Auto-fix successful: {fix.check_name}")
                else:
                    self.logger.warning(f"❌ Auto-fix failed: {fix.check_name} - {result.stderr}")
                    
            except Exception as e:
                self.logger.warning(f"Auto-fix error for {fix.check_name}: {e}")


class SplunkValidator:
    """Validator for Splunk Enterprise connectivity and functionality"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def validate_all(self) -> List[ValidationResult]:
        """Run all Splunk validation checks"""
        results = []
        
        # Check configuration
        results.append(self._validate_configuration())
        
        # Check connectivity (only if config is valid)
        config_valid = results[-1].success
        if config_valid:
            results.append(self._validate_connectivity())
            results.append(self._validate_authentication())
            results.append(self._validate_indexes())
            results.append(self._validate_hec())
        
        return results
    
    def _validate_configuration(self) -> ValidationResult:
        """Validate Splunk configuration"""
        try:
            from kepler.core.config import load_config
            config = load_config()
            
            if not config.splunk.host:
                return ValidationResult(
                    check_name="Splunk configuration",
                    category=ValidationCategory.CONFIGURATION,
                    level=ValidationLevel.CRITICAL,
                    success=False,
                    message="Splunk host not configured",
                    hint="Run: kepler config init and edit ~/.kepler/config.yml",
                    auto_fix_available=True,
                    auto_fix_command="kepler config init"
                )
            
            if not config.splunk.token:
                return ValidationResult(
                    check_name="Splunk configuration",
                    category=ValidationCategory.CONFIGURATION,
                    level=ValidationLevel.CRITICAL,
                    success=False,
                    message="Splunk token not configured",
                    hint="Add authentication token to ~/.kepler/config.yml",
                    documentation_url="https://docs.splunk.com/Documentation/Splunk/latest/Security/CreateAuthTokens"
                )
            
            return ValidationResult(
                check_name="Splunk configuration",
                category=ValidationCategory.CONFIGURATION,
                level=ValidationLevel.SUCCESS,
                success=True,
                message="Splunk configuration is valid",
                context={
                    "host": config.splunk.host,
                    "hec_configured": bool(config.splunk.hec_token)
                }
            )
            
        except Exception as e:
            return ValidationResult(
                check_name="Splunk configuration",
                category=ValidationCategory.CONFIGURATION,
                level=ValidationLevel.CRITICAL,
                success=False,
                message="Failed to load Splunk configuration",
                details=str(e),
                hint="Run: kepler config init"
            )
    
    def _validate_connectivity(self) -> ValidationResult:
        """Validate Splunk network connectivity using official SDK"""
        try:
            from kepler.connectors.splunk import create_splunk_connector
            from kepler.core.config import load_config
            
            config = load_config()
            
            # Test connectivity using official Splunk SDK (same as authentication)
            connector = create_splunk_connector(
                host=config.splunk.host,
                token=config.splunk.token,
                verify_ssl=config.splunk.verify_ssl
            )
            
            # Test basic server info via SDK
            server_info = connector.client.info
            splunk_version = server_info.get('version', 'Unknown')
            
            return ValidationResult(
                check_name="Splunk connectivity",
                category=ValidationCategory.CONNECTIVITY,
                level=ValidationLevel.SUCCESS,
                success=True,
                message="Splunk server is accessible via SDK",
                context={
                    "splunk_version": splunk_version,
                    "connection_method": "splunk-sdk"
                }
            )
        except Exception as e:
            return ValidationResult(
                check_name="Splunk connectivity",
                category=ValidationCategory.CONNECTIVITY,
                level=ValidationLevel.CRITICAL,
                success=False,
                message="Splunk connectivity test failed",
                details=str(e),
                hint="Check Splunk configuration and network access"
            )
    
    def _validate_authentication(self) -> ValidationResult:
        """Validate Splunk authentication"""
        try:
            from kepler.connectors.splunk import create_splunk_connector
            from kepler.core.config import load_config
            
            config = load_config()
            
            # Test authentication by creating connector
            connector = create_splunk_connector(
                host=config.splunk.host,
                token=config.splunk.token,
                verify_ssl=config.splunk.verify_ssl
            )
            
            # Test with simple search
            test_results = connector.search("search index=* | head 1", max_results=1)
            
            return ValidationResult(
                check_name="Splunk authentication",
                category=ValidationCategory.AUTHENTICATION,
                level=ValidationLevel.SUCCESS,
                success=True,
                message="Splunk authentication successful",
                hint="Authentication token is valid and has search permissions"
            )
            
        except Exception as e:
            return ValidationResult(
                check_name="Splunk authentication",
                category=ValidationCategory.AUTHENTICATION,
                level=ValidationLevel.CRITICAL,
                success=False,
                message="Splunk authentication failed",
                details=str(e),
                hint="Check authentication token validity and permissions",
                documentation_url="https://docs.splunk.com/Documentation/Splunk/latest/Security/CreateAuthTokens"
            )
    
    def _validate_indexes(self) -> ValidationResult:
        """Validate Splunk index access"""
        try:
            from kepler.connectors.splunk import create_splunk_connector
            from kepler.core.config import load_config
            
            config = load_config()
            connector = create_splunk_connector(
                host=config.splunk.host,
                token=config.splunk.token,
                verify_ssl=config.splunk.verify_ssl
            )
            
            # List available indexes
            indexes = connector.get_available_indexes()
            
            if len(indexes) == 0:
                return ValidationResult(
                    check_name="Splunk indexes",
                    category=ValidationCategory.PERMISSIONS,
                    level=ValidationLevel.WARNING,
                    success=False,
                    message="No accessible indexes found",
                    hint="Check index permissions or contact Splunk administrator"
                )
            
            return ValidationResult(
                check_name="Splunk indexes",
                category=ValidationCategory.PERMISSIONS,
                level=ValidationLevel.SUCCESS,
                success=True,
                message=f"Access to {len(indexes)} indexes confirmed",
                context={"accessible_indexes": indexes[:10]}  # First 10 for brevity
            )
            
        except Exception as e:
            return ValidationResult(
                check_name="Splunk indexes",
                category=ValidationCategory.PERMISSIONS,
                level=ValidationLevel.WARNING,
                success=False,
                message="Failed to validate index access",
                details=str(e),
                hint="Check index permissions and network connectivity"
            )
    
    def _validate_hec(self) -> ValidationResult:
        """Validate Splunk HEC (HTTP Event Collector) functionality"""
        try:
            from kepler.core.config import load_config
            config = load_config()
            
            if not config.splunk.hec_token:
                return ValidationResult(
                    check_name="Splunk HEC",
                    category=ValidationCategory.CONFIGURATION,
                    level=ValidationLevel.WARNING,
                    success=False,
                    message="HEC token not configured",
                    hint="Configure HEC token for writing model predictions back to Splunk",
                    documentation_url="https://docs.splunk.com/Documentation/Splunk/latest/Data/UsetheHTTPEventCollector"
                )
            
            # Test HEC connectivity using official HecWriter
            from kepler.connectors.hec import HecWriter
            
            hec_writer = HecWriter(
                hec_url=config.splunk.hec_url,
                hec_token=config.splunk.hec_token,
                verify_ssl=config.splunk.verify_ssl
            )
            
            # Test with simple validation event
            test_event = {
                "message": "Kepler validation test",
                "timestamp": datetime.now().isoformat(),
                "source": "kepler-validator"
            }
            
            success = hec_writer.write_event(
                event_data=test_event,
                index=config.splunk.metrics_index,
                source="kepler-validator",
                sourcetype="kepler:validation"
            )
            
            if success:
                return ValidationResult(
                    check_name="Splunk HEC",
                    category=ValidationCategory.FUNCTIONALITY,
                    level=ValidationLevel.SUCCESS,
                    success=True,
                    message="HEC is accessible and working",
                    hint="Model predictions will be written to Splunk automatically"
                )
            else:
                return ValidationResult(
                    check_name="Splunk HEC",
                    category=ValidationCategory.FUNCTIONALITY,
                    level=ValidationLevel.WARNING,
                    success=False,
                    message="HEC validation failed",
                    hint="Check HEC token and endpoint configuration"
                )
                
        except Exception as e:
            return ValidationResult(
                check_name="Splunk HEC",
                category=ValidationCategory.FUNCTIONALITY,
                level=ValidationLevel.WARNING,
                success=False,
                message="HEC validation failed",
                details=str(e),
                hint="Check HEC configuration and network access"
            )


class GCPValidator:
    """Validator for Google Cloud Platform services and authentication"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def validate_all(self) -> List[ValidationResult]:
        """Run all GCP validation checks"""
        results = []
        
        # Check gcloud CLI
        results.append(self._validate_gcloud_cli())
        
        # Check authentication (only if CLI available)
        if results[-1].success:
            results.append(self._validate_authentication())
            results.append(self._validate_project_access())
            results.append(self._validate_cloud_run_api())
            results.append(self._validate_artifact_registry())
        
        return results
    
    def _validate_gcloud_cli(self) -> ValidationResult:
        """Validate Google Cloud SDK installation"""
        try:
            result = subprocess.run(
                ["gcloud", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                version_info = result.stdout.split('\n')[0]
                return ValidationResult(
                    check_name="Google Cloud SDK",
                    category=ValidationCategory.PREREQUISITES,
                    level=ValidationLevel.SUCCESS,
                    success=True,
                    message="Google Cloud SDK is installed",
                    context={"version": version_info}
                )
            else:
                raise Exception("gcloud command failed")
                
        except FileNotFoundError:
            return ValidationResult(
                check_name="Google Cloud SDK",
                category=ValidationCategory.PREREQUISITES,
                level=ValidationLevel.CRITICAL,
                success=False,
                message="Google Cloud SDK not installed",
                hint="Install from: https://cloud.google.com/sdk/docs/install",
                documentation_url="https://cloud.google.com/sdk/docs/install"
            )
        except Exception as e:
            return ValidationResult(
                check_name="Google Cloud SDK",
                category=ValidationCategory.PREREQUISITES,
                level=ValidationLevel.CRITICAL,
                success=False,
                message="Google Cloud SDK validation failed",
                details=str(e),
                hint="Check gcloud installation and PATH configuration"
            )
    
    def _validate_authentication(self) -> ValidationResult:
        """Validate GCP authentication"""
        try:
            # Check active account
            result = subprocess.run(
                ["gcloud", "auth", "list", "--filter=status:ACTIVE", "--format=value(account)"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and result.stdout.strip():
                active_account = result.stdout.strip()
                return ValidationResult(
                    check_name="GCP authentication",
                    category=ValidationCategory.AUTHENTICATION,
                    level=ValidationLevel.SUCCESS,
                    success=True,
                    message="GCP authentication is active",
                    context={"account": active_account}
                )
            else:
                return ValidationResult(
                    check_name="GCP authentication",
                    category=ValidationCategory.AUTHENTICATION,
                    level=ValidationLevel.CRITICAL,
                    success=False,
                    message="No active GCP authentication",
                    hint="Run: gcloud auth login",
                    auto_fix_available=True,
                    auto_fix_command="gcloud auth login"
                )
                
        except Exception as e:
            return ValidationResult(
                check_name="GCP authentication",
                category=ValidationCategory.AUTHENTICATION,
                level=ValidationLevel.CRITICAL,
                success=False,
                message="GCP authentication check failed",
                details=str(e),
                hint="Ensure gcloud CLI is properly installed and configured"
            )
    
    def _validate_project_access(self) -> ValidationResult:
        """Validate GCP project access"""
        try:
            # Get current project
            result = subprocess.run(
                ["gcloud", "config", "get-value", "project"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and result.stdout.strip():
                project_id = result.stdout.strip()
                
                # Test project access
                result = subprocess.run(
                    ["gcloud", "projects", "describe", project_id, "--format=json"],
                    capture_output=True,
                    text=True,
                    timeout=15
                )
                
                if result.returncode == 0:
                    project_info = json.loads(result.stdout)
                    return ValidationResult(
                        check_name="GCP project access",
                        category=ValidationCategory.PERMISSIONS,
                        level=ValidationLevel.SUCCESS,
                        success=True,
                        message=f"Project access confirmed: {project_id}",
                        context={
                            "project_id": project_id,
                            "project_name": project_info.get("name"),
                            "project_state": project_info.get("lifecycleState")
                        }
                    )
                else:
                    return ValidationResult(
                        check_name="GCP project access",
                        category=ValidationCategory.PERMISSIONS,
                        level=ValidationLevel.CRITICAL,
                        success=False,
                        message=f"Cannot access project: {project_id}",
                        details=result.stderr,
                        hint="Check project permissions or set correct project: gcloud config set project YOUR_PROJECT"
                    )
            else:
                return ValidationResult(
                    check_name="GCP project access",
                    category=ValidationCategory.CONFIGURATION,
                    level=ValidationLevel.CRITICAL,
                    success=False,
                    message="No GCP project configured",
                    hint="Set project: gcloud config set project YOUR_PROJECT_ID"
                )
                
        except Exception as e:
            return ValidationResult(
                check_name="GCP project access",
                category=ValidationCategory.PERMISSIONS,
                level=ValidationLevel.CRITICAL,
                success=False,
                message="Project access validation failed",
                details=str(e),
                hint="Check gcloud configuration and project permissions"
            )
    
    def _validate_cloud_run_api(self) -> ValidationResult:
        """Validate Cloud Run API access"""
        try:
            # Check if Cloud Run API is enabled
            result = subprocess.run(
                ["gcloud", "services", "list", "--enabled", "--filter=name:run.googleapis.com", "--format=value(name)"],
                capture_output=True,
                text=True,
                timeout=15
            )
            
            if result.returncode == 0 and "run.googleapis.com" in result.stdout:
                return ValidationResult(
                    check_name="Cloud Run API",
                    category=ValidationCategory.PERMISSIONS,
                    level=ValidationLevel.SUCCESS,
                    success=True,
                    message="Cloud Run API is enabled",
                    hint="Ready for model deployment with kp.deploy.to_cloud_run()"
                )
            else:
                return ValidationResult(
                    check_name="Cloud Run API",
                    category=ValidationCategory.PERMISSIONS,
                    level=ValidationLevel.CRITICAL,
                    success=False,
                    message="Cloud Run API not enabled",
                    hint="Enable with: gcloud services enable run.googleapis.com",
                    auto_fix_available=True,
                    auto_fix_command="gcloud services enable run.googleapis.com"
                )
                
        except Exception as e:
            return ValidationResult(
                check_name="Cloud Run API",
                category=ValidationCategory.PERMISSIONS,
                level=ValidationLevel.CRITICAL,
                success=False,
                message="Cloud Run API validation failed",
                details=str(e),
                hint="Check gcloud configuration and API access permissions"
            )
    
    def _validate_artifact_registry(self) -> ValidationResult:
        """Validate Artifact Registry for container images"""
        try:
            # Check if Artifact Registry API is enabled
            result = subprocess.run(
                ["gcloud", "services", "list", "--enabled", "--filter=name:artifactregistry.googleapis.com", "--format=value(name)"],
                capture_output=True,
                text=True,
                timeout=15
            )
            
            if result.returncode == 0 and "artifactregistry.googleapis.com" in result.stdout:
                return ValidationResult(
                    check_name="Artifact Registry",
                    category=ValidationCategory.PERMISSIONS,
                    level=ValidationLevel.SUCCESS,
                    success=True,
                    message="Artifact Registry API is enabled",
                    hint="Container images will be stored in Artifact Registry during deployment"
                )
            else:
                return ValidationResult(
                    check_name="Artifact Registry",
                    category=ValidationCategory.PERMISSIONS,
                    level=ValidationLevel.WARNING,
                    success=False,
                    message="Artifact Registry API not enabled",
                    hint="Enable with: gcloud services enable artifactregistry.googleapis.com",
                    auto_fix_available=True,
                    auto_fix_command="gcloud services enable artifactregistry.googleapis.com"
                )
                
        except Exception as e:
            return ValidationResult(
                check_name="Artifact Registry",
                category=ValidationCategory.PERMISSIONS,
                level=ValidationLevel.WARNING,
                success=False,
                message="Artifact Registry validation failed",
                details=str(e),
                hint="Check API permissions (deployment will use alternative methods)"
            )


class PrerequisitesValidator:
    """Validator for development prerequisites and environment"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.lib_manager = LibraryManager(".")
    
    def validate_all(self) -> List[ValidationResult]:
        """Run all prerequisites validation checks"""
        results = []
        
        results.append(self._validate_python_version())
        results.append(self._validate_kepler_installation())
        results.append(self._validate_library_environment())
        results.append(self._validate_jupyter_availability())
        results.append(self._validate_splunk_sdk())
        
        return results
    
    def _validate_python_version(self) -> ValidationResult:
        """Validate Python version compatibility"""
        import sys
        
        version_info = sys.version_info
        version_string = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
        
        if version_info >= (3, 11):
            return ValidationResult(
                check_name="Python version",
                category=ValidationCategory.PREREQUISITES,
                level=ValidationLevel.SUCCESS,
                success=True,
                message=f"Python {version_string} is compatible",
                context={"version": version_string, "recommended": True}
            )
        elif version_info >= (3, 8):
            return ValidationResult(
                check_name="Python version",
                category=ValidationCategory.PREREQUISITES,
                level=ValidationLevel.WARNING,
                success=True,
                message=f"Python {version_string} is supported but not optimal",
                hint="Consider upgrading to Python 3.11+ for better performance",
                context={"version": version_string, "recommended": False}
            )
        else:
            return ValidationResult(
                check_name="Python version",
                category=ValidationCategory.PREREQUISITES,
                level=ValidationLevel.CRITICAL,
                success=False,
                message=f"Python {version_string} is not supported",
                hint="Upgrade to Python 3.8+ (recommended: 3.11+)",
                context={"version": version_string, "minimum_required": "3.8"}
            )
    
    def _validate_kepler_installation(self) -> ValidationResult:
        """Validate Kepler framework installation"""
        try:
            import kepler
            version = getattr(kepler, '__version__', 'unknown')
            
            # Check if main APIs are available
            required_apis = ['data', 'train_unified', 'automl', 'versioning', 'deploy']
            missing_apis = []
            
            for api in required_apis:
                if not hasattr(kepler, api):
                    missing_apis.append(api)
            
            if missing_apis:
                return ValidationResult(
                    check_name="Kepler installation",
                    category=ValidationCategory.PREREQUISITES,
                    level=ValidationLevel.CRITICAL,
                    success=False,
                    message="Kepler installation is incomplete",
                    details=f"Missing APIs: {', '.join(missing_apis)}",
                    hint="Reinstall Kepler framework"
                )
            
            return ValidationResult(
                check_name="Kepler installation",
                category=ValidationCategory.PREREQUISITES,
                level=ValidationLevel.SUCCESS,
                success=True,
                message=f"Kepler v{version} is properly installed",
                context={"version": version, "apis_available": required_apis}
            )
            
        except ImportError:
            return ValidationResult(
                check_name="Kepler installation",
                category=ValidationCategory.PREREQUISITES,
                level=ValidationLevel.CRITICAL,
                success=False,
                message="Kepler framework not installed",
                hint="Install with: pip install kepler-framework",
                auto_fix_available=True,
                auto_fix_command="pip install -e ."
            )
    
    def _validate_library_environment(self) -> ValidationResult:
        """Validate Python library environment"""
        try:
            env_report = self.lib_manager.validate_environment()
            
            total_libs = env_report.get('total_libraries', 0)
            successful_imports = env_report.get('successful_imports', 0)
            missing_libs = env_report.get('missing_libraries', [])
            
            if total_libs == 0:
                return ValidationResult(
                    check_name="Library environment",
                    category=ValidationCategory.PREREQUISITES,
                    level=ValidationLevel.INFO,
                    success=True,
                    message="No requirements.txt found - using base environment",
                    hint="Create requirements.txt with: kepler libs template --template ml"
                )
            
            success_rate = (successful_imports / total_libs) * 100 if total_libs > 0 else 0
            
            if success_rate >= 90:
                return ValidationResult(
                    check_name="Library environment",
                    category=ValidationCategory.PREREQUISITES,
                    level=ValidationLevel.SUCCESS,
                    success=True,
                    message=f"Library environment is healthy ({success_rate:.1f}% success rate)",
                    context={
                        "total_libraries": total_libs,
                        "successful_imports": successful_imports,
                        "success_rate": success_rate
                    }
                )
            elif success_rate >= 70:
                return ValidationResult(
                    check_name="Library environment",
                    category=ValidationCategory.PREREQUISITES,
                    level=ValidationLevel.WARNING,
                    success=False,
                    message=f"Library environment has issues ({success_rate:.1f}% success rate)",
                    details=f"Missing libraries: {', '.join(missing_libs[:5])}",
                    hint="Run: kepler libs install --library requirements.txt",
                    auto_fix_available=True,
                    auto_fix_command="kepler libs install"
                )
            else:
                return ValidationResult(
                    check_name="Library environment",
                    category=ValidationCategory.PREREQUISITES,
                    level=ValidationLevel.CRITICAL,
                    success=False,
                    message=f"Library environment is broken ({success_rate:.1f}% success rate)",
                    details=f"Missing libraries: {', '.join(missing_libs)}",
                    hint="Reinstall libraries: kepler libs install",
                    auto_fix_available=True,
                    auto_fix_command="kepler libs install"
                )
                
        except Exception as e:
            return ValidationResult(
                check_name="Library environment",
                category=ValidationCategory.PREREQUISITES,
                level=ValidationLevel.WARNING,
                success=False,
                message="Library environment validation failed",
                details=str(e),
                hint="Check Python environment and requirements.txt"
            )
    
    def _validate_jupyter_availability(self) -> ValidationResult:
        """Validate Jupyter notebook availability"""
        try:
            import jupyter
            
            return ValidationResult(
                check_name="Jupyter availability",
                category=ValidationCategory.PREREQUISITES,
                level=ValidationLevel.SUCCESS,
                success=True,
                message="Jupyter is available",
                hint="Use Jupyter notebooks for interactive analysis with Kepler"
            )
            
        except ImportError:
            return ValidationResult(
                check_name="Jupyter availability",
                category=ValidationCategory.PREREQUISITES,
                level=ValidationLevel.WARNING,
                success=False,
                message="Jupyter not installed",
                hint="Install with: pip install jupyter",
                auto_fix_available=True,
                auto_fix_command="pip install jupyter"
            )
    
    def _validate_splunk_sdk(self) -> ValidationResult:
        """Validate Splunk SDK availability"""
        try:
            import splunklib.client as client
            import splunklib.results as results
            
            # Test basic SDK functionality
            sdk_version = getattr(client, '__version__', 'Unknown')
            
            return ValidationResult(
                check_name="Splunk SDK",
                category=ValidationCategory.PREREQUISITES,
                level=ValidationLevel.SUCCESS,
                success=True,
                message="Splunk SDK is available and ready",
                context={
                    "sdk_version": sdk_version,
                    "modules": ["splunklib.client", "splunklib.results"]
                },
                hint="Use official Splunk SDK for all Splunk operations"
            )
            
        except ImportError as e:
            return ValidationResult(
                check_name="Splunk SDK",
                category=ValidationCategory.PREREQUISITES,
                level=ValidationLevel.CRITICAL,
                success=False,
                message="Splunk SDK not installed",
                details=str(e),
                hint="Install with: pip install splunk-sdk",
                auto_fix_available=True,
                auto_fix_command="pip install splunk-sdk"
            )
    
    def _validate_authentication(self) -> ValidationResult:
        """Validate GCP authentication"""
        try:
            result = subprocess.run(
                ["gcloud", "auth", "list", "--filter=status:ACTIVE", "--format=value(account)"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and result.stdout.strip():
                active_account = result.stdout.strip()
                return ValidationResult(
                    check_name="GCP authentication",
                    category=ValidationCategory.AUTHENTICATION,
                    level=ValidationLevel.SUCCESS,
                    success=True,
                    message="GCP authentication is active",
                    context={"account": active_account}
                )
            else:
                return ValidationResult(
                    check_name="GCP authentication",
                    category=ValidationCategory.AUTHENTICATION,
                    level=ValidationLevel.CRITICAL,
                    success=False,
                    message="No active GCP authentication",
                    hint="Run: gcloud auth login",
                    auto_fix_available=True,
                    auto_fix_command="gcloud auth login"
                )
        except Exception as e:
            return ValidationResult(
                check_name="GCP authentication",
                category=ValidationCategory.AUTHENTICATION,
                level=ValidationLevel.CRITICAL,
                success=False,
                message="GCP authentication check failed",
                details=str(e),
                hint="Check gcloud CLI installation and configuration"
            )
    
    def _validate_project_access(self) -> ValidationResult:
        """Validate GCP project access and permissions"""
        try:
            result = subprocess.run(
                ["gcloud", "config", "get-value", "project"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and result.stdout.strip():
                project_id = result.stdout.strip()
                
                # Test project describe (requires basic permissions)
                result = subprocess.run(
                    ["gcloud", "projects", "describe", project_id, "--format=json"],
                    capture_output=True,
                    text=True,
                    timeout=15
                )
                
                if result.returncode == 0:
                    project_info = json.loads(result.stdout)
                    return ValidationResult(
                        check_name="GCP project access",
                        category=ValidationCategory.PERMISSIONS,
                        level=ValidationLevel.SUCCESS,
                        success=True,
                        message=f"Project access confirmed: {project_id}",
                        context={
                            "project_id": project_id,
                            "project_name": project_info.get("name"),
                            "project_state": project_info.get("lifecycleState")
                        }
                    )
                else:
                    return ValidationResult(
                        check_name="GCP project access",
                        category=ValidationCategory.PERMISSIONS,
                        level=ValidationLevel.CRITICAL,
                        success=False,
                        message=f"Cannot access project: {project_id}",
                        details=result.stderr,
                        hint="Check project permissions or set correct project"
                    )
            else:
                return ValidationResult(
                    check_name="GCP project access",
                    category=ValidationCategory.CONFIGURATION,
                    level=ValidationLevel.CRITICAL,
                    success=False,
                    message="No GCP project configured",
                    hint="Set project: gcloud config set project YOUR_PROJECT_ID"
                )
        except Exception as e:
            return ValidationResult(
                check_name="GCP project access",
                category=ValidationCategory.PERMISSIONS,
                level=ValidationLevel.CRITICAL,
                success=False,
                message="Project access validation failed",
                details=str(e),
                hint="Check gcloud configuration and project permissions"
            )
    
    def _validate_cloud_run_api(self) -> ValidationResult:
        """Validate Cloud Run API is enabled"""
        try:
            result = subprocess.run(
                ["gcloud", "services", "list", "--enabled", "--filter=name:run.googleapis.com", "--format=value(name)"],
                capture_output=True,
                text=True,
                timeout=15
            )
            
            if result.returncode == 0 and "run.googleapis.com" in result.stdout:
                return ValidationResult(
                    check_name="Cloud Run API",
                    category=ValidationCategory.PERMISSIONS,
                    level=ValidationLevel.SUCCESS,
                    success=True,
                    message="Cloud Run API is enabled",
                    hint="Ready for model deployment"
                )
            else:
                return ValidationResult(
                    check_name="Cloud Run API",
                    category=ValidationCategory.PERMISSIONS,
                    level=ValidationLevel.CRITICAL,
                    success=False,
                    message="Cloud Run API not enabled",
                    hint="Enable with: gcloud services enable run.googleapis.com",
                    auto_fix_available=True,
                    auto_fix_command="gcloud services enable run.googleapis.com"
                )
        except Exception as e:
            return ValidationResult(
                check_name="Cloud Run API",
                category=ValidationCategory.PERMISSIONS,
                level=ValidationLevel.CRITICAL,
                success=False,
                message="Cloud Run API validation failed",
                details=str(e),
                hint="Check gcloud permissions and API access"
            )
    
    def _validate_artifact_registry(self) -> ValidationResult:
        """Validate Artifact Registry API for container storage"""
        try:
            result = subprocess.run(
                ["gcloud", "services", "list", "--enabled", "--filter=name:artifactregistry.googleapis.com", "--format=value(name)"],
                capture_output=True,
                text=True,
                timeout=15
            )
            
            if result.returncode == 0 and "artifactregistry.googleapis.com" in result.stdout:
                return ValidationResult(
                    check_name="Artifact Registry API",
                    category=ValidationCategory.PERMISSIONS,
                    level=ValidationLevel.SUCCESS,
                    success=True,
                    message="Artifact Registry API is enabled"
                )
            else:
                return ValidationResult(
                    check_name="Artifact Registry API",
                    category=ValidationCategory.PERMISSIONS,
                    level=ValidationLevel.WARNING,
                    success=False,
                    message="Artifact Registry API not enabled",
                    hint="Enable with: gcloud services enable artifactregistry.googleapis.com",
                    auto_fix_available=True,
                    auto_fix_command="gcloud services enable artifactregistry.googleapis.com"
                )
        except Exception as e:
            return ValidationResult(
                check_name="Artifact Registry API",
                category=ValidationCategory.PERMISSIONS,
                level=ValidationLevel.WARNING,
                success=False,
                message="Artifact Registry validation failed",
                details=str(e),
                hint="Check API permissions (fallback methods available)"
            )


# Global validator instance
_ecosystem_validator = None


def get_ecosystem_validator() -> EcosystemValidator:
    """Get or create global EcosystemValidator instance"""
    global _ecosystem_validator
    if _ecosystem_validator is None:
        _ecosystem_validator = EcosystemValidator()
    return _ecosystem_validator


# Convenience functions for SDK usage
def validate_ecosystem(include_optional: bool = True, auto_fix: bool = False) -> EcosystemValidationReport:
    """
    Validate complete Kepler ecosystem
    
    Args:
        include_optional: Include optional components (MLflow, DVC)
        auto_fix: Attempt automatic fixes for common issues
        
    Returns:
        Complete validation report
        
    Example:
        >>> import kepler as kp
        >>> report = kp.validate.ecosystem()
        >>> print(f"Overall status: {report.overall_status.value}")
        >>> print(f"Success rate: {report.success_rate:.1f}%")
        >>> 
        >>> for rec in report.recommendations:
        ...     print(rec)
    """
    validator = get_ecosystem_validator()
    return validator.validate_complete_ecosystem(include_optional, auto_fix)


def validate_splunk() -> List[ValidationResult]:
    """
    Validate Splunk connectivity and functionality
    
    Returns:
        List of Splunk validation results
        
    Example:
        >>> results = kp.validate.splunk()
        >>> for result in results:
        ...     print(f"{result.check_name}: {result.message}")
    """
    validator = get_ecosystem_validator()
    return validator.splunk_validator.validate_all()


def validate_gcp() -> List[ValidationResult]:
    """
    Validate GCP authentication and services
    
    Returns:
        List of GCP validation results
        
    Example:
        >>> results = kp.validate.gcp()
        >>> for result in results:
        ...     if not result.success:
        ...         print(f"❌ {result.check_name}: {result.hint}")
    """
    validator = get_ecosystem_validator()
    return validator.gcp_validator.validate_all()
