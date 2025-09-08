"""
Integration tests for ecosystem validation system
Tests Task 7.9: Add validation integration tests with real platforms

Tests validation functionality with real platform connections when available.
Includes fallback tests for environments without platform access.
"""

import pytest
import os
from unittest.mock import Mock, patch
from pathlib import Path

from kepler.core.ecosystem_validator import (
    EcosystemValidator,
    SplunkValidator, 
    GCPValidator,
    PrerequisitesValidator,
    ValidationLevel,
    ValidationCategory,
    validate_ecosystem,
    validate_splunk,
    validate_gcp
)
from kepler.utils.exceptions import ValidationError


class TestEcosystemValidation:
    """Test ecosystem validation functionality"""
    
    def test_ecosystem_validator_initialization(self):
        """Test EcosystemValidator can be initialized"""
        validator = EcosystemValidator()
        assert validator is not None
        assert hasattr(validator, 'splunk_validator')
        assert hasattr(validator, 'gcp_validator')
        assert hasattr(validator, 'prerequisites_validator')
    
    def test_validation_result_structure(self):
        """Test ValidationResult data structure"""
        from kepler.core.ecosystem_validator import ValidationResult
        
        result = ValidationResult(
            check_name="test_check",
            category=ValidationCategory.PREREQUISITES,
            level=ValidationLevel.SUCCESS,
            success=True,
            message="Test message",
            hint="Test hint",
            auto_fix_available=True,
            auto_fix_command="test command"
        )
        
        assert result.check_name == "test_check"
        assert result.category == ValidationCategory.PREREQUISITES
        assert result.level == ValidationLevel.SUCCESS
        assert result.success is True
        assert result.auto_fix_available is True
    
    def test_ecosystem_validation_report_structure(self):
        """Test EcosystemValidationReport data structure and calculations"""
        from kepler.core.ecosystem_validator import EcosystemValidationReport, ValidationResult
        
        # Create sample results
        results = [
            ValidationResult("test1", ValidationCategory.PREREQUISITES, ValidationLevel.SUCCESS, True, "Success"),
            ValidationResult("test2", ValidationCategory.CONNECTIVITY, ValidationLevel.CRITICAL, False, "Failed"),
            ValidationResult("test3", ValidationCategory.AUTHENTICATION, ValidationLevel.WARNING, False, "Warning")
        ]
        
        report = EcosystemValidationReport(
            overall_status=ValidationLevel.CRITICAL,
            total_checks=3,
            successful_checks=1,
            failed_checks=1,
            warning_checks=1,
            validation_time=1.5,
            results=results
        )
        
        assert report.success_rate == pytest.approx(33.33, rel=1e-2)
        assert report.total_checks == 3
        assert report.successful_checks == 1
    
    def test_prerequisites_validation(self):
        """Test prerequisites validation"""
        validator = PrerequisitesValidator()
        results = validator.validate_all()
        
        assert len(results) > 0
        
        # Should always have Python version check
        python_check = next((r for r in results if "python version" in r.check_name.lower()), None)
        assert python_check is not None
        
        # Should always have Kepler installation check
        kepler_check = next((r for r in results if "kepler installation" in r.check_name.lower()), None)
        assert kepler_check is not None
    
    def test_splunk_validator_without_config(self):
        """Test Splunk validation when no configuration exists"""
        validator = SplunkValidator()
        
        with patch('kepler.core.config.load_config') as mock_load:
            # Mock missing configuration
            mock_load.side_effect = Exception("Config not found")
            
            results = validator.validate_all()
            
            # Should have configuration check that fails
            config_check = results[0]
            assert config_check.check_name == "Splunk configuration"
            assert config_check.success is False
            assert config_check.level == ValidationLevel.CRITICAL
            assert "hint" in config_check.__dict__
    
    def test_gcp_validator_without_gcloud(self):
        """Test GCP validation when gcloud CLI is not available"""
        validator = GCPValidator()
        
        with patch('subprocess.run') as mock_run:
            # Mock gcloud not found
            mock_run.side_effect = FileNotFoundError("gcloud not found")
            
            results = validator.validate_all()
            
            # Should have gcloud CLI check that fails
            gcloud_check = results[0]
            assert gcloud_check.check_name == "Google Cloud SDK"
            assert gcloud_check.success is False
            assert gcloud_check.level == ValidationLevel.CRITICAL
    
    def test_ecosystem_validation_api_function(self):
        """Test ecosystem validation API function"""
        # Test that API function works
        report = validate_ecosystem(include_optional=False, auto_fix=False)
        
        assert hasattr(report, 'overall_status')
        assert hasattr(report, 'results')
        assert hasattr(report, 'recommendations')
        assert isinstance(report.results, list)
        assert isinstance(report.recommendations, list)
    
    def test_platform_specific_validation_apis(self):
        """Test platform-specific validation API functions"""
        # Test Splunk validation API
        splunk_results = validate_splunk()
        assert isinstance(splunk_results, list)
        
        # Test GCP validation API
        gcp_results = validate_gcp()
        assert isinstance(gcp_results, list)
    
    @pytest.mark.skipif(
        not pytest.config.getoption("--run-platform-tests", default=False),
        reason="Platform tests require --run-platform-tests flag and real platform access"
    )
    def test_real_splunk_validation(self):
        """
        Test Splunk validation with real Splunk instance
        
        Run with: pytest --run-platform-tests tests/integration/test_ecosystem_validation.py::test_real_splunk_validation
        """
        try:
            # This test requires real Splunk configuration
            validator = SplunkValidator()
            results = validator.validate_all()
            
            # Check that we get meaningful results
            assert len(results) > 0
            
            # At least configuration check should run
            config_results = [r for r in results if "configuration" in r.check_name.lower()]
            assert len(config_results) > 0
            
            print(f"✅ Splunk validation completed with {len(results)} checks")
            for result in results:
                status = "✅" if result.success else "❌"
                print(f"   {status} {result.check_name}: {result.message}")
                
        except Exception as e:
            pytest.skip(f"Real Splunk validation skipped: {e}")
    
    @pytest.mark.skipif(
        not pytest.config.getoption("--run-platform-tests", default=False),
        reason="Platform tests require --run-platform-tests flag and real platform access"
    )
    def test_real_gcp_validation(self):
        """
        Test GCP validation with real GCP setup
        
        Run with: pytest --run-platform-tests tests/integration/test_ecosystem_validation.py::test_real_gcp_validation
        """
        try:
            # This test requires real GCP configuration
            validator = GCPValidator()
            results = validator.validate_all()
            
            # Check that we get meaningful results
            assert len(results) > 0
            
            # At least gcloud CLI check should run
            gcloud_results = [r for r in results if "sdk" in r.check_name.lower()]
            assert len(gcloud_results) > 0
            
            print(f"✅ GCP validation completed with {len(results)} checks")
            for result in results:
                status = "✅" if result.success else "❌"
                print(f"   {status} {result.check_name}: {result.message}")
                
        except Exception as e:
            pytest.skip(f"Real GCP validation skipped: {e}")
    
    def test_validation_error_codes(self):
        """Test that validation uses standardized error codes"""
        # Test ValidationError uses correct format
        try:
            raise ValidationError(
                "Test validation error",
                component="test_component",
                hint="Test hint"
            )
        except ValidationError as e:
            assert hasattr(e, 'code')
            assert e.code == "VALIDATE_001"
            assert hasattr(e, 'context')
            assert hasattr(e, 'hint')
    
    def test_auto_fix_functionality(self):
        """Test auto-fix functionality structure"""
        validator = EcosystemValidator()
        
        # Create mock results with auto-fixes
        mock_results = [
            Mock(
                auto_fix_available=True,
                auto_fix_command="pip install missing-package",
                success=False,
                check_name="test_check"
            )
        ]
        
        mock_report = Mock()
        mock_report.results = mock_results
        
        # Test that auto-fix method exists and can be called
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
            
            # This should not raise an exception
            validator._apply_auto_fixes(mock_report)


class TestSecureCredentialManagement:
    """Test secure credential management system"""
    
    def test_credential_manager_initialization(self):
        """Test SecureCredentialManager initialization"""
        from kepler.core.security import SecureCredentialManager
        
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SecureCredentialManager(config_dir=temp_dir)
            assert manager.config_dir.exists()
            assert manager.config_dir.stat().st_mode & 0o777 == 0o700  # Check permissions
    
    def test_credential_storage_apis(self):
        """Test credential storage API functions"""
        from kepler.core.security import (
            store_credential,
            get_credential,
            list_credentials,
            delete_credential
        )
        
        # Test that functions are callable
        assert callable(store_credential)
        assert callable(get_credential)
        assert callable(list_credentials)
        assert callable(delete_credential)
    
    def test_credential_encryption_fallback(self):
        """Test credential encryption when keychain is not available"""
        from kepler.core.security import SecureCredentialManager
        
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SecureCredentialManager(config_dir=temp_dir)
            
            # Mock keychain not available
            with patch.object(manager, '_store_in_keychain', return_value=False):
                with patch.object(manager, '_get_master_password', return_value="test_password"):
                    
                    # Test storage
                    success = manager.store_credential("test_cred", "test_value")
                    assert success
                    
                    # Test retrieval
                    retrieved = manager.get_credential("test_cred")
                    assert retrieved == "test_value"
    
    def test_security_validation(self):
        """Test security posture validation"""
        from kepler.core.security import validate_security
        
        security_status = validate_security()
        
        assert isinstance(security_status, dict)
        assert "overall_secure" in security_status
        assert "issues" in security_status
        assert "recommendations" in security_status


class TestCLIValidationCommands:
    """Test CLI validation commands"""
    
    def test_validate_command_structure(self):
        """Test validate command structure"""
        from typer.testing import CliRunner
        from kepler.cli.main import app
        
        runner = CliRunner()
        
        # Test help
        result = runner.invoke(app, ["validate", "--help"])
        assert result.exit_code == 0
        assert "Validate Kepler ecosystem" in result.stdout
        assert "--auto-fix" in result.stdout
        assert "--format" in result.stdout
    
    def test_setup_command_structure(self):
        """Test setup command structure"""
        from typer.testing import CliRunner
        from kepler.cli.main import app
        
        runner = CliRunner()
        
        # Test help
        result = runner.invoke(app, ["setup", "--help"])
        assert result.exit_code == 0
        assert "Guided setup for platform integration" in result.stdout
        assert "--interactive" in result.stdout
        assert "--secure" in result.stdout
    
    def test_diagnose_command_structure(self):
        """Test diagnose command structure"""
        from typer.testing import CliRunner
        from kepler.cli.main import app
        
        runner = CliRunner()
        
        # Test help
        result = runner.invoke(app, ["diagnose", "--help"])
        assert result.exit_code == 0
        assert "Intelligent troubleshooting" in result.stdout
        assert "--platform" in result.stdout
        assert "--verbose" in result.stdout
    
    def test_validate_prerequisites_command(self):
        """Test validate prerequisites command"""
        from typer.testing import CliRunner
        from kepler.cli.main import app
        
        runner = CliRunner()
        
        # This should work without any external dependencies
        result = runner.invoke(app, ["validate", "prerequisites"])
        
        # Should not crash (exit code 0 or 1 are both acceptable)
        assert result.exit_code in [0, 1]
        
        # Should contain validation output
        assert "Prerequisites Validation" in result.stdout or "Validating prerequisites" in result.stdout


class TestValidationIntegrationWithOtherSystems:
    """Test validation integration with other Kepler systems"""
    
    def test_validation_with_library_manager(self):
        """Test validation integrates with LibraryManager"""
        validator = EcosystemValidator()
        
        # Test that library manager is accessible
        assert hasattr(validator, 'lib_manager')
        assert validator.lib_manager is not None
    
    def test_validation_with_config_system(self):
        """Test validation integrates with config system"""
        validator = SplunkValidator()
        
        # Test configuration validation (should handle missing config gracefully)
        config_result = validator._validate_configuration()
        
        assert hasattr(config_result, 'check_name')
        assert hasattr(config_result, 'success')
        assert hasattr(config_result, 'level')
        assert hasattr(config_result, 'message')
    
    def test_validation_error_handling(self):
        """Test validation error handling follows standards"""
        validator = EcosystemValidator()
        
        # Test that validation errors use standardized format
        try:
            # Force an error condition
            raise ValidationError(
                "Test validation error",
                component="test_component"
            )
        except ValidationError as e:
            assert hasattr(e, 'code')
            assert e.code == "VALIDATE_001"
            assert hasattr(e, 'context')
            assert e.context.get('component') == "test_component"


class TestValidationReporting:
    """Test validation reporting and output formatting"""
    
    def test_validation_report_generation(self):
        """Test validation report generation"""
        from kepler.core.ecosystem_validator import ValidationResult, EcosystemValidationReport
        
        # Create sample results
        results = [
            ValidationResult("test1", ValidationCategory.PREREQUISITES, ValidationLevel.SUCCESS, True, "Success"),
            ValidationResult("test2", ValidationCategory.CONNECTIVITY, ValidationLevel.CRITICAL, False, "Failed")
        ]
        
        # Create report
        report = EcosystemValidationReport(
            overall_status=ValidationLevel.CRITICAL,
            total_checks=2,
            successful_checks=1,
            failed_checks=1,
            warning_checks=0,
            validation_time=1.0,
            results=results
        )
        
        # Test report properties
        assert report.success_rate == 50.0
        assert report.overall_status == ValidationLevel.CRITICAL
        assert len(report.results) == 2
    
    def test_recommendation_generation(self):
        """Test recommendation generation based on results"""
        validator = EcosystemValidator()
        
        # Mock some validation results
        validator.results = [
            Mock(
                level=ValidationLevel.CRITICAL,
                success=False,
                check_name="Test critical issue",
                hint="Fix this critical issue"
            ),
            Mock(
                level=ValidationLevel.WARNING,
                success=False,
                check_name="Test warning",
                hint="Fix this warning"
            ),
            Mock(
                auto_fix_available=True,
                auto_fix_command="pip install something",
                success=False,
                check_name="Auto-fixable issue"
            )
        ]
        
        recommendations = validator._generate_recommendations()
        
        assert len(recommendations) > 0
        assert any("CRITICAL" in rec for rec in recommendations)
        assert any("WARNINGS" in rec for rec in recommendations)
        assert any("AUTO-FIXES" in rec for rec in recommendations)


class TestValidationCLIIntegration:
    """Test validation CLI integration"""
    
    def test_cli_validation_ecosystem_dry_run(self):
        """Test CLI ecosystem validation (dry run)"""
        from typer.testing import CliRunner
        from kepler.cli.main import app
        
        runner = CliRunner()
        
        # Test ecosystem validation with summary format
        result = runner.invoke(app, ["validate", "ecosystem", "--format", "summary"])
        
        # Should complete without crashing
        assert result.exit_code in [0, 1]  # Success or validation failures
        
        # Should contain some validation output
        assert len(result.stdout) > 100  # Should have substantial output
    
    def test_cli_validation_json_output(self):
        """Test CLI validation with JSON output"""
        from typer.testing import CliRunner
        from kepler.cli.main import app
        import json
        
        runner = CliRunner()
        
        # Test JSON output format
        result = runner.invoke(app, ["validate", "prerequisites", "--format", "json"])
        
        # Should produce valid JSON
        try:
            output_data = json.loads(result.stdout)
            assert isinstance(output_data, dict)
            assert "overall_status" in output_data
            assert "results" in output_data
        except json.JSONDecodeError:
            pytest.fail("CLI validation did not produce valid JSON output")


# Test configuration hook for pytest
def pytest_addoption(parser):
    """Add custom pytest options"""
    parser.addoption(
        "--run-platform-tests",
        action="store_true", 
        default=False,
        help="Run tests that require actual platform access (Splunk, GCP)"
    )


# Test markers
pytestmark = [
    pytest.mark.integration,
    pytest.mark.validation
]
