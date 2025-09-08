"""
End-to-End Cloud Run Deployment Integration Tests
Tests Task 6.10: Create end-to-end deployment integration tests

Tests the complete deployment pipeline:
- Model training → Deployment → Health checks → Predictions → Splunk integration

Note: These tests require actual GCP credentials and project access.
Use with caution in CI/CD environments.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import kepler as kp
from kepler.deployers.cloud_run_deployer import CloudRunDeployer, CloudRunConfig
from kepler.utils.exceptions import DeploymentError


class TestCloudRunDeploymentE2E:
    """End-to-end deployment tests (requires GCP access)"""
    
    @pytest.fixture
    def sample_training_data(self):
        """Create sample data for model training"""
        np.random.seed(42)
        return pd.DataFrame({
            'temperature': np.random.normal(25, 5, 1000),
            'pressure': np.random.normal(1013, 50, 1000),
            'vibration': np.random.normal(0.5, 0.2, 1000),
            'failure': np.random.randint(0, 2, 1000)
        })
    
    @pytest.fixture
    def trained_model(self, sample_training_data):
        """Train a sample model for deployment testing"""
        try:
            # Train XGBoost model (most reliable for testing)
            model = kp.train_unified.train(
                sample_training_data, 
                target="failure", 
                algorithm="xgboost"
            )
            return model
        except Exception:
            # Fallback to sklearn if XGBoost not available
            model = kp.train_unified.train(
                sample_training_data,
                target="failure", 
                algorithm="random_forest"
            )
            return model
    
    @pytest.fixture
    def mock_gcp_credentials(self):
        """Mock GCP credentials for testing"""
        with patch('subprocess.run') as mock_run:
            # Mock gcloud --version
            mock_run.return_value = Mock(returncode=0, stdout="Google Cloud SDK 400.0.0")
            yield mock_run
    
    def test_cloud_run_deployer_initialization(self):
        """Test CloudRunDeployer can be initialized"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="test-project")
            
            deployer = CloudRunDeployer("test-project", "us-central1")
            assert deployer.project_id == "test-project"
            assert deployer.region == "us-central1"
    
    def test_deployment_config_parsing(self):
        """Test deployment configuration parsing"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="test-project")
            
            deployer = CloudRunDeployer("test-project")
            
            config_dict = {
                "project_id": "test-project",
                "region": "us-west1",
                "service_name": "test-service",
                "memory": "2Gi",
                "cpu": "2"
            }
            
            config = deployer._parse_deployment_config(config_dict)
            
            assert config.project_id == "test-project"
            assert config.region == "us-west1"
            assert config.service_name == "test-service"
            assert config.memory == "2Gi"
            assert config.cpu == "2"
    
    def test_model_framework_detection(self, trained_model):
        """Test automatic framework detection from trained models"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="test-project")
            
            deployer = CloudRunDeployer("test-project")
            framework_info = deployer._detect_model_framework(trained_model)
            
            assert "framework" in framework_info
            assert "type" in framework_info
            assert "dependencies" in framework_info
            
            # Should detect either xgboost or sklearn
            assert framework_info["framework"] in ["xgboost", "sklearn"]
            assert framework_info["type"] == "traditional_ml"
    
    def test_fastapi_app_generation(self, trained_model):
        """Test FastAPI application generation for any model type"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="test-project")
            
            deployer = CloudRunDeployer("test-project")
            framework_info = deployer._detect_model_framework(trained_model)
            config = CloudRunConfig(project_id="test-project", service_name="test-service")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Generate FastAPI app
                deployer._generate_fastapi_app(trained_model, temp_path, config)
                
                # Verify app was generated
                app_file = temp_path / "main.py"
                assert app_file.exists()
                
                # Check app content
                app_content = app_file.read_text()
                assert "FastAPI" in app_content
                assert "/healthz" in app_content
                assert "/readyz" in app_content
                assert "/predict" in app_content
                assert "PredictionInput" in app_content
                assert "PredictionOutput" in app_content
    
    def test_dockerfile_generation(self, trained_model):
        """Test Dockerfile generation for different frameworks"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="test-project")
            
            deployer = CloudRunDeployer("test-project")
            config = CloudRunConfig(project_id="test-project", service_name="test-service")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Generate Dockerfile
                deployer._generate_dockerfile(trained_model, temp_path, config)
                
                # Verify Dockerfile was generated
                dockerfile = temp_path / "Dockerfile"
                assert dockerfile.exists()
                
                # Check Dockerfile content
                dockerfile_content = dockerfile.read_text()
                assert "FROM python:" in dockerfile_content
                assert "COPY requirements.txt" in dockerfile_content
                assert "pip install" in dockerfile_content
                assert "EXPOSE" in dockerfile_content
                assert "uvicorn" in dockerfile_content
                assert "HEALTHCHECK" in dockerfile_content
    
    def test_production_requirements_generation(self, trained_model):
        """Test production requirements.txt generation"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="test-project")
            
            deployer = CloudRunDeployer("test-project")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Generate requirements
                deployer._generate_production_requirements(trained_model, temp_path)
                
                # Verify requirements were generated
                requirements_file = temp_path / "requirements.txt"
                assert requirements_file.exists()
                
                # Check requirements content
                requirements_content = requirements_file.read_text()
                assert "fastapi>=" in requirements_content
                assert "uvicorn" in requirements_content
                assert "pydantic>=" in requirements_content
                assert "pandas>=" in requirements_content
                
                # Should include framework-specific dependencies
                framework_info = deployer._detect_model_framework(trained_model)
                if framework_info["framework"] == "xgboost":
                    assert "xgboost>=" in requirements_content
                elif framework_info["framework"] == "sklearn":
                    assert "scikit-learn>=" in requirements_content
    
    def test_service_yaml_generation(self, trained_model):
        """Test Cloud Run service YAML generation"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="test-project")
            
            deployer = CloudRunDeployer("test-project")
            config = CloudRunConfig(
                project_id="test-project",
                service_name="test-service",
                memory="2Gi",
                cpu="2",
                min_instances=1,
                max_instances=50
            )
            
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Generate service YAML
                deployer._generate_service_yaml(config, temp_path)
                
                # Verify YAML was generated
                service_yaml = temp_path / "service.yaml"
                assert service_yaml.exists()
                
                # Check YAML content
                import yaml
                with open(service_yaml) as f:
                    service_config = yaml.safe_load(f)
                
                assert service_config["apiVersion"] == "serving.knative.dev/v1"
                assert service_config["kind"] == "Service"
                assert service_config["metadata"]["name"] == "test-service"
                
                # Check resource limits
                container = service_config["spec"]["template"]["spec"]["containers"][0]
                assert container["resources"]["limits"]["memory"] == "2Gi"
                assert container["resources"]["limits"]["cpu"] == "2"
    
    def test_deployment_artifacts_creation(self, trained_model):
        """Test complete deployment artifacts creation"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="test-project")
            
            deployer = CloudRunDeployer("test-project")
            config = CloudRunConfig(project_id="test-project", service_name="test-service")
            
            # Create deployment artifacts
            temp_dir = deployer._create_deployment_artifacts(trained_model, config)
            temp_path = Path(temp_dir)
            
            try:
                # Verify all artifacts were created
                assert (temp_path / "main.py").exists()          # FastAPI app
                assert (temp_path / "Dockerfile").exists()       # Dockerfile
                assert (temp_path / "requirements.txt").exists() # Requirements
                assert (temp_path / "service.yaml").exists()     # Service config
                
                # Verify artifacts are non-empty
                assert (temp_path / "main.py").stat().st_size > 1000
                assert (temp_path / "Dockerfile").stat().st_size > 500
                assert (temp_path / "requirements.txt").stat().st_size > 100
                
            finally:
                # Cleanup
                import shutil
                shutil.rmtree(temp_dir)
    
    @pytest.mark.skipif(
        not pytest.config.getoption("--run-deployment-tests", default=False),
        reason="Deployment tests require --run-deployment-tests flag and GCP access"
    )
    def test_sdk_deployment_api(self, trained_model):
        """
        Test SDK deployment API (requires real GCP access)
        
        Run with: pytest --run-deployment-tests tests/integration/test_cloud_run_deployment.py::test_sdk_deployment_api
        """
        # This test requires actual GCP credentials
        project_id = "kepler-test-project"  # Replace with actual test project
        
        try:
            # Deploy model using SDK
            result = kp.deploy.to_cloud_run(
                trained_model,
                project_id=project_id,
                service_name=f"kepler-test-{int(time.time())}",
                region="us-central1",
                memory="1Gi",
                min_instances=0,
                max_instances=1  # Limit for testing
            )
            
            assert result["success"]
            assert result["service_url"]
            assert result["service_name"]
            
            service_name = result["service_name"]
            service_url = result["service_url"]
            
            # Wait for deployment to be ready
            time.sleep(30)
            
            # Test health endpoints
            import requests
            
            health_response = requests.get(f"{service_url}/healthz", timeout=10)
            assert health_response.status_code == 200
            
            ready_response = requests.get(f"{service_url}/readyz", timeout=10)
            assert ready_response.status_code == 200
            
            # Test prediction endpoint
            test_input = {
                "data": {
                    "temperature": 25.5,
                    "pressure": 1013.2,
                    "vibration": 0.3
                }
            }
            
            predict_response = requests.post(
                f"{service_url}/predict",
                json=test_input,
                timeout=10
            )
            assert predict_response.status_code == 200
            
            prediction_result = predict_response.json()
            assert "prediction" in prediction_result
            assert "model_info" in prediction_result
            
            # Validate deployment status
            status_info = kp.deploy.get_status(service_name, project_id)
            assert status_info["ready"]
            assert status_info["url"] == service_url
            
            print(f"✅ E2E deployment test successful: {service_url}")
            
        except Exception as e:
            pytest.fail(f"E2E deployment test failed: {e}")
        
        finally:
            # Cleanup: delete test service
            try:
                import subprocess
                subprocess.run([
                    "gcloud", "run", "services", "delete", service_name,
                    "--region", "us-central1",
                    "--quiet"
                ], timeout=60)
                print(f"🧹 Cleaned up test service: {service_name}")
            except Exception as cleanup_error:
                print(f"⚠️ Cleanup failed: {cleanup_error}")
    
    def test_cli_deployment_command_structure(self):
        """Test CLI deployment command structure and validation"""
        from typer.testing import CliRunner
        from kepler.cli.main import app
        
        runner = CliRunner()
        
        # Test help text
        result = runner.invoke(app, ["deploy", "--help"])
        assert result.exit_code == 0
        assert "Deploy trained model to cloud platform" in result.stdout
        assert "--cloud" in result.stdout
        assert "--project" in result.stdout
        assert "--service" in result.stdout
        
        # Test missing model file
        result = runner.invoke(app, ["deploy", "nonexistent.pkl"])
        assert result.exit_code == 1
        assert "Model file not found" in result.stdout
    
    def test_cli_status_command_structure(self):
        """Test CLI status command structure"""
        from typer.testing import CliRunner
        from kepler.cli.main import app
        
        runner = CliRunner()
        
        # Test help text
        result = runner.invoke(app, ["status", "--help"])
        assert result.exit_code == 0
        assert "Get status of deployed model service" in result.stdout
        assert "--project" in result.stdout
        assert "--region" in result.stdout
    
    def test_deployment_error_handling(self, trained_model):
        """Test error handling in deployment scenarios"""
        # Test with invalid project ID
        with pytest.raises(DeploymentError) as exc_info:
            deployer = CloudRunDeployer("invalid-project-id-12345")
            
        assert "DEPLOY_001" in str(exc_info.value)  # Should use standardized error code
    
    def test_fastapi_app_validation(self, trained_model):
        """Test generated FastAPI app structure and endpoints"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="test-project")
            
            deployer = CloudRunDeployer("test-project")
            framework_info = deployer._detect_model_framework(trained_model)
            config = CloudRunConfig(project_id="test-project", service_name="test-service")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Generate FastAPI app
                deployer._generate_fastapi_app(trained_model, temp_path, config)
                
                app_content = (temp_path / "main.py").read_text()
                
                # Validate required endpoints
                assert '@app.get("/healthz"' in app_content
                assert '@app.get("/readyz"' in app_content
                assert '@app.post("/predict"' in app_content
                assert '@app.get("/"' in app_content
                
                # Validate Pydantic models
                assert "class PredictionInput" in app_content
                assert "class PredictionOutput" in app_content
                assert "class HealthStatus" in app_content
                
                # Validate Splunk integration
                assert "write_prediction_to_splunk" in app_content
                assert "SPLUNK_HEC_URL" in app_content
                
                # Validate error handling
                assert "HTTPException" in app_content
                assert "status.HTTP_503_SERVICE_UNAVAILABLE" in app_content
    
    def test_dockerfile_optimization_by_framework(self):
        """Test Dockerfile generation optimization for different frameworks"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="test-project")
            
            deployer = CloudRunDeployer("test-project")
            
            # Test sklearn model Dockerfile
            sklearn_model = Mock()
            sklearn_model.algorithm = "random_forest"
            sklearn_model.model_type = "classification"
            
            framework_info = deployer._detect_model_framework(sklearn_model)
            assert framework_info["framework"] == "sklearn"
            
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                config = CloudRunConfig(project_id="test-project", service_name="sklearn-test")
                
                deployer._generate_dockerfile(sklearn_model, temp_path, config)
                
                dockerfile_content = (temp_path / "Dockerfile").read_text()
                assert "python:3.11-slim" in dockerfile_content
                assert "# No additional system dependencies needed" in dockerfile_content
    
    def test_health_check_validation_logic(self):
        """Test health check validation logic"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="test-project")
            
            deployer = CloudRunDeployer("test-project")
            
            # Mock successful health checks
            with patch('requests.get') as mock_get:
                # Mock /healthz response
                mock_healthz = Mock()
                mock_healthz.status_code = 200
                mock_healthz.json.return_value = {"status": "healthy", "model_loaded": True}
                
                # Mock /readyz response
                mock_readyz = Mock()
                mock_readyz.status_code = 200
                mock_readyz.json.return_value = {"status": "ready", "model_loaded": True}
                
                mock_get.side_effect = [mock_healthz, mock_readyz]
                
                # Test validation
                health_results = deployer._validate_deployment_health("https://test-service.run.app")
                
                assert health_results["overall_status"] == "healthy"
                assert health_results["healthz"]["status"] == "healthy"
                assert health_results["readyz"]["status"] == "ready"
    
    def test_complete_deployment_artifacts_pipeline(self, trained_model):
        """Test complete pipeline: artifacts → deployment structure → validation"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="test-project")
            
            deployer = CloudRunDeployer("test-project")
            config = CloudRunConfig(
                project_id="test-project",
                service_name="pipeline-test",
                memory="2Gi",
                cpu="2",
                environment_variables={
                    "SPLUNK_HEC_URL": "https://splunk.example.com/services/collector",
                    "SPLUNK_INDEX": "test_predictions"
                }
            )
            
            # Create complete deployment artifacts
            temp_dir = deployer._create_deployment_artifacts(trained_model, config)
            temp_path = Path(temp_dir)
            
            try:
                # Validate complete deployment structure
                required_files = ["main.py", "Dockerfile", "requirements.txt", "service.yaml"]
                for file_name in required_files:
                    file_path = temp_path / file_name
                    assert file_path.exists(), f"Missing required file: {file_name}"
                    assert file_path.stat().st_size > 0, f"Empty file: {file_name}"
                
                # Validate FastAPI app includes Splunk configuration
                app_content = (temp_path / "main.py").read_text()
                assert "SPLUNK_HEC_URL" in app_content
                assert "write_prediction_to_splunk" in app_content
                
                # Validate service YAML includes environment variables
                import yaml
                with open(temp_path / "service.yaml") as f:
                    service_config = yaml.safe_load(f)
                
                env_vars = service_config["spec"]["template"]["spec"]["containers"][0]["env"]
                env_names = [var["name"] for var in env_vars]
                assert "SPLUNK_HEC_URL" in env_names
                assert "SPLUNK_INDEX" in env_names
                
                print("✅ Complete deployment pipeline validation successful")
                
            finally:
                # Cleanup
                import shutil
                shutil.rmtree(temp_dir)
    
    def test_sdk_api_integration(self, trained_model):
        """Test SDK API integration for deployment"""
        # Test that SDK APIs are properly exposed
        assert hasattr(kp, 'deploy')
        assert hasattr(kp.deploy, 'to_cloud_run')
        assert hasattr(kp.deploy, 'validate')
        assert hasattr(kp.deploy, 'get_status')
        
        # Test API signatures
        import inspect
        
        # Check to_cloud_run signature
        sig = inspect.signature(kp.deploy.to_cloud_run)
        assert 'model' in sig.parameters
        assert 'project_id' in sig.parameters
        assert 'service_name' in sig.parameters
        
        # Check validate signature
        sig = inspect.signature(kp.deploy.validate)
        assert 'service_name' in sig.parameters
        assert 'project_id' in sig.parameters
    
    def test_framework_compatibility_matrix(self):
        """Test framework detection for different model types"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="test-project")
            
            deployer = CloudRunDeployer("test-project")
            
            # Test different algorithm types
            test_cases = [
                ("random_forest", "sklearn", "traditional_ml"),
                ("xgboost", "xgboost", "traditional_ml"),
                ("pytorch", "pytorch", "deep_learning"),
                ("transformers", "transformers", "generative_ai")
            ]
            
            for algorithm, expected_framework, expected_type in test_cases:
                mock_model = Mock()
                mock_model.algorithm = algorithm
                mock_model.model_type = "classification"
                
                framework_info = deployer._detect_model_framework(mock_model)
                
                assert framework_info["framework"] == expected_framework
                assert framework_info["type"] == expected_type
                assert isinstance(framework_info["dependencies"], list)
                assert len(framework_info["dependencies"]) > 0


class TestDeploymentIntegrationWithVersioning:
    """Test deployment integration with versioning system"""
    
    def test_deployment_with_versioned_model(self, sample_training_data):
        """Test deploying a versioned model"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="test-project")
            
            # Train and version model
            model = kp.train_unified.train(
                sample_training_data,
                target="failure", 
                algorithm="random_forest"
            )
            
            # Create versioned release
            version = kp.versioning.create_unified_version(
                "deployment-test-v1.0",
                experiment_name="deployment-integration-test"
            )
            
            # Test deployment with version metadata
            deployment_config = {
                "project_id": "test-project",
                "service_name": "versioned-model-test",
                "environment_variables": {
                    "MODEL_VERSION": version.version_id,
                    "GIT_COMMIT": version.git_commit
                }
            }
            
            deployer = CloudRunDeployer("test-project")
            
            # This would test the deployment in a real scenario
            # For now, we test the configuration parsing
            config = deployer._parse_deployment_config(deployment_config)
            assert config.environment_variables["MODEL_VERSION"] == version.version_id
            assert config.environment_variables["GIT_COMMIT"] == version.git_commit
    
    @pytest.fixture
    def sample_training_data(self):
        """Create sample data for model training"""
        np.random.seed(42)
        return pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100) * 10,
            'target': np.random.randint(0, 2, 100)
        })


# Test configuration hook for pytest
def pytest_addoption(parser):
    """Add custom pytest options"""
    parser.addoption(
        "--run-deployment-tests",
        action="store_true",
        default=False,
        help="Run tests that require actual GCP deployment (requires credentials)"
    )


# Test markers for different test types
pytestmark = [
    pytest.mark.integration,
    pytest.mark.deployment
]
