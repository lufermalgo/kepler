"""
Kepler Cloud Run Deployer - Task 6.2 Implementation
Automatic deployment to Google Cloud Run for ANY AI model type

Implements ModelDeployer interface for Google Cloud Run deployment with:
- Automatic Dockerfile generation for any AI framework
- FastAPI wrapper generation for any model type  
- Health checks (healthz/readyz) implementation
- Environment management and scaling configuration
- Integration with Kepler's unlimited library system

Philosophy: "From any model to production in one command"
"""

import os
import subprocess
import tempfile
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime

from kepler.core.interfaces import ModelDeployer
from kepler.utils.logging import get_logger
from kepler.utils.exceptions import DeploymentError, KeplerError
from kepler.core.library_manager import LibraryManager


@dataclass
class CloudRunConfig:
    """Configuration for Cloud Run deployment"""
    project_id: str
    region: str = "us-central1"
    service_name: Optional[str] = None
    memory: str = "1Gi"
    cpu: str = "1"
    min_instances: int = 0
    max_instances: int = 100
    port: int = 8080
    timeout: int = 300
    allow_unauthenticated: bool = True
    environment_variables: Dict[str, str] = None
    
    def __post_init__(self):
        if self.environment_variables is None:
            self.environment_variables = {}


@dataclass
class DeploymentResult:
    """Result of Cloud Run deployment"""
    success: bool
    service_name: str
    service_url: Optional[str] = None
    revision_name: Optional[str] = None
    deployment_time: Optional[float] = None
    error_message: Optional[str] = None
    logs: List[str] = None
    health_check_results: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.logs is None:
            self.logs = []
        if self.health_check_results is None:
            self.health_check_results = {}


class CloudRunDeployer(ModelDeployer):
    """
    Google Cloud Run deployer for ANY AI model type
    
    Supports deployment of models trained with:
    - Traditional ML: sklearn, XGBoost, LightGBM, CatBoost
    - Deep Learning: PyTorch, TensorFlow, Keras, JAX
    - Generative AI: transformers, langchain, OpenAI
    - Computer Vision: OpenCV, PIL
    - Custom frameworks: Any Python library
    
    Features:
    - Automatic Dockerfile generation based on model type
    - FastAPI wrapper generation with health checks
    - Environment optimization for production
    - Automatic scaling configuration
    - Integration with Kepler versioning system
    """
    
    def __init__(self, project_id: str, region: str = "us-central1"):
        """
        Initialize Cloud Run deployer
        
        Args:
            project_id: Google Cloud project ID
            region: GCP region for deployment (default: us-central1)
        """
        self.project_id = project_id
        self.region = region
        self.logger = get_logger(__name__)
        self.lib_manager = LibraryManager(".")
        
        # Validate GCP credentials and setup
        self._validate_gcp_setup()
        
        self.logger.info(f"CloudRunDeployer initialized for project: {project_id}, region: {region}")
    
    def _validate_gcp_setup(self) -> None:
        """Validate Google Cloud SDK and authentication"""
        try:
            # Check if gcloud CLI is available
            result = subprocess.run(
                ["gcloud", "--version"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            
            if result.returncode != 0:
                raise DeploymentError(
                    "Google Cloud SDK not found",
                    hint="Install gcloud CLI: https://cloud.google.com/sdk/docs/install"
                )
            
            # Check if authenticated
            result = subprocess.run(
                ["gcloud", "auth", "list", "--filter=status:ACTIVE", "--format=value(account)"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if not result.stdout.strip():
                raise DeploymentError(
                    "No active Google Cloud authentication",
                    hint="Run: gcloud auth login"
                )
            
            # Validate project access
            result = subprocess.run(
                ["gcloud", "config", "get-value", "project"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            current_project = result.stdout.strip()
            if current_project != self.project_id:
                self.logger.warning(f"Current gcloud project: {current_project}, expected: {self.project_id}")
                
                # Set project
                subprocess.run(
                    ["gcloud", "config", "set", "project", self.project_id],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
            self.logger.info("✅ GCP setup validated")
            
        except subprocess.TimeoutExpired:
            raise DeploymentError(
                "GCP validation timeout",
                hint="Check network connectivity and gcloud CLI installation"
            )
        except Exception as e:
            raise DeploymentError(
                f"GCP setup validation failed: {str(e)}",
                hint="Ensure gcloud CLI is installed and authenticated"
            )
    
    def deploy(self, model: Any, deployment_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deploy model to Google Cloud Run (implements ModelDeployer interface)
        
        Args:
            model: Trained Kepler model to deploy
            deployment_config: Cloud Run deployment configuration
            
        Returns:
            Dict with deployment results (service_url, status, etc.)
            
        Raises:
            DeploymentError: If deployment fails
        """
        self.logger.info("Starting Cloud Run deployment")
        start_time = time.time()
        
        try:
            # Parse deployment configuration
            config = self._parse_deployment_config(deployment_config)
            
            # Generate service name if not provided
            if not config.service_name:
                config.service_name = self._generate_service_name(model)
            
            # Create deployment artifacts
            temp_dir = self._create_deployment_artifacts(model, config)
            
            # Build and deploy to Cloud Run
            deployment_result = self._deploy_to_cloud_run(temp_dir, config)
            
            # Validate deployment health
            if deployment_result.success and deployment_result.service_url:
                health_results = self._validate_deployment_health(deployment_result.service_url)
                deployment_result.health_check_results = health_results
            
            deployment_result.deployment_time = time.time() - start_time
            
            self.logger.info(f"Deployment completed in {deployment_result.deployment_time:.2f}s")
            
            return {
                "success": deployment_result.success,
                "service_name": deployment_result.service_name,
                "service_url": deployment_result.service_url,
                "revision_name": deployment_result.revision_name,
                "deployment_time": deployment_result.deployment_time,
                "health_checks": deployment_result.health_check_results,
                "logs": deployment_result.logs
            }
            
        except Exception as e:
            raise DeploymentError(
                f"Cloud Run deployment failed: {str(e)}",
                service_name=config.service_name if 'config' in locals() else None,
                hint="Check GCP credentials, project permissions, and model compatibility"
            )
    
    def _parse_deployment_config(self, config_dict: Dict[str, Any]) -> CloudRunConfig:
        """Parse deployment configuration dictionary into CloudRunConfig"""
        return CloudRunConfig(
            project_id=config_dict.get("project_id", self.project_id),
            region=config_dict.get("region", self.region),
            service_name=config_dict.get("service_name"),
            memory=config_dict.get("memory", "1Gi"),
            cpu=config_dict.get("cpu", "1"),
            min_instances=config_dict.get("min_instances", 0),
            max_instances=config_dict.get("max_instances", 100),
            port=config_dict.get("port", 8080),
            timeout=config_dict.get("timeout", 300),
            allow_unauthenticated=config_dict.get("allow_unauthenticated", True),
            environment_variables=config_dict.get("environment_variables", {})
        )
    
    def _generate_service_name(self, model: Any) -> str:
        """Generate unique service name based on model"""
        # Extract model information
        model_type = getattr(model, 'model_type', 'unknown')
        algorithm = getattr(model, 'algorithm', 'model')
        
        # Generate timestamp-based unique name
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        service_name = f"kepler-{algorithm}-{model_type}-{timestamp}"
        
        # Ensure Cloud Run naming requirements (lowercase, alphanumeric + hyphens)
        service_name = service_name.lower().replace('_', '-')
        
        # Limit length (Cloud Run max 63 characters)
        if len(service_name) > 63:
            service_name = service_name[:60] + "001"
            
        self.logger.info(f"Generated service name: {service_name}")
        return service_name
    
    def _create_deployment_artifacts(self, model: Any, config: CloudRunConfig) -> str:
        """
        Create all deployment artifacts (Dockerfile, FastAPI app, requirements)
        
        Returns:
            Path to temporary directory with deployment artifacts
        """
        temp_dir = tempfile.mkdtemp(prefix="kepler_deploy_")
        temp_path = Path(temp_dir)
        
        self.logger.info(f"Creating deployment artifacts in: {temp_dir}")
        
        # 1. Generate FastAPI application
        self._generate_fastapi_app(model, temp_path, config)
        
        # 2. Generate Dockerfile
        self._generate_dockerfile(model, temp_path, config)
        
        # 3. Generate requirements.txt optimized for production
        self._generate_production_requirements(model, temp_path)
        
        # 4. Generate Cloud Run service configuration
        self._generate_service_yaml(config, temp_path)
        
        self.logger.info("✅ Deployment artifacts created successfully")
        return temp_dir
    
    def _generate_fastapi_app(self, model: Any, temp_path: Path, config: CloudRunConfig) -> None:
        """Generate FastAPI application wrapper for any model type"""
        
        # Detect model framework and type
        framework_info = self._detect_model_framework(model)
        
        # Generate FastAPI app based on model type
        app_code = self._create_fastapi_template(model, framework_info, config)
        
        # Write FastAPI application
        app_file = temp_path / "main.py"
        with open(app_file, 'w') as f:
            f.write(app_code)
        
        self.logger.info(f"✅ FastAPI app generated: {app_file}")
    
    def _detect_model_framework(self, model: Any) -> Dict[str, Any]:
        """Detect what framework was used to train the model"""
        
        framework_info = {
            "framework": "unknown",
            "type": "unknown", 
            "requires_special_handling": False,
            "dependencies": []
        }
        
        # Check model attributes and class
        model_class = str(type(model))
        
        if hasattr(model, 'algorithm'):
            algorithm = getattr(model, 'algorithm', 'unknown')
            framework_info["algorithm"] = algorithm
            
            # Map algorithm to framework
            if algorithm in ['random_forest', 'logistic_regression', 'linear_regression']:
                framework_info.update({
                    "framework": "sklearn",
                    "type": "traditional_ml",
                    "dependencies": ["scikit-learn", "joblib", "pandas", "numpy"]
                })
            elif algorithm in ['xgboost', 'gradient_boosting']:
                framework_info.update({
                    "framework": "xgboost", 
                    "type": "traditional_ml",
                    "dependencies": ["xgboost", "pandas", "numpy"]
                })
            elif algorithm in ['pytorch', 'neural_network']:
                framework_info.update({
                    "framework": "pytorch",
                    "type": "deep_learning",
                    "dependencies": ["torch", "pandas", "numpy"],
                    "requires_special_handling": True
                })
            elif algorithm in ['transformers', 'bert', 'gpt']:
                framework_info.update({
                    "framework": "transformers",
                    "type": "generative_ai", 
                    "dependencies": ["transformers", "torch", "pandas", "numpy"],
                    "requires_special_handling": True
                })
        
        # Check for specific model types
        if 'sklearn' in model_class:
            framework_info.update({
                "framework": "sklearn",
                "type": "traditional_ml",
                "dependencies": ["scikit-learn", "joblib"]
            })
        elif 'xgboost' in model_class:
            framework_info.update({
                "framework": "xgboost",
                "type": "traditional_ml", 
                "dependencies": ["xgboost"]
            })
        elif 'torch' in model_class or 'pytorch' in model_class.lower():
            framework_info.update({
                "framework": "pytorch",
                "type": "deep_learning",
                "dependencies": ["torch"],
                "requires_special_handling": True
            })
        
        self.logger.info(f"Detected model framework: {framework_info}")
        return framework_info
    
    def _create_fastapi_template(self, model: Any, framework_info: Dict[str, Any], config: CloudRunConfig) -> str:
        """Create FastAPI application template for any model type"""
        
        # Extract model information
        model_name = getattr(model, 'algorithm', 'kepler-model')
        model_type = getattr(model, 'model_type', 'prediction')
        target_column = getattr(model, 'target_column', 'target')
        feature_columns = getattr(model, 'feature_columns', [])
        
        # Generate imports based on framework
        imports = self._generate_framework_imports(framework_info)
        
        # Generate prediction logic based on model type
        prediction_logic = self._generate_prediction_logic(model, framework_info)
        
        # Create FastAPI template
        app_template = f'''"""
Kepler Auto-Generated FastAPI Application
Model: {model_name}
Framework: {framework_info.get("framework", "unknown")}
Generated: {datetime.now().isoformat()}
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import joblib
{imports}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Kepler Model API - {model_name}",
    description="Auto-generated API for {framework_info.get('framework', 'unknown')} model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Splunk HEC configuration (optional)
SPLUNK_HEC_URL = os.environ.get("SPLUNK_HEC_URL")
SPLUNK_HEC_TOKEN = os.environ.get("SPLUNK_HEC_TOKEN")
SPLUNK_INDEX = os.environ.get("SPLUNK_INDEX", "ml_predictions")

def write_prediction_to_splunk(prediction_data: Dict[str, Any]) -> bool:
    """Write prediction results to Splunk via HEC"""
    if not SPLUNK_HEC_URL or not SPLUNK_HEC_TOKEN:
        logger.warning("Splunk HEC not configured, skipping result write")
        return False
    
    try:
        import requests
        
        # Prepare HEC event
        event = {{
            "time": time.time(),
            "source": "kepler-model-api",
            "sourcetype": "kepler:prediction",
            "index": SPLUNK_INDEX,
            "event": {{
                "model_name": "{model_name}",
                "framework": "{framework_info.get('framework', 'unknown')}",
                "prediction": prediction_data.get("prediction"),
                "confidence": prediction_data.get("confidence"),
                "input_features": prediction_data.get("input_summary"),
                "timestamp": datetime.now().isoformat(),
                "service_name": os.environ.get("K_SERVICE", "unknown"),
                "revision": os.environ.get("K_REVISION", "unknown")
            }}
        }}
        
        # Send to Splunk HEC
        response = requests.post(
            SPLUNK_HEC_URL,
            headers={{
                "Authorization": f"Splunk {{SPLUNK_HEC_TOKEN}}",
                "Content-Type": "application/json"
            }},
            json=event,
            timeout=5
        )
        
        if response.status_code == 200:
            logger.info("✅ Prediction written to Splunk")
            return True
        else:
            logger.warning(f"Splunk write failed: {{response.status_code}} - {{response.text}}")
            return False
            
    except Exception as e:
        logger.warning(f"Failed to write to Splunk: {{e}}")
        return False

# Load model on startup
model_instance = None

@app.on_event("startup")
async def load_model():
    """Load model on application startup"""
    global model_instance
    try:
        # Load model from file (will be copied during deployment)
        model_instance = joblib.load("model.pkl")
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {{e}}")
        raise

# Input/Output models
class PredictionInput(BaseModel):
    """Input data for model prediction"""
    data: Dict[str, Any] = Field(..., description="Input features for prediction")
    
    class Config:
        schema_extra = {{
            "example": {{
                "data": {{{self._generate_example_input(feature_columns)}}}
            }}
        }}

class PredictionOutput(BaseModel):
    """Output from model prediction"""
    prediction: Any = Field(..., description="Model prediction result")
    confidence: Optional[float] = Field(None, description="Prediction confidence score")
    model_info: Dict[str, Any] = Field(..., description="Model metadata")
    
    class Config:
        schema_extra = {{
            "example": {{
                "prediction": 0.85 if "{model_type}" == "regression" else 1,
                "confidence": 0.92,
                "model_info": {{
                    "algorithm": "{model_name}",
                    "framework": "{framework_info.get('framework', 'unknown')}",
                    "version": "1.0.0"
                }}
            }}
        }}

class HealthStatus(BaseModel):
    """Health check status"""
    status: str = Field(..., description="Service health status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    framework: str = Field(..., description="AI framework used")
    version: str = Field(..., description="API version")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")

# Global startup time for uptime calculation
startup_time = None

@app.on_event("startup")
async def record_startup_time():
    """Record startup time for uptime calculation"""
    global startup_time
    startup_time = time.time()

# Health check endpoints (required for Cloud Run)
@app.get("/healthz", response_model=HealthStatus, tags=["Health"])
async def health_check():
    """
    Liveness probe endpoint
    
    Returns service health status for Cloud Run health monitoring.
    """
    global startup_time
    
    uptime = time.time() - startup_time if startup_time else 0
    
    return HealthStatus(
        status="healthy",
        model_loaded=model_instance is not None,
        framework="{framework_info.get('framework', 'unknown')}",
        version="1.0.0",
        uptime_seconds=uptime
    )

@app.get("/readyz", response_model=HealthStatus, tags=["Health"])  
async def readiness_check():
    """
    Readiness probe endpoint
    
    Checks if service is ready to handle requests.
    """
    global startup_time
    
    if model_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    uptime = time.time() - startup_time if startup_time else 0
    
    return HealthStatus(
        status="ready",
        model_loaded=True,
        framework="{framework_info.get('framework', 'unknown')}",
        version="1.0.0", 
        uptime_seconds=uptime
    )

# Prediction endpoint
@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict(input_data: PredictionInput):
    """
    Make prediction using the deployed model
    
    Accepts input data and returns model predictions with metadata.
    """
    if model_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not available"
        )
    
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([input_data.data])
        
        {prediction_logic}
        
        # Prepare result
        result = PredictionOutput(
            prediction=prediction_result,
            confidence=confidence_score,
            model_info={{
                "algorithm": "{model_name}",
                "framework": "{framework_info.get('framework', 'unknown')}",
                "model_type": "{model_type}",
                "target_column": "{target_column}",
                "version": "1.0.0"
            }}
        )
        
        # Write prediction to Splunk (async, non-blocking)
        try:
            prediction_data = {{
                "prediction": prediction_result,
                "confidence": confidence_score,
                "input_summary": {{k: str(v)[:100] for k, v in input_data.data.items()}}  # Truncate for logging
            }}
            write_prediction_to_splunk(prediction_data)
        except Exception as e:
            logger.warning(f"Splunk write failed (non-blocking): {{e}}")
        
        return result
        
    except Exception as e:
        logger.error(f"Prediction failed: {{e}}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {{str(e)}}"
        )

# Root endpoint
@app.get("/", tags=["Info"])
async def root():
    """
    Root endpoint with API information
    """
    return {{
        "message": "Kepler Model API - {model_name}",
        "framework": "{framework_info.get('framework', 'unknown')}",
        "model_type": "{model_type}",
        "docs": "/docs",
        "health": "/healthz",
        "readiness": "/readyz",
        "predict": "/predict"
    }}

# Run with Uvicorn (for local testing)
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", {config.port}))
    uvicorn.run(app, host="0.0.0.0", port=port)
'''
        
        return app_template
    
    def _generate_framework_imports(self, framework_info: Dict[str, Any]) -> str:
        """Generate framework-specific imports"""
        framework = framework_info.get("framework", "unknown")
        
        if framework == "sklearn":
            return "from sklearn.base import BaseEstimator"
        elif framework == "xgboost":
            return "import xgboost as xgb"
        elif framework == "pytorch":
            return "import torch\nimport torch.nn as nn"
        elif framework == "transformers":
            return "from transformers import pipeline, AutoTokenizer, AutoModel"
        else:
            return "# Framework-specific imports will be added here"
    
    def _generate_prediction_logic(self, model: Any, framework_info: Dict[str, Any]) -> str:
        """Generate prediction logic based on framework"""
        framework = framework_info.get("framework", "unknown")
        
        if framework in ["sklearn", "xgboost"]:
            return '''        # Traditional ML prediction
        prediction_result = model_instance.predict(df)[0]
        
        # Get confidence score if available
        confidence_score = None
        if hasattr(model_instance, 'predict_proba'):
            try:
                proba = model_instance.predict_proba(df)[0]
                confidence_score = float(max(proba))
            except:
                confidence_score = None'''
                
        elif framework == "pytorch":
            return '''        # PyTorch prediction
        model_instance.eval()
        with torch.no_grad():
            # Convert to tensor
            input_tensor = torch.FloatTensor(df.values)
            output = model_instance(input_tensor)
            
            # Handle different output types
            if hasattr(output, 'softmax'):
                prediction_result = output.softmax(dim=1).argmax().item()
                confidence_score = float(output.softmax(dim=1).max())
            else:
                prediction_result = float(output.squeeze())
                confidence_score = None'''
                
        elif framework == "transformers":
            return '''        # Transformers prediction
        # Assuming text input in 'text' field
        text_input = input_data.data.get('text', '')
        
        if hasattr(model_instance, 'tokenizer'):
            # Use model's tokenizer
            inputs = model_instance.tokenizer(text_input, return_tensors="pt", truncation=True)
            outputs = model_instance.model(**inputs)
            prediction_result = outputs.logits.argmax().item()
            confidence_score = float(torch.softmax(outputs.logits, dim=1).max())
        else:
            # Fallback to simple prediction
            prediction_result = str(model_instance.predict([text_input])[0])
            confidence_score = None'''
        else:
            return '''        # Generic prediction (fallback)
        try:
            prediction_result = model_instance.predict(df)[0]
            confidence_score = None
        except Exception as e:
            logger.warning(f"Using fallback prediction method: {e}")
            prediction_result = "prediction_not_available"
            confidence_score = None'''
    
    def _generate_example_input(self, feature_columns: List[str]) -> str:
        """Generate example input for API documentation"""
        if not feature_columns:
            return '"feature1": 1.0, "feature2": 2.0, "feature3": "value"'
        
        examples = []
        for i, col in enumerate(feature_columns[:5]):  # Limit to 5 features for readability
            if 'temperature' in col.lower():
                examples.append(f'"{col}": 25.5')
            elif 'pressure' in col.lower():
                examples.append(f'"{col}": 1013.2')
            elif 'id' in col.lower():
                examples.append(f'"{col}": "SENSOR_001"')
            else:
                examples.append(f'"{col}": {1.0 + i}')
        
        return ", ".join(examples)
    
    def _generate_dockerfile(self, model: Any, temp_path: Path, config: CloudRunConfig) -> None:
        """Generate optimized Dockerfile for the model framework"""
        
        framework_info = self._detect_model_framework(model)
        framework = framework_info.get("framework", "unknown")
        
        # Select base image based on framework
        if framework == "pytorch":
            base_image = "python:3.11-slim"  # CPU-only for Cloud Run
            system_deps = "RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && rm -rf /var/lib/apt/lists/*"
        elif framework == "transformers":
            base_image = "python:3.11-slim"
            system_deps = "RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ git && rm -rf /var/lib/apt/lists/*"
        else:
            base_image = "python:3.11-slim" 
            system_deps = "# No additional system dependencies needed"
        
        dockerfile_content = f'''# Kepler Auto-Generated Dockerfile
# Framework: {framework}
# Generated: {datetime.now().isoformat()}

FROM {base_image}

# Set working directory
WORKDIR /app

# Install system dependencies if needed
{system_deps}

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \\
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash kepler && \\
    chown -R kepler:kepler /app
USER kepler

# Expose port
EXPOSE {config.port}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:{config.port}/healthz || exit 1

# Run FastAPI with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "{config.port}"]
'''
        
        dockerfile_path = temp_path / "Dockerfile"
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
        
        self.logger.info(f"✅ Dockerfile generated: {dockerfile_path}")
    
    def _generate_production_requirements(self, model: Any, temp_path: Path) -> None:
        """Generate optimized requirements.txt for production"""
        
        framework_info = self._detect_model_framework(model)
        dependencies = framework_info.get("dependencies", [])
        
        # Base FastAPI requirements
        base_requirements = [
            "fastapi>=0.100.0",
            "uvicorn[standard]>=0.20.0",
            "pydantic>=2.0.0",
            "pandas>=2.0.0",
            "numpy>=1.24.0",
            "joblib>=1.3.0"
        ]
        
        # Add framework-specific requirements
        framework_requirements = []
        for dep in dependencies:
            if dep == "scikit-learn":
                framework_requirements.append("scikit-learn>=1.3.0")
            elif dep == "xgboost":
                framework_requirements.append("xgboost>=1.7.0")
            elif dep == "torch":
                # CPU-only PyTorch for Cloud Run
                framework_requirements.append("torch>=2.0.0+cpu --index-url https://download.pytorch.org/whl/cpu")
            elif dep == "transformers":
                framework_requirements.extend([
                    "transformers>=4.30.0",
                    "torch>=2.0.0+cpu --index-url https://download.pytorch.org/whl/cpu"
                ])
        
        # Combine all requirements
        all_requirements = base_requirements + framework_requirements
        
        requirements_path = temp_path / "requirements.txt"
        with open(requirements_path, 'w') as f:
            f.write("\n".join(all_requirements))
        
        self.logger.info(f"✅ Production requirements generated: {requirements_path}")
    
    def _generate_service_yaml(self, config: CloudRunConfig, temp_path: Path) -> None:
        """Generate Cloud Run service YAML configuration"""
        
        service_config = {
            "apiVersion": "serving.knative.dev/v1",
            "kind": "Service",
            "metadata": {
                "name": config.service_name,
                "annotations": {
                    "run.googleapis.com/ingress": "all"
                }
            },
            "spec": {
                "template": {
                    "metadata": {
                        "annotations": {
                            "autoscaling.knative.dev/minScale": str(config.min_instances),
                            "autoscaling.knative.dev/maxScale": str(config.max_instances),
                            "run.googleapis.com/execution-environment": "gen2"
                        }
                    },
                    "spec": {
                        "containerConcurrency": 80,
                        "timeoutSeconds": config.timeout,
                        "containers": [{
                            "image": f"gcr.io/{config.project_id}/{config.service_name}:latest",
                            "ports": [{
                                "containerPort": config.port
                            }],
                            "resources": {
                                "limits": {
                                    "cpu": config.cpu,
                                    "memory": config.memory
                                }
                            },
                            "env": [
                                {
                                    "name": "PORT",
                                    "value": str(config.port)
                                }
                            ] + [
                                {
                                    "name": key,
                                    "value": value
                                }
                                for key, value in config.environment_variables.items()
                            ]
                        }]
                    }
                }
            }
        }
        
        service_yaml_path = temp_path / "service.yaml"
        import yaml
        with open(service_yaml_path, 'w') as f:
            yaml.dump(service_config, f, default_flow_style=False)
        
        self.logger.info(f"✅ Service YAML generated: {service_yaml_path}")
    
    def _deploy_to_cloud_run(self, temp_dir: str, config: CloudRunConfig) -> DeploymentResult:
        """Execute deployment to Cloud Run"""
        
        self.logger.info("Building and deploying to Cloud Run...")
        
        try:
            # Change to deployment directory
            original_dir = os.getcwd()
            os.chdir(temp_dir)
            
            # Build and deploy using gcloud
            deploy_cmd = [
                "gcloud", "run", "deploy", config.service_name,
                "--source", ".",
                "--platform", "managed",
                "--region", config.region,
                "--memory", config.memory,
                "--cpu", config.cpu,
                "--min-instances", str(config.min_instances),
                "--max-instances", str(config.max_instances),
                "--port", str(config.port),
                "--timeout", str(config.timeout),
                "--format", "json"
            ]
            
            if config.allow_unauthenticated:
                deploy_cmd.append("--allow-unauthenticated")
            
            # Add environment variables
            for key, value in config.environment_variables.items():
                deploy_cmd.extend(["--set-env-vars", f"{key}={value}"])
            
            self.logger.info(f"Executing: {' '.join(deploy_cmd)}")
            
            # Execute deployment
            result = subprocess.run(
                deploy_cmd,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes timeout
            )
            
            os.chdir(original_dir)  # Return to original directory
            
            if result.returncode != 0:
                return DeploymentResult(
                    success=False,
                    service_name=config.service_name,
                    error_message=f"gcloud deploy failed: {result.stderr}",
                    logs=[result.stdout, result.stderr]
                )
            
            # Parse deployment result
            try:
                deploy_info = json.loads(result.stdout)
                service_url = deploy_info.get("status", {}).get("url")
                revision_name = deploy_info.get("status", {}).get("latestRevisionName")
                
                return DeploymentResult(
                    success=True,
                    service_name=config.service_name,
                    service_url=service_url,
                    revision_name=revision_name,
                    logs=[result.stdout]
                )
                
            except json.JSONDecodeError:
                # Fallback: extract URL from text output
                lines = result.stdout.split('\n')
                service_url = None
                for line in lines:
                    if 'https://' in line and 'run.app' in line:
                        service_url = line.strip()
                        break
                
                return DeploymentResult(
                    success=True,
                    service_name=config.service_name,
                    service_url=service_url,
                    logs=[result.stdout]
                )
                
        except subprocess.TimeoutExpired:
            os.chdir(original_dir)
            return DeploymentResult(
                success=False,
                service_name=config.service_name,
                error_message="Deployment timeout (30 minutes exceeded)",
                logs=["Deployment timed out"]
            )
        except Exception as e:
            os.chdir(original_dir)
            return DeploymentResult(
                success=False,
                service_name=config.service_name,
                error_message=f"Deployment error: {str(e)}",
                logs=[str(e)]
            )
    
    def _validate_deployment_health(self, service_url: str) -> Dict[str, Any]:
        """Validate deployment health using health check endpoints"""
        
        health_results = {
            "healthz": {"status": "unknown", "response_time_ms": None},
            "readyz": {"status": "unknown", "response_time_ms": None},
            "overall_status": "unknown"
        }
        
        import requests
        
        # Test /healthz endpoint
        try:
            start_time = time.time()
            response = requests.get(f"{service_url}/healthz", timeout=10)
            response_time = (time.time() - start_time) * 1000
            
            health_results["healthz"] = {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "status_code": response.status_code,
                "response_time_ms": response_time,
                "response_body": response.json() if response.status_code == 200 else response.text
            }
        except Exception as e:
            health_results["healthz"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Test /readyz endpoint
        try:
            start_time = time.time()
            response = requests.get(f"{service_url}/readyz", timeout=10)
            response_time = (time.time() - start_time) * 1000
            
            health_results["readyz"] = {
                "status": "ready" if response.status_code == 200 else "not_ready",
                "status_code": response.status_code,
                "response_time_ms": response_time,
                "response_body": response.json() if response.status_code == 200 else response.text
            }
        except Exception as e:
            health_results["readyz"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Determine overall status
        if (health_results["healthz"]["status"] == "healthy" and 
            health_results["readyz"]["status"] == "ready"):
            health_results["overall_status"] = "healthy"
        else:
            health_results["overall_status"] = "unhealthy"
        
        return health_results
    
    def validate_deployment(self, deployment_id: str) -> Dict[str, Any]:
        """
        Validate that deployment is healthy and accessible (implements ModelDeployer interface)
        
        Args:
            deployment_id: Service name or URL to validate
            
        Returns:
            Dict with health check results
        """
        try:
            # If deployment_id looks like a URL, use it directly
            if deployment_id.startswith('https://'):
                service_url = deployment_id
            else:
                # Construct URL from service name
                service_url = f"https://{deployment_id}-{self.region.replace('_', '-')}-{self.project_id}.run.app"
            
            return self._validate_deployment_health(service_url)
            
        except Exception as e:
            raise DeploymentError(
                f"Deployment validation failed: {str(e)}",
                service_name=deployment_id,
                hint="Check service name and ensure deployment completed successfully"
            )
    
    def get_deployment_status(self, deployment_id: str) -> Dict[str, Any]:
        """
        Get current status of deployment (implements ModelDeployer interface)
        
        Args:
            deployment_id: Service name to check
            
        Returns:
            Dict with current status, metrics, logs
        """
        try:
            # Get service details using gcloud
            result = subprocess.run([
                "gcloud", "run", "services", "describe", deployment_id,
                "--region", self.region,
                "--format", "json"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                raise DeploymentError(
                    f"Service {deployment_id} not found",
                    service_name=deployment_id,
                    hint="Check service name and region"
                )
            
            service_info = json.loads(result.stdout)
            
            # Extract key information
            status_info = {
                "service_name": deployment_id,
                "region": self.region,
                "url": service_info.get("status", {}).get("url"),
                "ready": service_info.get("status", {}).get("conditions", [{}])[0].get("status") == "True",
                "latest_revision": service_info.get("status", {}).get("latestRevisionName"),
                "traffic_allocation": service_info.get("status", {}).get("traffic", []),
                "creation_timestamp": service_info.get("metadata", {}).get("creationTimestamp"),
                "last_modified": service_info.get("status", {}).get("observedGeneration")
            }
            
            return status_info
            
        except subprocess.TimeoutExpired:
            raise DeploymentError(
                "Status check timeout",
                service_name=deployment_id,
                hint="Check network connectivity and service availability"
            )
        except Exception as e:
            raise DeploymentError(
                f"Failed to get deployment status: {str(e)}",
                service_name=deployment_id,
                hint="Ensure gcloud CLI is configured and service exists"
            )


# Convenience functions for SDK usage
def deploy_model_to_cloud_run(model: Any, 
                             project_id: str,
                             service_name: str = None,
                             region: str = "us-central1",
                             **kwargs) -> Dict[str, Any]:
    """
    Deploy any Kepler model to Google Cloud Run
    
    Args:
        model: Trained Kepler model (any framework)
        project_id: Google Cloud project ID
        service_name: Optional service name (auto-generated if None)
        region: GCP region (default: us-central1)
        **kwargs: Additional Cloud Run configuration
        
    Returns:
        Dict with deployment results
        
    Example:
        >>> import kepler as kp
        >>> 
        >>> # Train model
        >>> model = kp.train_unified.train(data, target="failure", algorithm="xgboost")
        >>> 
        >>> # Deploy to Cloud Run
        >>> result = deploy_model_to_cloud_run(
        ...     model,
        ...     project_id="my-ml-project",
        ...     service_name="predictive-maintenance",
        ...     memory="2Gi"
        ... )
        >>> 
        >>> print(f"Deployed to: {result['service_url']}")
    """
    deployer = CloudRunDeployer(project_id, region)
    
    deployment_config = {
        "project_id": project_id,
        "region": region,
        "service_name": service_name,
        **kwargs
    }
    
    return deployer.deploy(model, deployment_config)


def validate_cloud_run_deployment(service_name: str,
                                 project_id: str, 
                                 region: str = "us-central1") -> Dict[str, Any]:
    """
    Validate Cloud Run deployment health
    
    Args:
        service_name: Cloud Run service name
        project_id: Google Cloud project ID
        region: GCP region
        
    Returns:
        Dict with validation results
    """
    deployer = CloudRunDeployer(project_id, region)
    return deployer.validate_deployment(service_name)
