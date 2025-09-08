"""
Kepler Deploy Module - Automatic Model Deployment

Simple SDK interface for deploying ANY trained model to production.
Supports multiple deployment targets with unified API.

Usage:
    import kepler as kp
    
    # Train model
    model = kp.train_unified.train(data, target="failure", algorithm="xgboost")
    
    # Deploy to Cloud Run
    result = kp.deploy.to_cloud_run(model, project_id="my-project")
    print(f"Deployed to: {result['service_url']}")
"""

from typing import Dict, List, Optional, Any, Union
from kepler.deployers.cloud_run_deployer import (
    CloudRunDeployer,
    deploy_model_to_cloud_run,
    validate_cloud_run_deployment
)
from kepler.utils.logging import get_logger
from kepler.utils.exceptions import DeploymentError


def to_cloud_run(model: Any,
                project_id: str,
                service_name: str = None,
                region: str = "us-central1",
                memory: str = "1Gi",
                cpu: str = "1",
                min_instances: int = 0,
                max_instances: int = 100,
                **kwargs) -> Dict[str, Any]:
    """
    Deploy any Kepler model to Google Cloud Run
    
    Supports models trained with ANY framework:
    - Traditional ML: sklearn, XGBoost, LightGBM, CatBoost
    - Deep Learning: PyTorch, TensorFlow, Keras, JAX
    - Generative AI: transformers, langchain, OpenAI
    - Computer Vision: OpenCV, PIL
    - Custom frameworks: Any Python library
    
    Args:
        model: Trained Kepler model (any framework)
        project_id: Google Cloud project ID
        service_name: Optional service name (auto-generated if None)
        region: GCP region (default: us-central1)
        memory: Memory allocation (default: 1Gi)
        cpu: CPU allocation (default: 1)
        min_instances: Minimum instances (default: 0)
        max_instances: Maximum instances (default: 100)
        **kwargs: Additional Cloud Run configuration
        
    Returns:
        Dict with deployment results including service_url
        
    Raises:
        DeploymentError: If deployment fails
        
    Examples:
        # Deploy XGBoost model
        model = kp.train_unified.train(data, target="failure", algorithm="xgboost")
        result = kp.deploy.to_cloud_run(model, project_id="my-ml-project")
        
        # Deploy PyTorch model with custom config
        model = kp.train_unified.train(data, target="failure", algorithm="pytorch")
        result = kp.deploy.to_cloud_run(
            model,
            project_id="my-ml-project", 
            service_name="neural-net-api",
            memory="2Gi",
            cpu="2"
        )
        
        # Deploy transformer model
        model = kp.train_unified.train(text_data, target="sentiment", algorithm="transformers")
        result = kp.deploy.to_cloud_run(
            model,
            project_id="my-ml-project",
            memory="4Gi",  # More memory for transformers
            max_instances=50
        )
    """
    logger = get_logger(__name__)
    logger.info(f"Deploying model to Cloud Run: project={project_id}, region={region}")
    
    try:
        # Prepare deployment configuration
        deployment_config = {
            "project_id": project_id,
            "region": region,
            "service_name": service_name,
            "memory": memory,
            "cpu": cpu,
            "min_instances": min_instances,
            "max_instances": max_instances,
            **kwargs
        }
        
        # Use CloudRunDeployer for actual deployment
        return deploy_model_to_cloud_run(model, project_id, service_name, region, **kwargs)
        
    except Exception as e:
        raise DeploymentError(
            f"Cloud Run deployment failed: {str(e)}",
            service_name=service_name,
            hint="Check GCP credentials, project permissions, and model compatibility"
        )


def validate(service_name: str,
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
        
    Example:
        # Validate deployment
        health = kp.deploy.validate("my-model-api", "my-project")
        if health["overall_status"] == "healthy":
            print("✅ Deployment is healthy")
        else:
            print("❌ Deployment has issues")
    """
    logger = get_logger(__name__)
    logger.info(f"Validating Cloud Run deployment: {service_name}")
    
    return validate_cloud_run_deployment(service_name, project_id, region)


def get_status(service_name: str,
              project_id: str, 
              region: str = "us-central1") -> Dict[str, Any]:
    """
    Get detailed status of Cloud Run deployment
    
    Args:
        service_name: Cloud Run service name
        project_id: Google Cloud project ID
        region: GCP region
        
    Returns:
        Dict with detailed status information
        
    Example:
        # Get deployment status
        status = kp.deploy.get_status("my-model-api", "my-project")
        print(f"Service URL: {status['url']}")
        print(f"Ready: {status['ready']}")
        print(f"Latest revision: {status['latest_revision']}")
    """
    logger = get_logger(__name__)
    logger.info(f"Getting status for Cloud Run service: {service_name}")
    
    deployer = CloudRunDeployer(project_id, region)
    return deployer.get_deployment_status(service_name)


def list_deployments(project_id: str, region: str = "us-central1") -> List[Dict[str, Any]]:
    """
    List all Cloud Run services in project
    
    Args:
        project_id: Google Cloud project ID
        region: GCP region
        
    Returns:
        List of service information
        
    Example:
        # List all deployments
        services = kp.deploy.list_deployments("my-project")
        for service in services:
            print(f"{service['name']}: {service['url']}")
    """
    logger = get_logger(__name__)
    logger.info(f"Listing Cloud Run deployments in {project_id}/{region}")
    
    try:
        # List services using gcloud
        result = subprocess.run([
            "gcloud", "run", "services", "list",
            "--region", region,
            "--format", "json"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            raise DeploymentError(
                f"Failed to list services: {result.stderr}",
                hint="Check GCP credentials and project access"
            )
        
        services_info = json.loads(result.stdout)
        
        # Extract relevant information
        deployments = []
        for service in services_info:
            deployments.append({
                "name": service.get("metadata", {}).get("name"),
                "url": service.get("status", {}).get("url"),
                "ready": service.get("status", {}).get("conditions", [{}])[0].get("status") == "True",
                "region": region,
                "creation_time": service.get("metadata", {}).get("creationTimestamp"),
                "latest_revision": service.get("status", {}).get("latestRevisionName")
            })
        
        return deployments
        
    except subprocess.TimeoutExpired:
        raise DeploymentError(
            "List deployments timeout",
            hint="Check network connectivity"
        )
    except Exception as e:
        raise DeploymentError(
            f"Failed to list deployments: {str(e)}",
            hint="Ensure gcloud CLI is configured and project access is available"
        )
