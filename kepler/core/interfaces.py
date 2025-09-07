"""
Kepler Core Interfaces - Hexagonal Architecture (Ports & Adapters)

Defines the core interfaces (Ports) that must be implemented by adapters.
Following system-prompt.mdc requirements for hexagonal architecture.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
import pandas as pd
from pathlib import Path

from kepler.utils.exceptions import KeplerError


class DataConnector(ABC):
    """
    Port: Abstract interface for data extraction from any source
    
    All data connectors must implement this interface to be compatible
    with Kepler's core engine. This enables pluggable data sources.
    """
    
    @abstractmethod
    def extract(self, query: str, **kwargs) -> pd.DataFrame:
        """
        Extract data using source-specific query language
        
        Args:
            query: Query in source-specific language (SPL for Splunk, SQL for databases)
            **kwargs: Source-specific parameters
            
        Returns:
            DataFrame with extracted data
            
        Raises:
            KeplerError: If extraction fails
        """
        pass
    
    @abstractmethod
    def validate_connection(self) -> Dict[str, Any]:
        """
        Validate connection to data source
        
        Returns:
            Dict with validation results and error details
        """
        pass
    
    @abstractmethod
    def get_available_indexes(self) -> List[str]:
        """
        Get list of available data indexes/tables
        
        Returns:
            List of index/table names
        """
        pass


class ModelTrainer(ABC):
    """
    Port: Abstract interface for model training with any framework
    
    All model trainers must implement this interface to be compatible
    with Kepler's unified training API. This enables pluggable AI frameworks.
    """
    
    @abstractmethod
    def train(self, data: pd.DataFrame, target: str, **kwargs) -> Any:
        """
        Train model with framework-specific implementation
        
        Args:
            data: Training data DataFrame
            target: Target column name
            **kwargs: Framework-specific parameters
            
        Returns:
            Trained model object (framework-specific)
            
        Raises:
            KeplerError: If training fails
        """
        pass
    
    @abstractmethod
    def validate_data(self, data: pd.DataFrame, target: str) -> Dict[str, Any]:
        """
        Validate data compatibility with this trainer
        
        Args:
            data: Data to validate
            target: Target column name
            
        Returns:
            Dict with validation results
        """
        pass
    
    @abstractmethod
    def get_supported_algorithms(self) -> List[str]:
        """
        Get list of algorithms supported by this trainer
        
        Returns:
            List of algorithm names
        """
        pass
    
    @abstractmethod
    def estimate_training_time(self, data: pd.DataFrame, algorithm: str) -> float:
        """
        Estimate training time for given data and algorithm
        
        Args:
            data: Training data
            algorithm: Algorithm name
            
        Returns:
            Estimated training time in seconds
        """
        pass


class ModelDeployer(ABC):
    """
    Port: Abstract interface for model deployment to any platform
    
    All model deployers must implement this interface to be compatible
    with Kepler's deployment engine. This enables pluggable deployment targets.
    """
    
    @abstractmethod
    def deploy(self, model: Any, deployment_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deploy model to target platform
        
        Args:
            model: Trained model to deploy
            deployment_config: Platform-specific deployment configuration
            
        Returns:
            Dict with deployment results (endpoint URL, status, etc.)
            
        Raises:
            KeplerError: If deployment fails
        """
        pass
    
    @abstractmethod
    def validate_deployment(self, deployment_id: str) -> Dict[str, Any]:
        """
        Validate that deployment is healthy and accessible
        
        Args:
            deployment_id: Deployment identifier
            
        Returns:
            Dict with health check results
        """
        pass
    
    @abstractmethod
    def get_deployment_status(self, deployment_id: str) -> Dict[str, Any]:
        """
        Get current status of deployment
        
        Args:
            deployment_id: Deployment identifier
            
        Returns:
            Dict with current status, metrics, logs
        """
        pass


class ResultsWriter(ABC):
    """
    Port: Abstract interface for writing results to any destination
    
    All results writers must implement this interface to be compatible
    with Kepler's results pipeline. This enables pluggable result destinations.
    """
    
    @abstractmethod
    def write(self, data: pd.DataFrame, destination_config: Dict[str, Any]) -> bool:
        """
        Write results to destination
        
        Args:
            data: Results data to write
            destination_config: Destination-specific configuration
            
        Returns:
            True if write successful, False otherwise
            
        Raises:
            KeplerError: If write fails
        """
        pass
    
    @abstractmethod
    def validate_destination(self, destination_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that destination is accessible and writable
        
        Args:
            destination_config: Destination configuration
            
        Returns:
            Dict with validation results
        """
        pass


# Plugin Registry for Dynamic Loading
class PluginRegistry:
    """
    Registry for dynamically loading plugins that implement core interfaces
    
    Enables runtime discovery and loading of adapters without modifying core code.
    """
    
    def __init__(self):
        self._connectors: Dict[str, DataConnector] = {}
        self._trainers: Dict[str, ModelTrainer] = {}
        self._deployers: Dict[str, ModelDeployer] = {}
        self._writers: Dict[str, ResultsWriter] = {}
    
    def register_connector(self, name: str, connector: DataConnector) -> None:
        """Register a data connector adapter"""
        self._connectors[name] = connector
    
    def register_trainer(self, name: str, trainer: ModelTrainer) -> None:
        """Register a model trainer adapter"""
        self._trainers[name] = trainer
    
    def register_deployer(self, name: str, deployer: ModelDeployer) -> None:
        """Register a model deployer adapter"""
        self._deployers[name] = deployer
    
    def register_writer(self, name: str, writer: ResultsWriter) -> None:
        """Register a results writer adapter"""
        self._writers[name] = writer
    
    def get_connector(self, name: str) -> DataConnector:
        """Get registered data connector"""
        if name not in self._connectors:
            raise KeplerError(
                code="PLUGIN_001",
                message=f"Data connector '{name}' not registered",
                hint=f"Available connectors: {list(self._connectors.keys())}",
                context={"requested": name, "available": list(self._connectors.keys())},
                retryable=False
            )
        return self._connectors[name]
    
    def get_trainer(self, name: str) -> ModelTrainer:
        """Get registered model trainer"""
        if name not in self._trainers:
            raise KeplerError(
                code="PLUGIN_002", 
                message=f"Model trainer '{name}' not registered",
                hint=f"Available trainers: {list(self._trainers.keys())}",
                context={"requested": name, "available": list(self._trainers.keys())},
                retryable=False
            )
        return self._trainers[name]
    
    def get_deployer(self, name: str) -> ModelDeployer:
        """Get registered model deployer"""
        if name not in self._deployers:
            raise KeplerError(
                code="PLUGIN_003",
                message=f"Model deployer '{name}' not registered", 
                hint=f"Available deployers: {list(self._deployers.keys())}",
                context={"requested": name, "available": list(self._deployers.keys())},
                retryable=False
            )
        return self._deployers[name]
    
    def get_writer(self, name: str) -> ResultsWriter:
        """Get registered results writer"""
        if name not in self._writers:
            raise KeplerError(
                code="PLUGIN_004",
                message=f"Results writer '{name}' not registered",
                hint=f"Available writers: {list(self._writers.keys())}",
                context={"requested": name, "available": list(self._writers.keys())},
                retryable=False
            )
        return self._writers[name]


# Global plugin registry instance
_plugin_registry = PluginRegistry()


def get_plugin_registry() -> PluginRegistry:
    """Get global plugin registry instance"""
    return _plugin_registry
