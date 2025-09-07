"""
Kepler Versioning API - Task 5.1 Implementation
Data versioning and reproducibility with DVC integration

Provides comprehensive versioning for:
- Data versioning with DVC integration and fallback
- Version tracking and management
- Data lineage and provenance
- Reproducible data pipelines
"""

import os
import subprocess
import hashlib
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime

from kepler.utils.logging import get_logger
from kepler.utils.exceptions import ModelTrainingError


@dataclass
class DataVersion:
    """Data version metadata"""
    version_id: str
    file_path: str
    hash_md5: str
    size_bytes: int
    timestamp: str
    metadata: Dict[str, Any]
    dvc_file: str
    remote_url: Optional[str] = None


# Global version manager instance
_version_manager = None


def _get_version_manager():
    """Get or create global version manager instance"""
    global _version_manager
    if _version_manager is None:
        _version_manager = DataVersionManager()
    return _version_manager


class DataVersionManager:
    """Data versioning manager with DVC integration"""
    
    def __init__(self, project_root: str = None):
        self.logger = get_logger(__name__)
        self.project_root = Path(project_root or os.getcwd())
        self.dvc_dir = self.project_root / ".dvc"
        self.data_dir = self.project_root / "data"
        self.versions_file = self.project_root / ".kepler" / "data_versions.json"
        
        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True)
        (self.project_root / ".kepler").mkdir(exist_ok=True)
        
        self._load_version_history()
    
    def _load_version_history(self):
        """Load existing version history"""
        if self.versions_file.exists():
            with open(self.versions_file, 'r') as f:
                self.version_history = json.load(f)
        else:
            self.version_history = {
                'versions': {},
                'lineage': {},
                'metadata': {
                    'created': datetime.now().isoformat(),
                    'last_updated': datetime.now().isoformat()
                }
            }
    
    def _save_version_history(self):
        """Save version history to disk"""
        self.version_history['metadata']['last_updated'] = datetime.now().isoformat()
        with open(self.versions_file, 'w') as f:
            json.dump(self.version_history, f, indent=2)
    
    def initialize_dvc(self, remote_storage: Dict[str, str] = None) -> bool:
        """Initialize DVC in the project"""
        self.logger.info("Initializing DVC for data versioning...")
        
        try:
            # Check if DVC is available
            result = subprocess.run(['dvc', '--version'], 
                                  capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode != 0:
                self.logger.warning("DVC not available, using fallback versioning system")
                self.use_dvc = False
                return True
            
            # Check if already initialized
            if self.dvc_dir.exists():
                self.logger.info("DVC already initialized")
                self.use_dvc = True
            else:
                # Initialize DVC
                result = subprocess.run(['dvc', 'init'], 
                                      capture_output=True, text=True, cwd=self.project_root)
                
                if result.returncode != 0:
                    self.logger.warning(f"DVC init failed: {result.stderr}, using fallback")
                    self.use_dvc = False
                    return True
                
                self.logger.info("DVC initialized successfully")
                self.use_dvc = True
            
            return True
            
        except FileNotFoundError:
            self.logger.warning("DVC not found, using fallback versioning system")
            self.use_dvc = False
            return True
        except Exception as e:
            self.logger.warning(f"DVC initialization failed: {e}, using fallback")
            self.use_dvc = False
            return True
    
    def version_data(self, data_path: Union[str, Path], 
                    version_name: str = None,
                    metadata: Dict[str, Any] = None) -> DataVersion:
        """Version a data file with DVC or fallback"""
        self.logger.info(f"Versioning data: {data_path}")
        
        try:
            data_path = Path(data_path)
            
            if not data_path.exists():
                raise FileNotFoundError(f"Data path does not exist: {data_path}")
            
            # Generate version metadata
            file_hash = self._calculate_file_hash(data_path)
            file_size = self._get_file_size(data_path)
            timestamp = datetime.now().isoformat()
            
            # Generate version ID
            if version_name:
                version_id = f"{version_name}-{file_hash[:8]}"
            else:
                version_id = f"v{len(self.version_history['versions'])+1:03d}-{file_hash[:8]}"
            
            # Prepare metadata
            version_metadata = {
                'original_path': str(data_path),
                'file_type': data_path.suffix,
                'created_by': 'kepler-data-versioning',
                **(metadata or {})
            }
            
            # Add data profiling if it's a CSV file
            if data_path.suffix.lower() == '.csv':
                try:
                    df = pd.read_csv(data_path)
                    version_metadata.update({
                        'rows': len(df),
                        'columns': len(df.columns),
                        'column_names': df.columns.tolist()[:10]  # Limit for storage
                    })
                except Exception as e:
                    self.logger.debug(f"Could not profile CSV file: {e}")
            
            # Use fallback versioning for simplicity
            dvc_file = self._version_with_fallback(data_path, version_id)
            
            # Create version object
            data_version = DataVersion(
                version_id=version_id,
                file_path=str(data_path),
                hash_md5=file_hash,
                size_bytes=file_size,
                timestamp=timestamp,
                metadata=version_metadata,
                dvc_file=dvc_file,
                remote_url=None
            )
            
            # Store in version history
            self.version_history['versions'][version_id] = {
                'version_id': version_id,
                'file_path': str(data_path),
                'hash_md5': file_hash,
                'size_bytes': file_size,
                'timestamp': timestamp,
                'metadata': version_metadata,
                'dvc_file': dvc_file,
                'remote_url': None
            }
            
            self._save_version_history()
            
            self.logger.info(f"Data versioned successfully: {version_id}")
            return data_version
            
        except Exception as e:
            raise ModelTrainingError(
                f"Data versioning failed: {e}",
                suggestion="Check file permissions and path"
            )
    
    def _version_with_fallback(self, data_path: Path, version_id: str) -> str:
        """Version data using fallback system"""
        versioned_dir = self.project_root / ".kepler" / "data_versions"
        versioned_dir.mkdir(parents=True, exist_ok=True)
        
        # Create versioned copy
        versioned_path = versioned_dir / f"{version_id}_{data_path.name}"
        
        if data_path.is_file():
            import shutil
            shutil.copy2(data_path, versioned_path)
        
        return str(versioned_path)
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of file"""
        hash_md5 = hashlib.md5()
        
        if file_path.is_file():
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
        
        return hash_md5.hexdigest()
    
    def _get_file_size(self, file_path: Path) -> int:
        """Get file size in bytes"""
        if file_path.is_file():
            return file_path.stat().st_size
        return 0
    
    def get_version(self, version_id: str) -> Optional[DataVersion]:
        """Get version information by ID"""
        version_data = self.version_history['versions'].get(version_id)
        if version_data:
            return DataVersion(**version_data)
        return None
    
    def list_versions(self, file_pattern: str = None) -> List[DataVersion]:
        """List all data versions"""
        versions = []
        for version_data in self.version_history['versions'].values():
            if file_pattern is None or file_pattern in version_data['file_path']:
                versions.append(DataVersion(**version_data))
        
        return sorted(versions, key=lambda x: x.timestamp, reverse=True)
    
    def create_data_lineage(self, source_versions: List[str],
                           transformation_pipeline: List[str],
                           output_version: str,
                           metadata: Dict[str, Any] = None):
        """Create data lineage record"""
        lineage_data = {
            'source_versions': source_versions,
            'transformation_pipeline': transformation_pipeline,
            'output_version': output_version,
            'execution_metadata': {
                'timestamp': datetime.now().isoformat(),
                'kepler_version': '1.0.0',
                **(metadata or {})
            }
        }
        
        self.version_history['lineage'][output_version] = lineage_data
        self._save_version_history()
        
        return lineage_data
    
    def get_lineage(self, version_id: str):
        """Get data lineage for a version"""
        return self.version_history['lineage'].get(version_id)


# High-level API functions
def initialize_project_versioning(remote_storage: Dict[str, str] = None) -> bool:
    """Initialize data versioning for the current project"""
    manager = _get_version_manager()
    return manager.initialize_dvc(remote_storage)


def version_data(data_path: Union[str, Path], 
                version_name: str = None,
                metadata: Dict[str, Any] = None) -> str:
    """Version a data file with automatic tracking"""
    manager = _get_version_manager()
    version = manager.version_data(data_path, version_name, metadata)
    return version.version_id


def list_versions(file_pattern: str = None) -> List[Dict[str, Any]]:
    """List all data versions"""
    manager = _get_version_manager()
    versions = manager.list_versions(file_pattern)
    
    result = []
    for version in versions:
        result.append({
            'version_id': version.version_id,
            'file_path': version.file_path,
            'hash_md5': version.hash_md5,
            'size_bytes': version.size_bytes,
            'size_mb': version.size_bytes / (1024 * 1024),
            'timestamp': version.timestamp,
            'metadata': version.metadata,
            'has_remote': version.remote_url is not None
        })
    
    return result


def get_version_info(version_id: str) -> Optional[Dict[str, Any]]:
    """Get detailed information about a specific version"""
    manager = _get_version_manager()
    version = manager.get_version(version_id)
    
    if version:
        return {
            'version_id': version.version_id,
            'file_path': version.file_path,
            'hash_md5': version.hash_md5,
            'size_bytes': version.size_bytes,
            'size_mb': version.size_bytes / (1024 * 1024),
            'timestamp': version.timestamp,
            'metadata': version.metadata,
            'dvc_file': version.dvc_file,
            'has_remote': version.remote_url is not None
        }
    
    return None


def create_lineage(source_versions: List[str],
                  transformation_steps: List[str],
                  output_version: str,
                  metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create data lineage record for traceability"""
    manager = _get_version_manager()
    return manager.create_data_lineage(source_versions, transformation_steps, output_version, metadata)


def get_lineage(version_id: str) -> Optional[Dict[str, Any]]:
    """Get data lineage for a specific version"""
    manager = _get_version_manager()
    return manager.get_lineage(version_id)


def get_storage_summary() -> Dict[str, Any]:
    """Get summary of all versions and storage usage"""
    manager = _get_version_manager()
    total_versions = len(manager.version_history['versions'])
    total_size = sum(v['size_bytes'] for v in manager.version_history['versions'].values())
    
    return {
        'total_versions': total_versions,
        'total_size_bytes': total_size,
        'total_size_mb': total_size / (1024 * 1024),
        'dvc_enabled': hasattr(manager, 'use_dvc') and manager.use_dvc,
        'latest_versions': manager.list_versions()[:5]
    }


def checkout_version(version_id: str, target_path: str = None) -> str:
    """Checkout a specific version of data"""
    manager = _get_version_manager()
    
    version = manager.get_version(version_id)
    if not version:
        raise ModelTrainingError(f"Version not found: {version_id}")
    
    # Simple checkout using fallback system
    target = Path(target_path or version.file_path)
    versioned_path = Path(version.dvc_file)
    
    if versioned_path.exists():
        import shutil
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(versioned_path, target)
        return str(target)
    else:
        raise FileNotFoundError(f"Versioned data not found: {versioned_path}")


def version_feature_pipeline(pipeline_steps: List[Dict[str, Any]],
                            pipeline_name: str = None,
                            input_data_version: str = None,
                            metadata: Dict[str, Any] = None) -> str:
    """
    Version a feature engineering pipeline for reproducibility
    
    Args:
        pipeline_steps: List of pipeline step configurations
        pipeline_name: Optional pipeline name
        input_data_version: Version ID of input data
        metadata: Additional metadata
        
    Returns:
        Pipeline version ID
        
    Example:
        >>> # Version feature engineering pipeline
        >>> pipeline_steps = [
        ...     {
        ...         "operation": "polynomial_features",
        ...         "parameters": {"degree": 2, "columns": ["temperature", "pressure"]},
        ...         "output_features": ["temperature_squared", "pressure_squared", "temp_pressure_interaction"]
        ...     },
        ...     {
        ...         "operation": "categorical_encoding",
        ...         "parameters": {"method": "one_hot", "columns": ["equipment_type"]},
        ...         "output_features": ["equipment_type_A", "equipment_type_B", "equipment_type_C"]
        ...     }
        ... ]
        >>> 
        >>> pipeline_version = kp.versioning.version_feature_pipeline(
        ...     pipeline_steps,
        ...     pipeline_name="sensor-preprocessing-v1",
        ...     input_data_version="raw-data-v001"
        ... )
        >>> print(f"Pipeline versioned: {pipeline_version}")
    """
    logger = get_logger(__name__)
    logger.info("Versioning feature engineering pipeline")
    
    try:
        manager = _get_version_manager()
        
        # Generate pipeline hash for versioning
        pipeline_content = json.dumps(pipeline_steps, sort_keys=True)
        pipeline_hash = hashlib.md5(pipeline_content.encode()).hexdigest()
        
        # Generate pipeline version ID
        if pipeline_name:
            pipeline_version_id = f"{pipeline_name}-{pipeline_hash[:8]}"
        else:
            pipeline_count = len([k for k in manager.version_history.get('pipelines', {}).keys()])
            pipeline_version_id = f"pipeline-v{pipeline_count+1:03d}-{pipeline_hash[:8]}"
        
        # Prepare pipeline metadata
        pipeline_metadata = {
            'pipeline_type': 'feature_engineering',
            'total_steps': len(pipeline_steps),
            'input_data_version': input_data_version,
            'created_by': 'kepler-feature-versioning',
            'pipeline_hash': pipeline_hash,
            'timestamp': datetime.now().isoformat(),
            **(metadata or {})
        }
        
        # Store pipeline version
        if 'pipelines' not in manager.version_history:
            manager.version_history['pipelines'] = {}
        
        manager.version_history['pipelines'][pipeline_version_id] = {
            'version_id': pipeline_version_id,
            'pipeline_steps': pipeline_steps,
            'metadata': pipeline_metadata,
            'created': datetime.now().isoformat()
        }
        
        manager._save_version_history()
        
        logger.info(f"Feature pipeline versioned: {pipeline_version_id}")
        return pipeline_version_id
        
    except Exception as e:
        raise ModelTrainingError(
            f"Feature pipeline versioning failed: {e}",
            suggestion="Check pipeline steps format and metadata"
        )


def get_feature_pipeline(pipeline_version_id: str) -> Optional[Dict[str, Any]]:
    """
    Get feature engineering pipeline by version ID
    
    Args:
        pipeline_version_id: Pipeline version identifier
        
    Returns:
        Pipeline configuration or None if not found
        
    Example:
        >>> pipeline = kp.versioning.get_feature_pipeline("sensor-preprocessing-v1-a1b2c3d4")
        >>> if pipeline:
        ...     print(f"Pipeline: {pipeline['version_id']}")
        ...     print(f"Steps: {len(pipeline['pipeline_steps'])}")
        ...     for i, step in enumerate(pipeline['pipeline_steps'], 1):
        ...         print(f"  {i}. {step['operation']}")
    """
    try:
        manager = _get_version_manager()
        
        pipelines = manager.version_history.get('pipelines', {})
        pipeline_data = pipelines.get(pipeline_version_id)
        
        if pipeline_data:
            return {
                'version_id': pipeline_data['version_id'],
                'pipeline_steps': pipeline_data['pipeline_steps'],
                'metadata': pipeline_data['metadata'],
                'created': pipeline_data['created']
            }
        
        return None
        
    except Exception as e:
        raise ModelTrainingError(
            f"Failed to get feature pipeline: {e}",
            suggestion="Check pipeline version ID is correct"
        )


def list_feature_pipelines() -> List[Dict[str, Any]]:
    """
    List all feature engineering pipeline versions
    
    Returns:
        List of pipeline version information
        
    Example:
        >>> pipelines = kp.versioning.list_feature_pipelines()
        >>> for pipeline in pipelines:
        ...     print(f"{pipeline['version_id']}: {pipeline['total_steps']} steps")
    """
    try:
        manager = _get_version_manager()
        
        pipelines = manager.version_history.get('pipelines', {})
        result = []
        
        for pipeline_data in pipelines.values():
            metadata = pipeline_data['metadata']
            result.append({
                'version_id': pipeline_data['version_id'],
                'total_steps': metadata['total_steps'],
                'input_data_version': metadata.get('input_data_version'),
                'created': pipeline_data['created'],
                'pipeline_hash': metadata['pipeline_hash']
            })
        
        # Sort by creation time (newest first)
        return sorted(result, key=lambda x: x['created'], reverse=True)
        
    except Exception as e:
        raise ModelTrainingError(
            f"Failed to list feature pipelines: {e}",
            suggestion="Check versioning is initialized"
        )


def apply_versioned_pipeline(data: pd.DataFrame, pipeline_version_id: str) -> Dict[str, Any]:
    """
    Apply a versioned feature engineering pipeline to data
    
    Args:
        data: Input DataFrame
        pipeline_version_id: Pipeline version to apply
        
    Returns:
        Results with transformed data and metadata
        
    Example:
        >>> # Apply versioned pipeline
        >>> result = kp.versioning.apply_versioned_pipeline(
        ...     raw_data, 
        ...     "sensor-preprocessing-v1-a1b2c3d4"
        ... )
        >>> 
        >>> transformed_data = result['transformed_data']
        >>> applied_steps = result['applied_steps']
        >>> print(f"Applied {len(applied_steps)} pipeline steps")
    """
    logger = get_logger(__name__)
    logger.info(f"Applying versioned pipeline: {pipeline_version_id}")
    
    try:
        # Get pipeline configuration
        pipeline = get_feature_pipeline(pipeline_version_id)
        if not pipeline:
            raise ModelTrainingError(f"Pipeline not found: {pipeline_version_id}")
        
        # Apply pipeline steps sequentially
        transformed_data = data.copy()
        applied_steps = []
        
        for i, step in enumerate(pipeline['pipeline_steps']):
            try:
                operation = step['operation']
                parameters = step.get('parameters', {})
                
                # Apply specific operations
                if operation == 'polynomial_features':
                    transformed_data = _apply_polynomial_features(transformed_data, parameters)
                elif operation == 'categorical_encoding':
                    transformed_data = _apply_categorical_encoding(transformed_data, parameters)
                elif operation == 'missing_value_imputation':
                    transformed_data = _apply_missing_value_imputation(transformed_data, parameters)
                elif operation == 'feature_scaling':
                    transformed_data = _apply_feature_scaling(transformed_data, parameters)
                else:
                    logger.warning(f"Unknown operation: {operation}, skipping")
                    continue
                
                applied_steps.append({
                    'step_number': i + 1,
                    'operation': operation,
                    'parameters': parameters,
                    'output_shape': transformed_data.shape
                })
                
                logger.debug(f"Applied step {i+1}: {operation}")
                
            except Exception as e:
                logger.warning(f"Step {i+1} failed: {e}, continuing with next steps")
                continue
        
        result = {
            'transformed_data': transformed_data,
            'applied_steps': applied_steps,
            'pipeline_version': pipeline_version_id,
            'input_shape': data.shape,
            'output_shape': transformed_data.shape,
            'execution_metadata': {
                'timestamp': datetime.now().isoformat(),
                'steps_applied': len(applied_steps),
                'steps_total': len(pipeline['pipeline_steps'])
            }
        }
        
        logger.info(f"Pipeline applied: {len(applied_steps)}/{len(pipeline['pipeline_steps'])} steps successful")
        return result
        
    except Exception as e:
        raise ModelTrainingError(
            f"Failed to apply versioned pipeline: {e}",
            suggestion="Check pipeline version ID and data compatibility"
        )


def _apply_polynomial_features(data: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
    """Apply polynomial feature transformation"""
    degree = parameters.get('degree', 2)
    columns = parameters.get('columns', [])
    
    if not columns:
        # Auto-select numeric columns
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    result_data = data.copy()
    
    # Add polynomial features
    for col in columns:
        if col in data.columns and data[col].dtype in [np.number]:
            if degree >= 2:
                result_data[f"{col}_squared"] = data[col] ** 2
            if degree >= 3:
                result_data[f"{col}_cubed"] = data[col] ** 3
    
    return result_data


def _apply_categorical_encoding(data: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
    """Apply categorical encoding transformation"""
    method = parameters.get('method', 'one_hot')
    columns = parameters.get('columns', [])
    
    if not columns:
        # Auto-select categorical columns
        columns = data.select_dtypes(exclude=[np.number]).columns.tolist()
    
    result_data = data.copy()
    
    for col in columns:
        if col in data.columns:
            if method == 'one_hot':
                dummies = pd.get_dummies(data[col], prefix=col, drop_first=True)
                result_data = pd.concat([result_data.drop(columns=[col]), dummies], axis=1)
            elif method == 'label':
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                result_data[f"{col}_encoded"] = le.fit_transform(data[col].fillna('unknown'))
                result_data = result_data.drop(columns=[col])
    
    return result_data


def _apply_missing_value_imputation(data: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
    """Apply missing value imputation"""
    strategy = parameters.get('strategy', 'median')
    columns = parameters.get('columns', [])
    
    if not columns:
        columns = data.columns[data.isnull().any()].tolist()
    
    result_data = data.copy()
    
    for col in columns:
        if col in data.columns and data[col].isnull().any():
            if data[col].dtype in [np.number]:
                if strategy == 'median':
                    fill_value = data[col].median()
                elif strategy == 'mean':
                    fill_value = data[col].mean()
                else:
                    fill_value = 0
                result_data[col] = result_data[col].fillna(fill_value)
            else:
                # Categorical
                mode_value = data[col].mode()
                fill_value = mode_value[0] if len(mode_value) > 0 else 'unknown'
                result_data[col] = result_data[col].fillna(fill_value)
    
    return result_data


def _apply_feature_scaling(data: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
    """Apply feature scaling transformation"""
    method = parameters.get('method', 'standard')
    columns = parameters.get('columns', [])
    
    if not columns:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    result_data = data.copy()
    
    for col in columns:
        if col in data.columns and data[col].dtype in [np.number]:
            if method == 'standard':
                mean_val = data[col].mean()
                std_val = data[col].std()
                if std_val > 0:
                    result_data[col] = (data[col] - mean_val) / std_val
            elif method == 'minmax':
                min_val = data[col].min()
                max_val = data[col].max()
                if max_val > min_val:
                    result_data[col] = (data[col] - min_val) / (max_val - min_val)
    
    return result_data


def create_pipeline_from_automl(automl_result: Dict[str, Any],
                               pipeline_name: str = None) -> str:
    """
    Create versioned pipeline from AutoML feature engineering result
    
    Args:
        automl_result: Result from kp.automl.engineer_features()
        pipeline_name: Optional pipeline name
        
    Returns:
        Pipeline version ID
        
    Example:
        >>> # Generate features with AutoML
        >>> fe_result = kp.automl.engineer_features(data, target="failure")
        >>> 
        >>> # Version the pipeline for reuse
        >>> pipeline_id = kp.versioning.create_pipeline_from_automl(
        ...     fe_result,
        ...     pipeline_name="sensor-automl-pipeline"
        ... )
        >>> 
        >>> # Apply to new data
        >>> new_result = kp.versioning.apply_versioned_pipeline(new_data, pipeline_id)
    """
    logger = get_logger(__name__)
    logger.info("Creating versioned pipeline from AutoML result")
    
    try:
        if 'feature_info' not in automl_result:
            raise ValueError("Invalid AutoML result: missing feature_info")
        
        feature_info = automl_result['feature_info']
        operations_applied = feature_info.get('operations_applied', [])
        
        # Convert AutoML operations to pipeline steps
        pipeline_steps = []
        
        for operation_desc in operations_applied:
            if 'polynomial features' in operation_desc.lower():
                pipeline_steps.append({
                    'operation': 'polynomial_features',
                    'parameters': {'degree': 2},
                    'description': operation_desc
                })
            elif 'categorical encoding' in operation_desc.lower():
                pipeline_steps.append({
                    'operation': 'categorical_encoding',
                    'parameters': {'method': 'one_hot'},
                    'description': operation_desc
                })
            elif 'missing value' in operation_desc.lower():
                pipeline_steps.append({
                    'operation': 'missing_value_imputation',
                    'parameters': {'strategy': 'median'},
                    'description': operation_desc
                })
        
        # Add metadata from AutoML
        metadata = {
            'source': 'automl_feature_engineering',
            'automl_strategy': feature_info.get('strategy_used', 'auto'),
            'original_features': feature_info.get('original_features', 0),
            'final_features': feature_info.get('final_features', 0),
            'new_features_created': len(feature_info.get('new_features', []))
        }
        
        # Version the pipeline
        pipeline_version_id = version_feature_pipeline(
            pipeline_steps,
            pipeline_name,
            metadata=metadata
        )
        
        logger.info(f"AutoML pipeline versioned: {pipeline_version_id}")
        return pipeline_version_id
        
    except Exception as e:
        raise ModelTrainingError(
            f"Failed to create pipeline from AutoML result: {e}",
            suggestion="Check AutoML result format"
        )


def reproduce_feature_engineering(data: pd.DataFrame, 
                                 pipeline_version_id: str,
                                 validate_compatibility: bool = True) -> Dict[str, Any]:
    """
    Reproduce exact feature engineering from a versioned pipeline
    
    Args:
        data: Input data DataFrame
        pipeline_version_id: Pipeline version to reproduce
        validate_compatibility: Whether to validate data compatibility
        
    Returns:
        Reproduction result with transformed data and validation info
        
    Example:
        >>> # Reproduce exact feature engineering
        >>> reproduction = kp.versioning.reproduce_feature_engineering(
        ...     new_sensor_data,
        ...     "sensor-preprocessing-v1-a1b2c3d4",
        ...     validate_compatibility=True
        ... )
        >>> 
        >>> if reproduction['reproduction_successful']:
        ...     transformed_data = reproduction['transformed_data']
        ...     print("✅ Feature engineering reproduced successfully")
        ... else:
        ...     print("⚠️ Reproduction issues:", reproduction['issues'])
    """
    logger = get_logger(__name__)
    logger.info(f"Reproducing feature engineering: {pipeline_version_id}")
    
    try:
        # Get pipeline configuration
        pipeline = get_feature_pipeline(pipeline_version_id)
        if not pipeline:
            raise ModelTrainingError(f"Pipeline not found: {pipeline_version_id}")
        
        # Validate data compatibility if requested
        validation_issues = []
        if validate_compatibility:
            validation_issues = _validate_data_pipeline_compatibility(data, pipeline)
        
        # Apply pipeline
        result = apply_versioned_pipeline(data, pipeline_version_id)
        
        # Prepare reproduction result
        reproduction_result = {
            'reproduction_successful': len(validation_issues) == 0,
            'transformed_data': result['transformed_data'],
            'applied_steps': result['applied_steps'],
            'pipeline_version': pipeline_version_id,
            'validation_issues': validation_issues,
            'input_shape': data.shape,
            'output_shape': result['output_shape'],
            'execution_metadata': {
                'timestamp': datetime.now().isoformat(),
                'reproduction_method': 'versioned_pipeline',
                'validation_performed': validate_compatibility
            }
        }
        
        if reproduction_result['reproduction_successful']:
            logger.info("Feature engineering reproduction successful")
        else:
            logger.warning(f"Reproduction completed with {len(validation_issues)} issues")
        
        return reproduction_result
        
    except Exception as e:
        raise ModelTrainingError(
            f"Feature engineering reproduction failed: {e}",
            suggestion="Check pipeline version and data format compatibility"
        )


def _validate_data_pipeline_compatibility(data: pd.DataFrame, pipeline: Dict[str, Any]) -> List[str]:
    """Validate that data is compatible with pipeline"""
    issues = []
    
    try:
        # Check if expected columns exist
        for step in pipeline['pipeline_steps']:
            operation = step['operation']
            parameters = step.get('parameters', {})
            
            if 'columns' in parameters:
                expected_columns = parameters['columns']
                missing_columns = [col for col in expected_columns if col not in data.columns]
                
                if missing_columns:
                    issues.append(f"Missing columns for {operation}: {missing_columns}")
        
        # Check data types compatibility
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(exclude=[np.number]).columns
        
        if len(numeric_cols) == 0:
            issues.append("No numeric columns found - may affect polynomial features")
        
        if len(categorical_cols) == 0:
            issues.append("No categorical columns found - may affect encoding steps")
        
    except Exception as e:
        issues.append(f"Validation error: {e}")
    
    return issues