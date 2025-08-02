"""
Base model trainer interface for Kepler framework

Provides abstract base classes and common functionality for ML model training.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, Tuple, List
from pathlib import Path
import pandas as pd
from dataclasses import dataclass
from datetime import datetime
import joblib

from kepler.utils.logging import get_logger
from kepler.utils.exceptions import ModelTrainingError


@dataclass
class TrainingResult:
    """Results from model training"""
    model: Any  # Trained model object
    model_path: Optional[str]  # Path where model was saved
    metrics: Dict[str, float]  # Training metrics
    training_time: float  # Training time in seconds
    feature_names: List[str]  # Names of features used
    target_name: str  # Name of target variable
    preprocessing_info: Dict[str, Any]  # Info about preprocessing applied
    model_type: str  # Type of model trained
    hyperparameters: Dict[str, Any]  # Hyperparameters used


@dataclass
class TrainingConfig:
    """Configuration for model training"""
    algorithm: str = "random_forest"
    test_size: float = 0.2
    random_state: int = 42
    target_column: str = "target"
    feature_columns: Optional[List[str]] = None  # If None, use all except target
    preprocessing: Dict[str, Any] = None  # Preprocessing configuration
    hyperparameters: Dict[str, Any] = None  # Model-specific hyperparameters
    save_model: bool = True
    model_output_path: Optional[str] = None
    cross_validation: bool = False
    cv_folds: int = 5


class BaseModelTrainer(ABC):
    """
    Abstract base class for model trainers
    
    All model trainers in Kepler should inherit from this class.
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize trainer with configuration
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self.model = None
        self.is_trained = False
        self.training_result = None
        
        # Validate configuration
        self._validate_config()
        
        self.logger.info(f"Initialized {self.__class__.__name__} trainer")
    
    def _validate_config(self) -> None:
        """Validate training configuration"""
        if not 0 < self.config.test_size < 1:
            raise ModelTrainingError(
                f"test_size must be between 0 and 1, got {self.config.test_size}",
                suggestion="Set test_size to a value like 0.2 (20% for testing)"
            )
        
        if self.config.cv_folds < 2:
            raise ModelTrainingError(
                f"cv_folds must be >= 2, got {self.config.cv_folds}",
                suggestion="Set cv_folds to at least 2 for cross-validation"
            )
    
    @abstractmethod
    def _create_model(self) -> Any:
        """
        Create and return the ML model instance
        
        Returns:
            Initialized model object
        """
        pass
    
    @abstractmethod
    def _get_default_hyperparameters(self) -> Dict[str, Any]:
        """
        Get default hyperparameters for this model type
        
        Returns:
            Dictionary of default hyperparameters
        """
        pass
    
    def _prepare_hyperparameters(self) -> Dict[str, Any]:
        """
        Prepare final hyperparameters by merging defaults with config
        
        Returns:
            Dictionary of hyperparameters to use
        """
        defaults = self._get_default_hyperparameters()
        
        if self.config.hyperparameters:
            # Merge config hyperparameters with defaults
            hyperparams = {**defaults, **self.config.hyperparameters}
        else:
            hyperparams = defaults
        
        # Add random state if supported
        if 'random_state' in hyperparams:
            hyperparams['random_state'] = self.config.random_state
        
        return hyperparams
    
    def _prepare_data(self, 
                     df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Prepare data for training (split features/target, train/test)
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        from sklearn.model_selection import train_test_split
        
        # Validate target column exists
        if self.config.target_column not in df.columns:
            raise ModelTrainingError(
                f"Target column '{self.config.target_column}' not found in data",
                data_info=f"Available columns: {list(df.columns)}",
                suggestion="Check your target column name in the configuration"
            )
        
        # Prepare features and target
        if self.config.feature_columns:
            # Use specified feature columns
            missing_cols = [col for col in self.config.feature_columns if col not in df.columns]
            if missing_cols:
                raise ModelTrainingError(
                    f"Feature columns not found in data: {missing_cols}",
                    data_info=f"Available columns: {list(df.columns)}",
                    suggestion="Check your feature column names"
                )
            X = df[self.config.feature_columns].copy()
        else:
            # Use all columns except target
            feature_cols = [col for col in df.columns if col != self.config.target_column]
            X = df[feature_cols].copy()
        
        y = df[self.config.target_column].copy()
        
        self.logger.info(f"Prepared features: {list(X.columns)}")
        self.logger.info(f"Target: {self.config.target_column}")
        self.logger.info(f"Data shape: {X.shape}, Target shape: {y.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y if self._is_classification(y) else None
        )
        
        self.logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def _is_classification(self, y: pd.Series) -> bool:
        """
        Determine if this is a classification problem
        
        Args:
            y: Target series
            
        Returns:
            True if classification, False if regression
        """
        # Simple heuristic: if target has few unique values relative to size, it's classification
        unique_ratio = y.nunique() / len(y)
        is_numeric_with_few_classes = (
            pd.api.types.is_numeric_dtype(y) and 
            y.nunique() <= 20 and 
            unique_ratio < 0.1
        )
        is_categorical = y.dtype in ['object', 'category']
        
        return is_categorical or is_numeric_with_few_classes
    
    def _evaluate_model(self, 
                       model: Any, 
                       X_test: pd.DataFrame, 
                       y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate trained model and return metrics
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary of evaluation metrics
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            mean_squared_error, mean_absolute_error, r2_score
        )
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        metrics = {}
        
        if self._is_classification(y_test):
            # Classification metrics
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
            
            # Handle multiclass vs binary
            if y_test.nunique() > 2:
                metrics['precision'] = precision_score(y_test, y_pred, average='weighted')
                metrics['recall'] = recall_score(y_test, y_pred, average='weighted')
                metrics['f1'] = f1_score(y_test, y_pred, average='weighted')
            else:
                metrics['precision'] = precision_score(y_test, y_pred)
                metrics['recall'] = recall_score(y_test, y_pred)
                metrics['f1'] = f1_score(y_test, y_pred)
        else:
            # Regression metrics
            metrics['mse'] = mean_squared_error(y_test, y_pred)
            metrics['mae'] = mean_absolute_error(y_test, y_pred)
            metrics['r2'] = r2_score(y_test, y_pred)
            metrics['rmse'] = metrics['mse'] ** 0.5
        
        self.logger.info(f"Model evaluation metrics: {metrics}")
        return metrics
    
    def _save_model(self, model: Any, X_train: pd.DataFrame) -> Optional[str]:
        """
        Save trained model to disk
        
        Args:
            model: Trained model to save
            X_train: Training features (for feature names)
            
        Returns:
            Path where model was saved, or None if not saved
        """
        if not self.config.save_model:
            return None
        
        # Determine output path
        if self.config.model_output_path:
            model_path = Path(self.config.model_output_path)
        else:
            # Default naming
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = Path(f"model_{self.config.algorithm}_{timestamp}.pkl")
        
        # Ensure directory exists
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare model metadata
        model_metadata = {
            'model': model,
            'feature_names': list(X_train.columns),
            'target_name': self.config.target_column,
            'model_type': self.config.algorithm,
            'hyperparameters': self._prepare_hyperparameters(),
            'training_timestamp': datetime.now().isoformat(),
            'kepler_version': '0.1.0'  # TODO: Get from package
        }
        
        # Save using joblib
        joblib.dump(model_metadata, model_path)
        
        self.logger.info(f"Model saved to: {model_path}")
        return str(model_path)
    
    def train(self, df: pd.DataFrame) -> TrainingResult:
        """
        Train the model on provided data
        
        Args:
            df: Training data DataFrame
            
        Returns:
            TrainingResult with model and metrics
            
        Raises:
            ModelTrainingError: If training fails
        """
        self.logger.info(f"Starting model training with {self.config.algorithm}")
        start_time = datetime.now()
        
        try:
            # Prepare data
            X_train, X_test, y_train, y_test = self._prepare_data(df)
            
            # Create model with hyperparameters
            hyperparams = self._prepare_hyperparameters()
            self.logger.info(f"Using hyperparameters: {hyperparams}")
            
            model = self._create_model()
            model.set_params(**hyperparams)
            
            # Train model
            self.logger.info("Training model...")
            model.fit(X_train, y_train)
            
            # Evaluate model
            metrics = self._evaluate_model(model, X_test, y_test)
            
            # Save model
            model_path = self._save_model(model, X_train)
            
            # Calculate training time
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = TrainingResult(
                model=model,
                model_path=model_path,
                metrics=metrics,
                training_time=training_time,
                feature_names=list(X_train.columns),
                target_name=self.config.target_column,
                preprocessing_info={},  # TODO: Add preprocessing info
                model_type=self.config.algorithm,
                hyperparameters=hyperparams
            )
            
            self.model = model
            self.is_trained = True
            self.training_result = result
            
            self.logger.info(f"Training completed in {training_time:.2f} seconds")
            return result
            
        except Exception as e:
            if isinstance(e, ModelTrainingError):
                raise
            else:
                raise ModelTrainingError(
                    f"Model training failed: {e}",
                    suggestion="Check your data format and configuration"
                ) from e
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Make predictions with trained model
        
        Args:
            X: Features for prediction
            
        Returns:
            Predictions as pandas Series
            
        Raises:
            ModelTrainingError: If model not trained or prediction fails
        """
        if not self.is_trained or self.model is None:
            raise ModelTrainingError(
                "Model has not been trained yet",
                suggestion="Call train() method first"
            )
        
        try:
            predictions = self.model.predict(X)
            return pd.Series(predictions, index=X.index)
            
        except Exception as e:
            raise ModelTrainingError(
                f"Prediction failed: {e}",
                suggestion="Check that input features match training data"
            ) from e


def load_model(model_path: str) -> Dict[str, Any]:
    """
    Load a saved Kepler model
    
    Args:
        model_path: Path to saved model file
        
    Returns:
        Dictionary with model and metadata
        
    Raises:
        ModelTrainingError: If loading fails
    """
    try:
        model_data = joblib.load(model_path)
        
        # Validate model data structure
        required_keys = ['model', 'feature_names', 'target_name', 'model_type']
        missing_keys = [key for key in required_keys if key not in model_data]
        if missing_keys:
            raise ModelTrainingError(
                f"Invalid model file: missing keys {missing_keys}",
                suggestion="Model file may be corrupted or from incompatible version"
            )
        
        return model_data
        
    except Exception as e:
        if isinstance(e, ModelTrainingError):
            raise
        else:
            raise ModelTrainingError(
                f"Failed to load model from {model_path}: {e}",
                suggestion="Check that the file exists and is a valid Kepler model"
            ) from e