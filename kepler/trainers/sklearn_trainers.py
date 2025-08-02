"""
Scikit-learn model trainers for Kepler framework

Implements specific trainers for sklearn models.
"""

from typing import Dict, Any
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

from kepler.trainers.base import BaseModelTrainer, TrainingConfig
from kepler.utils.logging import get_logger
from kepler.utils.exceptions import ModelTrainingError


class RandomForestTrainer(BaseModelTrainer):
    """
    Random Forest trainer supporting both classification and regression
    
    Automatically detects whether to use RandomForestClassifier or RandomForestRegressor
    based on the target variable characteristics.
    """
    
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self._model_class = None  # Will be determined during training
    
    def _get_default_hyperparameters(self) -> Dict[str, Any]:
        """Get default hyperparameters for Random Forest"""
        return {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',  # 'sqrt' for classification, 'log2' also good
            'bootstrap': True,
            'random_state': 42,
            'n_jobs': -1,  # Use all available cores
            'class_weight': None  # Only used for classification
        }
    
    def _create_model(self) -> Any:
        """
        Create Random Forest model
        
        Model type (classifier vs regressor) will be determined during training
        when we have access to the target variable.
        """
        # This will be called again during training with proper model class
        if self._model_class is None:
            return RandomForestRegressor()  # Default, will be replaced
        
        return self._model_class()
    
    def _determine_model_class(self, y):
        """Determine whether to use classifier or regressor based on target"""
        if self._is_classification(y):
            self.logger.info("Detected classification problem - using RandomForestClassifier")
            return RandomForestClassifier
        else:
            self.logger.info("Detected regression problem - using RandomForestRegressor")
            return RandomForestRegressor
    
    def train(self, df):
        """Override train to determine model class based on target"""
        # Quick peek at target to determine model class
        if self.config.target_column not in df.columns:
            raise ModelTrainingError(
                f"Target column '{self.config.target_column}' not found in data"
            )
        
        y_sample = df[self.config.target_column]
        self._model_class = self._determine_model_class(y_sample)
        
        # Call parent train method
        return super().train(df)
    
    def _prepare_hyperparameters(self) -> Dict[str, Any]:
        """
        Prepare hyperparameters, filtering out invalid ones based on model type
        """
        hyperparams = super()._prepare_hyperparameters()
        
        # Filter out parameters that don't apply to the current model class
        if self._model_class == RandomForestRegressor:
            # Remove classification-only parameters
            hyperparams.pop('class_weight', None)
        elif self._model_class == RandomForestClassifier:
            # Classification parameters are fine
            pass
        
        return hyperparams


class LinearModelTrainer(BaseModelTrainer):
    """
    Linear model trainer supporting both regression and classification
    
    Uses LinearRegression for regression and LogisticRegression for classification.
    """
    
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self._model_class = None
    
    def _get_default_hyperparameters(self) -> Dict[str, Any]:
        """Get default hyperparameters for Linear models"""
        return {
            'random_state': 42,
            'max_iter': 1000,  # For LogisticRegression
            'fit_intercept': True,
            'class_weight': None  # Only for classification
        }
    
    def _create_model(self) -> Any:
        """Create Linear model"""
        if self._model_class is None:
            return LinearRegression()  # Default
        return self._model_class()
    
    def _determine_model_class(self, y):
        """Determine whether to use LinearRegression or LogisticRegression"""
        if self._is_classification(y):
            self.logger.info("Detected classification problem - using LogisticRegression")
            return LogisticRegression
        else:
            self.logger.info("Detected regression problem - using LinearRegression")
            return LinearRegression
    
    def train(self, df):
        """Override train to determine model class based on target"""
        if self.config.target_column not in df.columns:
            raise ModelTrainingError(
                f"Target column '{self.config.target_column}' not found in data"
            )
        
        y_sample = df[self.config.target_column]
        self._model_class = self._determine_model_class(y_sample)
        
        return super().train(df)
    
    def _prepare_hyperparameters(self) -> Dict[str, Any]:
        """
        Prepare hyperparameters, filtering out invalid ones based on model type
        """
        hyperparams = super()._prepare_hyperparameters()
        
        # Filter out parameters that don't apply to the current model class
        if self._model_class == LinearRegression:
            # Remove classification-only parameters and parameters not supported by LinearRegression
            hyperparams.pop('class_weight', None)
            hyperparams.pop('max_iter', None)  # LinearRegression doesn't have max_iter
            hyperparams.pop('random_state', None)  # LinearRegression doesn't have random_state
        elif self._model_class == LogisticRegression:
            # Logistic regression parameters are fine
            pass
        
        return hyperparams


# Factory function to create trainers
def create_trainer(algorithm: str, config: TrainingConfig) -> BaseModelTrainer:
    """
    Factory function to create model trainers
    
    Args:
        algorithm: Algorithm name ('random_forest', 'linear')
        config: Training configuration
        
    Returns:
        Initialized trainer instance
        
    Raises:
        ModelTrainingError: If algorithm not supported
    """
    algorithm = algorithm.lower()
    
    if algorithm in ['random_forest', 'rf']:
        return RandomForestTrainer(config)
    elif algorithm in ['linear', 'linear_regression', 'logistic_regression']:
        return LinearModelTrainer(config)
    else:
        supported_algorithms = ['random_forest', 'rf', 'linear', 'linear_regression', 'logistic_regression']
        raise ModelTrainingError(
            f"Unsupported algorithm: {algorithm}",
            suggestion=f"Supported algorithms: {supported_algorithms}"
        )