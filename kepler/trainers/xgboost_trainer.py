"""
XGBoost Trainer for Kepler Framework

Implements XGBoost training with sklearn-like interface following official documentation.
Supports both XGBClassifier and XGBRegressor with comprehensive parameter configuration.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import joblib
from pathlib import Path

from kepler.trainers.base import BaseModelTrainer, TrainingResult, TrainingConfig
from kepler.utils.logging import get_logger
from kepler.utils.exceptions import ModelTrainingError

# Dynamic import of XGBoost (might not be installed)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class XGBoostTrainer(BaseModelTrainer):
    """
    XGBoost trainer implementing official XGBoost patterns
    
    Supports:
    - Binary and multi-class classification (XGBClassifier)
    - Regression (XGBRegressor)
    - Early stopping and cross-validation
    - GPU acceleration (if available)
    - Model saving in JSON format (recommended by XGBoost)
    - Feature importance analysis
    """
    
    def __init__(self, config: TrainingConfig):
        if not XGBOOST_AVAILABLE:
            raise ModelTrainingError(
                "XGBoost not available",
                suggestion="Install with: kepler libs install --library xgboost"
            )
        
        super().__init__(config)
        self.model_type = self._determine_model_type()
        
    def _determine_model_type(self) -> str:
        """Determine if this is classification or regression based on config"""
        if self.config.task_type == "auto":
            # Will be determined during training based on target data
            return "auto"
        elif self.config.task_type in ["classification", "binary", "multiclass"]:
            return "classification"
        elif self.config.task_type == "regression":
            return "regression"
        else:
            raise ModelTrainingError(f"Unsupported task type: {self.config.task_type}")
    
    def _create_model(self):
        """Create XGBoost model based on task type"""
        base_params = {
            'random_state': self.config.random_state,
            'n_jobs': -1,  # Use all available cores
            'tree_method': 'hist',  # Recommended by XGBoost docs
        }
        
        if self.model_type == "classification":
            return xgb.XGBClassifier(**base_params)
        elif self.model_type == "regression":
            return xgb.XGBRegressor(**base_params)
        else:
            raise ModelTrainingError(f"Cannot create model for type: {self.model_type}")
    
    def _prepare_hyperparameters(self) -> Dict[str, Any]:
        """Prepare XGBoost-specific hyperparameters"""
        # Default XGBoost hyperparameters following official recommendations
        hyperparams = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.3,
            'subsample': 1.0,
            'colsample_bytree': 1.0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'min_child_weight': 1,
            'gamma': 0
        }
        
        # Task-specific parameters
        if self.model_type == "classification":
            # Binary classification by default, will auto-adjust for multiclass
            hyperparams['objective'] = 'binary:logistic'
            hyperparams['eval_metric'] = 'logloss'
        elif self.model_type == "regression":
            hyperparams['objective'] = 'reg:squarederror'
            hyperparams['eval_metric'] = 'rmse'
        
        # Override with user-provided hyperparameters
        if self.config.hyperparameters:
            hyperparams.update(self.config.hyperparameters)
        
        return hyperparams
    
    def _prepare_data(self, df: pd.DataFrame) -> tuple:
        """Prepare data for XGBoost training"""
        # Handle missing target column
        if self.config.target_column not in df.columns:
            raise ModelTrainingError(
                f"Target column '{self.config.target_column}' not found in data",
                data_info=f"Available columns: {list(df.columns)}",
                suggestion="Check your target column name"
            )
        
        # Separate features and target
        X = df.drop(columns=[self.config.target_column])
        y = df[self.config.target_column]
        
        # Auto-detect task type if needed
        if self.model_type == "auto":
            unique_values = y.nunique()
            if unique_values <= 10 and y.dtype in ['object', 'category', 'bool']:
                self.model_type = "classification"
                self.logger.info(f"Auto-detected task: classification ({unique_values} unique values)")
            elif unique_values <= 20:
                self.model_type = "classification"
                self.logger.info(f"Auto-detected task: classification ({unique_values} unique values)")
            else:
                self.model_type = "regression"
                self.logger.info(f"Auto-detected task: regression ({unique_values} unique values)")
        
        # Handle categorical variables (XGBoost can handle them natively)
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns
        if len(categorical_columns) > 0:
            self.logger.info(f"Found {len(categorical_columns)} categorical columns: {list(categorical_columns)}")
            # XGBoost can handle string categoricals directly with enable_categorical=True
            for col in categorical_columns:
                X[col] = X[col].astype('category')
        
        # Handle missing values (XGBoost can handle them, but let's be explicit)
        missing_counts = X.isnull().sum()
        if missing_counts.sum() > 0:
            self.logger.info(f"Found missing values in {(missing_counts > 0).sum()} columns")
            # XGBoost handles missing values natively, so we keep them
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y if self.model_type == "classification" and y.nunique() > 1 else None
        )
        
        self.logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def _evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate XGBoost model performance"""
        predictions = model.predict(X_test)
        
        metrics = {}
        
        if self.model_type == "classification":
            # Classification metrics
            accuracy = accuracy_score(y_test, predictions)
            metrics['accuracy'] = accuracy
            
            # Get probabilities for additional metrics
            try:
                probabilities = model.predict_proba(X_test)
                if probabilities.shape[1] == 2:
                    # Binary classification
                    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
                    
                    metrics['roc_auc'] = roc_auc_score(y_test, probabilities[:, 1])
                    metrics['precision'] = precision_score(y_test, predictions, average='weighted', zero_division=0)
                    metrics['recall'] = recall_score(y_test, predictions, average='weighted', zero_division=0)
                    metrics['f1_score'] = f1_score(y_test, predictions, average='weighted', zero_division=0)
                else:
                    # Multi-class classification
                    from sklearn.metrics import precision_score, recall_score, f1_score
                    
                    metrics['precision'] = precision_score(y_test, predictions, average='weighted', zero_division=0)
                    metrics['recall'] = recall_score(y_test, predictions, average='weighted', zero_division=0)
                    metrics['f1_score'] = f1_score(y_test, predictions, average='weighted', zero_division=0)
                    
            except Exception as e:
                self.logger.warning(f"Could not compute additional classification metrics: {e}")
            
            # Log classification report
            report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
            self.logger.info("Classification Report:")
            for class_name, class_metrics in report.items():
                if isinstance(class_metrics, dict):
                    self.logger.info(f"  {class_name}: precision={class_metrics.get('precision', 0):.3f}, "
                                   f"recall={class_metrics.get('recall', 0):.3f}, "
                                   f"f1-score={class_metrics.get('f1-score', 0):.3f}")
            
        else:
            # Regression metrics
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, predictions)
            
            metrics['mse'] = mse
            metrics['rmse'] = rmse
            metrics['r2_score'] = r2
            
            # Mean Absolute Error
            mae = np.mean(np.abs(y_test - predictions))
            metrics['mae'] = mae
            
            self.logger.info(f"Regression Metrics: RMSE={rmse:.4f}, R²={r2:.4f}, MAE={mae:.4f}")
        
        # XGBoost-specific metrics
        try:
            # Feature importance
            feature_importance = model.feature_importances_
            if hasattr(model, 'feature_names_in_'):
                feature_names = model.feature_names_in_
            else:
                feature_names = [f"feature_{i}" for i in range(len(feature_importance))]
            
            # Log top 5 most important features
            importance_pairs = list(zip(feature_names, feature_importance))
            importance_pairs.sort(key=lambda x: x[1], reverse=True)
            
            self.logger.info("Top 5 Feature Importances:")
            for name, importance in importance_pairs[:5]:
                self.logger.info(f"  {name}: {importance:.4f}")
                
            metrics['feature_importance_top'] = dict(importance_pairs[:5])
            
        except Exception as e:
            self.logger.warning(f"Could not compute feature importance: {e}")
        
        return metrics
    
    def _save_model(self, model, X_train: pd.DataFrame) -> Optional[str]:
        """Save XGBoost model using recommended JSON format"""
        if not self.config.save_model:
            return None
        
        try:
            # Create models directory
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)
            
            # Generate model filename
            timestamp = self._get_timestamp()
            model_filename = f"xgboost_{self.config.algorithm}_{timestamp}.json"
            model_path = models_dir / model_filename
            
            # Save using XGBoost's native JSON format (recommended)
            model.save_model(str(model_path))
            
            # Also save metadata
            metadata = {
                'model_type': self.model_type,
                'algorithm': 'xgboost',
                'target_column': self.config.target_column,
                'feature_columns': list(X_train.columns),
                'training_timestamp': timestamp,
                'xgboost_version': xgb.__version__,
                'hyperparameters': model.get_params()
            }
            
            metadata_path = models_dir / f"xgboost_{self.config.algorithm}_{timestamp}_metadata.json"
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Model saved to {model_path}")
            self.logger.info(f"Metadata saved to {metadata_path}")
            
            return str(model_path)
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            return None


class LightGBMTrainer(BaseModelTrainer):
    """
    LightGBM trainer for gradient boosting
    
    Note: This is a placeholder for future implementation.
    Will be implemented when LightGBM integration is prioritized.
    """
    
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        raise ModelTrainingError(
            "LightGBM trainer not yet implemented",
            suggestion="Use XGBoost trainer for now: kp.train.xgboost()"
        )
    
    def _create_model(self):
        pass
    
    def _prepare_hyperparameters(self) -> Dict[str, Any]:
        pass


class CatBoostTrainer(BaseModelTrainer):
    """
    CatBoost trainer for gradient boosting with categorical features
    
    Note: This is a placeholder for future implementation.
    Will be implemented when CatBoost integration is prioritized.
    """
    
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        raise ModelTrainingError(
            "CatBoost trainer not yet implemented",
            suggestion="Use XGBoost trainer for now: kp.train.xgboost()"
        )
    
    def _create_model(self):
        pass
    
    def _prepare_hyperparameters(self) -> Dict[str, Any]:
        pass


def create_xgboost_trainer(config: TrainingConfig) -> XGBoostTrainer:
    """
    Create XGBoost trainer with configuration
    
    Args:
        config: Training configuration
        
    Returns:
        Configured XGBoost trainer
        
    Example:
        from kepler.trainers.base import TrainingConfig
        from kepler.trainers.xgboost_trainer import create_xgboost_trainer
        
        config = TrainingConfig(
            algorithm="xgboost",
            target_column="target",
            task_type="classification"
        )
        
        trainer = create_xgboost_trainer(config)
        result = trainer.train(data)
    """
    return XGBoostTrainer(config)


def create_lightgbm_trainer(config: TrainingConfig) -> LightGBMTrainer:
    """Create LightGBM trainer (placeholder for future implementation)"""
    return LightGBMTrainer(config)


def create_catboost_trainer(config: TrainingConfig) -> CatBoostTrainer:
    """Create CatBoost trainer (placeholder for future implementation)"""
    return CatBoostTrainer(config)
