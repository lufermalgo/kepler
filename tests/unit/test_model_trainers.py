"""
Unit tests for Kepler model trainers

Tests the base trainer class and sklearn trainers.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from pathlib import Path

from kepler.trainers.base import (
    BaseModelTrainer, TrainingConfig, TrainingResult, load_model
)
from kepler.trainers.sklearn_trainers import (
    RandomForestTrainer, LinearModelTrainer, create_trainer
)
from kepler.utils.exceptions import ModelTrainingError


class TestTrainingConfig:
    """Test TrainingConfig dataclass"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = TrainingConfig()
        
        assert config.algorithm == "random_forest"
        assert config.test_size == 0.2
        assert config.random_state == 42
        assert config.target_column == "target"
        assert config.feature_columns is None
        assert config.save_model is True
        assert config.cross_validation is False
        assert config.cv_folds == 5
    
    def test_custom_config(self):
        """Test custom configuration values"""
        config = TrainingConfig(
            algorithm="linear",
            test_size=0.3,
            target_column="label",
            feature_columns=["f1", "f2"],
            hyperparameters={"n_estimators": 200}
        )
        
        assert config.algorithm == "linear"
        assert config.test_size == 0.3
        assert config.target_column == "label"
        assert config.feature_columns == ["f1", "f2"]
        assert config.hyperparameters == {"n_estimators": 200}


class MockTrainer(BaseModelTrainer):
    """Mock trainer for testing BaseModelTrainer"""
    
    def _create_model(self):
        """Create mock model"""
        model = Mock()
        model.fit = Mock()
        model.predict = Mock(return_value=np.array([1, 0, 1]))
        model.set_params = Mock()
        return model
    
    def _get_default_hyperparameters(self):
        """Get mock hyperparameters"""
        return {"param1": "value1", "random_state": 42}


class TestBaseModelTrainer:
    """Test BaseModelTrainer abstract class"""
    
    def test_initialization(self):
        """Test trainer initialization"""
        config = TrainingConfig()
        trainer = MockTrainer(config)
        
        assert trainer.config == config
        assert trainer.model is None
        assert trainer.is_trained is False
        assert trainer.training_result is None
    
    def test_config_validation_invalid_test_size(self):
        """Test config validation with invalid test size"""
        config = TrainingConfig(test_size=1.5)
        
        with pytest.raises(ModelTrainingError) as exc_info:
            MockTrainer(config)
        
        assert "test_size must be between 0 and 1" in str(exc_info.value)
    
    def test_config_validation_invalid_cv_folds(self):
        """Test config validation with invalid CV folds"""
        config = TrainingConfig(cv_folds=1)
        
        with pytest.raises(ModelTrainingError) as exc_info:
            MockTrainer(config)
        
        assert "cv_folds must be >= 2" in str(exc_info.value)
    
    def test_prepare_hyperparameters_defaults(self):
        """Test hyperparameter preparation with defaults"""
        config = TrainingConfig()
        trainer = MockTrainer(config)
        
        hyperparams = trainer._prepare_hyperparameters()
        
        assert hyperparams["param1"] == "value1"
        assert hyperparams["random_state"] == 42
    
    def test_prepare_hyperparameters_with_config(self):
        """Test hyperparameter preparation with config overrides"""
        config = TrainingConfig(hyperparameters={"param1": "override", "param2": "new"})
        trainer = MockTrainer(config)
        
        hyperparams = trainer._prepare_hyperparameters()
        
        assert hyperparams["param1"] == "override"  # Overridden
        assert hyperparams["param2"] == "new"       # New parameter
        assert hyperparams["random_state"] == 42    # Default preserved
    
    def test_is_classification_categorical(self):
        """Test classification detection with categorical data"""
        config = TrainingConfig()
        trainer = MockTrainer(config)
        
        y_cat = pd.Series(['A', 'B', 'A', 'B', 'C'])
        assert trainer._is_classification(y_cat) is True
    
    def test_is_classification_numeric_few_classes(self):
        """Test classification detection with numeric few classes"""
        config = TrainingConfig()
        trainer = MockTrainer(config)
        
        y_numeric = pd.Series([0, 1, 0, 1, 0] * 20)  # Low unique ratio
        assert trainer._is_classification(y_numeric) is True
    
    def test_is_classification_regression(self):
        """Test classification detection with regression data"""
        config = TrainingConfig()
        trainer = MockTrainer(config)
        
        y_regression = pd.Series(np.random.normal(0, 1, 100))
        assert trainer._is_classification(y_regression) is False
    
    def test_prepare_data_basic(self):
        """Test basic data preparation"""
        # Create sample data
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'feature2': [10, 20, 30, 40],
            'target': [0, 1, 0, 1]
        })
        
        config = TrainingConfig(target_column='target')
        trainer = MockTrainer(config)
        
        X_train, X_test, y_train, y_test = trainer._prepare_data(df)
        
        # Check that data was split
        assert len(X_train) + len(X_test) == len(df)
        assert len(y_train) + len(y_test) == len(df)
        
        # Check feature columns
        expected_features = ['feature1', 'feature2']
        assert list(X_train.columns) == expected_features
        assert list(X_test.columns) == expected_features
    
    def test_prepare_data_specific_features(self):
        """Test data preparation with specific feature columns"""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'feature2': [10, 20, 30, 40], 
            'feature3': [100, 200, 300, 400],
            'target': [0, 1, 0, 1]
        })
        
        config = TrainingConfig(
            target_column='target',
            feature_columns=['feature1', 'feature3']
        )
        trainer = MockTrainer(config)
        
        X_train, X_test, y_train, y_test = trainer._prepare_data(df)
        
        # Check only specified features are included
        expected_features = ['feature1', 'feature3']
        assert list(X_train.columns) == expected_features
    
    def test_prepare_data_missing_target(self):
        """Test data preparation with missing target column"""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'feature2': [10, 20, 30, 40]
        })
        
        config = TrainingConfig(target_column='missing_target')
        trainer = MockTrainer(config)
        
        with pytest.raises(ModelTrainingError) as exc_info:
            trainer._prepare_data(df)
        
        assert "Target column 'missing_target' not found" in str(exc_info.value)
    
    def test_prepare_data_missing_features(self):
        """Test data preparation with missing feature columns"""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'target': [0, 1, 0, 1]
        })
        
        config = TrainingConfig(
            target_column='target',
            feature_columns=['feature1', 'missing_feature']
        )
        trainer = MockTrainer(config)
        
        with pytest.raises(ModelTrainingError) as exc_info:
            trainer._prepare_data(df)
        
        assert "Feature columns not found" in str(exc_info.value)
    
    @patch('kepler.trainers.base.joblib.dump')
    def test_save_model(self, mock_dump):
        """Test model saving"""
        config = TrainingConfig(save_model=True, model_output_path="test_model.pkl")
        trainer = MockTrainer(config)
        
        model = Mock()
        X_train = pd.DataFrame({'feature1': [1, 2], 'feature2': [10, 20]})
        
        result_path = trainer._save_model(model, X_train)
        
        assert result_path == "test_model.pkl"
        mock_dump.assert_called_once()
        
        # Check the metadata passed to joblib.dump
        call_args = mock_dump.call_args
        metadata = call_args[0][0]  # First argument to dump()
        
        assert metadata['model'] == model
        assert metadata['feature_names'] == ['feature1', 'feature2']
        assert metadata['target_name'] == 'target'
        assert metadata['model_type'] == 'random_forest'
    
    def test_save_model_disabled(self):
        """Test that model is not saved when disabled"""
        config = TrainingConfig(save_model=False)
        trainer = MockTrainer(config)
        
        model = Mock()
        X_train = pd.DataFrame({'feature1': [1, 2]})
        
        result_path = trainer._save_model(model, X_train)
        
        assert result_path is None
    
    def test_predict_not_trained(self):
        """Test prediction with untrained model"""
        config = TrainingConfig()
        trainer = MockTrainer(config)
        
        X = pd.DataFrame({'feature1': [1, 2]})
        
        with pytest.raises(ModelTrainingError) as exc_info:
            trainer.predict(X)
        
        assert "Model has not been trained yet" in str(exc_info.value)


class TestRandomForestTrainer:
    """Test RandomForestTrainer"""
    
    def test_initialization(self):
        """Test RandomForest trainer initialization"""
        config = TrainingConfig()
        trainer = RandomForestTrainer(config)
        
        assert trainer._model_class is None
    
    def test_default_hyperparameters(self):
        """Test default hyperparameters"""
        config = TrainingConfig()
        trainer = RandomForestTrainer(config)
        
        hyperparams = trainer._get_default_hyperparameters()
        
        assert hyperparams['n_estimators'] == 100
        assert hyperparams['max_features'] == 'sqrt'
        assert hyperparams['random_state'] == 42
        assert hyperparams['n_jobs'] == -1


class TestLinearModelTrainer:
    """Test LinearModelTrainer"""
    
    def test_initialization(self):
        """Test Linear model trainer initialization"""
        config = TrainingConfig()
        trainer = LinearModelTrainer(config)
        
        assert trainer._model_class is None
    
    def test_default_hyperparameters(self):
        """Test default hyperparameters"""
        config = TrainingConfig()
        trainer = LinearModelTrainer(config)
        
        hyperparams = trainer._get_default_hyperparameters()
        
        assert hyperparams['random_state'] == 42
        assert hyperparams['max_iter'] == 1000
        assert hyperparams['fit_intercept'] is True


class TestCreateTrainer:
    """Test trainer factory function"""
    
    def test_create_random_forest_trainer(self):
        """Test creating RandomForest trainer"""
        config = TrainingConfig()
        
        trainer = create_trainer("random_forest", config)
        
        assert isinstance(trainer, RandomForestTrainer)
    
    def test_create_random_forest_trainer_alias(self):
        """Test creating RandomForest trainer with alias"""
        config = TrainingConfig()
        
        trainer = create_trainer("rf", config)
        
        assert isinstance(trainer, RandomForestTrainer)
    
    def test_create_linear_trainer(self):
        """Test creating Linear trainer"""
        config = TrainingConfig()
        
        trainer = create_trainer("linear", config)
        
        assert isinstance(trainer, LinearModelTrainer)
    
    def test_create_unsupported_trainer(self):
        """Test creating unsupported trainer"""
        config = TrainingConfig()
        
        with pytest.raises(ModelTrainingError) as exc_info:
            create_trainer("unsupported_algorithm", config)
        
        assert "Unsupported algorithm" in str(exc_info.value)


class TestLoadModel:
    """Test model loading functionality"""
    
    @patch('kepler.trainers.base.joblib.load')
    def test_load_model_success(self, mock_load):
        """Test successful model loading"""
        # Mock the loaded model data
        mock_model_data = {
            'model': Mock(),
            'feature_names': ['feature1', 'feature2'],
            'target_name': 'target',
            'model_type': 'random_forest',
            'hyperparameters': {'n_estimators': 100}
        }
        mock_load.return_value = mock_model_data
        
        result = load_model("test_model.pkl")
        
        assert result == mock_model_data
        mock_load.assert_called_once_with("test_model.pkl")
    
    @patch('kepler.trainers.base.joblib.load')
    def test_load_model_missing_keys(self, mock_load):
        """Test loading model with missing required keys"""
        # Mock incomplete model data
        mock_model_data = {
            'model': Mock(),
            'feature_names': ['feature1', 'feature2']
            # Missing 'target_name' and 'model_type'
        }
        mock_load.return_value = mock_model_data
        
        with pytest.raises(ModelTrainingError) as exc_info:
            load_model("test_model.pkl")
        
        assert "Invalid model file: missing keys" in str(exc_info.value)
    
    @patch('kepler.trainers.base.joblib.load')
    def test_load_model_file_error(self, mock_load):
        """Test loading model with file error"""
        mock_load.side_effect = FileNotFoundError("File not found")
        
        with pytest.raises(ModelTrainingError) as exc_info:
            load_model("nonexistent_model.pkl")
        
        assert "Failed to load model" in str(exc_info.value)