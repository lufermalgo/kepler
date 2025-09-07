"""
Unit tests for XGBoost integration in Kepler Framework

Tests XGBoost training functionality and integration with unlimited library system.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from kepler.train import xgboost, KeplerModel
from kepler.utils.exceptions import ModelTrainingError


class TestXGBoostIntegration:
    """Test XGBoost training integration"""
    
    def setup_method(self):
        """Setup test data"""
        # Create synthetic classification dataset
        np.random.seed(42)
        n_samples = 1000  # Larger dataset for proper stratification
        n_features = 5
        
        # Features
        X = np.random.randn(n_samples, n_features)
        
        # Binary classification target
        y_binary = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        # Regression target
        y_regression = X[:, 0] * 2 + X[:, 1] * 1.5 + np.random.normal(0, 0.1, n_samples)
        
        # Create DataFrames
        feature_names = [f"feature_{i}" for i in range(n_features)]
        
        self.classification_data = pd.DataFrame(X, columns=feature_names)
        self.classification_data['target'] = y_binary
        
        self.regression_data = pd.DataFrame(X, columns=feature_names)
        self.regression_data['target'] = y_regression
        
        # Categorical data
        self.categorical_data = self.classification_data.copy()
        self.categorical_data['category'] = np.random.choice(['A', 'B', 'C'], n_samples)
        
    def test_xgboost_available_check(self):
        """Test that XGBoost availability is checked properly"""
        
        with patch('kepler.core.library_manager.LibraryManager.dynamic_import') as mock_import:
            mock_import.side_effect = Exception("XGBoost not found")
            
            with pytest.raises(ModelTrainingError) as exc_info:
                xgboost(self.classification_data, target="target")
            
        assert "XGBoost not available" in str(exc_info.value)
        # Check suggestion separately
        assert hasattr(exc_info.value, 'suggestion')
        assert "kepler libs install" in exc_info.value.suggestion
    
    @patch('kepler.core.library_manager.LibraryManager.dynamic_import')
    def test_xgboost_binary_classification(self, mock_import):
        """Test XGBoost binary classification training"""
        
        # Mock XGBoost import
        mock_xgb = MagicMock()
        mock_classifier = MagicMock()
        mock_classifier.fit.return_value = None
        # Mock will be configured per test
        mock_classifier.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.2, 0.8]])
        mock_classifier.feature_importances_ = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
        
        mock_xgb.XGBClassifier.return_value = mock_classifier
        mock_import.return_value = mock_xgb
        
        # Train model
        model = xgboost(
            self.classification_data, 
            target="target",
            task="classification",
            n_estimators=50
        )
        
        # Verify model creation
        assert isinstance(model, KeplerModel)
        assert model.trained == True
        assert model.model_type == "classification"
        assert model.target_column == "target"
        assert len(model.feature_columns) == 5
        
        # Verify XGBoost was called with correct parameters
        mock_xgb.XGBClassifier.assert_called_once()
        call_kwargs = mock_xgb.XGBClassifier.call_args[1]
        assert call_kwargs['n_estimators'] == 50
        assert call_kwargs['tree_method'] == 'hist'
        assert call_kwargs['objective'] == 'binary:logistic'
        
        # Verify training was called
        mock_classifier.fit.assert_called_once()
    
    @patch('kepler.core.library_manager.LibraryManager.dynamic_import')
    def test_xgboost_regression(self, mock_import):
        """Test XGBoost regression training"""
        
        # Mock XGBoost import
        mock_xgb = MagicMock()
        mock_regressor = MagicMock()
        mock_regressor.fit.return_value = None
        mock_regressor.predict.return_value = np.array([1.5, 2.3, 0.8, 1.9] * 50)  # Match test set size
        mock_regressor.feature_importances_ = np.array([0.4, 0.3, 0.15, 0.1, 0.05])
        
        mock_xgb.XGBRegressor.return_value = mock_regressor
        mock_import.return_value = mock_xgb
        
        # Train model
        model = xgboost(
            self.regression_data,
            target="target", 
            task="regression",
            learning_rate=0.1,
            max_depth=8
        )
        
        # Verify model creation
        assert isinstance(model, KeplerModel)
        assert model.model_type == "regression"
        
        # Verify XGBoost was called with correct parameters
        mock_xgb.XGBRegressor.assert_called_once()
        call_kwargs = mock_xgb.XGBRegressor.call_args[1]
        assert call_kwargs['learning_rate'] == 0.1
        assert call_kwargs['max_depth'] == 8
        assert call_kwargs['objective'] == 'reg:squarederror'
    
    @patch('kepler.core.library_manager.LibraryManager.dynamic_import')
    def test_xgboost_auto_task_detection(self, mock_import):
        """Test automatic task type detection"""
        
        mock_xgb = MagicMock()
        mock_classifier = MagicMock()
        mock_classifier.fit.return_value = None
        mock_classifier.predict.return_value = np.array([0, 1])
        mock_classifier.feature_importances_ = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        mock_xgb.XGBClassifier.return_value = mock_classifier
        mock_import.return_value = mock_xgb
        
        # Test with binary data (should auto-detect classification)
        model = xgboost(self.classification_data, target="target", task="auto")
        
        assert model.model_type == "classification"
        mock_xgb.XGBClassifier.assert_called_once()
    
    @patch('kepler.core.library_manager.LibraryManager.dynamic_import')
    def test_xgboost_categorical_features(self, mock_import):
        """Test XGBoost with categorical features"""
        
        mock_xgb = MagicMock()
        mock_classifier = MagicMock()
        mock_classifier.fit.return_value = None
        mock_classifier.predict.return_value = np.array([0, 1, 0])
        mock_classifier.feature_importances_ = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
        mock_xgb.XGBClassifier.return_value = mock_classifier
        mock_import.return_value = mock_xgb
        
        # Train with categorical data
        model = xgboost(
            self.categorical_data,
            target="target",
            enable_categorical=True
        )
        
        # Verify enable_categorical was passed
        call_kwargs = mock_xgb.XGBClassifier.call_args[1]
        assert call_kwargs['enable_categorical'] == True
    
    @patch('kepler.core.library_manager.LibraryManager.dynamic_import')
    def test_xgboost_custom_parameters(self, mock_import):
        """Test XGBoost with custom parameters via kwargs"""
        
        mock_xgb = MagicMock()
        mock_classifier = MagicMock()
        mock_classifier.fit.return_value = None
        mock_classifier.predict.return_value = np.array([0, 1])
        mock_classifier.feature_importances_ = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        mock_xgb.XGBClassifier.return_value = mock_classifier
        mock_import.return_value = mock_xgb
        
        # Train with custom parameters
        model = xgboost(
            self.classification_data,
            target="target",
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1
        )
        
        # Verify custom parameters were passed
        call_kwargs = mock_xgb.XGBClassifier.call_args[1]
        assert call_kwargs['subsample'] == 0.8
        assert call_kwargs['colsample_bytree'] == 0.8
        assert call_kwargs['reg_alpha'] == 0.1
        assert call_kwargs['reg_lambda'] == 0.1
    
    @patch('kepler.core.library_manager.LibraryManager.dynamic_import')
    def test_xgboost_early_stopping(self, mock_import):
        """Test XGBoost early stopping for larger datasets"""
        
        mock_xgb = MagicMock()
        mock_classifier = MagicMock()
        mock_classifier.fit.return_value = None
        mock_classifier.predict.return_value = np.array([0] * 40)  # Large test set
        mock_classifier.feature_importances_ = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        mock_xgb.XGBClassifier.return_value = mock_classifier
        mock_import.return_value = mock_xgb
        
        # Create larger dataset to trigger early stopping
        large_data = pd.concat([self.classification_data] * 3, ignore_index=True)
        
        model = xgboost(large_data, target="target")
        
        # Verify fit was called with eval_set and early_stopping_rounds
        fit_call = mock_classifier.fit.call_args
        assert 'eval_set' in fit_call[1]
        assert 'early_stopping_rounds' in fit_call[1]
        assert fit_call[1]['early_stopping_rounds'] == 10
    
    def test_xgboost_feature_autodetection(self):
        """Test automatic feature detection"""
        
        with patch('kepler.core.library_manager.LibraryManager.dynamic_import') as mock_import:
            mock_xgb = MagicMock()
            mock_classifier = MagicMock()
            mock_classifier.fit.return_value = None
            mock_classifier.predict.return_value = np.array([0, 1])
            mock_classifier.feature_importances_ = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
            mock_xgb.XGBClassifier.return_value = mock_classifier
            mock_import.return_value = mock_xgb
            
            # Train without specifying features (should auto-detect)
            model = xgboost(self.classification_data, target="target")
            
            # Should auto-detect all columns except target
            expected_features = [col for col in self.classification_data.columns if col != "target"]
            assert model.feature_columns == expected_features


class TestMLTrainingWrappers:
    """Test traditional ML training wrappers (sklearn, XGBoost, etc.)"""
    
    def test_lightgbm_placeholder(self):
        """Test that LightGBM raises appropriate error with suggestion"""
        
        data = pd.DataFrame({'feature1': [1, 2, 3], 'target': [0, 1, 0]})
        
        with pytest.raises(ModelTrainingError) as exc_info:
            from kepler.train import lightgbm
            lightgbm(data, target="target")
        
        assert "LightGBM trainer not yet implemented" in str(exc_info.value)
        assert hasattr(exc_info.value, 'suggestion')
        assert "Use XGBoost for now" in exc_info.value.suggestion
    
    def test_catboost_placeholder(self):
        """Test that CatBoost raises appropriate error with suggestion"""
        
        data = pd.DataFrame({'feature1': [1, 2, 3], 'target': [0, 1, 0]})
        
        with pytest.raises(ModelTrainingError) as exc_info:
            from kepler.train import catboost
            catboost(data, target="target")
        
        assert "CatBoost trainer not yet implemented" in str(exc_info.value)
        assert hasattr(exc_info.value, 'suggestion')
        assert "Use XGBoost for now" in exc_info.value.suggestion


class TestTrainingWrapperIntegration:
    """Test integration between training wrappers and unlimited library system"""
    
    def test_dynamic_library_loading_in_training(self):
        """Test that training functions use dynamic library loading"""
        
        with patch('kepler.core.library_manager.LibraryManager.dynamic_import') as mock_import:
            mock_xgb = MagicMock()
            mock_classifier = MagicMock()
            mock_classifier.fit.return_value = None
            mock_classifier.predict.return_value = np.array([0, 1])
            mock_classifier.feature_importances_ = np.array([0.5, 0.5])
            mock_xgb.XGBClassifier.return_value = mock_classifier
            mock_import.return_value = mock_xgb
            
            data = pd.DataFrame({
                'feature1': list(range(100)),
                'feature2': [i * 0.5 for i in range(100)], 
                'target': [i % 2 for i in range(100)]
            })
            
            model = xgboost(data, target="target")
            
            # Verify dynamic import was called
            mock_import.assert_called_once_with("xgboost")
            
            # Verify model was created and trained
            assert isinstance(model, KeplerModel)
            assert model.trained == True


class TestPRDComplianceML:
    """Test compliance with PRD requirements for ML training"""
    
    def test_prd_requirement_ml_wrappers(self):
        """
        Test PRD requirement: 'Create training wrappers for ML (sklearn, XGBoost, LightGBM, CatBoost)'
        """
        
        # Test that all required ML algorithms are accessible
        from kepler.train import random_forest, linear_model, xgboost, lightgbm, catboost
        
        # Verify functions exist
        assert callable(random_forest)
        assert callable(linear_model)
        assert callable(xgboost)
        assert callable(lightgbm)
        assert callable(catboost)
        
        # Verify they have proper documentation
        assert "Random Forest" in random_forest.__doc__
        assert "XGBoost" in xgboost.__doc__
        assert "LightGBM" in lightgbm.__doc__
        assert "CatBoost" in catboost.__doc__
    
    def test_prd_requirement_unified_api(self):
        """
        Test PRD requirement: 'Create unified training API that works with ANY framework'
        """
        
        # All training functions should have consistent API
        from kepler.train import random_forest, linear_model, xgboost
        
        # Test consistent parameter signature
        import inspect
        
        rf_sig = inspect.signature(random_forest)
        xgb_sig = inspect.signature(xgboost)
        
        # Core parameters should be consistent
        core_params = ['data', 'target', 'features', 'test_size', 'task']
        
        for param in core_params:
            if param in rf_sig.parameters and param in xgb_sig.parameters:
                # Parameters should have same defaults where applicable
                rf_param = rf_sig.parameters[param]
                xgb_param = xgb_sig.parameters[param]
                
                if rf_param.default != inspect.Parameter.empty and xgb_param.default != inspect.Parameter.empty:
                    # For test_size, should be consistent
                    if param == 'test_size':
                        assert rf_param.default == xgb_param.default
    
    def test_prd_requirement_simple_api(self):
        """
        Test PRD requirement: 'permitir entrenamiento de modelos con una API simple'
        """
        
        # API should be simple enough for data scientists
        data = pd.DataFrame({
            'temperature': [20, 25, 30, 35, 40, 45, 50, 55] * 10,
            'pressure': [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7] * 10,
            'status': ['normal', 'warning', 'normal', 'alert', 'normal', 'warning', 'normal', 'alert'] * 10
        })
        
        with patch('kepler.core.library_manager.LibraryManager.dynamic_import') as mock_import:
            mock_xgb = MagicMock()
            mock_classifier = MagicMock()
            mock_classifier.fit.return_value = None
            # Mock will be configured per test
            mock_classifier.feature_importances_ = np.array([0.6, 0.4])
            mock_xgb.XGBClassifier.return_value = mock_classifier
            mock_import.return_value = mock_xgb
            
            # Simple API usage - should work with minimal parameters
            model = xgboost(data, target="status")
            
            # Should auto-detect everything
            assert model.target_column == "status"
            assert len(model.feature_columns) == 2  # temperature, pressure
            assert model.trained == True
    
    def test_prd_requirement_automatic_serialization(self):
        """
        Test PRD requirement: 'El sistema DEBE serializar automáticamente modelos entrenados'
        """
        
        with patch('kepler.core.library_manager.LibraryManager.dynamic_import') as mock_import:
            mock_xgb = MagicMock()
            mock_classifier = MagicMock()
            mock_classifier.fit.return_value = None
            mock_classifier.predict.return_value = np.array([0, 1])
            mock_classifier.feature_importances_ = np.array([0.5, 0.5])
            mock_xgb.XGBClassifier.return_value = mock_classifier
            mock_import.return_value = mock_xgb
            
            data = pd.DataFrame({
                'feature1': list(range(100)),
                'target': [i % 2 for i in range(100)]
            })
            
            model = xgboost(data, target="target")
            
            # Model should be serializable
            assert hasattr(model, 'save')
            assert callable(model.save)
            
            # KeplerModel should contain all necessary metadata for serialization
            assert hasattr(model, 'model_type')
            assert hasattr(model, 'target_column')
            assert hasattr(model, 'feature_columns')
            assert hasattr(model, 'performance')
