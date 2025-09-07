"""
Unit tests for Unified Training API
Tests Task 1.8: Create unified training API that works with ANY framework
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from kepler.train_unified import UnifiedTrainer, FrameworkType, train, list_algorithms
from kepler.utils.exceptions import ModelTrainingError


class TestUnifiedTrainer:
    """Test UnifiedTrainer class"""
    
    @pytest.fixture
    def trainer(self):
        """Create UnifiedTrainer instance"""
        return UnifiedTrainer()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        np.random.seed(42)
        return pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100) * 10,
            'feature3': np.random.randint(0, 5, 100),
            'target': np.random.randint(0, 2, 100)
        })
    
    def test_framework_registry_initialization(self, trainer):
        """Test that framework registry is properly initialized"""
        assert len(trainer._framework_registry) > 0
        
        # Check that key frameworks are registered
        assert 'sklearn' in trainer._framework_registry
        assert 'xgboost' in trainer._framework_registry
        assert 'pytorch' in trainer._framework_registry
        assert 'transformers' in trainer._framework_registry
        
        # Verify framework structure
        sklearn_info = trainer._framework_registry['sklearn']
        assert 'type' in sklearn_info
        assert 'algorithms' in sklearn_info
        assert 'module' in sklearn_info
        assert 'function_map' in sklearn_info
        assert sklearn_info['type'] == FrameworkType.TRADITIONAL_ML
    
    def test_auto_detect_framework_sklearn(self, trainer):
        """Test framework auto-detection for sklearn algorithms"""
        test_cases = [
            ('random_forest', 'sklearn'),
            ('logistic_regression', 'sklearn'),
            ('decision_tree', 'sklearn')
        ]
        
        for algorithm, expected_framework in test_cases:
            framework_info = trainer.auto_detect_framework(algorithm)
            assert framework_info['framework'] == expected_framework
            assert framework_info['type'] == FrameworkType.TRADITIONAL_ML
    
    def test_auto_detect_framework_deep_learning(self, trainer):
        """Test framework auto-detection for deep learning algorithms"""
        test_cases = [
            ('pytorch', 'pytorch'),
            ('neural_network', 'pytorch'),
            ('mlp', 'pytorch'),
            ('tensorflow', 'tensorflow')
        ]
        
        for algorithm, expected_framework in test_cases:
            framework_info = trainer.auto_detect_framework(algorithm)
            assert framework_info['framework'] == expected_framework
            assert framework_info['type'] == FrameworkType.DEEP_LEARNING
    
    def test_auto_detect_framework_generative_ai(self, trainer):
        """Test framework auto-detection for generative AI algorithms"""
        test_cases = [
            ('transformers', 'transformers'),
            ('bert', 'transformers'),
            ('gpt', 'transformers'),
            ('llm', 'transformers')
        ]
        
        for algorithm, expected_framework in test_cases:
            framework_info = trainer.auto_detect_framework(algorithm)
            assert framework_info['framework'] == expected_framework
            assert framework_info['type'] == FrameworkType.GENERATIVE_AI
    
    def test_auto_detect_framework_fallback(self, trainer):
        """Test framework auto-detection fallback for unknown algorithms"""
        # Unknown algorithm should fall back to sklearn
        framework_info = trainer.auto_detect_framework("unknown_algorithm")
        assert framework_info['framework'] == 'sklearn'
        assert framework_info['type'] == FrameworkType.TRADITIONAL_ML
    
    def test_auto_select_algorithm_small_dataset(self, trainer):
        """Test automatic algorithm selection for small datasets"""
        small_data = pd.DataFrame({
            'feature1': np.random.rand(500),  # Small dataset
            'target': np.random.randint(0, 2, 500)
        })
        
        selected = trainer._auto_select_algorithm(small_data, 'target')
        assert selected == "random_forest"  # Should prefer simple algorithms for small data
    
    def test_auto_select_algorithm_large_dataset(self, trainer):
        """Test automatic algorithm selection for large datasets"""
        # Mock large dataset characteristics
        large_data = pd.DataFrame({
            'feature1': np.random.rand(50000),  # Large dataset
            'feature2': np.random.rand(50000),
            'category': np.random.choice(['A', 'B', 'C'], 50000),  # Categorical feature
            'target': np.random.randint(0, 2, 50000)
        })
        
        selected = trainer._auto_select_algorithm(large_data, 'target')
        assert selected == "xgboost"  # Should prefer gradient boosting for large data with categoricals
    
    def test_auto_select_algorithm_high_dimensional(self, trainer):
        """Test automatic algorithm selection for high-dimensional data"""
        # Create high-dimensional data
        n_features = 150
        n_samples = 5000
        
        feature_data = {}
        for i in range(n_features):
            feature_data[f'feature_{i}'] = np.random.rand(n_samples)
        feature_data['target'] = np.random.randint(0, 2, n_samples)
        
        high_dim_data = pd.DataFrame(feature_data)
        
        selected = trainer._auto_select_algorithm(high_dim_data, 'target')
        # Should prefer neural networks for high-dimensional data with sufficient samples
        assert selected in ["pytorch", "random_forest"]
    
    def test_list_available_algorithms(self, trainer):
        """Test listing of available algorithms"""
        algorithms = trainer.list_available_algorithms()
        
        assert isinstance(algorithms, dict)
        assert FrameworkType.TRADITIONAL_ML.value in algorithms
        assert FrameworkType.DEEP_LEARNING.value in algorithms
        assert FrameworkType.GENERATIVE_AI.value in algorithms
        
        # Check that algorithms are properly categorized
        ml_algorithms = algorithms[FrameworkType.TRADITIONAL_ML.value]
        assert 'random_forest' in ml_algorithms
        assert 'xgboost' in ml_algorithms
        
        dl_algorithms = algorithms[FrameworkType.DEEP_LEARNING.value]
        assert 'pytorch' in dl_algorithms
    
    def test_register_custom_framework(self, trainer):
        """Test registration of custom frameworks"""
        custom_config = {
            'type': FrameworkType.TRADITIONAL_ML,
            'algorithms': ['custom_algo'],
            'module': 'custom_ml.train',
            'function_map': {'custom_algo': 'train_model'}
        }
        
        trainer.register_custom_framework('custom_ml', custom_config)
        
        assert 'custom_ml' in trainer._framework_registry
        assert trainer._framework_registry['custom_ml'] == custom_config
        
        # Should be able to detect custom framework
        framework_info = trainer.auto_detect_framework('custom_algo')
        assert framework_info['framework'] == 'custom_ml'


class TestUnifiedTrainingAPI:
    """Test unified training API functions"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        np.random.seed(42)
        return pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100) * 10,
            'target': np.random.randint(0, 2, 100)
        })
    
    @patch('kepler.train_unified._unified_trainer.train')
    def test_unified_train_function(self, mock_train, sample_data):
        """Test main unified train function"""
        mock_model = Mock()
        mock_train.return_value = mock_model
        
        result = train(sample_data, target="target", algorithm="random_forest")
        
        assert result == mock_model
        mock_train.assert_called_once_with(
            sample_data, "target", "random_forest", None, 0.2, 42
        )
    
    @patch('kepler.train_unified._unified_trainer.list_available_algorithms')
    def test_list_algorithms_function(self, mock_list):
        """Test list_algorithms function"""
        expected_algorithms = {
            'traditional_ml': ['random_forest', 'xgboost'],
            'deep_learning': ['pytorch', 'tensorflow']
        }
        mock_list.return_value = expected_algorithms
        
        result = list_algorithms()
        
        assert result == expected_algorithms
        mock_list.assert_called_once()
    
    @patch('kepler.train_unified._unified_trainer.get_framework_info')
    def test_get_algorithm_info_function(self, mock_get_info):
        """Test get_algorithm_info function"""
        from kepler.train_unified import get_algorithm_info
        
        expected_info = {
            'framework': 'sklearn',
            'type': FrameworkType.TRADITIONAL_ML,
            'module': 'kepler.train'
        }
        mock_get_info.return_value = expected_info
        
        result = get_algorithm_info("random_forest")
        
        assert result == expected_info
        mock_get_info.assert_called_once_with("random_forest")
    
    @patch('kepler.train_unified._unified_trainer.register_custom_framework')
    def test_register_framework_function(self, mock_register):
        """Test register_framework function"""
        from kepler.train_unified import register_framework
        
        custom_config = {
            'type': FrameworkType.TRADITIONAL_ML,
            'algorithms': ['custom_algo']
        }
        
        register_framework('custom_ml', custom_config)
        
        mock_register.assert_called_once_with('custom_ml', custom_config)
    
    @patch('kepler.train_unified.train')
    def test_auto_train_convenience_function(self, mock_train, sample_data):
        """Test auto_train convenience function"""
        from kepler.train_unified import auto_train
        
        mock_model = Mock()
        mock_train.return_value = mock_model
        
        result = auto_train(sample_data, target="target", epochs=50)
        
        assert result == mock_model
        mock_train.assert_called_once_with(sample_data, "target", algorithm="auto", epochs=50)
    
    @patch('kepler.train_unified.train')
    def test_quick_train_convenience_function(self, mock_train, sample_data):
        """Test quick_train convenience function"""
        from kepler.train_unified import quick_train
        
        mock_model = Mock()
        mock_train.return_value = mock_model
        
        result = quick_train(sample_data, "target", "xgboost", max_depth=6)
        
        assert result == mock_model
        mock_train.assert_called_once_with(sample_data, "target", "xgboost", max_depth=6)


class TestUnifiedParameterMapping:
    """Test parameter mapping between unified API and frameworks"""
    
    @pytest.fixture
    def trainer(self):
        return UnifiedTrainer()
    
    def test_prepare_unified_params_sklearn(self, trainer):
        """Test parameter preparation for sklearn"""
        # Mock sklearn function signature
        mock_function = Mock()
        mock_function.__name__ = 'random_forest'
        
        with patch.object(trainer, '_get_training_function') as mock_get_func, \
             patch('inspect.signature') as mock_signature:
            
            mock_get_func.return_value = mock_function
            
            # Mock signature with sklearn-compatible parameters
            mock_param = Mock()
            mock_param.name = 'data'
            mock_signature.return_value.parameters = {
                'data': mock_param, 'target': mock_param, 'features': mock_param,
                'test_size': mock_param, 'random_state': mock_param
            }
            
            framework_info = {
                'type': FrameworkType.TRADITIONAL_ML,
                'framework': 'sklearn'
            }
            
            data = pd.DataFrame({'a': [1, 2], 'target': [0, 1]})
            params = trainer._prepare_unified_params(
                framework_info, data, 'target', ['a'], 0.2, 42
            )
            
            assert params['data'] is data
            assert params['target'] == 'target'
            assert params['features'] == ['a']
            assert params['test_size'] == 0.2
            assert params['random_state'] == 42
    
    def test_prepare_unified_params_deep_learning(self, trainer):
        """Test parameter preparation for deep learning frameworks"""
        mock_function = Mock()
        mock_function.__name__ = 'pytorch'
        
        with patch.object(trainer, '_get_training_function') as mock_get_func, \
             patch('inspect.signature') as mock_signature:
            
            mock_get_func.return_value = mock_function
            
            # Mock signature with deep learning parameters
            mock_param = Mock()
            mock_signature.return_value.parameters = {
                'data': mock_param, 'target': mock_param, 'epochs': mock_param,
                'batch_size': mock_param, 'learning_rate': mock_param, 'hidden_sizes': mock_param
            }
            
            framework_info = {
                'type': FrameworkType.DEEP_LEARNING,
                'framework': 'pytorch'
            }
            
            data = pd.DataFrame({'a': [1, 2], 'target': [0, 1]})
            params = trainer._prepare_unified_params(
                framework_info, data, 'target', ['a'], 0.2, 42,
                epochs=100, batch_size=32, learning_rate=0.001, hidden_sizes=[64, 32]
            )
            
            assert params['epochs'] == 100
            assert params['batch_size'] == 32
            assert params['learning_rate'] == 0.001
            assert params['hidden_sizes'] == [64, 32]


class TestFrameworkIntegration:
    """Test integration with actual training frameworks"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        np.random.seed(42)
        return pd.DataFrame({
            'feature1': np.random.rand(50),
            'feature2': np.random.rand(50) * 10,
            'target': np.random.randint(0, 2, 50)
        })
    
    def test_unified_api_with_sklearn(self, sample_data):
        """Test unified API with sklearn backend"""
        with patch('kepler.train_unified._unified_trainer._get_training_function') as mock_get_func, \
             patch('kepler.train_unified._unified_trainer.lib_manager.dynamic_import'):
            
            # Mock sklearn training function
            mock_sklearn_train = Mock()
            mock_model = Mock()
            mock_model.framework_info = {'framework': 'sklearn'}
            mock_sklearn_train.return_value = mock_model
            mock_get_func.return_value = mock_sklearn_train
            
            result = train(sample_data, target="target", algorithm="random_forest")
            
            assert result == mock_model
            mock_sklearn_train.assert_called_once()
    
    def test_unified_api_with_pytorch(self, sample_data):
        """Test unified API with PyTorch backend"""
        with patch('kepler.train_unified._unified_trainer._get_training_function') as mock_get_func, \
             patch('kepler.train_unified._unified_trainer.lib_manager.dynamic_import'):
            
            # Mock PyTorch training function
            mock_pytorch_train = Mock()
            mock_model = Mock()
            mock_model.framework_info = {'framework': 'pytorch'}
            mock_pytorch_train.return_value = mock_model
            mock_get_func.return_value = mock_pytorch_train
            
            result = train(sample_data, target="target", algorithm="pytorch", epochs=50)
            
            assert result == mock_model
            mock_pytorch_train.assert_called_once()
    
    def test_unified_api_auto_selection(self, sample_data):
        """Test unified API with automatic algorithm selection"""
        with patch('kepler.train_unified._unified_trainer._auto_select_algorithm') as mock_auto, \
             patch('kepler.train_unified._unified_trainer._get_training_function') as mock_get_func, \
             patch('kepler.train_unified._unified_trainer.lib_manager.dynamic_import'):
            
            mock_auto.return_value = "xgboost"
            
            mock_xgb_train = Mock()
            mock_model = Mock()
            mock_xgb_train.return_value = mock_model
            mock_get_func.return_value = mock_xgb_train
            
            result = train(sample_data, target="target", algorithm="auto")
            
            assert result == mock_model
            mock_auto.assert_called_once_with(sample_data, "target")


class TestPRDCompliance:
    """Test compliance with PRD requirements for unified training"""
    
    def test_prd_requirement_unified_api(self):
        """
        Test PRD Requirement: "Create unified training API that works with ANY framework"
        """
        from kepler.train_unified import train, auto_train, quick_train
        
        # Unified training functions should be available
        assert callable(train)
        assert callable(auto_train)
        assert callable(quick_train)
    
    def test_prd_requirement_any_framework_support(self):
        """
        Test PRD Requirement: Support for ANY AI framework
        """
        trainer = UnifiedTrainer()
        
        # Should support multiple framework types
        frameworks = trainer._framework_registry
        framework_types = [info['type'] for info in frameworks.values()]
        
        assert FrameworkType.TRADITIONAL_ML in framework_types
        assert FrameworkType.DEEP_LEARNING in framework_types
        assert FrameworkType.GENERATIVE_AI in framework_types
    
    def test_prd_requirement_consistent_api(self):
        """
        Test PRD Requirement: Consistent API regardless of framework
        """
        # All training functions should accept the same base parameters
        from kepler.train_unified import train
        
        import inspect
        sig = inspect.signature(train)
        
        # Should have consistent base parameters
        required_params = ['data', 'target']
        optional_params = ['algorithm', 'features', 'test_size', 'random_state']
        
        for param in required_params:
            assert param in sig.parameters
            
        for param in optional_params:
            assert param in sig.parameters
    
    def test_prd_requirement_framework_extensibility(self):
        """
        Test PRD Requirement: Ability to register custom frameworks
        """
        from kepler.train_unified import register_framework
        
        # Should be able to register custom frameworks
        assert callable(register_framework)
        
        trainer = UnifiedTrainer()
        
        # Should be able to register and use custom framework
        custom_config = {
            'type': FrameworkType.TRADITIONAL_ML,
            'algorithms': ['test_algo'],
            'module': 'test.train',
            'function_map': {'test_algo': 'train_test'}
        }
        
        trainer.register_custom_framework('test_framework', custom_config)
        
        # Should be detectable after registration
        framework_info = trainer.auto_detect_framework('test_algo')
        assert framework_info['framework'] == 'test_framework'


class TestErrorHandling:
    """Test error handling in unified training API"""
    
    def test_missing_framework_error(self):
        """Test error when framework is not available"""
        trainer = UnifiedTrainer()
        
        with patch.object(trainer, '_get_training_function') as mock_get_func:
            mock_get_func.side_effect = ModelTrainingError("Framework not found")
            
            with pytest.raises(ModelTrainingError):
                trainer.train(
                    pd.DataFrame({'a': [1, 2], 'target': [0, 1]}),
                    target="target",
                    algorithm="nonexistent_framework"
                )
    
    def test_invalid_data_error(self):
        """Test error handling for invalid data"""
        trainer = UnifiedTrainer()
        
        # Empty DataFrame should raise error
        empty_data = pd.DataFrame()
        
        with pytest.raises(ModelTrainingError):
            trainer.train(empty_data, target="nonexistent", algorithm="random_forest")
    
    def test_missing_target_column_error(self):
        """Test error when target column doesn't exist"""
        trainer = UnifiedTrainer()
        
        data = pd.DataFrame({'feature1': [1, 2, 3]})
        
        with pytest.raises(ModelTrainingError):
            trainer.train(data, target="nonexistent_target", algorithm="random_forest")
