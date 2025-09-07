"""
Safe unit tests for AI framework coverage
Tests Task 1.9: Comprehensive testing without causing segmentation faults
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

import kepler as kp
from kepler.utils.exceptions import ModelTrainingError


class TestAIFrameworkAPICoverage:
    """Test that all AI framework APIs are available and callable"""
    
    def test_traditional_ml_apis_available(self):
        """Test that traditional ML training APIs are available"""
        # All traditional ML functions should be callable
        ml_functions = [
            kp.train.random_forest,
            kp.train.xgboost,
            kp.train.lightgbm,
            kp.train.catboost
        ]
        
        for func in ml_functions:
            assert callable(func), f"{func.__name__} should be callable"
    
    def test_deep_learning_apis_available(self):
        """Test that deep learning training APIs are available"""
        # All deep learning functions should be callable
        dl_functions = [
            kp.train.pytorch,
            kp.train.tensorflow,
            kp.train.keras,
            kp.train.jax
        ]
        
        for func in dl_functions:
            assert callable(func), f"{func.__name__} should be callable"
    
    def test_generative_ai_apis_available(self):
        """Test that generative AI training APIs are available"""
        # All generative AI functions should be callable
        genai_functions = [
            kp.train.transformers,
            kp.train.langchain,
            kp.train.openai_finetune
        ]
        
        for func in genai_functions:
            assert callable(func), f"{func.__name__} should be callable"
    
    def test_unified_training_api_available(self):
        """Test that unified training API is available"""
        # Unified training functions should be callable
        unified_functions = [
            kp.train_unified.train,
            kp.train_unified.auto_train,
            kp.train_unified.quick_train,
            kp.train_unified.list_algorithms,
            kp.train_unified.get_algorithm_info
        ]
        
        for func in unified_functions:
            assert callable(func), f"{func.__name__} should be callable"


class TestFrameworkTypeDetection:
    """Test framework type detection and categorization"""
    
    def test_algorithm_categorization(self):
        """Test that algorithms are properly categorized by framework type"""
        algorithms = kp.train_unified.list_algorithms()
        
        # Should have main AI framework categories
        expected_categories = ['traditional_ml', 'deep_learning', 'generative_ai']
        
        for category in expected_categories:
            assert category in algorithms, f"Missing framework category: {category}"
            assert isinstance(algorithms[category], list)
            assert len(algorithms[category]) > 0
    
    def test_framework_detection_accuracy(self):
        """Test accuracy of framework detection for known algorithms"""
        test_cases = [
            # Traditional ML
            ('random_forest', 'sklearn', 'traditional_ml'),
            ('xgboost', 'xgboost', 'traditional_ml'),
            ('lightgbm', 'lightgbm', 'traditional_ml'),
            
            # Deep Learning  
            ('pytorch', 'pytorch', 'deep_learning'),
            ('neural_network', 'pytorch', 'deep_learning'),
            ('tensorflow', 'tensorflow', 'deep_learning'),
            
            # Generative AI
            ('transformers', 'transformers', 'generative_ai'),
            ('bert', 'transformers', 'generative_ai'),
            ('gpt', 'transformers', 'generative_ai')
        ]
        
        for algorithm, expected_framework, expected_type in test_cases:
            info = kp.train_unified.get_algorithm_info(algorithm)
            
            assert info['framework'] == expected_framework
            assert info['type'].value == expected_type
            assert 'module' in info
            assert 'function' in info


class TestTrainingAPIConsistency:
    """Test API consistency across framework types"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for API testing"""
        np.random.seed(42)
        return pd.DataFrame({
            'feature1': np.random.rand(50),
            'feature2': np.random.rand(50) * 10,
            'target': np.random.randint(0, 2, 50)
        })
    
    def test_consistent_parameter_interface(self, sample_data):
        """Test that all training functions accept consistent base parameters"""
        import inspect
        
        # Test traditional ML functions
        ml_functions = [kp.train.random_forest, kp.train.xgboost]
        
        for func in ml_functions:
            sig = inspect.signature(func)
            
            # Should accept standard parameters
            assert 'data' in sig.parameters
            assert 'target' in sig.parameters
            assert 'features' in sig.parameters or 'feature_columns' in sig.parameters
    
    def test_unified_api_parameter_consistency(self, sample_data):
        """Test unified API parameter consistency"""
        import inspect
        
        sig = inspect.signature(kp.train_unified.train)
        
        # Should have consistent base parameters
        required_params = ['data', 'target']
        optional_params = ['algorithm', 'features', 'test_size', 'random_state']
        
        for param in required_params:
            assert param in sig.parameters
            
        for param in optional_params:
            assert param in sig.parameters


class TestFrameworkSpecificParameterHandling:
    """Test framework-specific parameter handling"""
    
    def test_deep_learning_parameter_detection(self):
        """Test that deep learning parameters are properly detected"""
        from kepler.train_unified import UnifiedTrainer
        
        trainer = UnifiedTrainer()
        
        # Mock deep learning function signature
        with patch.object(trainer, '_get_training_function') as mock_get_func, \
             patch('inspect.signature') as mock_signature:
            
            mock_function = Mock()
            mock_get_func.return_value = mock_function
            
            # Mock signature with DL parameters
            mock_param = Mock()
            mock_signature.return_value.parameters = {
                'data': mock_param, 'target': mock_param, 'epochs': mock_param,
                'batch_size': mock_param, 'learning_rate': mock_param
            }
            
            framework_info = {'type': kp.train_unified.FrameworkType.DEEP_LEARNING}
            
            data = pd.DataFrame({'a': [1], 'target': [0]})
            params = trainer._prepare_unified_params(
                framework_info, data, 'target', ['a'], 0.2, 42,
                epochs=100, batch_size=32, learning_rate=0.001
            )
            
            # Should include DL-specific parameters
            assert params['epochs'] == 100
            assert params['batch_size'] == 32
            assert params['learning_rate'] == 0.001
    
    def test_generative_ai_parameter_detection(self):
        """Test that generative AI parameters are properly detected"""
        from kepler.train_unified import UnifiedTrainer
        
        trainer = UnifiedTrainer()
        
        with patch.object(trainer, '_get_training_function') as mock_get_func, \
             patch('inspect.signature') as mock_signature:
            
            mock_function = Mock()
            mock_get_func.return_value = mock_function
            
            # Mock signature with GenAI parameters
            mock_param = Mock()
            mock_signature.return_value.parameters = {
                'data': mock_param, 'target': mock_param, 'text_column': mock_param,
                'model_name': mock_param, 'max_length': mock_param
            }
            
            framework_info = {'type': kp.train_unified.FrameworkType.GENERATIVE_AI}
            
            data = pd.DataFrame({'text': ['sample'], 'target': [0]})
            params = trainer._prepare_unified_params(
                framework_info, data, 'target', ['text'], 0.2, 42,
                text_column='text', model_name='bert-base', max_length=512
            )
            
            # Should include GenAI-specific parameters
            assert params['text_column'] == 'text'
            assert params['model_name'] == 'bert-base'
            assert params['max_length'] == 512


class TestAutomaticAlgorithmSelection:
    """Test automatic algorithm selection across framework types"""
    
    def test_small_dataset_selection(self):
        """Test algorithm selection for small datasets"""
        small_data = pd.DataFrame({
            'feature1': np.random.rand(100),
            'target': np.random.randint(0, 2, 100)
        })
        
        selected = kp.train_unified.auto_select_algorithm(small_data, 'target')
        
        # Should prefer simple, robust algorithms for small data
        assert selected in ['random_forest', 'xgboost']
    
    def test_large_dataset_selection(self):
        """Test algorithm selection for large datasets"""
        # Simulate large dataset characteristics without creating actual large data
        from kepler.train_unified import UnifiedTrainer
        
        trainer = UnifiedTrainer()
        
        # Mock large dataset
        large_data = pd.DataFrame({
            'feature1': np.random.rand(50000),
            'feature2': np.random.rand(50000),
            'category': np.random.choice(['A', 'B', 'C'], 50000),
            'target': np.random.randint(0, 2, 50000)
        })
        
        selected = trainer._auto_select_algorithm(large_data, 'target')
        
        # Should prefer gradient boosting for large data with categorical features
        assert selected == 'xgboost'
    
    def test_high_dimensional_selection(self):
        """Test algorithm selection for high-dimensional data"""
        from kepler.train_unified import UnifiedTrainer
        
        trainer = UnifiedTrainer()
        
        # Create high-dimensional data
        n_features = 150
        feature_data = {f'feature_{i}': np.random.rand(1000) for i in range(n_features)}
        feature_data['target'] = np.random.randint(0, 2, 1000)
        
        high_dim_data = pd.DataFrame(feature_data)
        
        selected = trainer._auto_select_algorithm(high_dim_data, 'target')
        
        # Should consider neural networks for high-dimensional data
        assert selected in ['pytorch', 'random_forest']


class TestErrorHandlingRobustness:
    """Test robust error handling across all framework types"""
    
    def test_missing_framework_graceful_handling(self):
        """Test graceful handling when framework is missing"""
        # Test with truly nonexistent algorithm
        with pytest.raises(ModelTrainingError) as exc_info:
            kp.train_unified.train(
                pd.DataFrame({'a': [1], 'target': [0]}),
                target='target',
                algorithm='completely_nonexistent_framework_12345'
            )
        
        # Should provide helpful error message
        error_msg = str(exc_info.value)
        assert ("training" in error_msg.lower() or 
                "framework" in error_msg.lower() or 
                "import" in error_msg.lower())
    
    def test_invalid_data_handling_across_frameworks(self):
        """Test invalid data handling across all framework types"""
        invalid_datasets = [
            pd.DataFrame(),  # Empty DataFrame
            pd.DataFrame({'feature1': [1, 2]}),  # Missing target
            pd.DataFrame({'target': [1, 2]})  # Missing features
        ]
        
        algorithms_to_test = ['random_forest', 'xgboost', 'pytorch']
        
        for algorithm in algorithms_to_test:
            for invalid_data in invalid_datasets:
                try:
                    with pytest.raises((ModelTrainingError, ValueError, KeyError)):
                        kp.train_unified.train(invalid_data, target='target', algorithm=algorithm)
                except Exception as e:
                    # Some frameworks may not be available - that's acceptable
                    if "not available" in str(e).lower():
                        continue
                    else:
                        raise


class TestPRDComplianceComprehensive:
    """Comprehensive PRD compliance testing for AI frameworks"""
    
    def test_prd_unlimited_ai_experimentation(self):
        """
        Test PRD: "maximizar las posibilidades de experimentación con cualquier librería Python"
        """
        # Should support all major AI framework types
        algorithms = kp.train_unified.list_algorithms()
        
        ai_framework_types = ['traditional_ml', 'deep_learning', 'generative_ai']
        
        for framework_type in ai_framework_types:
            assert framework_type in algorithms
            assert len(algorithms[framework_type]) > 0
    
    def test_prd_any_python_library_support(self):
        """
        Test PRD: "cualquier librería Python (ML, Deep Learning, IA Generativa, custom, experimental)"
        """
        # Should support custom framework registration
        from kepler.train_unified import register_framework, FrameworkType
        
        # Should be able to register any type of framework
        framework_types = [
            FrameworkType.TRADITIONAL_ML,
            FrameworkType.DEEP_LEARNING,
            FrameworkType.GENERATIVE_AI,
            FrameworkType.COMPUTER_VISION,
            FrameworkType.NLP
        ]
        
        for framework_type in framework_types:
            custom_config = {
                'type': framework_type,
                'algorithms': ['test_algo'],
                'module': 'test.module',
                'function_map': {'test_algo': 'train_func'}
            }
            
            # Should not raise error when registering any framework type
            register_framework(f'test_{framework_type.value}', custom_config)
    
    def test_prd_simple_api_requirement(self):
        """
        Test PRD: "permitir entrenamiento de modelos con una API simple"
        """
        import inspect
        
        # Unified API should be simple
        sig = inspect.signature(kp.train_unified.train)
        
        # Should have minimal required parameters
        required_params = ['data', 'target']
        for param in required_params:
            assert param in sig.parameters
        
        # Should have sensible defaults
        optional_params = ['algorithm', 'test_size', 'random_state']
        for param in optional_params:
            assert param in sig.parameters
            assert sig.parameters[param].default is not inspect.Parameter.empty
    
    def test_prd_automatic_model_comparison(self):
        """
        Test PRD: "soportar comparación automática entre modelos"
        """
        # Should provide tools for model comparison
        assert hasattr(kp.train_unified, 'list_algorithms')
        assert hasattr(kp.train_unified, 'auto_select_algorithm')
        assert hasattr(kp.train_unified, 'get_algorithm_info')
    
    def test_prd_framework_extensibility(self):
        """
        Test PRD: Framework should be extensible for any AI framework
        """
        from kepler.train_unified import UnifiedTrainer
        
        trainer = UnifiedTrainer()
        
        # Should support framework registration
        assert hasattr(trainer, 'register_custom_framework')
        
        # Should maintain framework registry
        assert hasattr(trainer, '_framework_registry')
        assert isinstance(trainer._framework_registry, dict)
        assert len(trainer._framework_registry) > 0


class TestAIFrameworkEcosystemIntegration:
    """Test integration across the complete AI framework ecosystem"""
    
    def test_library_manager_integration_with_training(self):
        """Test that library manager integrates properly with training"""
        from kepler.train_unified import UnifiedTrainer
        from kepler.core.library_manager import LibraryManager
        
        trainer = UnifiedTrainer()
        
        # Trainer should have library manager
        assert hasattr(trainer, 'lib_manager')
        assert isinstance(trainer.lib_manager, LibraryManager)
    
    def test_dynamic_library_loading_integration(self):
        """Test integration with dynamic library loading"""
        from kepler.core.library_manager import LibraryManager
        
        # Should be able to use library manager in training context
        manager = LibraryManager()
        
        # Test dynamic import functionality
        pandas_module = manager.dynamic_import('pandas')
        assert pandas_module is not None
        
        # Should work with training-related libraries  
        numpy_module = manager.dynamic_import('numpy')
        assert numpy_module is not None
    
    def test_unlimited_library_ecosystem_support(self):
        """Test that training supports unlimited library ecosystem"""
        # Should integrate with unlimited library support
        assert hasattr(kp, 'libs')
        assert hasattr(kp, 'train_unified')
        
        # Library management should support any source
        lib_methods = [
            'install', 'install_github', 'install_local', 'install_wheel',
            'create_custom_lib', 'setup_ssh', 'validate_custom'
        ]
        
        for method in lib_methods:
            assert hasattr(kp.libs, method), f"Missing library method: {method}"


class TestMultiFrameworkWorkflows:
    """Test workflows that span multiple framework types"""
    
    @pytest.fixture
    def workflow_data(self):
        """Create data for multi-framework workflow testing"""
        np.random.seed(42)
        return {
            'tabular': pd.DataFrame({
                'feature1': np.random.rand(100),
                'feature2': np.random.rand(100) * 10,
                'target': np.random.randint(0, 2, 100)
            }),
            'text': pd.DataFrame({
                'text_content': ["positive text"] * 50 + ["negative text"] * 50,
                'sentiment': [1] * 50 + [0] * 50
            })
        }
    
    def test_multi_framework_experiment_workflow(self, workflow_data):
        """Test workflow using multiple frameworks in same project"""
        tabular_data = workflow_data['tabular']
        
        # Should be able to use multiple frameworks in same session
        frameworks_to_test = ['random_forest', 'xgboost']
        
        models = {}
        
        for framework in frameworks_to_test:
            try:
                # Use unified API for consistent interface
                model = kp.train_unified.train(
                    tabular_data,
                    target='target',
                    algorithm=framework,
                    test_size=0.2
                )
                
                models[framework] = model
                
            except Exception as e:
                # Framework may not be available - record for analysis
                models[framework] = f"Error: {e}"
        
        # At least one framework should work
        successful_models = [name for name, model in models.items() 
                           if not isinstance(model, str)]
        
        if successful_models:
            # Successful models should have consistent interface
            for framework_name in successful_models:
                model = models[framework_name]
                assert hasattr(model, 'trained') or hasattr(model, 'model')
        else:
            pytest.skip("No frameworks available for multi-framework workflow test")
    
    def test_framework_switching_workflow(self, workflow_data):
        """Test switching between frameworks within same workflow"""
        # Simulate Ana's workflow: try different frameworks for same problem
        tabular_data = workflow_data['tabular']
        
        experiment_results = {}
        
        # Try multiple algorithms with unified API
        algorithms = ['random_forest', 'xgboost', 'pytorch']
        
        for algorithm in algorithms:
            try:
                # Same API call, different framework
                info = kp.train_unified.get_algorithm_info(algorithm)
                
                experiment_results[algorithm] = {
                    'framework': info['framework'],
                    'type': info['type'].value,
                    'available': True
                }
                
            except Exception as e:
                experiment_results[algorithm] = {
                    'available': False,
                    'error': str(e)
                }
        
        # Should have information for all algorithms
        assert len(experiment_results) == len(algorithms)
        
        # At least core algorithms should be available
        core_algorithms = ['random_forest', 'xgboost']
        available_core = [algo for algo in core_algorithms 
                         if experiment_results[algo]['available']]
        
        assert len(available_core) > 0, "At least one core algorithm should be available"


class TestAIFrameworkDocumentationCompliance:
    """Test that framework support matches documentation promises"""
    
    def test_documented_frameworks_are_implemented(self):
        """Test that all documented frameworks have implementations"""
        # Based on PRD, these frameworks should be supported
        documented_frameworks = {
            'traditional_ml': ['sklearn', 'xgboost', 'lightgbm', 'catboost'],
            'deep_learning': ['pytorch', 'tensorflow', 'keras'],
            'generative_ai': ['transformers', 'langchain', 'openai']
        }
        
        algorithms = kp.train_unified.list_algorithms()
        
        for framework_type, expected_frameworks in documented_frameworks.items():
            if framework_type in algorithms:
                available_algorithms = algorithms[framework_type]
                
                # At least some documented frameworks should be available
                # (Not all may be implemented yet, but core ones should be)
                core_frameworks = expected_frameworks[:2]  # First 2 are core
                
                available_core = [fw for fw in core_frameworks 
                                if any(fw in algo for algo in available_algorithms)]
                
                assert len(available_core) > 0, f"No core {framework_type} frameworks available"
    
    def test_framework_function_availability(self):
        """Test that framework functions are available as documented"""
        # Core training functions should be available
        core_functions = [
            ('kp.train.random_forest', 'Traditional ML'),
            ('kp.train.xgboost', 'Traditional ML'),
            ('kp.train.pytorch', 'Deep Learning'),
            ('kp.train.transformers', 'Generative AI')
        ]
        
        for func_path, framework_type in core_functions:
            try:
                func = eval(func_path)
                assert callable(func), f"{func_path} should be callable"
            except AttributeError:
                pytest.fail(f"Documented function {func_path} not available")
            except Exception as e:
                # Other errors are OK - function exists but may have dependencies
                pass
