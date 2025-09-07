"""
Simplified unit tests for AutoML - Task 1.11
Tests core AutoML functionality without complex dependencies
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

import kepler.automl as automl_api
from kepler.utils.exceptions import ModelTrainingError


class TestAutoMLBasicFunctionality:
    """Test basic AutoML functionality"""
    
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
    
    def test_data_characteristics_analysis(self, sample_data):
        """Test data characteristics analysis"""
        characteristics = automl_api.analyze_data_characteristics(sample_data, 'target')
        
        assert characteristics.n_samples == 100
        assert characteristics.n_features == 3
        assert characteristics.target_type == 'classification'
        assert characteristics.n_classes == 2
        assert characteristics.complexity_level.value in ['simple', 'moderate']
    
    def test_algorithm_selection_api(self, sample_data):
        """Test algorithm selection API"""
        algorithm = automl_api.select_algorithm(sample_data, 'target')
        
        assert isinstance(algorithm, str)
        assert algorithm in automl_api.ALGORITHM_REGISTRY
        assert len(algorithm) > 0
    
    def test_algorithm_recommendations_api(self, sample_data):
        """Test algorithm recommendations API"""
        recommendations = automl_api.recommend_algorithms(sample_data, 'target', top_k=2)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 2
        assert all('algorithm' in rec for rec in recommendations)
        assert all('confidence' in rec for rec in recommendations)
        assert all('framework' in rec for rec in recommendations)
    
    def test_data_analysis_api(self, sample_data):
        """Test data analysis API"""
        analysis = automl_api.analyze_data(sample_data, 'target')
        
        assert isinstance(analysis, dict)
        assert 'task_type' in analysis
        assert 'complexity' in analysis
        assert 'n_samples' in analysis
        assert 'n_features' in analysis
        assert 'quick_recommendations' in analysis
        
        # Verify analysis content
        assert analysis['task_type'] == 'classification'
        assert analysis['n_samples'] == 100
        assert analysis['n_features'] == 3
    
    def test_algorithm_explanation_api(self, sample_data):
        """Test algorithm explanation API"""
        explanation = automl_api.explain_algorithm_choice('random_forest', sample_data, 'target')
        
        assert isinstance(explanation, str)
        assert len(explanation) > 50
        assert 'random_forest' in explanation
        assert 'sklearn' in explanation
    
    def test_list_algorithms_api(self):
        """Test list algorithms API"""
        algorithms = automl_api.list_available_algorithms()
        
        assert isinstance(algorithms, list)
        assert len(algorithms) > 0
        
        # Check algorithm structure
        for algo in algorithms:
            assert 'name' in algo
            assert 'framework' in algo
            assert 'types' in algo
            assert 'description' in algo
    
    @patch('kepler.automl.train')
    def test_auto_train_api(self, mock_train, sample_data):
        """Test auto train API"""
        # Mock the training function
        mock_model = Mock()
        mock_model.framework_info = {}
        mock_train.return_value = mock_model
        
        model = automl_api.auto_train(sample_data, 'target')
        
        assert model is not None
        mock_train.assert_called_once()
        
        # Verify specific algorithm was selected
        call_kwargs = mock_train.call_args[1]
        assert 'algorithm' in call_kwargs
        assert call_kwargs['algorithm'] in automl_api.ALGORITHM_REGISTRY


class TestAlgorithmScoring:
    """Test algorithm scoring logic"""
    
    def test_algorithm_registry_structure(self):
        """Test algorithm registry has proper structure"""
        registry = automl_api.ALGORITHM_REGISTRY
        
        assert isinstance(registry, dict)
        assert len(registry) > 0
        
        # Check required algorithms
        expected_algorithms = ['random_forest', 'xgboost', 'pytorch']
        for algo in expected_algorithms:
            assert algo in registry
    
    def test_score_algorithm_function(self):
        """Test algorithm scoring function"""
        # Create simple characteristics
        characteristics = automl_api.DataCharacteristics(
            n_samples=1000,
            n_features=5,
            n_numeric_features=5,
            n_categorical_features=0,
            target_type='classification',
            n_classes=2,
            class_imbalance_ratio=1.5,
            missing_percentage=2.0,
            complexity_level=automl_api.DataComplexity.SIMPLE,
            has_text_features=False
        )
        
        # Test Random Forest scoring
        rf_info = automl_api.ALGORITHM_REGISTRY['random_forest']
        rf_score = automl_api.score_algorithm(characteristics, rf_info)
        
        assert isinstance(rf_score, float)
        assert 0.0 <= rf_score <= 1.0
        assert rf_score > 0.5  # Should score well for simple data
    
    def test_different_data_complexities(self):
        """Test algorithm selection for different complexity levels"""
        complexities = [
            automl_api.DataComplexity.SIMPLE,
            automl_api.DataComplexity.MODERATE,
            automl_api.DataComplexity.COMPLEX,
            automl_api.DataComplexity.HIGH_DIMENSIONAL
        ]
        
        for complexity in complexities:
            characteristics = automl_api.DataCharacteristics(
                n_samples=1000,
                n_features=10,
                n_numeric_features=8,
                n_categorical_features=2,
                target_type='classification',
                n_classes=2,
                class_imbalance_ratio=1.0,
                missing_percentage=5.0,
                complexity_level=complexity,
                has_text_features=False
            )
            
            # Score algorithms for this complexity
            scores = {}
            for algo_name, algo_info in automl_api.ALGORITHM_REGISTRY.items():
                score = automl_api.score_algorithm(characteristics, algo_info)
                scores[algo_name] = score
            
            # At least one algorithm should have a positive score
            max_score = max(scores.values())
            assert max_score > 0.0, f"No algorithm scored > 0 for {complexity.value} data"


class TestErrorHandling:
    """Test error handling in AutoML"""
    
    def test_missing_target_column(self):
        """Test error handling for missing target column"""
        data = pd.DataFrame({'feature1': [1, 2, 3]})
        
        with pytest.raises(ModelTrainingError):
            automl_api.select_algorithm(data, 'nonexistent_target')
    
    def test_empty_dataframe(self):
        """Test error handling for empty DataFrame"""
        empty_data = pd.DataFrame()
        
        with pytest.raises(Exception):  # Should raise some form of error
            automl_api.analyze_data_characteristics(empty_data, 'target')
    
    def test_invalid_algorithm_explanation(self):
        """Test explanation for invalid algorithm"""
        data = pd.DataFrame({'feature1': [1, 2], 'target': [0, 1]})
        
        explanation = automl_api.explain_algorithm_choice('nonexistent_algorithm', data, 'target')
        
        assert "not found in registry" in explanation


class TestPRDCompliance:
    """Test PRD compliance for AutoML"""
    
    def test_automl_api_availability(self):
        """Test that AutoML APIs are available as specified in PRD"""
        # PRD Requirement: "El sistema DEBE proporcionar capacidades AutoML"
        required_functions = [
            'select_algorithm',
            'recommend_algorithms', 
            'analyze_data',
            'explain_algorithm_choice',
            'list_available_algorithms',
            'auto_train'
        ]
        
        for func_name in required_functions:
            assert hasattr(automl_api, func_name), f"Missing AutoML function: {func_name}"
            assert callable(getattr(automl_api, func_name)), f"AutoML function not callable: {func_name}"
    
    def test_automatic_algorithm_selection_requirement(self):
        """Test PRD requirement for automatic algorithm selection"""
        # Should be able to select algorithm automatically
        data = pd.DataFrame({
            'feature1': np.random.rand(50),
            'target': np.random.randint(0, 2, 50)
        })
        
        algorithm = automl_api.select_algorithm(data, 'target')
        assert isinstance(algorithm, str)
        assert algorithm in automl_api.ALGORITHM_REGISTRY
    
    def test_industrial_constraints_support(self):
        """Test support for industrial constraints"""
        data = pd.DataFrame({
            'feature1': np.random.rand(100),
            'target': np.random.randint(0, 2, 100)
        })
        
        # Should accept industrial constraints
        constraints = {
            'interpretability': 'high',
            'max_training_time_minutes': 30
        }
        
        # Should not crash with constraints
        recommendations = automl_api.recommend_algorithms(
            data, 'target', constraints=constraints
        )
        
        assert isinstance(recommendations, list)
    
    def test_unified_api_integration(self):
        """Test integration with unified training API"""
        # AutoML should integrate with kp.train_unified
        with patch('kepler.automl.train') as mock_train:
            mock_model = Mock()
            mock_train.return_value = mock_model
            
            data = pd.DataFrame({
                'feature1': np.random.rand(50),
                'target': np.random.randint(0, 2, 50)
            })
            
            model = automl_api.auto_train(data, 'target')
            
            # Should call unified training API
            mock_train.assert_called_once()
            assert model is not None
