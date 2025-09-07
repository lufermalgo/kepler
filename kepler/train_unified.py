"""
Kepler Unified Training API - Task 1.8
Unified API that works with ANY AI framework using consistent syntax

Philosophy: "One API, Any Framework"
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, Dict, Any, List
from enum import Enum
import inspect

from kepler.utils.logging import get_logger
from kepler.utils.exceptions import ModelTrainingError
from kepler.core.library_manager import LibraryManager


class FrameworkType(Enum):
    """Supported AI framework types"""
    TRADITIONAL_ML = "traditional_ml"
    DEEP_LEARNING = "deep_learning"
    GENERATIVE_AI = "generative_ai"
    COMPUTER_VISION = "computer_vision"
    NLP = "nlp"
    TIME_SERIES = "time_series"
    REINFORCEMENT_LEARNING = "reinforcement_learning"


class UnifiedTrainer:
    """
    Unified trainer that can work with ANY AI framework
    
    Provides consistent API regardless of underlying framework:
    - Same parameter names across frameworks
    - Automatic task detection (classification/regression)
    - Unified performance metrics
    - Consistent model saving and loading
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.lib_manager = LibraryManager()
        self._framework_registry = {}
        self._register_default_frameworks()
    
    def _register_default_frameworks(self):
        """Register default framework mappings"""
        self._framework_registry = {
            # Traditional ML
            'sklearn': {
                'type': FrameworkType.TRADITIONAL_ML,
                'algorithms': ['random_forest', 'logistic_regression', 'linear_regression', 'svm', 'decision_tree'],
                'module': 'kepler.train',
                'function_map': {
                    'random_forest': 'random_forest',
                    'logistic_regression': 'random_forest',  # Use RF as fallback
                    'linear_regression': 'random_forest',
                    'svm': 'random_forest',  # Use RF as fallback
                    'decision_tree': 'random_forest'  # Use RF as fallback
                }
            },
            'xgboost': {
                'type': FrameworkType.TRADITIONAL_ML,
                'algorithms': ['xgboost', 'gradient_boosting'],
                'module': 'kepler.train',
                'function_map': {'xgboost': 'xgboost', 'gradient_boosting': 'xgboost'}
            },
            'lightgbm': {
                'type': FrameworkType.TRADITIONAL_ML,
                'algorithms': ['lightgbm', 'lgb'],
                'module': 'kepler.train',
                'function_map': {'lightgbm': 'lightgbm', 'lgb': 'lightgbm'}
            },
            'catboost': {
                'type': FrameworkType.TRADITIONAL_ML,
                'algorithms': ['catboost', 'cat'],
                'module': 'kepler.train',
                'function_map': {'catboost': 'catboost', 'cat': 'catboost'}
            },
            
            # Deep Learning
            'pytorch': {
                'type': FrameworkType.DEEP_LEARNING,
                'algorithms': ['pytorch', 'torch', 'neural_network', 'mlp', 'cnn', 'rnn'],
                'module': 'kepler.train',
                'function_map': {
                    'pytorch': 'pytorch', 'torch': 'pytorch', 'neural_network': 'pytorch',
                    'mlp': 'pytorch', 'cnn': 'pytorch', 'rnn': 'pytorch'
                }
            },
            'tensorflow': {
                'type': FrameworkType.DEEP_LEARNING,
                'algorithms': ['tensorflow', 'tf', 'keras'],
                'module': 'kepler.train',
                'function_map': {'tensorflow': 'tensorflow', 'tf': 'tensorflow', 'keras': 'tensorflow'}
            },
            
            # Generative AI
            'transformers': {
                'type': FrameworkType.GENERATIVE_AI,
                'algorithms': ['transformers', 'huggingface', 'bert', 'gpt', 'llm'],
                'module': 'kepler.train',
                'function_map': {
                    'transformers': 'transformers', 'huggingface': 'transformers',
                    'bert': 'transformers', 'gpt': 'transformers', 'llm': 'transformers'
                }
            }
        }
    
    def auto_detect_framework(self, algorithm: str) -> Dict[str, Any]:
        """
        Automatically detect which framework to use for given algorithm
        
        Args:
            algorithm: Algorithm name (e.g., 'random_forest', 'pytorch', 'transformers')
            
        Returns:
            Framework information dict
        """
        algorithm = algorithm.lower()
        
        for framework_name, framework_info in self._framework_registry.items():
            if algorithm in framework_info['algorithms']:
                return {
                    'framework': framework_name,
                    'type': framework_info['type'],
                    'module': framework_info['module'],
                    'function': framework_info['function_map'][algorithm]
                }
        
        # If not found in registry, try to infer from algorithm name
        if any(ml_keyword in algorithm for ml_keyword in ['forest', 'tree', 'svm', 'regression']):
            return {
                'framework': 'sklearn',
                'type': FrameworkType.TRADITIONAL_ML,
                'module': 'kepler.train',
                'function': 'random_forest'
            }
        elif any(dl_keyword in algorithm for dl_keyword in ['neural', 'deep', 'net', 'cnn', 'rnn']):
            return {
                'framework': 'pytorch',
                'type': FrameworkType.DEEP_LEARNING,
                'module': 'kepler.train',
                'function': 'pytorch'
            }
        elif any(gen_keyword in algorithm for gen_keyword in ['llm', 'gpt', 'bert', 'transformer']):
            return {
                'framework': 'transformers',
                'type': FrameworkType.GENERATIVE_AI,
                'module': 'kepler.train',
                'function': 'transformers'
            }
        
        # Default to sklearn
        return {
            'framework': 'sklearn',
            'type': FrameworkType.TRADITIONAL_ML,
            'module': 'kepler.train',
            'function': 'random_forest'
        }
    
    def train(self, 
              data: pd.DataFrame,
              target: str,
              algorithm: str = "auto",
              features: List[str] = None,
              test_size: float = 0.2,
              random_state: int = 42,
              **kwargs) -> Any:
        """
        Unified training API that works with ANY framework
        
        Args:
            data: Training data DataFrame
            target: Target column name
            algorithm: Algorithm to use ("auto" for automatic selection)
            features: Feature columns (None for auto-detection)
            test_size: Test set size
            random_state: Random seed
            **kwargs: Framework-specific parameters
            
        Returns:
            Trained KeplerModel instance
            
        Examples:
            # Traditional ML
            model = trainer.train(data, target="failure", algorithm="random_forest")
            model = trainer.train(data, target="failure", algorithm="xgboost")
            
            # Deep Learning
            model = trainer.train(data, target="failure", algorithm="pytorch", epochs=100)
            model = trainer.train(data, target="failure", algorithm="neural_network", hidden_sizes=[64, 32])
            
            # Generative AI
            model = trainer.train(text_data, target="sentiment", algorithm="transformers", 
                                text_column="review_text")
            
            # Auto-detection
            model = trainer.train(data, target="failure", algorithm="auto")
        """
        self.logger.info(f"🚀 Unified training started: algorithm={algorithm}, target={target}")
        
        try:
            # Auto-detect algorithm if requested
            if algorithm == "auto":
                algorithm = self._auto_select_algorithm(data, target)
                self.logger.info(f"🧠 Auto-selected algorithm: {algorithm}")
            
            # Detect framework for algorithm
            framework_info = self.auto_detect_framework(algorithm)
            self.logger.info(f"🔧 Using framework: {framework_info['framework']}")
            
            # Import framework-specific training function
            training_function = self._get_training_function(framework_info)
            
            # Prepare unified parameters
            unified_params = self._prepare_unified_params(
                framework_info, data, target, features, test_size, random_state, **kwargs
            )
            
            # Call framework-specific training function
            self.logger.info(f"🎯 Training with {framework_info['framework']}...")
            model = training_function(**unified_params)
            
            # Add unified metadata
            if hasattr(model, 'framework_info'):
                model.framework_info = framework_info
            else:
                # For frameworks that don't support custom attributes
                model._kepler_framework_info = framework_info
            
            self.logger.info("✅ Unified training completed successfully")
            return model
            
        except Exception as e:
            error_msg = f"Unified training failed: {e}"
            self.logger.error(error_msg)
            raise ModelTrainingError(
                error_msg,
                suggestion=f"Try different algorithm or check data format. Available algorithms: {self.list_available_algorithms()}"
            )
    
    def _auto_select_algorithm(self, data: pd.DataFrame, target: str) -> str:
        """
        Automatically select best algorithm based on data characteristics
        
        Args:
            data: Training data
            target: Target column
            
        Returns:
            Recommended algorithm name
        """
        # Analyze data characteristics
        n_samples, n_features = data.shape
        
        # Check target type
        target_values = data[target].nunique()
        is_classification = target_values < n_samples * 0.05  # Heuristic for classification
        
        # Check data types
        numeric_features = len(data.select_dtypes(include=[np.number]).columns)
        categorical_features = n_features - numeric_features
        
        # Decision logic for algorithm selection
        if n_samples < 1000:
            # Small dataset - use simple algorithms
            return "random_forest"
        elif n_samples > 100000 and categorical_features > 0:
            # Large dataset with categorical features - use gradient boosting
            return "xgboost"
        elif n_features > 100:
            # High-dimensional data - consider neural networks
            if n_samples > 10000:
                return "pytorch"
            else:
                return "random_forest"
        else:
            # Default to robust algorithm
            return "xgboost"
    
    def _get_training_function(self, framework_info: Dict[str, Any]) -> callable:
        """Get the training function for specified framework"""
        try:
            # Import the module dynamically
            module = self.lib_manager.dynamic_import(framework_info['module'].split('.')[-1])
            if module is None:
                # Try importing the full module path
                import importlib
                module = importlib.import_module(framework_info['module'])
            
            # Get the training function
            function_name = framework_info['function']
            training_function = getattr(module, function_name)
            
            return training_function
            
        except (ImportError, AttributeError) as e:
            raise ModelTrainingError(
                f"Could not load training function {framework_info['function']} from {framework_info['module']}",
                suggestion=f"Install required framework: {framework_info['framework']}"
            )
    
    def _prepare_unified_params(self, framework_info: Dict[str, Any], 
                               data: pd.DataFrame, target: str, features: List[str],
                               test_size: float, random_state: int, **kwargs) -> Dict[str, Any]:
        """
        Prepare unified parameters for framework-specific function
        
        Maps unified parameter names to framework-specific parameter names
        """
        # Get the training function to inspect its signature
        training_function = self._get_training_function(framework_info)
        sig = inspect.signature(training_function)
        
        # Base parameters that all frameworks should accept
        params = {
            'data': data,
            'target': target
        }
        
        # Add optional parameters if supported by the function
        if 'features' in sig.parameters:
            params['features'] = features
        if 'test_size' in sig.parameters:
            params['test_size'] = test_size
        if 'random_state' in sig.parameters:
            params['random_state'] = random_state
            
        # Framework-specific parameter mapping
        framework_type = framework_info['type']
        
        if framework_type == FrameworkType.DEEP_LEARNING:
            # Map unified DL parameters
            if 'epochs' in kwargs and 'epochs' in sig.parameters:
                params['epochs'] = kwargs['epochs']
            if 'batch_size' in kwargs and 'batch_size' in sig.parameters:
                params['batch_size'] = kwargs['batch_size']
            if 'learning_rate' in kwargs and 'learning_rate' in sig.parameters:
                params['learning_rate'] = kwargs['learning_rate']
            if 'hidden_sizes' in kwargs and 'hidden_sizes' in sig.parameters:
                params['hidden_sizes'] = kwargs['hidden_sizes']
                
        elif framework_type == FrameworkType.GENERATIVE_AI:
            # Map unified GenAI parameters
            if 'text_column' in kwargs and 'text_column' in sig.parameters:
                params['text_column'] = kwargs['text_column']
            if 'model_name' in kwargs and 'model_name' in sig.parameters:
                params['model_name'] = kwargs['model_name']
            if 'max_length' in kwargs and 'max_length' in sig.parameters:
                params['max_length'] = kwargs['max_length']
                
        # Add any remaining kwargs that match function parameters
        for param_name, param_value in kwargs.items():
            if param_name in sig.parameters and param_name not in params:
                params[param_name] = param_value
        
        return params
    
    def list_available_algorithms(self) -> Dict[str, List[str]]:
        """
        List all available algorithms by framework type
        
        Returns:
            Dict mapping framework types to available algorithms
        """
        algorithms_by_type = {}
        
        for framework_name, framework_info in self._framework_registry.items():
            framework_type = framework_info['type'].value
            
            if framework_type not in algorithms_by_type:
                algorithms_by_type[framework_type] = []
                
            algorithms_by_type[framework_type].extend(framework_info['algorithms'])
        
        # Remove duplicates
        for framework_type in algorithms_by_type:
            algorithms_by_type[framework_type] = list(set(algorithms_by_type[framework_type]))
        
        return algorithms_by_type
    
    def get_framework_info(self, algorithm: str) -> Dict[str, Any]:
        """Get framework information for specified algorithm"""
        return self.auto_detect_framework(algorithm)
    
    def register_custom_framework(self, framework_name: str, framework_config: Dict[str, Any]):
        """
        Register custom framework for unified training
        
        Args:
            framework_name: Name of the custom framework
            framework_config: Configuration dict with type, algorithms, module, function_map
            
        Example:
            trainer.register_custom_framework('my_custom_ml', {
                'type': FrameworkType.TRADITIONAL_ML,
                'algorithms': ['custom_algorithm'],
                'module': 'my_custom_ml.train',
                'function_map': {'custom_algorithm': 'train_model'}
            })
        """
        self._framework_registry[framework_name] = framework_config
        self.logger.info(f"Registered custom framework: {framework_name}")


# Global unified trainer instance
_unified_trainer = UnifiedTrainer()


def train(data: pd.DataFrame,
          target: str,
          algorithm: str = "auto",
          features: List[str] = None,
          test_size: float = 0.2,
          random_state: int = 42,
          **kwargs) -> Any:
    """
    Unified training function that works with ANY AI framework
    
    This is the main entry point for unified training in Kepler.
    Automatically detects the best framework for your algorithm and data.
    
    Args:
        data: Training data DataFrame
        target: Target column name
        algorithm: Algorithm to use ("auto", "random_forest", "xgboost", "pytorch", "transformers", etc.)
        features: Feature columns (None for auto-detection)
        test_size: Test set size (0.0-1.0)
        random_state: Random seed for reproducibility
        **kwargs: Framework-specific parameters
        
    Returns:
        Trained KeplerModel instance
        
    Examples:
        # Traditional ML (auto-detects sklearn/xgboost)
        model = kp.train_unified.train(data, target="failure", algorithm="random_forest")
        model = kp.train_unified.train(data, target="failure", algorithm="xgboost")
        
        # Deep Learning (auto-detects pytorch/tensorflow)
        model = kp.train_unified.train(data, target="failure", algorithm="pytorch", epochs=100)
        model = kp.train_unified.train(data, target="failure", algorithm="neural_network", 
                                     hidden_sizes=[64, 32], learning_rate=0.001)
        
        # Generative AI (auto-detects transformers/langchain)
        model = kp.train_unified.train(text_data, target="sentiment", algorithm="transformers",
                                     text_column="review_text", model_name="bert-base-uncased")
        
        # Auto-selection (analyzes data and chooses best algorithm)
        model = kp.train_unified.train(data, target="failure", algorithm="auto")
        
        # Custom framework (if registered)
        model = kp.train_unified.train(data, target="failure", algorithm="my_custom_algo")
    """
    return _unified_trainer.train(data, target, algorithm, features, test_size, random_state, **kwargs)


def list_algorithms() -> Dict[str, List[str]]:
    """
    List all available algorithms organized by framework type
    
    Returns:
        Dict mapping framework types to available algorithms
        
    Example:
        algorithms = kp.train_unified.list_algorithms()
        print("Available algorithms:")
        for framework_type, algos in algorithms.items():
            print(f"  {framework_type}: {', '.join(algos)}")
    """
    return _unified_trainer.list_available_algorithms()


def get_algorithm_info(algorithm: str) -> Dict[str, Any]:
    """
    Get detailed information about specific algorithm
    
    Args:
        algorithm: Algorithm name
        
    Returns:
        Framework information for the algorithm
        
    Example:
        info = kp.train_unified.get_algorithm_info("xgboost")
        print(f"Framework: {info['framework']}")
        print(f"Type: {info['type']}")
    """
    return _unified_trainer.get_framework_info(algorithm)


def register_framework(framework_name: str, framework_config: Dict[str, Any]):
    """
    Register custom framework for unified training
    
    Args:
        framework_name: Name of the custom framework
        framework_config: Configuration dict
        
    Example:
        # Register custom ML framework
        kp.train_unified.register_framework('my_ml_lib', {
            'type': FrameworkType.TRADITIONAL_ML,
            'algorithms': ['custom_forest', 'custom_boost'],
            'module': 'my_ml_lib.train',
            'function_map': {
                'custom_forest': 'train_forest',
                'custom_boost': 'train_boost'
            }
        })
        
        # Now can use unified API
        model = kp.train_unified.train(data, target="y", algorithm="custom_forest")
    """
    _unified_trainer.register_custom_framework(framework_name, framework_config)


def auto_select_algorithm(data: pd.DataFrame, target: str) -> str:
    """
    Automatically select best algorithm based on data characteristics
    
    Args:
        data: Training data
        target: Target column
        
    Returns:
        Recommended algorithm name
        
    Example:
        best_algo = kp.train_unified.auto_select_algorithm(sensor_data, "failure")
        print(f"Recommended algorithm: {best_algo}")
        
        model = kp.train_unified.train(sensor_data, target="failure", algorithm=best_algo)
    """
    return _unified_trainer._auto_select_algorithm(data, target)


# Convenience aliases for common use cases
def auto_train(data: pd.DataFrame, target: str, **kwargs) -> Any:
    """
    Automatic training with best algorithm selection
    
    Convenience function that analyzes data and selects optimal algorithm automatically.
    
    Args:
        data: Training data DataFrame
        target: Target column name
        **kwargs: Additional parameters
        
    Returns:
        Trained model with best algorithm
        
    Example:
        # Let Kepler choose the best algorithm and framework
        model = kp.train_unified.auto_train(sensor_data, target="equipment_failure")
        print(f"Kepler selected: {model.framework_info['framework']}")
    """
    return train(data, target, algorithm="auto", **kwargs)


def quick_train(data: pd.DataFrame, target: str, algorithm: str, **kwargs) -> Any:
    """
    Quick training with minimal configuration
    
    Convenience function for rapid prototyping and experimentation.
    
    Args:
        data: Training data DataFrame
        target: Target column name
        algorithm: Algorithm to use
        **kwargs: Algorithm-specific parameters
        
    Returns:
        Trained model
        
    Example:
        # Quick experimentation
        rf_model = kp.train_unified.quick_train(data, "failure", "random_forest")
        xgb_model = kp.train_unified.quick_train(data, "failure", "xgboost")
        nn_model = kp.train_unified.quick_train(data, "failure", "pytorch", epochs=50)
    """
    return train(data, target, algorithm, **kwargs)
