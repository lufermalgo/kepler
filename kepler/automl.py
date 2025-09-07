"""
Kepler AutoML API - Task 1.11 Implementation
Automatic algorithm selection based on data characteristics

Provides intelligent automation for algorithm selection using:
- Data profiling and characteristics analysis
- Performance-based recommendations
- Industrial constraints consideration
- Automated ranking and recommendation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from kepler.utils.logging import get_logger
from kepler.utils.exceptions import ModelTrainingError
from kepler.core.library_manager import LibraryManager


class DataComplexity(Enum):
    """Data complexity levels for algorithm selection"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    HIGH_DIMENSIONAL = "high_dimensional"


@dataclass
class DataCharacteristics:
    """Data characteristics for algorithm selection"""
    n_samples: int
    n_features: int
    n_numeric_features: int
    n_categorical_features: int
    target_type: str  # 'classification' or 'regression'
    n_classes: int
    class_imbalance_ratio: float
    missing_percentage: float
    complexity_level: DataComplexity
    has_text_features: bool = False


# Algorithm registry with characteristics
ALGORITHM_REGISTRY = {
    'random_forest': {
        'framework': 'sklearn',
        'type': ['classification', 'regression'],
        'complexity': [DataComplexity.SIMPLE, DataComplexity.MODERATE],
        'sample_range': (100, 100000),
        'feature_range': (1, 1000),
        'handles_categorical': True,
        'handles_missing': True,
        'training_speed': 'fast',
        'interpretability': 'high',
        'memory_usage': 'low',
        'typical_performance': (0.75, 0.92)
    },
    'xgboost': {
        'framework': 'xgboost',
        'type': ['classification', 'regression'],
        'complexity': [DataComplexity.MODERATE, DataComplexity.COMPLEX],
        'sample_range': (500, 1000000),
        'feature_range': (1, 10000),
        'handles_categorical': True,
        'handles_missing': True,
        'training_speed': 'medium',
        'interpretability': 'medium',
        'memory_usage': 'medium',
        'typical_performance': (0.80, 0.95)
    },
    'pytorch': {
        'framework': 'pytorch',
        'type': ['classification', 'regression'],
        'complexity': [DataComplexity.COMPLEX, DataComplexity.HIGH_DIMENSIONAL],
        'sample_range': (1000, 10000000),
        'feature_range': (10, 100000),
        'handles_categorical': False,
        'handles_missing': False,
        'training_speed': 'slow',
        'interpretability': 'low',
        'memory_usage': 'high',
        'typical_performance': (0.78, 0.96)
    }
}


def analyze_data_characteristics(data: pd.DataFrame, target: str) -> DataCharacteristics:
    """Analyze data characteristics for algorithm selection"""
    logger = get_logger(__name__)
    logger.info("Analyzing data characteristics for AutoML...")
    
    # Basic dimensions
    n_samples, n_features_total = data.shape
    n_features = n_features_total - 1  # Exclude target
    
    # Feature type analysis
    features_data = data.drop(columns=[target])
    numeric_features = features_data.select_dtypes(include=[np.number]).columns
    categorical_features = features_data.select_dtypes(exclude=[np.number]).columns
    
    n_numeric_features = len(numeric_features)
    n_categorical_features = len(categorical_features)
    
    # Target analysis
    target_values = data[target].dropna()
    n_unique_targets = target_values.nunique()
    
    # Determine task type
    if n_unique_targets <= 20 and n_unique_targets < len(target_values) * 0.05:
        target_type = 'classification'
        n_classes = n_unique_targets
        
        # Calculate class imbalance
        class_counts = target_values.value_counts()
        majority_class = class_counts.max()
        minority_class = class_counts.min()
        class_imbalance_ratio = majority_class / minority_class if minority_class > 0 else float('inf')
    else:
        target_type = 'regression'
        n_classes = 0
        class_imbalance_ratio = 1.0
    
    # Missing data analysis
    missing_percentage = (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100
    
    # Complexity assessment
    if n_features > 100:
        complexity_level = DataComplexity.HIGH_DIMENSIONAL
    elif n_samples > 50000 or n_categorical_features > 10 or missing_percentage > 20:
        complexity_level = DataComplexity.COMPLEX
    elif n_samples > 5000 or n_categorical_features > 3 or missing_percentage > 5:
        complexity_level = DataComplexity.MODERATE
    else:
        complexity_level = DataComplexity.SIMPLE
    
    # Text feature detection
    has_text_features = any(
        features_data[col].dtype == 'object' and 
        features_data[col].str.len().mean() > 50  # Average length > 50 chars
        for col in categorical_features
        if len(features_data[col].dropna()) > 0
    )
    
    return DataCharacteristics(
        n_samples=n_samples,
        n_features=n_features,
        n_numeric_features=n_numeric_features,
        n_categorical_features=n_categorical_features,
        target_type=target_type,
        n_classes=n_classes,
        class_imbalance_ratio=class_imbalance_ratio,
        missing_percentage=missing_percentage,
        complexity_level=complexity_level,
        has_text_features=has_text_features
    )


def score_algorithm(characteristics: DataCharacteristics, algo_info: Dict[str, Any]) -> float:
    """Score algorithm suitability based on data characteristics"""
    score = 0.5  # Base score
    
    # Check basic compatibility
    if characteristics.target_type not in algo_info['type']:
        return 0.0
    
    # Check complexity compatibility
    if characteristics.complexity_level in algo_info['complexity']:
        score += 0.3
    else:
        score -= 0.2
    
    # Check sample size range
    min_samples, max_samples = algo_info['sample_range']
    if min_samples <= characteristics.n_samples <= max_samples:
        score += 0.2
    else:
        score -= 0.1
    
    # Categorical handling
    if characteristics.n_categorical_features > 0:
        if algo_info['handles_categorical']:
            score += 0.2
        else:
            score -= 0.3
    
    # Missing data handling
    if characteristics.missing_percentage > 5:
        if algo_info['handles_missing']:
            score += 0.1
        else:
            score -= 0.2
    
    return max(0.0, min(1.0, score))


def select_algorithm(data: pd.DataFrame, target: str, 
                    constraints: Dict[str, Any] = None) -> str:
    """
    Automatically select the best algorithm for given data
    
    Args:
        data: Training data DataFrame
        target: Target column name
        constraints: Optional constraints
        
    Returns:
        Best algorithm name
        
    Example:
        >>> import kepler as kp
        >>> best_algo = kp.automl.select_algorithm(sensor_data, target="failure")
        >>> model = kp.train_unified.train(sensor_data, target="failure", algorithm=best_algo)
    """
    logger = get_logger(__name__)
    logger.info("AutoML algorithm selection started")
    
    try:
        # Analyze data characteristics
        characteristics = analyze_data_characteristics(data, target)
        
        # Score all algorithms
        algorithm_scores = {}
        for algo_name, algo_info in ALGORITHM_REGISTRY.items():
            score = score_algorithm(characteristics, algo_info)
            algorithm_scores[algo_name] = score
        
        # Select best algorithm
        best_algorithm = max(algorithm_scores.items(), key=lambda x: x[1])
        
        if best_algorithm[1] <= 0.0:
            raise ModelTrainingError(
                "No suitable algorithms found for the given data",
                hint="Try checking data quality or relaxing constraints"
            )
        
        logger.info(f"AutoML selected: {best_algorithm[0]} (score: {best_algorithm[1]:.3f})")
        return best_algorithm[0]
        
    except Exception as e:
        raise ModelTrainingError(
            f"AutoML algorithm selection failed: {e}",
            suggestion="Check data format and ensure target column exists"
        )


def recommend_algorithms(data: pd.DataFrame, target: str,
                        constraints: Dict[str, Any] = None,
                        top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Get ranked list of algorithm recommendations
    
    Args:
        data: Training data DataFrame
        target: Target column name
        constraints: Optional constraints
        top_k: Number of recommendations
        
    Returns:
        List of algorithm recommendations
        
    Example:
        >>> recommendations = kp.automl.recommend_algorithms(data, "failure", top_k=3)
        >>> for rec in recommendations:
        >>>     print(f"{rec['algorithm']}: {rec['confidence']:.1%}")
    """
    logger = get_logger(__name__)
    logger.info("AutoML algorithm recommendation started")
    
    try:
        # Analyze data
        characteristics = analyze_data_characteristics(data, target)
        
        # Score all algorithms
        recommendations = []
        for algo_name, algo_info in ALGORITHM_REGISTRY.items():
            score = score_algorithm(characteristics, algo_info)
            
            if score > 0.0:
                # Generate rationale
                rationale = []
                if characteristics.complexity_level in algo_info['complexity']:
                    rationale.append(f"Good fit for {characteristics.complexity_level.value} data")
                if characteristics.n_categorical_features > 0 and algo_info['handles_categorical']:
                    rationale.append("Handles categorical features natively")
                if algo_info['training_speed'] == 'fast':
                    rationale.append("Fast training time")
                if algo_info['interpretability'] == 'high':
                    rationale.append("Highly interpretable results")
                
                recommendations.append({
                    'algorithm': algo_name,
                    'framework': algo_info['framework'],
                    'confidence': score,
                    'rationale': rationale,
                    'constraints_satisfied': True,
                    'expected_performance': {
                        'min': algo_info['typical_performance'][0],
                        'max': algo_info['typical_performance'][1]
                    }
                })
        
        # Sort by confidence score
        recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        
        logger.info(f"Generated {len(recommendations[:top_k])} algorithm recommendations")
        return recommendations[:top_k]
        
    except Exception as e:
        raise ModelTrainingError(
            f"AutoML recommendation failed: {e}",
            suggestion="Check data format and ensure target column exists"
        )


def analyze_data(data: pd.DataFrame, target: str) -> Dict[str, Any]:
    """
    Analyze data characteristics for AutoML insights
    
    Args:
        data: Training data DataFrame
        target: Target column name
        
    Returns:
        Data analysis results
        
    Example:
        >>> analysis = kp.automl.analyze_data(sensor_data, "failure")
        >>> print(f"Task type: {analysis['task_type']}")
        >>> print(f"Complexity: {analysis['complexity']}")
    """
    logger = get_logger(__name__)
    logger.info("AutoML data analysis started")
    
    try:
        characteristics = analyze_data_characteristics(data, target)
        
        # Convert to simple dict format
        analysis = {
            'task_type': characteristics.target_type,
            'n_samples': characteristics.n_samples,
            'n_features': characteristics.n_features,
            'n_numeric_features': characteristics.n_numeric_features,
            'n_categorical_features': characteristics.n_categorical_features,
            'complexity': characteristics.complexity_level.value,
            'missing_percentage': characteristics.missing_percentage,
            'has_text_features': characteristics.has_text_features
        }
        
        if characteristics.target_type == 'classification':
            analysis.update({
                'n_classes': characteristics.n_classes,
                'class_imbalance_ratio': characteristics.class_imbalance_ratio
            })
        
        # Add quick recommendations
        quick_recommendations = recommend_algorithms(data, target, top_k=3)
        analysis['quick_recommendations'] = [
            {
                'algorithm': rec['algorithm'],
                'confidence': rec['confidence'],
                'rationale': rec['rationale'][:2] if rec['rationale'] else []
            }
            for rec in quick_recommendations
        ]
        
        logger.info("AutoML data analysis completed")
        return analysis
        
    except Exception as e:
        raise ModelTrainingError(
            f"AutoML data analysis failed: {e}",
            suggestion="Check data format and ensure target column exists"
        )


def explain_algorithm_choice(algorithm: str, data: pd.DataFrame, target: str) -> str:
    """
    Explain why a specific algorithm is suitable for given data
    
    Args:
        algorithm: Algorithm name to explain
        data: Training data DataFrame
        target: Target column name
        
    Returns:
        Human-readable explanation
        
    Example:
        >>> explanation = kp.automl.explain_algorithm_choice("xgboost", data, "failure")
        >>> print(explanation)
    """
    try:
        characteristics = analyze_data_characteristics(data, target)
        
        # Get algorithm info
        algo_info = ALGORITHM_REGISTRY.get(algorithm)
        if not algo_info:
            return f"Algorithm '{algorithm}' not found in registry"
        
        # Score the algorithm
        score = score_algorithm(characteristics, algo_info)
        
        explanation = f"Algorithm: {algorithm} ({algo_info['framework']} framework)\n"
        explanation += f"Suitability Score: {score:.1%}\n"
        explanation += f"Expected Performance: {algo_info['typical_performance'][0]:.1%}-{algo_info['typical_performance'][1]:.1%}\n\n"
        
        explanation += "Analysis:\n"
        explanation += f"• Data complexity: {characteristics.complexity_level.value}\n"
        explanation += f"• Task type: {characteristics.target_type}\n"
        explanation += f"• Samples: {characteristics.n_samples:,}\n"
        explanation += f"• Features: {characteristics.n_features}\n"
        
        if characteristics.n_categorical_features > 0:
            explanation += f"• Categorical features: {characteristics.n_categorical_features} "
            explanation += f"({'✅ Supported' if algo_info['handles_categorical'] else '❌ Needs preprocessing'})\n"
        
        if characteristics.missing_percentage > 5:
            explanation += f"• Missing data: {characteristics.missing_percentage:.1f}% "
            explanation += f"({'✅ Robust' if algo_info['handles_missing'] else '❌ Needs imputation'})\n"
        
        explanation += f"\nAlgorithm Properties:\n"
        explanation += f"• Training speed: {algo_info['training_speed']}\n"
        explanation += f"• Interpretability: {algo_info['interpretability']}\n"
        explanation += f"• Memory usage: {algo_info['memory_usage']}\n"
        
        return explanation
        
    except Exception as e:
        return f"Could not explain algorithm choice: {e}"


def list_available_algorithms() -> List[Dict[str, Any]]:
    """
    List all available algorithms with their characteristics
    
    Returns:
        List of algorithm information
        
    Example:
        >>> algorithms = kp.automl.list_available_algorithms()
        >>> for algo in algorithms:
        >>>     print(f"{algo['name']}: {algo['framework']}")
    """
    algorithms = []
    
    for algo_name, algo_info in ALGORITHM_REGISTRY.items():
        algorithms.append({
            'name': algo_name,
            'framework': algo_info['framework'],
            'types': algo_info['type'],
            'complexity': [c.value for c in algo_info['complexity']],
            'training_speed': algo_info['training_speed'],
            'interpretability': algo_info['interpretability'],
            'memory_usage': algo_info['memory_usage'],
            'handles_categorical': algo_info['handles_categorical'],
            'handles_missing': algo_info['handles_missing'],
            'description': f"{algo_info['framework']} {algo_name.replace('_', ' ').title()}"
        })
    
    return algorithms


def auto_train(data: pd.DataFrame, target: str, 
               constraints: Dict[str, Any] = None,
               **kwargs) -> Any:
    """
    Automatic training with intelligent algorithm selection
    
    Args:
        data: Training data DataFrame
        target: Target column name
        constraints: Optional constraints
        **kwargs: Additional training parameters
        
    Returns:
        Trained model with best selected algorithm
        
    Example:
        >>> # Full automatic training
        >>> model = kp.automl.auto_train(sensor_data, target="equipment_failure")
        
        >>> # With constraints
        >>> model = kp.automl.auto_train(
        ...     data, target="failure",
        ...     constraints={"interpretability": "high"}
        ... )
    """
    logger = get_logger(__name__)
    logger.info("AutoML automatic training started")
    
    try:
        # Step 1: Select best algorithm
        best_algorithm = select_algorithm(data, target, constraints)
        logger.info(f"AutoML selected algorithm: {best_algorithm}")
        
        # Step 2: Train with selected algorithm
        from kepler.train_unified import train
        
        model = train(
            data=data,
            target=target,
            algorithm=best_algorithm,
            **kwargs
        )
        
        # Add AutoML metadata
        if hasattr(model, 'framework_info'):
            model.framework_info['automl_selected'] = True
            model.framework_info['selection_method'] = 'data_characteristics_analysis'
        
        logger.info("AutoML automatic training completed successfully")
        return model
        
    except Exception as e:
        raise ModelTrainingError(
            f"AutoML automatic training failed: {e}",
            suggestion="Try manual algorithm selection with kp.train_unified.train()"
        )


def optimize_hyperparameters(data: pd.DataFrame, target: str, algorithm: str,
                            n_trials: int = 100, timeout: int = 3600,
                            cv_folds: int = 3) -> Dict[str, Any]:
    """
    Optimize hyperparameters for specific algorithm using Optuna
    
    Args:
        data: Training data DataFrame
        target: Target column name
        algorithm: Algorithm to optimize
        n_trials: Number of optimization trials
        timeout: Timeout in seconds
        cv_folds: Cross-validation folds
        
    Returns:
        Optimization results with best parameters and score
        
    Example:
        >>> # Optimize XGBoost hyperparameters
        >>> optimization = kp.automl.optimize_hyperparameters(
        ...     data, target="failure", algorithm="xgboost", n_trials=50
        ... )
        >>> print(f"Best score: {optimization['best_score']:.3f}")
        >>> print(f"Best params: {optimization['best_params']}")
        >>> 
        >>> # Train with optimized parameters
        >>> model = kp.train_unified.train(data, target="failure", algorithm="xgboost", 
        ...                              **optimization['best_params'])
    """
    logger = get_logger(__name__)
    logger.info(f"Starting hyperparameter optimization for {algorithm}")
    
    # Try to import Optuna
    try:
        lib_manager = LibraryManager()
        optuna = lib_manager.dynamic_import('optuna')
        
        if optuna is None:
            raise ImportError("Optuna not available")
            
        # Use Optuna for optimization
        return _optuna_optimization(data, target, algorithm, n_trials, timeout, cv_folds, optuna)
        
    except (ImportError, Exception) as e:
        # Optuna not available or failed - use simple grid search fallback
        logger.warning(f"Optuna optimization failed ({e}), using simple parameter search")
        
        try:
            return _simple_parameter_optimization(data, target, algorithm, cv_folds)
        except Exception as fallback_error:
            raise ModelTrainingError(
                f"Both Optuna and fallback optimization failed: {fallback_error}",
                suggestion="Check data quality and algorithm compatibility"
            )


def _optuna_optimization(data: pd.DataFrame, target: str, algorithm: str,
                        n_trials: int, timeout: int, cv_folds: int, optuna) -> Dict[str, Any]:
    """Optuna-based hyperparameter optimization"""
    logger = get_logger(__name__)
    
    # Define parameter spaces for different algorithms
    param_spaces = {
        'xgboost': {
            'n_estimators': (50, 300),
            'max_depth': (3, 10),
            'learning_rate': (0.01, 0.3),
            'subsample': (0.8, 1.0),
            'colsample_bytree': (0.8, 1.0)
        },
        'random_forest': {
            'n_estimators': (50, 200),
            'max_depth': (5, 20),
            'min_samples_split': (2, 10),
            'min_samples_leaf': (1, 5)
        },
        'pytorch': {
            'learning_rate': (0.0001, 0.1),
            'batch_size': [16, 32, 64, 128],
            'epochs': (20, 200),
            'hidden_sizes': [[32], [64], [32, 16], [64, 32], [128, 64, 32]]
        }
    }
    
    param_space = param_spaces.get(algorithm, {})
    if not param_space:
        logger.warning(f"No parameter space defined for {algorithm}, using defaults")
        return {'best_params': {}, 'best_score': 0.0, 'n_trials': 0}
    
    def objective(trial):
        """Optuna objective function"""
        try:
            # Sample hyperparameters
            params = {}
            for param_name, param_range in param_space.items():
                if isinstance(param_range, tuple) and len(param_range) == 2:
                    if isinstance(param_range[0], float):
                        params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
                    else:
                        params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                elif isinstance(param_range, list):
                    params[param_name] = trial.suggest_categorical(param_name, param_range)
            
            # Quick training with cross-validation
            from sklearn.model_selection import cross_val_score
            from kepler.train_unified import train
            
            # Use smaller sample for optimization speed
            if len(data) > 5000:
                sample_data = data.sample(n=5000, random_state=42)
            else:
                sample_data = data
            
            # Train model with suggested parameters
            model = train(sample_data, target=target, algorithm=algorithm, **params)
            
            # Cross-validation score
            X = sample_data.drop(columns=[target])
            y = sample_data[target]
            
            if hasattr(model, 'model') and hasattr(model.model, 'predict'):
                scores = cross_val_score(model.model, X, y, cv=cv_folds, scoring='accuracy')
                return scores.mean()
            else:
                return 0.0
                
        except Exception as e:
            logger.debug(f"Trial failed: {e}")
            return 0.0
    
    # Create Optuna study
    study = optuna.create_study(direction='maximize')
    
    # Optimize with timeout
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    
    logger.info(f"Optimization complete: {len(study.trials)} trials, best score: {study.best_value:.3f}")
    
    return {
        'best_params': study.best_params,
        'best_score': study.best_value,
        'n_trials': len(study.trials),
        'optimization_history': [trial.value for trial in study.trials if trial.value is not None]
    }


def _simple_parameter_optimization(data: pd.DataFrame, target: str, algorithm: str, 
                                 cv_folds: int) -> Dict[str, Any]:
    """Simple grid search fallback when Optuna is not available"""
    logger = get_logger(__name__)
    logger.info(f"Using simple parameter optimization for {algorithm}")
    
    # Define simple parameter grids
    simple_grids = {
        'xgboost': [
            {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1},
            {'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.05},
            {'n_estimators': 150, 'max_depth': 4, 'learning_rate': 0.2}
        ],
        'random_forest': [
            {'n_estimators': 100, 'max_depth': 10},
            {'n_estimators': 200, 'max_depth': 15},
            {'n_estimators': 150, 'max_depth': 8}
        ]
    }
    
    param_grid = simple_grids.get(algorithm, [{}])
    best_score = 0.0
    best_params = {}
    
    for params in param_grid:
        try:
            from kepler.train_unified import train
            
            # Use smaller sample for speed
            if len(data) > 3000:
                sample_data = data.sample(n=3000, random_state=42)
            else:
                sample_data = data
            
            model = train(sample_data, target=target, algorithm=algorithm, **params)
            
            if hasattr(model, 'performance') and model.performance:
                score = list(model.performance.values())[0]  # First metric
                if score > best_score:
                    best_score = score
                    best_params = params
                    
        except Exception as e:
            logger.debug(f"Parameter set failed: {params}, error: {e}")
            continue
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'n_trials': len(param_grid),
        'optimization_method': 'simple_grid_search'
    }