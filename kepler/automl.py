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


def engineer_features(data: pd.DataFrame, target: str = None, 
                     strategy: str = "auto") -> Dict[str, Any]:
    """
    Automatic feature engineering and selection
    
    Args:
        data: Input data DataFrame
        target: Target column name (optional for unsupervised)
        strategy: Feature engineering strategy ("auto", "minimal", "aggressive")
        
    Returns:
        Feature engineering results with transformed data and metadata
        
    Example:
        >>> # Automatic feature engineering
        >>> result = kp.automl.engineer_features(sensor_data, target="failure")
        >>> transformed_data = result['transformed_data']
        >>> feature_info = result['feature_info']
        >>> print(f"Generated {len(feature_info['new_features'])} new features")
    """
    logger = get_logger(__name__)
    logger.info(f"Starting automatic feature engineering with {strategy} strategy")
    
    try:
        # Analyze data first
        original_features = list(data.columns)
        if target and target in original_features:
            feature_columns = [col for col in original_features if col != target]
        else:
            feature_columns = original_features
        
        # Initialize transformed data
        transformed_data = data.copy()
        new_features = []
        feature_operations = []
        
        # Apply feature engineering based on strategy
        if strategy in ["auto", "aggressive"]:
            # Polynomial features for numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if target:
                numeric_cols = [col for col in numeric_cols if col != target]
            
            if len(numeric_cols) >= 2 and len(numeric_cols) <= 10:  # Reasonable range
                poly_features = _create_polynomial_features(data, numeric_cols, degree=2)
                transformed_data = pd.concat([transformed_data, poly_features], axis=1)
                new_features.extend(poly_features.columns.tolist())
                feature_operations.append(f"Polynomial features (degree=2) for {len(numeric_cols)} numeric columns")
            
            # Interaction features
            if len(numeric_cols) >= 2:
                interaction_features = _create_interaction_features(data, numeric_cols[:5])  # Limit to 5 cols
                transformed_data = pd.concat([transformed_data, interaction_features], axis=1)
                new_features.extend(interaction_features.columns.tolist())
                feature_operations.append(f"Interaction features for top {min(5, len(numeric_cols))} numeric columns")
        
        if strategy in ["auto", "minimal", "aggressive"]:
            # Handle missing values
            missing_cols = data.columns[data.isnull().any()].tolist()
            if target and target in missing_cols:
                missing_cols.remove(target)
            
            if missing_cols:
                imputed_data = _handle_missing_values(transformed_data, missing_cols, target)
                transformed_data = imputed_data
                feature_operations.append(f"Missing value imputation for {len(missing_cols)} columns")
            
            # Encode categorical variables
            categorical_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()
            if target and target in categorical_cols:
                categorical_cols.remove(target)
            
            if categorical_cols:
                encoded_data = _encode_categorical_features(transformed_data, categorical_cols)
                transformed_data = encoded_data
                feature_operations.append(f"Categorical encoding for {len(categorical_cols)} columns")
        
        # Feature selection if too many features
        final_feature_cols = [col for col in transformed_data.columns if col != target]
        
        if len(final_feature_cols) > 50 and target:  # Too many features
            selected_data = _select_best_features(transformed_data, target, max_features=50)
            transformed_data = selected_data
            feature_operations.append(f"Feature selection: reduced to top 50 features")
        
        # Prepare results
        final_features = [col for col in transformed_data.columns if col != target]
        
        feature_info = {
            'original_features': len(feature_columns),
            'final_features': len(final_features),
            'new_features': new_features,
            'operations_applied': feature_operations,
            'strategy_used': strategy
        }
        
        logger.info(f"Feature engineering complete: {len(feature_columns)} → {len(final_features)} features")
        
        return {
            'transformed_data': transformed_data,
            'feature_info': feature_info,
            'feature_names': final_features,
            'target_column': target
        }
        
    except Exception as e:
        raise ModelTrainingError(
            f"Automatic feature engineering failed: {e}",
            suggestion="Try strategy='minimal' or check data types"
        )


def _create_polynomial_features(data: pd.DataFrame, numeric_cols: List[str], degree: int = 2) -> pd.DataFrame:
    """Create polynomial features for numeric columns"""
    poly_features = pd.DataFrame(index=data.index)
    
    # Quadratic features (x^2)
    for col in numeric_cols:
        if data[col].var() > 1e-6:  # Avoid constant columns
            poly_features[f"{col}_squared"] = data[col] ** 2
    
    # Cross-product features (x*y) for degree 2
    if degree >= 2:
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                if data[col1].var() > 1e-6 and data[col2].var() > 1e-6:
                    poly_features[f"{col1}_{col2}_interaction"] = data[col1] * data[col2]
    
    return poly_features


def _create_interaction_features(data: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """Create interaction features between numeric columns"""
    interaction_features = pd.DataFrame(index=data.index)
    
    for i, col1 in enumerate(numeric_cols):
        for col2 in numeric_cols[i+1:]:
            # Ratio features
            if (data[col2] != 0).all():
                interaction_features[f"{col1}_{col2}_ratio"] = data[col1] / data[col2]
            
            # Sum and difference features
            interaction_features[f"{col1}_{col2}_sum"] = data[col1] + data[col2]
            interaction_features[f"{col1}_{col2}_diff"] = data[col1] - data[col2]
    
    return interaction_features


def _handle_missing_values(data: pd.DataFrame, missing_cols: List[str], target: str = None) -> pd.DataFrame:
    """Handle missing values with appropriate imputation"""
    imputed_data = data.copy()
    
    for col in missing_cols:
        if col == target:
            continue
            
        if data[col].dtype in ['object', 'category']:
            # Categorical: fill with mode
            mode_value = data[col].mode()
            if len(mode_value) > 0:
                imputed_data[col] = imputed_data[col].fillna(mode_value[0])
            else:
                imputed_data[col] = imputed_data[col].fillna('unknown')
        else:
            # Numeric: fill with median
            median_value = data[col].median()
            imputed_data[col] = imputed_data[col].fillna(median_value)
            
            # Create missing indicator
            imputed_data[f"{col}_was_missing"] = data[col].isnull().astype(int)
    
    return imputed_data


def _encode_categorical_features(data: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
    """Encode categorical features using appropriate encoding"""
    encoded_data = data.copy()
    
    for col in categorical_cols:
        unique_values = data[col].nunique()
        
        if unique_values <= 10:  # Low cardinality - use one-hot encoding
            dummies = pd.get_dummies(data[col], prefix=col, drop_first=True)
            encoded_data = pd.concat([encoded_data, dummies], axis=1)
            encoded_data = encoded_data.drop(columns=[col])
        else:  # High cardinality - use label encoding
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            encoded_data[f"{col}_encoded"] = le.fit_transform(data[col].fillna('unknown'))
            encoded_data = encoded_data.drop(columns=[col])
    
    return encoded_data


def _select_best_features(data: pd.DataFrame, target: str, max_features: int = 50) -> pd.DataFrame:
    """Select best features using correlation-based selection"""
    try:
        # Simple correlation-based feature selection
        feature_cols = [col for col in data.columns if col != target]
        
        if len(feature_cols) <= max_features:
            return data
        
        # Calculate correlations with target
        correlations = {}
        for col in feature_cols:
            try:
                if data[col].dtype in [np.number]:
                    corr = abs(data[col].corr(data[target]))
                    if not np.isnan(corr):
                        correlations[col] = corr
            except:
                continue
        
        # Select top features by correlation
        if correlations:
            sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
            selected_features = [feat[0] for feat in sorted_features[:max_features]]
            
            # Always include target
            final_columns = selected_features + [target]
            return data[final_columns]
        else:
            # Fallback: select first max_features
            selected_cols = feature_cols[:max_features] + [target]
            return data[selected_cols]
            
    except Exception as e:
        logger = get_logger(__name__)
        logger.warning(f"Feature selection failed: {e}, keeping all features")
        return data


def run_experiment_suite(data: pd.DataFrame, target: str,
                        algorithms: List[str] = None,
                        feature_engineering: bool = True,
                        hyperparameter_optimization: bool = False,
                        cv_folds: int = 3,
                        max_parallel_jobs: int = 3,
                        timeout_per_algorithm: int = 300) -> Dict[str, Any]:
    """
    Run parallel experiments with multiple algorithms and rank results
    
    Args:
        data: Training data DataFrame
        target: Target column name
        algorithms: List of algorithms to test (None for auto-selection)
        feature_engineering: Whether to apply automatic feature engineering
        hyperparameter_optimization: Whether to optimize hyperparameters
        cv_folds: Cross-validation folds
        max_parallel_jobs: Maximum parallel experiments
        timeout_per_algorithm: Timeout per algorithm in seconds
        
    Returns:
        Experiment results with rankings and detailed metrics
        
    Example:
        >>> # Run full experiment suite
        >>> results = kp.automl.run_experiment_suite(
        ...     sensor_data, target="failure",
        ...     algorithms=["random_forest", "xgboost", "pytorch"],
        ...     feature_engineering=True,
        ...     hyperparameter_optimization=True
        ... )
        >>> 
        >>> # Check rankings
        >>> for rank, result in enumerate(results['rankings'], 1):
        ...     print(f"{rank}. {result['algorithm']}: {result['score']:.3f}")
    """
    logger = get_logger(__name__)
    logger.info("Starting parallel experiment suite")
    
    try:
        # Auto-select algorithms if not provided
        if algorithms is None:
            recommendations = recommend_algorithms(data, target, top_k=5)
            algorithms = [rec['algorithm'] for rec in recommendations]
            logger.info(f"Auto-selected algorithms: {algorithms}")
        
        # Apply feature engineering if requested
        working_data = data
        feature_info = {}
        
        if feature_engineering:
            logger.info("Applying automatic feature engineering...")
            fe_result = engineer_features(data, target, strategy='auto')
            working_data = fe_result['transformed_data']
            feature_info = fe_result['feature_info']
            logger.info(f"Feature engineering: {feature_info['original_features']} → {feature_info['final_features']} features")
        
        # Run parallel experiments
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time
        
        experiment_results = {}
        start_time = time.time()
        
        logger.info(f"Running {len(algorithms)} experiments in parallel (max {max_parallel_jobs} jobs)")
        
        with ThreadPoolExecutor(max_workers=max_parallel_jobs) as executor:
            future_to_algorithm = {}
            
            for algorithm in algorithms:
                future = executor.submit(
                    _run_single_experiment,
                    working_data, target, algorithm, 
                    hyperparameter_optimization, cv_folds, timeout_per_algorithm
                )
                future_to_algorithm[future] = algorithm
            
            # Collect results
            for future in as_completed(future_to_algorithm, timeout=len(algorithms) * timeout_per_algorithm):
                algorithm = future_to_algorithm[future]
                
                try:
                    result = future.result()
                    experiment_results[algorithm] = result
                    logger.info(f"Experiment completed: {algorithm} - Score: {result.get('score', 0.0):.3f}")
                    
                except Exception as e:
                    logger.warning(f"Experiment failed: {algorithm} - {e}")
                    experiment_results[algorithm] = {
                        'status': 'failed',
                        'error': str(e),
                        'score': 0.0,
                        'algorithm': algorithm
                    }
        
        # Rank results by performance
        rankings = _rank_experiment_results(experiment_results)
        
        total_time = time.time() - start_time
        
        # Prepare comprehensive results
        suite_results = {
            'rankings': rankings,
            'detailed_results': experiment_results,
            'feature_engineering': feature_info,
            'experiment_config': {
                'algorithms_tested': algorithms,
                'feature_engineering_applied': feature_engineering,
                'hyperparameter_optimization': hyperparameter_optimization,
                'cv_folds': cv_folds,
                'total_experiments': len(algorithms),
                'successful_experiments': len([r for r in experiment_results.values() if r.get('status') != 'failed']),
                'total_time_seconds': total_time
            }
        }
        
        logger.info(f"Experiment suite completed: {len(rankings)} successful experiments in {total_time:.1f}s")
        return suite_results
        
    except Exception as e:
        raise ModelTrainingError(
            f"Experiment suite failed: {e}",
            suggestion="Try reducing number of algorithms or increasing timeout"
        )


def _run_single_experiment(data: pd.DataFrame, target: str, algorithm: str,
                          optimize_hyperparameters: bool, cv_folds: int,
                          timeout: int) -> Dict[str, Any]:
    """Run single experiment with algorithm"""
    logger = get_logger(__name__)
    
    try:
        experiment_start = time.time()
        
        # Optimize hyperparameters if requested
        best_params = {}
        if optimize_hyperparameters:
            try:
                optimization = optimize_hyperparameters(
                    data, target, algorithm, 
                    n_trials=20,  # Reduced for speed
                    timeout=timeout // 2  # Half timeout for optimization
                )
                best_params = optimization['best_params']
                logger.debug(f"{algorithm}: Optimized parameters: {best_params}")
            except Exception as e:
                logger.debug(f"{algorithm}: Hyperparameter optimization failed: {e}")
        
        # Train model with best parameters
        from kepler.train_unified import train
        
        model = train(
            data=data,
            target=target,
            algorithm=algorithm,
            **best_params
        )
        
        # Evaluate with cross-validation
        from sklearn.model_selection import cross_val_score
        
        X = data.drop(columns=[target])
        y = data[target]
        
        if hasattr(model, 'model') and hasattr(model.model, 'predict'):
            cv_scores = cross_val_score(model.model, X, y, cv=cv_folds, scoring='accuracy')
            mean_score = cv_scores.mean()
            std_score = cv_scores.std()
        else:
            # Fallback to model's reported performance
            if hasattr(model, 'performance') and model.performance:
                mean_score = list(model.performance.values())[0]
                std_score = 0.0
            else:
                mean_score = 0.0
                std_score = 0.0
        
        experiment_time = time.time() - experiment_start
        
        return {
            'status': 'success',
            'algorithm': algorithm,
            'score': mean_score,
            'score_std': std_score,
            'cv_scores': cv_scores.tolist() if 'cv_scores' in locals() else [],
            'hyperparameters': best_params,
            'hyperparameters_optimized': optimize_hyperparameters and bool(best_params),
            'training_time': experiment_time,
            'model_info': {
                'framework': getattr(model, 'framework_info', {}).get('framework', algorithm),
                'model_type': getattr(model, 'model_type', 'unknown')
            }
        }
        
    except Exception as e:
        return {
            'status': 'failed',
            'algorithm': algorithm,
            'score': 0.0,
            'error': str(e),
            'training_time': 0.0
        }


def _rank_experiment_results(experiment_results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rank experiment results by performance"""
    
    # Filter successful experiments
    successful_experiments = [
        result for result in experiment_results.values()
        if result.get('status') == 'success' and result.get('score', 0) > 0
    ]
    
    # Sort by score (descending)
    successful_experiments.sort(key=lambda x: x.get('score', 0), reverse=True)
    
    # Add ranking information
    rankings = []
    for rank, result in enumerate(successful_experiments, 1):
        ranking_entry = {
            'rank': rank,
            'algorithm': result['algorithm'],
            'score': result['score'],
            'score_std': result.get('score_std', 0.0),
            'training_time': result.get('training_time', 0.0),
            'hyperparameters_optimized': result.get('hyperparameters_optimized', False),
            'hyperparameters': result.get('hyperparameters', {}),
            'framework': result.get('model_info', {}).get('framework', result['algorithm'])
        }
        rankings.append(ranking_entry)
    
    return rankings


def get_experiment_leaderboard(experiment_results: Dict[str, Any]) -> str:
    """
    Generate human-readable leaderboard from experiment results
    
    Args:
        experiment_results: Results from run_experiment_suite()
        
    Returns:
        Formatted leaderboard string
        
    Example:
        >>> results = kp.automl.run_experiment_suite(data, "failure")
        >>> leaderboard = kp.automl.get_experiment_leaderboard(results)
        >>> print(leaderboard)
    """
    rankings = experiment_results.get('rankings', [])
    
    if not rankings:
        return "No successful experiments to display"
    
    leaderboard = "🏆 EXPERIMENT LEADERBOARD\n"
    leaderboard += "=" * 50 + "\n"
    leaderboard += f"{'Rank':<4} {'Algorithm':<15} {'Score':<8} {'Time':<8} {'Optimized':<10}\n"
    leaderboard += "-" * 50 + "\n"
    
    for result in rankings:
        optimized = "✅" if result['hyperparameters_optimized'] else "❌"
        leaderboard += f"{result['rank']:<4} {result['algorithm']:<15} {result['score']:<8.3f} {result['training_time']:<8.1f}s {optimized:<10}\n"
    
    # Add experiment summary
    config = experiment_results.get('experiment_config', {})
    leaderboard += "\n📊 EXPERIMENT SUMMARY\n"
    leaderboard += f"Total algorithms tested: {config.get('total_experiments', 0)}\n"
    leaderboard += f"Successful experiments: {config.get('successful_experiments', 0)}\n"
    leaderboard += f"Total time: {config.get('total_time_seconds', 0):.1f}s\n"
    
    if experiment_results.get('feature_engineering'):
        fe_info = experiment_results['feature_engineering']
        leaderboard += f"Feature engineering: {fe_info['original_features']} → {fe_info['final_features']} features\n"
    
    return leaderboard


def automl_pipeline(data: pd.DataFrame, target: str,
                   industrial_constraints: Dict[str, Any] = None,
                   optimization_time: str = "30m",
                   interpretability_required: bool = True,
                   max_inference_latency_ms: int = 200,
                   max_model_size_mb: int = 50) -> Dict[str, Any]:
    """
    Complete AutoML pipeline with industrial constraints
    
    Args:
        data: Training data DataFrame
        target: Target column name
        industrial_constraints: Specific industrial constraints
        optimization_time: Total optimization time ("30m", "1h", "2h")
        interpretability_required: Whether model must be interpretable
        max_inference_latency_ms: Maximum inference latency in milliseconds
        max_model_size_mb: Maximum model size in megabytes
        
    Returns:
        Complete AutoML pipeline results with best model and analysis
        
    Example:
        >>> # Industrial AutoML with constraints
        >>> pipeline_result = kp.automl.automl_pipeline(
        ...     sensor_data, target="equipment_failure",
        ...     optimization_time="1h",
        ...     interpretability_required=True,
        ...     max_inference_latency_ms=100
        ... )
        >>> 
        >>> best_model = pipeline_result['best_model']
        >>> deployment_ready = pipeline_result['deployment_ready']
        >>> print(f"Best: {pipeline_result['best_algorithm']} - {pipeline_result['best_score']:.3f}")
    """
    logger = get_logger(__name__)
    logger.info("Starting complete AutoML pipeline with industrial constraints")
    
    try:
        # Parse optimization time
        time_minutes = _parse_time_string(optimization_time)
        logger.info(f"AutoML pipeline budget: {time_minutes} minutes")
        
        # Prepare industrial constraints
        if industrial_constraints is None:
            industrial_constraints = {}
        
        # Add default industrial constraints
        constraints = {
            'interpretability': 'high' if interpretability_required else 'any',
            'max_training_time_minutes': time_minutes * 0.6,  # 60% for training
            'max_inference_latency_ms': max_inference_latency_ms,
            'max_model_size_mb': max_model_size_mb,
            'robustness_required': True,
            **industrial_constraints
        }
        
        logger.info(f"Industrial constraints: {constraints}")
        
        # Step 1: Data analysis and algorithm pre-filtering
        logger.info("Step 1: Analyzing data and filtering algorithms...")
        data_analysis = analyze_data(data, target)
        
        # Filter algorithms by industrial constraints
        suitable_algorithms = _filter_algorithms_for_industrial_use(constraints)
        logger.info(f"Algorithms suitable for constraints: {suitable_algorithms}")
        
        # Step 2: Feature engineering with industrial focus
        logger.info("Step 2: Applying industrial-focused feature engineering...")
        fe_result = engineer_features(data, target, strategy='auto')
        engineered_data = fe_result['transformed_data']
        
        # Step 3: Run constrained experiment suite
        logger.info("Step 3: Running constrained experiment suite...")
        experiment_results = run_experiment_suite(
            engineered_data,
            target=target,
            algorithms=suitable_algorithms,
            feature_engineering=False,  # Already done
            hyperparameter_optimization=True,
            cv_folds=3,
            max_parallel_jobs=3,
            timeout_per_algorithm=int(time_minutes * 60 * 0.3 / len(suitable_algorithms))  # Distribute time
        )
        
        # Step 4: Validate industrial constraints
        logger.info("Step 4: Validating industrial constraints...")
        validated_results = _validate_industrial_constraints(
            experiment_results['rankings'], constraints
        )
        
        # Step 5: Select best model that satisfies all constraints
        best_model_info = _select_best_industrial_model(validated_results, constraints)
        
        # Step 6: Prepare deployment-ready package
        deployment_package = _prepare_deployment_package(
            best_model_info, fe_result, constraints
        )
        
        # Compile final results
        pipeline_results = {
            'best_algorithm': best_model_info['algorithm'],
            'best_score': best_model_info['score'],
            'best_model': best_model_info,
            'all_rankings': validated_results,
            'data_analysis': data_analysis,
            'feature_engineering': fe_result['feature_info'],
            'constraints_applied': constraints,
            'deployment_ready': deployment_package,
            'pipeline_summary': {
                'total_time_minutes': time_minutes,
                'algorithms_tested': len(suitable_algorithms),
                'constraints_satisfied': best_model_info.get('constraints_satisfied', False),
                'ready_for_production': deployment_package['production_ready']
            }
        }
        
        logger.info(f"AutoML pipeline completed: Best algorithm {best_model_info['algorithm']} with score {best_model_info['score']:.3f}")
        return pipeline_results
        
    except Exception as e:
        raise ModelTrainingError(
            f"AutoML pipeline failed: {e}",
            suggestion="Try relaxing constraints or increasing optimization time"
        )


def _parse_time_string(time_str: str) -> int:
    """Parse time string to minutes"""
    time_str = time_str.lower().strip()
    
    if time_str.endswith('m'):
        return int(time_str[:-1])
    elif time_str.endswith('h'):
        return int(time_str[:-1]) * 60
    elif time_str.endswith('min'):
        return int(time_str[:-3])
    else:
        # Assume minutes
        return int(time_str)


def _filter_algorithms_for_industrial_use(constraints: Dict[str, Any]) -> List[str]:
    """Filter algorithms suitable for industrial constraints"""
    suitable_algorithms = []
    
    for algo_name, algo_info in ALGORITHM_REGISTRY.items():
        # Check interpretability
        if constraints.get('interpretability') == 'high':
            if algo_info['interpretability'] != 'high':
                continue
        
        # Check training time
        max_time = constraints.get('max_training_time_minutes', float('inf'))
        speed_map = {'fast': 5, 'medium': 15, 'slow': 45, 'very_slow': 120}
        algo_time = speed_map.get(algo_info['training_speed'], 30)
        
        if algo_time > max_time:
            continue
        
        # Check memory usage for model size constraint
        max_size = constraints.get('max_model_size_mb', float('inf'))
        if max_size < 10:  # Very small models only
            if algo_info['memory_usage'] in ['high', 'very_high']:
                continue
        
        suitable_algorithms.append(algo_name)
    
    # Ensure at least one algorithm is available
    if not suitable_algorithms:
        # Fallback to most basic algorithm
        suitable_algorithms = ['random_forest']
    
    return suitable_algorithms


def _validate_industrial_constraints(rankings: List[Dict[str, Any]], 
                                   constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Validate experiment results against industrial constraints"""
    
    validated_rankings = []
    
    for result in rankings:
        # Copy original result
        validated_result = result.copy()
        validated_result['constraints_satisfied'] = True
        validated_result['constraint_violations'] = []
        
        # Check inference latency constraint
        max_latency = constraints.get('max_inference_latency_ms', float('inf'))
        if max_latency < 200:  # Strict latency requirement
            algo_info = ALGORITHM_REGISTRY.get(result['algorithm'], {})
            if algo_info.get('training_speed') in ['slow', 'very_slow']:
                validated_result['constraints_satisfied'] = False
                validated_result['constraint_violations'].append(
                    f"Algorithm may not meet {max_latency}ms latency requirement"
                )
        
        # Check model size constraint
        max_size = constraints.get('max_model_size_mb', float('inf'))
        if max_size < 50:  # Strict size requirement
            algo_info = ALGORITHM_REGISTRY.get(result['algorithm'], {})
            if algo_info.get('memory_usage') in ['high', 'very_high']:
                validated_result['constraints_satisfied'] = False
                validated_result['constraint_violations'].append(
                    f"Model may exceed {max_size}MB size limit"
                )
        
        # Check interpretability
        if constraints.get('interpretability') == 'high':
            algo_info = ALGORITHM_REGISTRY.get(result['algorithm'], {})
            if algo_info.get('interpretability') != 'high':
                validated_result['constraints_satisfied'] = False
                validated_result['constraint_violations'].append(
                    "Algorithm does not meet high interpretability requirement"
                )
        
        validated_rankings.append(validated_result)
    
    return validated_rankings


def _select_best_industrial_model(validated_results: List[Dict[str, Any]], 
                                 constraints: Dict[str, Any]) -> Dict[str, Any]:
    """Select best model that satisfies industrial constraints"""
    
    # First, try to find models that satisfy all constraints
    fully_compliant = [r for r in validated_results if r.get('constraints_satisfied', False)]
    
    if fully_compliant:
        # Return best performing compliant model
        return fully_compliant[0]
    else:
        # No fully compliant models - return best with warnings
        if validated_results:
            best_model = validated_results[0]
            best_model['warning'] = "Best model does not satisfy all industrial constraints"
            return best_model
        else:
            raise ModelTrainingError(
                "No models found that satisfy industrial constraints",
                suggestion="Try relaxing constraints or using different algorithms"
            )


def _prepare_deployment_package(best_model_info: Dict[str, Any], 
                               fe_result: Dict[str, Any],
                               constraints: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare deployment-ready package with all necessary components"""
    
    deployment_package = {
        'production_ready': best_model_info.get('constraints_satisfied', False),
        'model_info': {
            'algorithm': best_model_info['algorithm'],
            'framework': best_model_info.get('framework', 'unknown'),
            'performance_score': best_model_info['score'],
            'hyperparameters': best_model_info.get('hyperparameters', {}),
            'training_time': best_model_info.get('training_time', 0.0)
        },
        'feature_engineering': {
            'operations_applied': fe_result['feature_info']['operations_applied'],
            'feature_names': fe_result['feature_names'],
            'original_to_final_features': f"{fe_result['feature_info']['original_features']} → {fe_result['feature_info']['final_features']}"
        },
        'industrial_compliance': {
            'constraints_satisfied': best_model_info.get('constraints_satisfied', False),
            'constraint_violations': best_model_info.get('constraint_violations', []),
            'interpretability': constraints.get('interpretability', 'any'),
            'max_latency_ms': constraints.get('max_inference_latency_ms', 'unlimited'),
            'max_size_mb': constraints.get('max_model_size_mb', 'unlimited')
        },
        'deployment_recommendations': []
    }
    
    # Add deployment recommendations
    if best_model_info.get('constraints_satisfied', False):
        deployment_package['deployment_recommendations'].append(
            "✅ Model meets all industrial constraints - ready for production deployment"
        )
    else:
        deployment_package['deployment_recommendations'].append(
            "⚠️ Model has constraint violations - review before production deployment"
        )
    
    if best_model_info['score'] > 0.85:
        deployment_package['deployment_recommendations'].append(
            "✅ High performance model - excellent for production use"
        )
    elif best_model_info['score'] > 0.75:
        deployment_package['deployment_recommendations'].append(
            "✅ Good performance model - suitable for production with monitoring"
        )
    else:
        deployment_package['deployment_recommendations'].append(
            "⚠️ Lower performance model - consider additional data or feature engineering"
        )
    
    return deployment_package


def industrial_automl(data: pd.DataFrame, target: str,
                     use_case: str = "predictive_maintenance",
                     optimization_budget: str = "1h",
                     production_environment: str = "edge") -> Dict[str, Any]:
    """
    Complete industrial AutoML pipeline with predefined constraints
    
    Args:
        data: Training data DataFrame
        target: Target column name
        use_case: Industrial use case ("predictive_maintenance", "quality_control", "anomaly_detection")
        optimization_budget: Time budget for optimization
        production_environment: Target environment ("edge", "cloud", "hybrid")
        
    Returns:
        Production-ready AutoML results
        
    Example:
        >>> # Predictive maintenance AutoML
        >>> result = kp.automl.industrial_automl(
        ...     sensor_data, 
        ...     target="equipment_failure",
        ...     use_case="predictive_maintenance",
        ...     production_environment="edge"
        ... )
        >>> 
        >>> if result['deployment_ready']:
        ...     print(f"✅ Ready for deployment: {result['best_algorithm']}")
        ... else:
        ...     print(f"⚠️ Needs review: {result['issues']}")
    """
    logger = get_logger(__name__)
    logger.info(f"Starting industrial AutoML for {use_case} use case")
    
    # Define industrial constraint templates
    constraint_templates = {
        'predictive_maintenance': {
            'interpretability_required': True,
            'max_inference_latency_ms': 100,
            'max_model_size_mb': 20,
            'robustness_required': True,
            'real_time_capable': True
        },
        'quality_control': {
            'interpretability_required': True,
            'max_inference_latency_ms': 50,
            'max_model_size_mb': 10,
            'accuracy_threshold': 0.95,
            'false_positive_tolerance': 'low'
        },
        'anomaly_detection': {
            'interpretability_required': False,
            'max_inference_latency_ms': 200,
            'max_model_size_mb': 100,
            'sensitivity_required': 'high',
            'false_negative_tolerance': 'very_low'
        }
    }
    
    # Environment-specific constraints
    environment_constraints = {
        'edge': {
            'max_model_size_mb': 20,
            'max_inference_latency_ms': 100,
            'cpu_only': True,
            'memory_efficient': True
        },
        'cloud': {
            'max_model_size_mb': 500,
            'max_inference_latency_ms': 1000,
            'gpu_available': True,
            'scalable': True
        },
        'hybrid': {
            'max_model_size_mb': 50,
            'max_inference_latency_ms': 200,
            'offline_capable': True,
            'sync_efficient': True
        }
    }
    
    # Merge constraints
    use_case_constraints = constraint_templates.get(use_case, {})
    env_constraints = environment_constraints.get(production_environment, {})
    
    final_constraints = {
        **use_case_constraints,
        **env_constraints
    }
    
    try:
        # Run complete AutoML pipeline
        pipeline_result = automl_pipeline(
            data, target,
            industrial_constraints=final_constraints,
            optimization_time=optimization_budget
        )
        
        # Add industrial-specific metadata
        industrial_result = {
            **pipeline_result,
            'use_case': use_case,
            'production_environment': production_environment,
            'industrial_compliance': _assess_industrial_compliance(
                pipeline_result, final_constraints, use_case
            )
        }
        
        logger.info(f"Industrial AutoML completed for {use_case}")
        return industrial_result
        
    except Exception as e:
        raise ModelTrainingError(
            f"Industrial AutoML pipeline failed: {e}",
            suggestion=f"Try different use_case or production_environment settings"
        )


def _assess_industrial_compliance(pipeline_result: Dict[str, Any], 
                                constraints: Dict[str, Any],
                                use_case: str) -> Dict[str, Any]:
    """Assess compliance with industrial requirements"""
    
    compliance = {
        'overall_compliant': True,
        'compliance_score': 1.0,
        'critical_issues': [],
        'warnings': [],
        'recommendations': []
    }
    
    best_model = pipeline_result['best_model']
    
    # Check performance threshold
    score = best_model['score']
    if use_case == 'quality_control' and score < 0.95:
        compliance['critical_issues'].append(f"Score {score:.3f} below quality control threshold (0.95)")
        compliance['overall_compliant'] = False
    elif score < 0.80:
        compliance['warnings'].append(f"Score {score:.3f} may be too low for industrial use")
        compliance['compliance_score'] -= 0.2
    
    # Check constraint satisfaction
    if not best_model.get('constraints_satisfied', True):
        violations = best_model.get('constraint_violations', [])
        for violation in violations:
            compliance['critical_issues'].append(violation)
        compliance['overall_compliant'] = False
    
    # Add recommendations
    if compliance['overall_compliant']:
        compliance['recommendations'].append("✅ Model ready for production deployment")
        compliance['recommendations'].append(f"✅ Suitable for {use_case} use case")
    else:
        compliance['recommendations'].append("⚠️ Address critical issues before deployment")
        compliance['recommendations'].append("💡 Consider relaxing constraints or gathering more data")
    
    return compliance


def _prepare_deployment_package(best_model_info: Dict[str, Any], 
                               fe_result: Dict[str, Any],
                               constraints: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare deployment-ready package"""
    
    return {
        'production_ready': best_model_info.get('constraints_satisfied', False),
        'model_config': {
            'algorithm': best_model_info['algorithm'],
            'hyperparameters': best_model_info.get('hyperparameters', {}),
            'expected_performance': best_model_info['score']
        },
        'feature_pipeline': {
            'operations': fe_result['feature_info']['operations_applied'],
            'input_features': fe_result['feature_info']['original_features'],
            'output_features': fe_result['feature_info']['final_features']
        },
        'deployment_config': {
            'max_latency_ms': constraints.get('max_inference_latency_ms', 1000),
            'max_memory_mb': constraints.get('max_model_size_mb', 100),
            'interpretability': constraints.get('interpretability', 'any'),
            'environment': constraints.get('production_environment', 'cloud')
        },
        'monitoring_recommendations': [
            "Monitor inference latency and memory usage",
            "Set up data drift detection",
            "Configure performance alerts",
            "Implement model retraining triggers"
        ]
    }