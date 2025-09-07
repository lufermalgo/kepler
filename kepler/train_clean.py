"""
Kepler Train Module - Unlimited AI Framework Training

Supports ANY AI framework through dynamic library loading:
- Traditional ML: sklearn, XGBoost, LightGBM, CatBoost
- Deep Learning: PyTorch, TensorFlow, Keras, JAX (future)
- Generative AI: transformers, langchain, openai (future)
- Computer Vision: opencv, pillow (future)
- NLP: spacy, nltk (future)

Philosophy: "Si está en Python, Kepler lo entrena"
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, Dict, Any, List
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, r2_score, classification_report, mean_squared_error
import joblib
import os
from datetime import datetime

from kepler.utils.logging import get_logger
from kepler.utils.exceptions import ModelTrainingError


class KeplerModel:
    """Simple wrapper for ML models with Kepler-specific functionality"""
    
    def __init__(self, model, model_type: str, target_column: str, feature_columns: List[str]):
        self.model = model
        self.model_type = model_type  # 'classification' or 'regression'
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.trained = False
        self.performance = {}
        
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data"""
        if not self.trained:
            raise ValueError("❌ Model not trained yet. Call train() first.")
            
        # Select only the features used in training
        X = data[self.feature_columns]
        return self.model.predict(X)
    
    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities (classification only)"""
        if self.model_type != 'classification':
            raise ValueError("❌ predict_proba only available for classification models")
            
        if not self.trained:
            raise ValueError("❌ Model not trained yet. Call train() first.")
            
        X = data[self.feature_columns]
        return self.model.predict_proba(X)
    
    def save(self, filename: str = None) -> str:
        """Save trained model to disk"""
        if not self.trained:
            raise ValueError("❌ Cannot save untrained model")
            
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"kepler_model_{self.model_type}_{timestamp}.pkl"
        
        # Save to models directory
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)
        filepath = os.path.join(models_dir, filename)
        
        # Save model and metadata
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'target_column': self.target_column,
            'feature_columns': self.feature_columns,
            'performance': self.performance,
            'trained_at': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        print(f"✅ Model saved to: {filepath}")
        return filepath


def random_forest(
    data: pd.DataFrame,
    target: str,
    features: List[str] = None,
    test_size: float = 0.2,
    n_estimators: int = 100,
    task: str = "auto"
) -> KeplerModel:
    """
    Train a Random Forest model in the simplest way possible.
    
    Args:
        data: DataFrame with features and target
        target: Name of target column
        features: List of feature columns (auto-detect if None)
        test_size: Fraction for test set (default: 0.2)
        n_estimators: Number of trees (default: 100)
        task: 'classification', 'regression', or 'auto'
        
    Returns:
        KeplerModel: Trained model ready for predictions
        
    Example:
        >>> import kepler as kp
        >>> data = kp.data.from_splunk("sensor_data")
        >>> model = kp.train.random_forest(data, target="anomaly")
        >>> predictions = model.predict(new_data)
    """
    
    print(f"🌲 Training Random Forest model...")
    
    # Auto-detect features if not provided
    if features is None:
        features = [col for col in data.columns if col != target]
        print(f"📊 Auto-detected {len(features)} feature columns")
    
    # Prepare data
    X = data[features]
    y = data[target]
    
    # Auto-detect task type
    if task == "auto":
        unique_values = y.nunique()
        if unique_values <= 10:  # Likely classification
            task = "classification"
            print(f"🎯 Auto-detected task: classification ({unique_values} unique values)")
        else:
            task = "regression"
            print(f"🎯 Auto-detected task: regression ({unique_values} unique values)")
    
    # Handle missing values (simple approach)
    X = X.fillna(X.mean())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Choose model based on task
    if task == "classification":
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    
    # Train model
    print(f"🚀 Training on {len(X_train):,} samples...")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    
    performance = {}
    if task == "classification":
        accuracy = accuracy_score(y_test, y_pred)
        performance['accuracy'] = accuracy
        performance['classification_report'] = classification_report(y_test, y_pred)
        print(f"✅ Model trained! Accuracy: {accuracy:.3f}")
    else:
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        performance['r2_score'] = r2
        performance['mse'] = mse
        print(f"✅ Model trained! R² Score: {r2:.3f}")
    
    # Create Kepler model wrapper
    kepler_model = KeplerModel(model, task, target, features)
    kepler_model.trained = True
    kepler_model.performance = performance
    
    return kepler_model


def linear_model(
    data: pd.DataFrame,
    target: str,
    features: List[str] = None,
    test_size: float = 0.2,
    task: str = "auto"
) -> KeplerModel:
    """
    Train a Linear model (Logistic Regression or Linear Regression).
    
    Args:
        data: DataFrame with features and target
        target: Name of target column
        features: List of feature columns (auto-detect if None)
        test_size: Fraction for test set (default: 0.2)
        task: 'classification', 'regression', or 'auto'
        
    Returns:
        KeplerModel: Trained model ready for predictions
    """
    
    print(f"📈 Training Linear model...")
    
    # Auto-detect features
    if features is None:
        features = [col for col in data.columns if col != target]
        print(f"📊 Auto-detected {len(features)} feature columns")
    
    # Prepare data
    X = data[features]
    y = data[target]
    
    # Auto-detect task type
    if task == "auto":
        unique_values = y.nunique()
        if unique_values <= 10:
            task = "classification"
            print(f"🎯 Auto-detected task: classification")
        else:
            task = "regression"
            print(f"🎯 Auto-detected task: regression")
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Choose model
    if task == "classification":
        model = LogisticRegression(random_state=42, max_iter=1000)
    else:
        model = LinearRegression()
    
    # Train
    print(f"🚀 Training on {len(X_train):,} samples...")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    
    performance = {}
    if task == "classification":
        accuracy = accuracy_score(y_test, y_pred)
        performance['accuracy'] = accuracy
        print(f"✅ Model trained! Accuracy: {accuracy:.3f}")
    else:
        r2 = r2_score(y_test, y_pred)
        performance['r2_score'] = r2
        print(f"✅ Model trained! R² Score: {r2:.3f}")
    
    # Create wrapper
    kepler_model = KeplerModel(model, task, target, features)
    kepler_model.trained = True
    kepler_model.performance = performance
    
    return kepler_model


def load_model(filepath: str) -> KeplerModel:
    """
    Load a previously saved Kepler model.
    
    Args:
        filepath: Path to saved model file
        
    Returns:
        KeplerModel: Loaded model ready for predictions
    """
    
    try:
        model_data = joblib.load(filepath)
        
        kepler_model = KeplerModel(
            model=model_data['model'],
            model_type=model_data['model_type'],
            target_column=model_data['target_column'],
            feature_columns=model_data['feature_columns']
        )
        kepler_model.trained = True
        kepler_model.performance = model_data.get('performance', {})
        
        print(f"✅ Model loaded from: {filepath}")
        print(f"🎯 Type: {model_data['model_type']}")
        print(f"📊 Features: {len(model_data['feature_columns'])}")
        
        return kepler_model
        
    except Exception as e:
        raise ValueError(f"❌ Error loading model: {e}")


def xgboost(
    data: pd.DataFrame,
    target: str,
    features: List[str] = None,
    test_size: float = 0.2,
    task: str = "auto",
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.3,
    enable_categorical: bool = True,
    **kwargs
) -> KeplerModel:
    """
    Train an XGBoost model (Gradient Boosting) with advanced capabilities.
    
    XGBoost is an optimized distributed gradient boosting library designed for 
    efficiency, flexibility, and portability. Excellent for structured data.
    
    Args:
        data: DataFrame with features and target
        target: Name of target column
        features: List of feature columns (auto-detect if None)
        test_size: Fraction for test set (default: 0.2)
        task: 'classification', 'regression', or 'auto'
        n_estimators: Number of boosting rounds (default: 100)
        max_depth: Maximum tree depth (default: 6)
        learning_rate: Learning rate (default: 0.3)
        enable_categorical: Enable categorical feature support (default: True)
        **kwargs: Additional XGBoost parameters
        
    Returns:
        KeplerModel: Trained XGBoost model ready for predictions
        
    Example:
        import kepler as kp
        
        # Load data from Splunk
        data = kp.data.from_splunk("search index=sensors")
        
        # Train XGBoost classifier
        model = kp.train.xgboost(data, target="status", task="classification")
        
        # Train XGBoost regressor with custom parameters
        model = kp.train.xgboost(
            data, 
            target="temperature",
            task="regression",
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1
        )
    """
    
    # Dynamic import of XGBoost
    try:
        from kepler.core.library_manager import LibraryManager
        lib_manager = LibraryManager(".")
        xgb = lib_manager.dynamic_import("xgboost")
    except Exception as e:
        raise ModelTrainingError(
            "XGBoost not available",
            suggestion="Install with: kepler libs install --library xgboost>=1.7.0"
        )
    
    logger = get_logger(__name__)
    logger.info(f"Training XGBoost model...")
    
    # Auto-detect features
    if features is None:
        features = [col for col in data.columns if col != target]
        logger.info(f"Auto-detected {len(features)} feature columns")
    
    # Prepare data
    X = data[features].copy()
    y = data[target].copy()
    
    # Auto-detect task type
    if task == "auto":
        unique_values = y.nunique()
        if unique_values <= 10 and y.dtype in ['object', 'category', 'bool']:
            task = "classification"
            logger.info(f"Auto-detected task: classification ({unique_values} unique values)")
        elif unique_values <= 20:
            task = "classification"
            logger.info(f"Auto-detected task: classification ({unique_values} unique values)")
        else:
            task = "regression"
            logger.info(f"Auto-detected task: regression ({unique_values} unique values)")
    
    # Handle categorical variables
    categorical_columns = X.select_dtypes(include=['object', 'category']).columns
    if len(categorical_columns) > 0 and enable_categorical:
        logger.info(f"Converting {len(categorical_columns)} categorical columns for XGBoost")
        for col in categorical_columns:
            X[col] = X[col].astype('category')
    elif len(categorical_columns) > 0:
        logger.info(f"Encoding {len(categorical_columns)} categorical columns")
        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    
    # Handle missing values (XGBoost handles them natively)
    missing_counts = X.isnull().sum()
    if missing_counts.sum() > 0:
        logger.info(f"Found missing values in {(missing_counts > 0).sum()} columns (XGBoost will handle them)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=42,
        stratify=y if task == "classification" and y.nunique() > 1 else None
    )
    
    # Create model with parameters
    base_params = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist',  # Recommended by XGBoost docs
        'enable_categorical': enable_categorical
    }
    
    # Add task-specific parameters and additional kwargs
    if task == "classification":
        base_params['objective'] = 'binary:logistic'
        base_params['eval_metric'] = 'logloss'
        model = xgb.XGBClassifier(**base_params, **kwargs)
    else:
        base_params['objective'] = 'reg:squarederror'
        base_params['eval_metric'] = 'rmse'
        model = xgb.XGBRegressor(**base_params, **kwargs)
    
    # Train with early stopping for larger datasets
    logger.info(f"Training on {len(X_train):,} samples...")
    
    if len(X_train) > 100:
        # Use early stopping and validation
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=10,
            verbose=False
        )
        logger.info(f"Training completed with early stopping")
    else:
        # Simple training for small datasets
        model.fit(X_train, y_train)
        logger.info(f"Training completed")
    
    # Evaluate
    y_pred = model.predict(X_test)
    
    performance = {}
    if task == "classification":
        accuracy = accuracy_score(y_test, y_pred)
        performance['accuracy'] = accuracy
        
        # Additional classification metrics
        try:
            from sklearn.metrics import precision_score, recall_score, f1_score
            performance['precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            performance['recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            performance['f1_score'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # ROC AUC for binary classification
            if y.nunique() == 2:
                try:
                    from sklearn.metrics import roc_auc_score
                    y_proba = model.predict_proba(X_test)[:, 1]
                    performance['roc_auc'] = roc_auc_score(y_test, y_proba)
                except Exception:
                    pass
                    
        except Exception as e:
            logger.warning(f"Could not compute additional metrics: {e}")
        
        logger.info(f"XGBoost Classification Results:")
        logger.info(f"  Accuracy: {performance['accuracy']:.4f}")
        if 'precision' in performance:
            logger.info(f"  Precision: {performance['precision']:.4f}")
            logger.info(f"  Recall: {performance['recall']:.4f}")
            logger.info(f"  F1-Score: {performance['f1_score']:.4f}")
        if 'roc_auc' in performance:
            logger.info(f"  ROC AUC: {performance['roc_auc']:.4f}")
            
    else:
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mae = np.mean(np.abs(y_test - y_pred))
        
        performance['mse'] = mse
        performance['rmse'] = rmse
        performance['r2_score'] = r2
        performance['mae'] = mae
        
        logger.info(f"XGBoost Regression Results:")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  R² Score: {r2:.4f}")
        logger.info(f"  MAE: {mae:.4f}")
    
    # Feature importance
    try:
        feature_importance = model.feature_importances_
        importance_pairs = list(zip(features, feature_importance))
        importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        logger.info("Top 5 Feature Importances:")
        for name, importance in importance_pairs[:5]:
            logger.info(f"  {name}: {importance:.4f}")
            
    except Exception as e:
        logger.warning(f"Could not compute feature importance: {e}")
    
    # Create Kepler model
    kepler_model = KeplerModel(
        model=model,
        model_type=task,
        target_column=target,
        feature_columns=features
    )
    kepler_model.trained = True
    kepler_model.performance = performance
    
    return kepler_model


def lightgbm(
    data: pd.DataFrame,
    target: str,
    features: List[str] = None,
    **kwargs
) -> KeplerModel:
    """
    Train a LightGBM model (Gradient Boosting).
    
    Note: LightGBM integration is planned for future implementation.
    For now, use XGBoost as an alternative.
    
    Args:
        data: DataFrame with features and target
        target: Name of target column
        features: List of feature columns
        **kwargs: LightGBM parameters
        
    Returns:
        KeplerModel: Trained model
        
    Example:
        import kepler as kp
        
        # This will be available in future versions
        model = kp.train.lightgbm(data, target="temperature")
    """
    raise ModelTrainingError(
        "LightGBM trainer not yet implemented",
        suggestion="Use XGBoost for now: kp.train.xgboost(data, target)"
    )


def catboost(
    data: pd.DataFrame,
    target: str,
    features: List[str] = None,
    **kwargs
) -> KeplerModel:
    """
    Train a CatBoost model (Gradient Boosting with categorical features).
    
    Note: CatBoost integration is planned for future implementation.
    For now, use XGBoost as an alternative.
    
    Args:
        data: DataFrame with features and target
        target: Name of target column
        features: List of feature columns
        **kwargs: CatBoost parameters
        
    Returns:
        KeplerModel: Trained model
        
    Example:
        import kepler as kp
        
        # This will be available in future versions
        model = kp.train.catboost(data, target="category")
    """
    raise ModelTrainingError(
        "CatBoost trainer not yet implemented",
        suggestion="Use XGBoost for now: kp.train.xgboost(data, target)"
    )
