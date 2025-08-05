"""
Kepler Train Module - Simple model training for data scientists
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