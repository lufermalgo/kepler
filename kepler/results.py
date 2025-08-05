"""
Kepler Results Module - Simple results writing for data scientists
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, Dict, Any, List
from datetime import datetime
import os


def to_splunk(
    data: Union[pd.DataFrame, Dict, List[Dict]],
    index: str = "ml_predictions",
    source: str = "kepler_model",
    metrics: bool = False
) -> bool:
    """
    Write ML results back to Splunk in the simplest way possible.
    
    Args:
        data: Results to write (DataFrame, dict, or list of dicts)
        index: Target Splunk index (default: ml_predictions)
        source: Source name for the data (default: kepler_model)
        metrics: Write as metrics instead of events (default: False)
        
    Returns:
        bool: True if successful
        
    Example:
        >>> import kepler as kp
        >>> predictions = model.predict(data)
        >>> success = kp.results.to_splunk(predictions, index="anomaly_predictions")
    """
    
    print(f"📡 Writing results to Splunk index '{index}'...")
    
    # Import inside function to avoid circular imports
    from kepler.core.config import load_config
    from kepler.connectors.hec import HecWriter
    
    try:
        # Auto-load configuration (transparent)
        config = load_config()
        hec_token = os.getenv('SPLUNK_HEC_TOKEN')
        
        if not hec_token:
            raise ValueError("❌ SPLUNK_HEC_TOKEN not found. Check your .env file.")
        
        # Create HEC writer
        hec_writer = HecWriter(
            hec_url=config.splunk.hec_url,
            hec_token=hec_token,
            verify_ssl=config.splunk.verify_ssl
        )
        
        # Convert data to proper format
        if isinstance(data, pd.DataFrame):
            # Convert DataFrame to list of dicts
            records = data.to_dict('records')
        elif isinstance(data, dict):
            # Single record
            records = [data]
        elif isinstance(data, (list, np.ndarray)):
            # Handle different list types
            if isinstance(data, np.ndarray):
                records = [{"prediction": val, "timestamp": datetime.now().isoformat()} for val in data]
            else:
                records = data if all(isinstance(item, dict) for item in data) else [{"prediction": val} for val in data]
        else:
            raise ValueError(f"❌ Unsupported data type: {type(data)}")
        
        # Enrich records with metadata
        enriched_records = []
        for record in records:
            enriched_record = {
                "time": record.get("time", datetime.now().timestamp()),
                "source": source,
                "sourcetype": "kepler_prediction",
                "index": index,
                "event": record
            }
            
            # Add metric format if requested
            if metrics:
                enriched_record["event"] = "metric"
                # Convert numeric fields to metrics format
                fields = {}
                for key, value in record.items():
                    if isinstance(value, (int, float)):
                        fields[f"metric_name:{key}"] = value
                    else:
                        fields[key] = value
                enriched_record["fields"] = fields
            
            enriched_records.append(enriched_record)
        
        # Write to Splunk
        if metrics:
            success = hec_writer.write_metrics(enriched_records)
        else:
            success = hec_writer.write_events(enriched_records)
        
        if success:
            print(f"✅ Successfully wrote {len(enriched_records):,} records to Splunk")
            return True
        else:
            print(f"❌ Failed to write records to Splunk")
            return False
            
    except Exception as e:
        print(f"❌ Error writing to Splunk: {e}")
        return False


def create_prediction_summary(
    model_predictions: Union[np.ndarray, List],
    original_data: pd.DataFrame,
    model_name: str = "kepler_model",
    confidence_scores: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Create a comprehensive summary of model predictions.
    
    Args:
        model_predictions: Model predictions
        original_data: Original input data
        model_name: Name of the model
        confidence_scores: Confidence scores (optional)
        
    Returns:
        DataFrame: Summary ready to write to Splunk
        
    Example:
        >>> predictions = model.predict(data)
        >>> confidence = model.predict_proba(data)
        >>> summary = kp.results.create_prediction_summary(
        ...     predictions, data, "anomaly_detector", confidence
        ... )
        >>> kp.results.to_splunk(summary)
    """
    
    print(f"📋 Creating prediction summary...")
    
    # Create base summary
    summary = pd.DataFrame({
        'model_name': model_name,
        'prediction': model_predictions,
        'prediction_timestamp': datetime.now().isoformat(),
        'data_timestamp': datetime.now().isoformat()
    })
    
    # Add confidence scores if available
    if confidence_scores is not None:
        if confidence_scores.ndim == 2:  # Multi-class probabilities
            summary['confidence_score'] = np.max(confidence_scores, axis=1)
            summary['predicted_class_probability'] = np.max(confidence_scores, axis=1)
        else:  # Single confidence score
            summary['confidence_score'] = confidence_scores
    
    # Add sample of original features (first few columns)
    if not original_data.empty:
        # Add key features for context
        feature_cols = original_data.select_dtypes(include=[np.number]).columns[:5]
        for col in feature_cols:
            if col in original_data.columns:
                summary[f'feature_{col}'] = original_data[col].values
    
    # Add prediction metadata
    summary['total_features'] = len(original_data.columns) if not original_data.empty else 0
    summary['prediction_id'] = [f"{model_name}_{i:06d}" for i in range(len(summary))]
    
    print(f"✅ Created summary with {len(summary):,} predictions")
    return summary


def write_model_metrics(
    model_performance: Dict[str, Any],
    model_name: str = "kepler_model",
    index: str = "ml_model_metrics"
) -> bool:
    """
    Write model performance metrics to Splunk.
    
    Args:
        model_performance: Dictionary with performance metrics
        model_name: Name of the model
        index: Target metrics index
        
    Returns:
        bool: True if successful
        
    Example:
        >>> model = kp.train.random_forest(data, target="anomaly")
        >>> kp.results.write_model_metrics(
        ...     model.performance, "anomaly_detector"
        ... )
    """
    
    # Prepare metrics record
    metrics_record = {
        'model_name': model_name,
        'evaluation_timestamp': datetime.now().isoformat(),
        **model_performance
    }
    
    return to_splunk(metrics_record, index=index, source="kepler_model_evaluation", metrics=True)


def write_anomaly_alerts(
    anomaly_predictions: Union[np.ndarray, List],
    threshold: float = 0.5,
    data_context: Optional[pd.DataFrame] = None,
    index: str = "ml_anomaly_alerts"
) -> bool:
    """
    Write anomaly detection alerts to Splunk.
    
    Args:
        anomaly_predictions: Binary predictions or anomaly scores
        threshold: Threshold for anomaly detection (default: 0.5)
        data_context: Original data for context (optional)
        index: Target index for alerts
        
    Returns:
        bool: True if successful
        
    Example:
        >>> anomaly_scores = model.predict_proba(data)[:, 1]
        >>> kp.results.write_anomaly_alerts(
        ...     anomaly_scores, threshold=0.8, data_context=data
        ... )
    """
    
    # Filter only anomalies above threshold
    if isinstance(anomaly_predictions, np.ndarray):
        anomaly_mask = anomaly_predictions > threshold
        anomaly_indices = np.where(anomaly_mask)[0]
        anomaly_scores = anomaly_predictions[anomaly_mask]
    else:
        anomaly_indices = [i for i, score in enumerate(anomaly_predictions) if score > threshold]
        anomaly_scores = [score for score in anomaly_predictions if score > threshold]
    
    if len(anomaly_indices) == 0:
        print("✅ No anomalies detected above threshold")
        return True
    
    # Create alerts
    alerts = []
    for idx, score in zip(anomaly_indices, anomaly_scores):
        alert = {
            'alert_type': 'ANOMALY_DETECTED',
            'anomaly_score': float(score),
            'severity': 'HIGH' if score > 0.8 else 'MEDIUM',
            'data_point_index': int(idx),
            'detection_timestamp': datetime.now().isoformat(),
            'threshold_used': threshold
        }
        
        # Add context if available
        if data_context is not None and idx < len(data_context):
            context_row = data_context.iloc[idx]
            alert['context'] = context_row.to_dict()
        
        alerts.append(alert)
    
    print(f"🚨 Writing {len(alerts)} anomaly alerts to Splunk")
    return to_splunk(alerts, index=index, source="kepler_anomaly_detector")