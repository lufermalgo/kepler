"""
HTTP Event Collector (HEC) writer for Splunk

Provides efficient batch writing of events and metrics to Splunk HEC endpoints.
"""

import json
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import requests
import pandas as pd

from kepler.utils.logging import get_logger
from kepler.utils.exceptions import SplunkConnectionError, DataExtractionError


class HecWriter:
    """
    HTTP Event Collector writer for Splunk
    
    Supports both events and metrics endpoints with batching optimization.
    """
    
    def __init__(self, 
                 hec_url: str, 
                 hec_token: str, 
                 verify_ssl: bool = True,
                 timeout: int = 30,
                 batch_size: int = 100):
        """
        Initialize HEC writer
        
        Args:
            hec_url: Full HEC endpoint URL (e.g., https://localhost:8088/services/collector)
            hec_token: HEC token for authentication
            verify_ssl: Whether to verify SSL certificates
            timeout: Request timeout in seconds
            batch_size: Number of events to batch per request
        """
        self.hec_url = hec_url.rstrip('/')
        self.hec_token = hec_token
        self.verify_ssl = verify_ssl
        self.timeout = timeout
        self.batch_size = batch_size
        self.logger = get_logger(f"{__name__}.HecWriter")
        
        # Initialize session
        self._session = None
        
        self.logger.debug(f"Initialized HecWriter for {self.hec_url}")
    
    @property
    def session(self) -> requests.Session:
        """Get or create HTTP session for HEC requests"""
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({
                'Authorization': f'Splunk {self.hec_token}',
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            })
            self._session.verify = self.verify_ssl
            self.logger.debug("HTTP session initialized for HEC")
        
        return self._session
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on HEC endpoint
        
        Returns:
            Dictionary with health check results
        """
        self.logger.info("Performing HEC health check")
        health_info = {
            'connected': False,
            'response_time_ms': None,
            'hec_available': False,
            'error': None
        }
        
        start_time = time.time()
        
        try:
            # Use the health endpoint if available, otherwise try a simple event
            health_url = f"{self.hec_url}/health"
            
            response = self.session.get(health_url, timeout=self.timeout)
            
            response_time = (time.time() - start_time) * 1000
            health_info['response_time_ms'] = round(response_time, 2)
            
            if response.status_code == 200:
                health_info.update({
                    'connected': True,
                    'hec_available': True
                })
                self.logger.info(f"HEC health check successful")
            else:
                health_info['error'] = f"HTTP {response.status_code}: {response.text}"
                self.logger.warning(f"HEC health check failed: {health_info['error']}")
                
        except requests.exceptions.Timeout:
            health_info['error'] = f"HEC timeout after {self.timeout}s"
            self.logger.error(health_info['error'])
        except requests.exceptions.ConnectionError as e:
            health_info['error'] = f"HEC connection error: {e}"
            self.logger.error(health_info['error'])
        except Exception as e:
            health_info['error'] = f"Unexpected HEC error: {e}"
            self.logger.error(health_info['error'])
        
        return health_info
    
    def validate_connection(self) -> bool:
        """
        Validate that HEC connection is working
        
        Returns:
            True if connection is valid
            
        Raises:
            SplunkConnectionError: If HEC validation fails
        """
        self.logger.info("Validating HEC connection")
        
        health_info = self.health_check()
        
        if not health_info['connected']:
            error_msg = health_info.get('error', 'Unknown HEC connection error')
            raise SplunkConnectionError(
                f"HEC connection validation failed: {error_msg}",
                splunk_host=self.hec_url,
                suggestion="Verify HEC is enabled and token is valid"
            )
        
        self.logger.info("HEC connection validated successfully")
        return True
    
    def write_event(self, 
                   event_data: Dict[str, Any],
                   source: Optional[str] = None,
                   sourcetype: Optional[str] = None,
                   index: Optional[str] = None,
                   host: Optional[str] = None,
                   timestamp: Optional[Union[datetime, float, int]] = None) -> bool:
        """
        Write a single event to Splunk HEC
        
        Args:
            event_data: Event data as dictionary
            source: Event source
            sourcetype: Event sourcetype  
            index: Target index
            host: Event host
            timestamp: Event timestamp
            
        Returns:
            True if successful
            
        Raises:
            DataExtractionError: If write fails
        """
        events = [{
            'event': event_data,
            'source': source,
            'sourcetype': sourcetype,
            'index': index,
            'host': host,
            'time': self._format_timestamp(timestamp) if timestamp else None
        }]
        
        # Remove None values
        events[0] = {k: v for k, v in events[0].items() if v is not None}
        
        return self.write_events(events)
    
    def write_events(self, events: List[Dict[str, Any]]) -> bool:
        """
        Write multiple events to Splunk HEC with batching
        
        Args:
            events: List of event dictionaries
            
        Returns:
            True if all events written successfully
            
        Raises:
            DataExtractionError: If write fails
        """
        if not events:
            self.logger.warning("No events to write")
            return True
        
        self.logger.info(f"Writing {len(events)} events to HEC")
        
        try:
            # Process events in batches
            for i in range(0, len(events), self.batch_size):
                batch = events[i:i + self.batch_size]
                self._write_batch(batch, endpoint_type='event')
                
            self.logger.info(f"Successfully wrote {len(events)} events")
            return True
            
        except Exception as e:
            raise DataExtractionError(
                f"Failed to write events to HEC: {e}",
                suggestion="Check HEC token and endpoint configuration"
            )
    
    def write_metric(self,
                    metric_name: str,
                    metric_value: Union[int, float],
                    dimensions: Optional[Dict[str, str]] = None,
                    timestamp: Optional[Union[datetime, float, int]] = None,
                    source: Optional[str] = None,
                    host: Optional[str] = None,
                    index: Optional[str] = None) -> bool:
        """
        Write a single metric to Splunk HEC metrics endpoint
        
        Args:
            metric_name: Name of the metric
            metric_value: Numeric value of the metric
            dimensions: Metric dimensions (key-value pairs)
            timestamp: Metric timestamp
            source: Metric source
            host: Metric host
            index: Target index
            
        Returns:
            True if successful
        """
        metrics = [{
            'metric': metric_name,
            'value': metric_value,
            'dimensions': dimensions or {},
            'source': source,
            'host': host,
            'index': index,
            'time': self._format_timestamp(timestamp) if timestamp else None
        }]
        
        # Remove None values
        metrics[0] = {k: v for k, v in metrics[0].items() if v is not None}
        
        return self.write_metrics(metrics)
    
    def write_metrics(self, metrics: List[Dict[str, Any]]) -> bool:
        """
        Write multiple metrics to Splunk HEC metrics endpoint
        
        Args:
            metrics: List of metric dictionaries
            
        Returns:
            True if all metrics written successfully
        """
        if not metrics:
            self.logger.warning("No metrics to write")
            return True
        
        self.logger.info(f"Writing {len(metrics)} metrics to HEC")
        
        try:
            # Process metrics in batches
            for i in range(0, len(metrics), self.batch_size):
                batch = metrics[i:i + self.batch_size]
                self._write_batch(batch, endpoint_type='metric')
                
            self.logger.info(f"Successfully wrote {len(metrics)} metrics")
            return True
            
        except Exception as e:
            raise DataExtractionError(
                f"Failed to write metrics to HEC: {e}",
                suggestion="Check HEC token and metrics endpoint configuration"
            )
    
    def write_dataframe_as_events(self,
                                 df: pd.DataFrame,
                                 source: Optional[str] = None,
                                 sourcetype: Optional[str] = "kepler:dataframe",
                                 index: Optional[str] = None,
                                 host: Optional[str] = None,
                                 timestamp_column: Optional[str] = "_time") -> bool:
        """
        Write pandas DataFrame as events to Splunk HEC
        
        Args:
            df: DataFrame to write
            source: Event source
            sourcetype: Event sourcetype
            index: Target index
            host: Event host
            timestamp_column: Column to use as timestamp
            
        Returns:
            True if successful
        """
        if df.empty:
            self.logger.warning("DataFrame is empty, nothing to write")
            return True
        
        events = []
        
        for _, row in df.iterrows():
            event_data = row.to_dict()
            
            # Extract timestamp if specified
            timestamp = None
            if timestamp_column and timestamp_column in event_data:
                timestamp = event_data.pop(timestamp_column)
            
            event = {
                'event': event_data,
                'source': source,
                'sourcetype': sourcetype,
                'index': index,
                'host': host,
                'time': self._format_timestamp(timestamp) if timestamp else None
            }
            
            # Remove None values
            event = {k: v for k, v in event.items() if v is not None}
            events.append(event)
        
        return self.write_events(events)
    
    def write_dataframe_as_metrics(self,
                                  df: pd.DataFrame,
                                  metric_columns: List[str],
                                  dimension_columns: Optional[List[str]] = None,
                                  timestamp_column: Optional[str] = "_time",
                                  source: Optional[str] = None,
                                  host: Optional[str] = None,
                                  index: Optional[str] = None) -> bool:
        """
        Write pandas DataFrame as metrics to Splunk HEC
        
        Args:
            df: DataFrame to write
            metric_columns: Columns containing metric values
            dimension_columns: Columns to use as dimensions
            timestamp_column: Column to use as timestamp
            source: Metric source
            host: Metric host
            index: Target index
            
        Returns:
            True if successful
        """
        if df.empty:
            self.logger.warning("DataFrame is empty, nothing to write")
            return True
        
        metrics = []
        dimension_columns = dimension_columns or []
        
        for _, row in df.iterrows():
            # Extract timestamp
            timestamp = None
            if timestamp_column and timestamp_column in df.columns:
                timestamp = row[timestamp_column]
            
            # Extract dimensions
            dimensions = {}
            for dim_col in dimension_columns:
                if dim_col in df.columns:
                    dimensions[dim_col] = str(row[dim_col])
            
            # Create metrics for each metric column
            for metric_col in metric_columns:
                if metric_col in df.columns and pd.notna(row[metric_col]):
                    metric = {
                        'metric': metric_col,
                        'value': float(row[metric_col]),
                        'dimensions': dimensions,
                        'source': source,
                        'host': host,
                        'index': index,
                        'time': self._format_timestamp(timestamp) if timestamp else None
                    }
                    
                    # Remove None values
                    metric = {k: v for k, v in metric.items() if v is not None}
                    metrics.append(metric)
        
        return self.write_metrics(metrics)
    
    def _write_batch(self, batch: List[Dict[str, Any]], endpoint_type: str = 'event') -> None:
        """
        Write a batch of events/metrics to HEC
        
        Args:
            batch: Batch of events or metrics
            endpoint_type: 'event' or 'metric'
        """
        # Both events and metrics use the same endpoint: /services/collector
        # The difference is in the data format (event field)
        # Según documentación oficial: docs.splunk.com
        if self.hec_url.endswith('/metrics') or self.hec_url.endswith('/event'):
            # Si la URL termina con un endpoint específico, usar base
            base_url = self.hec_url.rstrip('/metrics').rstrip('/event')
            url = base_url
        else:
            # Si es la URL base, usar tal como está
            url = self.hec_url
        
        # Convert batch to newline-delimited JSON
        payload = '\n'.join(json.dumps(item) for item in batch)
        
        self.logger.debug(f"Writing batch of {len(batch)} {endpoint_type}s to {url}")
        
        response = self.session.post(
            url,
            data=payload,
            timeout=self.timeout
        )
        
        if response.status_code != 200:
            raise DataExtractionError(
                f"HEC {endpoint_type} write failed: HTTP {response.status_code}",
                suggestion=f"Response: {response.text}"
            )
        
        # Check for HEC-specific errors in response
        try:
            response_data = response.json()
            if response_data.get('code') != 0:
                raise DataExtractionError(
                    f"HEC returned error: {response_data.get('text', 'Unknown error')}",
                    suggestion="Check HEC configuration and token permissions"
                )
        except (json.JSONDecodeError, AttributeError):
            # Response might not be JSON, which is OK for successful writes
            pass
    
    def _format_timestamp(self, timestamp: Union[datetime, float, int, str]) -> float:
        """
        Format timestamp for HEC
        
        Args:
            timestamp: Timestamp to format
            
        Returns:
            Unix timestamp as float
        """
        if isinstance(timestamp, datetime):
            return timestamp.timestamp()
        elif isinstance(timestamp, (int, float)):
            return float(timestamp)
        elif isinstance(timestamp, str):
            # Try to parse string timestamp
            try:
                return pd.to_datetime(timestamp).timestamp()
            except Exception as e:
                raise ValueError(f"Could not parse timestamp string '{timestamp}': {e}")
        else:
            raise ValueError(f"Unsupported timestamp type: {type(timestamp)}")
    
    def close(self) -> None:
        """Close HEC connection and clean up resources"""
        if self._session:
            self._session.close()
            self._session = None
        
        self.logger.debug("HecWriter connections closed")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


def create_hec_writer(hec_url: str, hec_token: str, **kwargs) -> HecWriter:
    """
    Factory function to create and validate HecWriter
    
    Args:
        hec_url: HEC endpoint URL
        hec_token: HEC authentication token
        **kwargs: Additional arguments for HecWriter
        
    Returns:
        Validated HecWriter instance
    """
    writer = HecWriter(hec_url=hec_url, hec_token=hec_token, **kwargs)
    writer.validate_connection()
    return writer