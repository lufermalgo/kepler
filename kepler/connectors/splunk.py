"""
Splunk connector for Kepler framework

Provides connection, authentication, and basic operations with Splunk Enterprise.
"""

import os
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import requests
import pandas as pd
from splunklib import client, results

from kepler.utils.logging import get_logger
from kepler.utils.exceptions import SplunkConnectionError, DataExtractionError
from kepler.core.interfaces import DataConnector


class SplunkConnector(DataConnector):
    """
    Handles connection and operations with Splunk Enterprise
    
    Supports both splunk-sdk and direct REST API calls for maximum flexibility.
    """
    
    def __init__(self, host: str, token: str, verify_ssl: bool = True, timeout: int = 30, auto_fallback: bool = True):
        """
        Initialize Splunk connector with resilient connection handling
        
        Args:
            host: Splunk server URL (e.g., https://localhost:8089)
            token: Authentication token for Splunk API
            verify_ssl: Whether to verify SSL certificates
            timeout: Request timeout in seconds
            auto_fallback: Enable automatic fallback mechanisms (SSL, protocol)
        """
        self.original_host = host
        self.host = host.rstrip('/')
        self.token = token
        self.verify_ssl = verify_ssl
        self.timeout = timeout
        self.auto_fallback = auto_fallback
        self.logger = get_logger(f"{__name__}.SplunkConnector")
        
        # Connection state tracking for fallback mechanisms
        self._ssl_failed = False
        self._connection_tested = False
        self._fallback_attempted = False
        self._last_error = None
        
        # Initialize splunk-sdk client
        self._client = None
        self._session = None
        
        self.logger.debug(f"Initialized SplunkConnector for {self.host}")
        self.logger.debug(f"SSL verification: {self.verify_ssl}, Auto-fallback: {self.auto_fallback}")
    
    @property
    def client(self) -> client.Service:
        """Get or create splunk-sdk client"""
        if self._client is None:
            try:
                # Parse host to get hostname and port
                if '://' in self.host:
                    protocol, host_port = self.host.split('://', 1)
                    if ':' in host_port:
                        hostname, port = host_port.split(':', 1)
                        port = int(port)
                    else:
                        hostname = host_port
                        port = 8089 if protocol == 'https' else 8089
                else:
                    hostname = self.host
                    port = 8089
                    protocol = 'https'
                
                self._client = client.connect(
                    host=hostname,
                    port=port,
                    token=self.token,
                    scheme=protocol,
                    verify=self.verify_ssl
                )
                self.logger.debug("Splunk SDK client connected successfully")
                
            except Exception as e:
                raise SplunkConnectionError(
                    f"Failed to connect to Splunk via SDK: {e}",
                    splunk_host=self.host,
                    hint="Check your Splunk host URL and authentication token"
                )
        
        return self._client
    
    @property
    def session(self) -> requests.Session:
        """Get or create HTTP session for REST API calls"""
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({
                'Authorization': f'Bearer {self.token}',
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            })
            self._session.verify = self.verify_ssl
            self.logger.debug("HTTP session initialized for REST API")
        
        return self._session
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on Splunk connection
        
        Returns:
            Dictionary with health check results
        """
        self.logger.info("Performing Splunk health check")
        health_info = {
            'connected': False,
            'response_time_ms': None,
            'splunk_version': None,
            'server_name': None,
            'error': None
        }
        
        start_time = time.time()
        
        try:
            # Try to get server info via REST API first (faster)
            response = self.session.get(
                f"{self.host}/services/server/info?output_mode=json",
                timeout=self.timeout
            )
            
            response_time = (time.time() - start_time) * 1000
            health_info['response_time_ms'] = round(response_time, 2)
            
            if response.status_code == 200:
                server_info = response.json()
                if 'entry' in server_info and len(server_info['entry']) > 0:
                    content = server_info['entry'][0].get('content', {})
                    health_info.update({
                        'connected': True,
                        'splunk_version': content.get('version'),
                        'server_name': content.get('serverName'),
                    })
                    self.logger.info(f"Health check successful - Splunk {content.get('version')}")
                else:
                    health_info['connected'] = True
                    self.logger.info("Health check successful - basic connectivity confirmed")
            else:
                health_info['error'] = f"HTTP {response.status_code}: {response.text}"
                self.logger.warning(f"Health check failed: {health_info['error']}")
                
        except requests.exceptions.Timeout:
            health_info['error'] = f"Connection timeout after {self.timeout}s"
            self.logger.error(health_info['error'])
        except requests.exceptions.ConnectionError as e:
            health_info['error'] = f"Connection error: {e}"
            self.logger.error(health_info['error'])
        except Exception as e:
            health_info['error'] = f"Unexpected error: {e}"
            self.logger.error(health_info['error'])
        
        return health_info
    
    def validate_connection(self) -> bool:
        """
        Validate connection to Splunk server with automatic fallback mechanisms
        
        Attempts multiple connection strategies:
        1. HTTPS with SSL verification
        2. HTTPS without SSL verification (if auto_fallback enabled)
        3. HTTP (if auto_fallback enabled and HTTPS fails)
        
        Returns:
            True if connection is valid, False otherwise
            
        Raises:
            SplunkConnectionError: If all connection attempts fail
        """
        if self._connection_tested:
            return True
        
        connection_attempts = []
        
        try:
            self.logger.info("Validating Splunk connection")
            
            # Attempt 1: Try with current configuration
            try:
                health_info = self._attempt_health_check("Primary connection")
                if health_info['connected']:
                    self.logger.info("Splunk connection validation successful")
                    self._connection_tested = True
                    return True
                else:
                    raise Exception(health_info.get('error', 'Health check failed'))
                    
            except Exception as e:
                connection_attempts.append(f"Primary: {e}")
                self._last_error = e
                
                if not self.auto_fallback:
                    raise
                
                self.logger.warning(f"Primary connection failed: {e}")
            
            # Attempt 2: Try without SSL verification if SSL error
            if self.verify_ssl and self._is_ssl_error(self._last_error):
                try:
                    self.logger.warning("Attempting fallback: Disabling SSL verification")
                    original_verify = self.verify_ssl
                    self.verify_ssl = False
                    self._ssl_failed = True
                    
                    # Recreate client with new SSL setting
                    self._client = None
                    self._session = None
                    
                    health_info = self._attempt_health_check("SSL fallback")
                    if health_info['connected']:
                        self.logger.warning("⚠️  Connected with SSL verification disabled")
                        self._connection_tested = True
                        return True
                    else:
                        raise Exception(health_info.get('error', 'SSL fallback health check failed'))
                    
                except Exception as e:
                    connection_attempts.append(f"SSL fallback: {e}")
                    self.verify_ssl = original_verify  # Restore original setting
                    self.logger.warning(f"SSL fallback failed: {e}")
            
            # Attempt 3: Try HTTP if HTTPS failed
            if self.host.startswith('https://'):
                try:
                    self.logger.warning("Attempting fallback: HTTP instead of HTTPS")
                    original_host = self.host
                    self.host = self.host.replace('https://', 'http://').replace(':8089', ':8000')
                    self._fallback_attempted = True
                    
                    # Recreate client with new host
                    self._client = None
                    self._session = None
                    
                    health_info = self._attempt_health_check("HTTP fallback")
                    if health_info['connected']:
                        self.logger.warning(f"⚠️  Connected via HTTP fallback: {self.host}")
                        self._connection_tested = True
                        return True
                    else:
                        raise Exception(health_info.get('error', 'HTTP fallback health check failed'))
                    
                except Exception as e:
                    connection_attempts.append(f"HTTP fallback: {e}")
                    self.host = original_host  # Restore original host
                    self.logger.warning(f"HTTP fallback failed: {e}")
            
            # All attempts failed
            error_summary = "; ".join(connection_attempts)
            raise SplunkConnectionError(
                f"All connection attempts failed: {error_summary}",
                splunk_host=self.original_host,
                hint=self._get_connection_hint()
            )
            
        except SplunkConnectionError:
            raise
        except Exception as e:
            raise SplunkConnectionError(
                f"Unexpected error during connection validation: {e}",
                splunk_host=self.host,
                hint="Check Splunk server status and network connectivity"
            )
    
    def _attempt_health_check(self, attempt_name: str) -> Dict[str, Any]:
        """Attempt a single health check"""
        self.logger.debug(f"Attempting {attempt_name} to {self.host}")
        health_info = self.health_check()
        self.logger.debug(f"{attempt_name} health check result: {health_info}")
        return health_info
    
    def _is_ssl_error(self, error: Exception) -> bool:
        """Check if error is SSL-related"""
        error_str = str(error).lower()
        ssl_indicators = [
            'ssl', 'certificate', 'handshake', 'tls',
            'certificate_verify_failed', 'self-signed'
        ]
        return any(indicator in error_str for indicator in ssl_indicators)
    
    def _get_connection_hint(self) -> str:
        """Get contextual hint based on connection failures"""
        if self._ssl_failed:
            return ("SSL certificate issues detected. "
                   "For development: set verify_ssl=False in configuration. "
                   "For production: install proper SSL certificates")
        elif self._fallback_attempted:
            return ("HTTPS connection failed. Check if Splunk is running on port 8089 "
                   "and firewall allows connections")
        else:
            return ("Verify Splunk is running and accessible. "
                   "Check host URL, authentication token, and network connectivity")
    
    def test_authentication(self) -> bool:
        """
        Test authentication by attempting to access user info
        
        Returns:
            True if authentication is valid
            
        Raises:
            SplunkConnectionError: If authentication fails
        """
        self.logger.info("Testing Splunk authentication")
        
        try:
            response = self.session.get(
                f"{self.host}/services/authentication/current-context?output_mode=json",
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                self.logger.info("Authentication test successful")
                return True
            elif response.status_code == 401:
                raise SplunkConnectionError(
                    "Authentication failed - invalid token",
                    splunk_host=self.host,
                    hint="Check your authentication token in kepler.yml"
                )
            else:
                raise SplunkConnectionError(
                    f"Authentication test failed: HTTP {response.status_code}",
                    splunk_host=self.host
                )
                
        except requests.exceptions.RequestException as e:
            raise SplunkConnectionError(
                f"Authentication test failed: {e}",
                splunk_host=self.host,
                hint="Check network connectivity to Splunk server"
            )
    
    def extract(self, query: str, **kwargs) -> pd.DataFrame:
        """
        Extract data using SPL query (implements DataConnector interface)
        
        Args:
            query: SPL query string
            **kwargs: Additional parameters (earliest_time, latest_time, max_results, etc.)
            
        Returns:
            DataFrame with extracted data
            
        Raises:
            KeplerError: If extraction fails
        """
        # Extract parameters from kwargs
        earliest_time = kwargs.get('earliest_time')
        latest_time = kwargs.get('latest_time') 
        max_results = kwargs.get('max_results', 10000)
        timeout = kwargs.get('timeout')
        
        # Use existing search method
        results = self.search(query, earliest_time, latest_time, max_results, timeout)
        
        # Convert to DataFrame
        if not results:
            return pd.DataFrame()
        
        return pd.DataFrame(results)
    
    def search(self, 
               query: str, 
               earliest_time: Optional[str] = None,
               latest_time: Optional[str] = None,
               max_results: int = 10000,
               timeout: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Execute a Splunk search query
        
        Args:
            query: SPL query to execute
            earliest_time: Earliest time for search (e.g., '-1d', '2024-01-01T00:00:00')
            latest_time: Latest time for search (e.g., 'now', '2024-01-02T00:00:00')
            max_results: Maximum number of results to return
            timeout: Search timeout in seconds (defaults to self.timeout * 3)
            
        Returns:
            List of result dictionaries
            
        Raises:
            DataExtractionError: If search execution fails
        """
        self.logger.info(f"Executing Splunk search: {query[:100]}{'...' if len(query) > 100 else ''}")
        
        if timeout is None:
            timeout = self.timeout * 3  # Allow more time for searches
        
        try:
            # Prepare search parameters
            search_params = {
                'search': query if query.startswith('search ') or query.startswith('|') else f'search {query}',
                'output_mode': 'json',
                'count': min(max_results, 50000),  # Splunk limit
                'timeout': timeout
            }
            
            if earliest_time:
                search_params['earliest_time'] = earliest_time
            if latest_time:
                search_params['latest_time'] = latest_time
                
            self.logger.debug(f"Search parameters: {search_params}")
            
            # Execute search via REST API (oneshot for simplicity)
            response = self.session.post(
                f"{self.host}/services/search/jobs/oneshot",
                data=search_params,
                timeout=timeout
            )
            
            if response.status_code != 200:
                raise DataExtractionError(
                    f"Search execution failed: HTTP {response.status_code}",
                    query=query,
                    hint="Check your SPL query syntax and permissions"
                )
            
            # Parse results
            result_data = response.json()
            
            # Check for Splunk-specific errors in messages
            if 'messages' in result_data and result_data['messages']:
                error_messages = []
                for msg in result_data['messages']:
                    if msg.get('type') in ['FATAL', 'ERROR']:
                        error_messages.append(msg.get('text', 'Unknown error'))
                
                if error_messages:
                    error_text = '; '.join(error_messages)
                    raise DataExtractionError(
                        f"Splunk query error: {error_text}",
                        query=query,
                        hint="Check the SPL syntax according to Splunk documentation"
                    )
            
            results_list = result_data.get('results', [])
            
            self.logger.info(f"Search completed successfully - {len(results_list)} results")
            return results_list
            
        except requests.exceptions.Timeout:
            raise DataExtractionError(
                f"Search timed out after {timeout} seconds",
                query=query,
                hint="Try reducing the time range or adding more specific filters"
            )
        except requests.exceptions.RequestException as e:
            raise DataExtractionError(
                f"Search request failed: {e}",
                query=query,
                hint="Check network connectivity and query syntax"
            )
        except Exception as e:
            raise DataExtractionError(
                f"Unexpected error during search: {e}",
                query=query
            )
    
    def search_to_dataframe(self, 
                           query: str,
                           earliest_time: Optional[str] = None,
                           latest_time: Optional[str] = None,
                           max_results: int = 10000,
                           timeout: Optional[int] = None,
                           optimize_for_metrics: bool = None) -> pd.DataFrame:
        """
        Execute search and return results as pandas DataFrame
        
        Args:
            query: SPL query to execute
            earliest_time: Earliest time for search
            latest_time: Latest time for search  
            max_results: Maximum number of results
            timeout: Search timeout in seconds
            optimize_for_metrics: Auto-detect if None, True for metrics optimization
            
        Returns:
            pandas DataFrame with search results
        """
        # Auto-detect metrics queries if not specified
        if optimize_for_metrics is None:
            optimize_for_metrics = self._is_metrics_query(query)
        
        if optimize_for_metrics:
            self.logger.info("Using metrics-optimized search")
            
        results = self.search(
            query=query,
            earliest_time=earliest_time,
            latest_time=latest_time,
            max_results=max_results,
            timeout=timeout
        )
        
        if not results:
            self.logger.warning("Search returned no results")
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        self.logger.info(f"Created DataFrame with shape {df.shape}")
        
        # Enhanced cleanup for metrics data
        self._clean_dataframe(df, is_metrics=optimize_for_metrics)
        
        return df
    
    def search_metrics(self,
                      metric_name: str,
                      earliest_time: str = "-1h",
                      latest_time: str = "now", 
                      span: str = "5m",
                      by_fields: Optional[List[str]] = None,
                      aggregation: str = "avg",
                      where_clause: str = "",
                      max_results: int = 10000) -> pd.DataFrame:
        """
        Optimized search for metrics data using mstats
        
        Args:
            metric_name: Name of the metric to search for
            earliest_time: Start time for the search
            latest_time: End time for the search
            span: Time span for aggregation (e.g., '5m', '1h')
            by_fields: Fields to group metrics by
            aggregation: Aggregation function (avg, sum, count, max, min)
            where_clause: Additional WHERE conditions
            max_results: Maximum results to return
            
        Returns:
            pandas DataFrame with aggregated metrics data
        """
        # Build mstats query
        by_clause = f" by {', '.join(by_fields)}" if by_fields else ""
        where_clause = f" WHERE {where_clause}" if where_clause else ""
        
        query = f"| mstats {aggregation}({metric_name}) as {metric_name}_{aggregation}{where_clause} span={span}{by_clause}"
        
        self.logger.info(f"Executing optimized metrics search: {query}")
        
        return self.search_to_dataframe(
            query=query,
            earliest_time=earliest_time,
            latest_time=latest_time,
            max_results=max_results,
            optimize_for_metrics=True
        )
    
    def _is_metrics_query(self, query: str) -> bool:
        """
        Detect if query is metrics-optimized (uses mstats, mcatalog, etc.)
        
        Args:
            query: SPL query to analyze
            
        Returns:
            True if query appears to be metrics-focused
        """
        query_lower = query.lower().strip()
        
        # Remove leading search command if present
        if query_lower.startswith('search '):
            query_lower = query_lower[7:].strip()
        
        # Check for metrics-specific commands
        metrics_commands = ['mstats', 'mcatalog', 'mpreview', 'mrollup']
        
        for cmd in metrics_commands:
            if query_lower.startswith(f'| {cmd}') or query_lower.startswith(cmd):
                return True
        
        # Check for metrics-specific functions
        metrics_functions = ['avg(', 'sum(', 'count(', 'max(', 'min(', 'stdev(', 'perc']
        if any(func in query_lower for func in metrics_functions):
            # Also check for common metrics patterns
            if any(pattern in query_lower for pattern in ['span=', 'by _time', 'timechart']):
                return True
        
        return False
    
    def _clean_dataframe(self, df: pd.DataFrame, is_metrics: bool = False) -> None:
        """
        Clean and optimize DataFrame (in-place)
        
        Args:
            df: DataFrame to clean
            is_metrics: True if this is metrics data (more aggressive numeric conversion)
        """
        if df.empty:
            return
        
        # Convert numeric columns
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert to numeric
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                if not numeric_series.isna().all():
                    # For metrics data, be more aggressive with conversion
                    nan_threshold = 0.7 if is_metrics else 0.5
                    if numeric_series.isna().sum() / len(df) < nan_threshold:
                        df[col] = numeric_series
                        self.logger.debug(f"Converted column {col} to numeric (metrics: {is_metrics})")
        
        # Convert time columns to datetime
        time_columns = ['_time', '_span', 'latest', 'earliest']
        for time_col in time_columns:
            if time_col in df.columns:
                try:
                    df[time_col] = pd.to_datetime(df[time_col])
                    self.logger.debug(f"Converted {time_col} column to datetime")
                except Exception as e:
                    self.logger.warning(f"Could not convert {time_col} to datetime: {e}")
        
        # For metrics data, ensure proper sorting by time if _time exists  
        if is_metrics and '_time' in df.columns and pd.api.types.is_datetime64_any_dtype(df['_time']):
            df.sort_values('_time', inplace=True)
            df.reset_index(drop=True, inplace=True)
            self.logger.debug("Sorted metrics data by _time")
    
    def close(self) -> None:
        """Close connections and clean up resources"""
        if self._session:
            self._session.close()
            self._session = None
        
        if self._client:
            try:
                self._client.logout()
            except:
                pass  # Ignore errors during logout
            self._client = None
        
        self.logger.debug("SplunkConnector connections closed")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

    
    # =====================================
    # INDEX MANAGEMENT AND VALIDATION
    # =====================================
    
    def get_available_indexes(self) -> List[str]:
        """
        Get list of available index names (implements DataConnector interface)
        
        Returns:
            List of index names
        """
        indexes_info = self.list_indexes()
        return [idx.get('name', '') for idx in indexes_info if idx.get('name')]
    
    def list_indexes(self) -> List[Dict[str, Any]]:
        """
        List all available indexes in Splunk
        
        Returns:
            List of index information dictionaries
        """
        self.logger.info("Retrieving list of Splunk indexes")
        
        try:
            response = self.session.get(
                f"{self.host}/services/data/indexes?output_mode=json",
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                indexes = []
                
                for entry in data.get('entry', []):
                    content = entry.get('content', {})
                    indexes.append({
                        'name': entry.get('name'),
                        'datatype': content.get('datatype', 'event'),
                        'maxDataSize': content.get('maxDataSize'),
                        'maxTime': content.get('maxTime'),
                        'minTime': content.get('minTime'),
                        'currentDBSizeMB': content.get('currentDBSizeMB'),
                        'totalEventCount': content.get('totalEventCount'),
                        'disabled': content.get('disabled', False)
                    })
                
                self.logger.info(f"Found {len(indexes)} indexes")
                return indexes
            else:
                raise SplunkConnectionError(
                    f"Failed to retrieve indexes: HTTP {response.status_code}",
                    splunk_host=self.host
                )
                
        except requests.exceptions.RequestException as e:
            raise SplunkConnectionError(
                f"Error retrieving indexes: {e}",
                splunk_host=self.host
            )
    
    def index_exists(self, index_name: str) -> bool:
        """
        Check if an index exists in Splunk
        
        Args:
            index_name: Name of the index to check
            
        Returns:
            True if index exists, False otherwise
        """
        self.logger.info(f"Checking if index '{index_name}' exists")
        
        try:
            response = self.session.get(
                f"{self.host}/services/data/indexes/{index_name}?output_mode=json",
                timeout=self.timeout
            )
            
            exists = response.status_code == 200
            self.logger.info(f"Index '{index_name}': {'exists' if exists else 'does not exist'}")
            return exists
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error checking index existence: {e}")
            return False
    
    def validate_index_access(self, index_name: str) -> Dict[str, Any]:
        """
        Validate if an index is accessible and queryable
        
        Args:
            index_name: Name of the index to validate
            
        Returns:
            Dictionary with validation results
        """
        self.logger.info(f"Validating access to index '{index_name}'")
        
        validation = {
            'exists': False,
            'accessible': False,
            'has_data': False,
            'event_count': 0,
            'size_mb': 0,
            'data_range': None,
            'error': None
        }
        
        try:
            # Check if index exists and get details
            response = self.session.get(
                f"{self.host}/services/data/indexes/{index_name}?output_mode=json",
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                validation['exists'] = True
                data = response.json()
                
                if 'entry' in data and len(data['entry']) > 0:
                    content = data['entry'][0].get('content', {})
                    
                    validation['accessible'] = True
                    validation['event_count'] = int(content.get('totalEventCount', 0))
                    validation['size_mb'] = float(content.get('currentDBSizeMB', 0))
                    validation['has_data'] = validation['event_count'] > 0
                    
                    # Get data time range if available
                    min_time = content.get('minTime')
                    max_time = content.get('maxTime')
                    if min_time and max_time:
                        validation['data_range'] = {
                            'earliest': min_time,
                            'latest': max_time
                        }
                        
                    self.logger.info(f"Index '{index_name}' validation: "
                                   f"{validation['event_count']} events, "
                                   f"{validation['size_mb']:.2f} MB")
            else:
                validation['error'] = f"Index not accessible: HTTP {response.status_code}"
                self.logger.warning(validation['error'])
                
        except requests.exceptions.RequestException as e:
            validation['error'] = f"Error validating index: {e}"
            self.logger.error(validation['error'])
            
        return validation
    
    def check_index_data(self, index_name: str, sample_query: Optional[str] = None) -> Dict[str, Any]:
        """
        Check if an index has queryable data
        
        Args:
            index_name: Name of the index to check
            sample_query: Optional SPL query to test (defaults to simple search)
            
        Returns:
            Dictionary with data check results
        """
        self.logger.info(f"Checking data in index '{index_name}'")
        
        if not sample_query:
            sample_query = f"search index={index_name} | head 5"
            
        try:
            results = self.search(
                query=sample_query,
                earliest_time="-24h",
                max_results=5
            )
            
            data_check = {
                'has_data': len(results) > 0,
                'sample_count': len(results),
                'queryable': True,
                'sample_events': results[:3] if results else [],
                'error': None
            }
            
            self.logger.info(f"Index '{index_name}' data check: "
                           f"{'✅ Has data' if data_check['has_data'] else '❌ No data'} "
                           f"({data_check['sample_count']} sample events)")
            
            return data_check
            
        except Exception as e:
            return {
                'has_data': False,
                'sample_count': 0,
                'queryable': False,
                'sample_events': [],
                'error': str(e)
            }
    
    def create_index(self, index_name: str, index_type: str = "event", **kwargs) -> bool:
        """
        Create a new index in Splunk
        
        Args:
            index_name: Name of the index to create
            index_type: Type of index ('event' or 'metric')
            **kwargs: Additional index configuration parameters
            
        Returns:
            True if index was created successfully, False otherwise
        """
        self.logger.info(f"Creating {index_type} index '{index_name}'")
        
        # Prepare index configuration
        index_config = {
            'name': index_name,
            'datatype': index_type
        }
        
        # Add common configuration
        if index_type == "metric":
            index_config.update({
                'maxDataSize': kwargs.get('maxDataSize', 'auto'),
                'maxHotBuckets': kwargs.get('maxHotBuckets', 3),
                'maxWarmDBCount': kwargs.get('maxWarmDBCount', 300)
            })
        else:  # event index
            index_config.update({
                'maxDataSize': kwargs.get('maxDataSize', 'auto'),
                'maxHotBuckets': kwargs.get('maxHotBuckets', 10),
                'maxWarmDBCount': kwargs.get('maxWarmDBCount', 300)
            })
        
        # Add any additional configuration
        index_config.update(kwargs)
        
        try:
            response = self.session.post(
                f"{self.host}/services/data/indexes?output_mode=json",
                data=index_config,
                timeout=self.timeout
            )
            
            if response.status_code in [200, 201]:
                self.logger.info(f"✅ Successfully created {index_type} index '{index_name}'")
                return True
            else:
                error_msg = f"Failed to create index: HTTP {response.status_code}"
                if response.text:
                    error_msg += f" - {response.text}"
                self.logger.error(error_msg)
                return False
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error creating index '{index_name}': {e}")
            return False
    
    def validate_project_indexes(self, project_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate all indexes required by a project configuration
        
        Args:
            project_config: Project configuration with splunk.indexes settings
            
        Returns:
            Dictionary with comprehensive validation results
        """
        self.logger.info("Validating project indexes configuration")
        
        splunk_config = project_config.get('splunk', {})
        
        # Get required indexes from configuration
        required_indexes = {
            'events_index': splunk_config.get('events_index', 'kepler_lab'),
            'metrics_index': splunk_config.get('metrics_index', 'kepler_metrics'),
            'default_index': splunk_config.get('default_index', 'main')
        }
        
        validation_results = {
            'overall_status': 'unknown',
            'indexes_checked': len(required_indexes),
            'indexes_valid': 0,
            'indexes_missing': [],
            'indexes_created': [],
            'validation_details': {},
            'recommendations': []
        }
        
        for index_type, index_name in required_indexes.items():
            self.logger.info(f"Validating {index_type}: '{index_name}'")
            
            # Validate index
            index_validation = self.validate_index_access(index_name)
            validation_results['validation_details'][index_name] = index_validation
            
            if index_validation['exists'] and index_validation['accessible']:
                validation_results['indexes_valid'] += 1
                
                # Check for data and provide recommendations
                if not index_validation['has_data']:
                    validation_results['recommendations'].append(
                        f"Index '{index_name}' exists but has no data - consider running 'kepler lab generate' to create test data"
                    )
            else:
                validation_results['indexes_missing'].append(index_name)
                
                # Attempt to create missing index
                if index_validation['error'] and "not accessible" in index_validation['error']:
                    suggested_type = "metric" if "metric" in index_type else "event"
                    
                    self.logger.info(f"Attempting to create missing {suggested_type} index '{index_name}'")
                    if self.create_index(index_name, suggested_type):
                        validation_results['indexes_created'].append(index_name)
                        validation_results['indexes_valid'] += 1
                        validation_results['recommendations'].append(
                            f"✅ Created {suggested_type} index '{index_name}' - ready for data ingestion"
                        )
                    else:
                        validation_results['recommendations'].append(
                            f"❌ Failed to create index '{index_name}' - manual creation required"
                        )
        
        # Determine overall status
        if validation_results['indexes_valid'] == validation_results['indexes_checked']:
            validation_results['overall_status'] = 'success'
        elif validation_results['indexes_valid'] > 0:
            validation_results['overall_status'] = 'partial'
        else:
            validation_results['overall_status'] = 'failed'
            
        self.logger.info(f"Project index validation completed: "
                       f"{validation_results['indexes_valid']}/{validation_results['indexes_checked']} indexes valid")
        
        return validation_results


def create_splunk_connector(host: str, token: str, **kwargs) -> SplunkConnector:
    """
    Factory function to create and validate SplunkConnector
    
    Args:
        host: Splunk server URL
        token: Authentication token
        **kwargs: Additional arguments for SplunkConnector
        
    Returns:
        Validated SplunkConnector instance
    """
    connector = SplunkConnector(host=host, token=token, **kwargs)
    connector.validate_connection()
    return connector