"""Tests for Splunk connector functionality"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import requests
import json

from kepler.connectors.splunk import SplunkConnector, create_splunk_connector
from kepler.utils.exceptions import SplunkConnectionError, DataExtractionError


class TestSplunkConnector:
    """Test SplunkConnector class"""
    
    def setup_method(self):
        """Set up test environment"""
        self.host = "https://localhost:8089"
        self.token = "test_token_123"
        self.connector = SplunkConnector(host=self.host, token=self.token, verify_ssl=False)
    
    def test_initialization(self):
        """Test SplunkConnector initialization"""
        assert self.connector.host == "https://localhost:8089"
        assert self.connector.token == "test_token_123"
        assert self.connector.verify_ssl == False
        assert self.connector.timeout == 30
    
    def test_host_cleanup(self):
        """Test that trailing slashes are removed from host"""
        connector = SplunkConnector("https://localhost:8089/", "token")
        assert connector.host == "https://localhost:8089"
    
    @patch('requests.Session.get')
    def test_health_check_success(self, mock_get):
        """Test successful health check"""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'entry': [{
                'content': {
                    'version': '9.1.0',
                    'serverName': 'test-splunk'
                }
            }]
        }
        mock_get.return_value = mock_response
        
        with patch('time.time', side_effect=[0, 0.1]):  # Mock 100ms response time
            health_info = self.connector.health_check()
        
        assert health_info['connected'] == True
        assert health_info['response_time_ms'] == 100.0
        assert health_info['splunk_version'] == '9.1.0'
        assert health_info['server_name'] == 'test-splunk'
        assert health_info['error'] is None
    
    @patch('requests.Session.get')
    def test_health_check_failure(self, mock_get):
        """Test health check failure"""
        # Mock failed response
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_get.return_value = mock_response
        
        with patch('time.time', return_value=0):
            health_info = self.connector.health_check()
        
        assert health_info['connected'] == False
        assert health_info['error'] == "HTTP 401: Unauthorized"
    
    @patch('requests.Session.get')
    def test_health_check_timeout(self, mock_get):
        """Test health check timeout"""
        mock_get.side_effect = requests.exceptions.Timeout()
        
        health_info = self.connector.health_check()
        
        assert health_info['connected'] == False
        assert "timeout" in health_info['error'].lower()
    
    @patch('requests.Session.get')
    def test_health_check_connection_error(self, mock_get):
        """Test health check connection error"""
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")
        
        health_info = self.connector.health_check()
        
        assert health_info['connected'] == False
        assert "Connection error" in health_info['error']
    
    @patch.object(SplunkConnector, 'health_check')
    def test_validate_connection_success(self, mock_health_check):
        """Test successful connection validation"""
        mock_health_check.return_value = {'connected': True, 'error': None}
        
        result = self.connector.validate_connection()
        assert result == True
    
    @patch.object(SplunkConnector, 'health_check')
    def test_validate_connection_failure(self, mock_health_check):
        """Test connection validation failure"""
        mock_health_check.return_value = {
            'connected': False, 
            'error': 'Connection refused'
        }
        
        with pytest.raises(SplunkConnectionError) as exc_info:
            self.connector.validate_connection()
        
        assert "Connection refused" in str(exc_info.value)
    
    @patch('requests.Session.get')
    def test_test_authentication_success(self, mock_get):
        """Test successful authentication"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        result = self.connector.test_authentication()
        assert result == True
    
    @patch('requests.Session.get')
    def test_test_authentication_invalid_token(self, mock_get):
        """Test authentication with invalid token"""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_get.return_value = mock_response
        
        with pytest.raises(SplunkConnectionError) as exc_info:
            self.connector.test_authentication()
        
        assert "invalid token" in str(exc_info.value).lower()
    
    @patch('requests.Session.post')
    def test_search_success(self, mock_post):
        """Test successful search execution"""
        # Mock successful search response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'results': [
                {'_time': '2024-01-01T00:00:00', 'host': 'server1', 'count': '100'},
                {'_time': '2024-01-01T01:00:00', 'host': 'server2', 'count': '200'}
            ]
        }
        mock_post.return_value = mock_response
        
        results = self.connector.search("index=test | stats count by host")
        
        assert len(results) == 2
        assert results[0]['host'] == 'server1'
        assert results[1]['count'] == '200'
        
        # Verify search parameters
        call_args = mock_post.call_args
        assert 'search index=test | stats count by host' in call_args[1]['data']['search']
    
    @patch('requests.Session.post')
    def test_search_with_time_range(self, mock_post):
        """Test search with time range parameters"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'results': []}
        mock_post.return_value = mock_response
        
        self.connector.search(
            "index=test",
            earliest_time="-1d",
            latest_time="now",
            max_results=500
        )
        
        call_args = mock_post.call_args
        search_data = call_args[1]['data']
        assert search_data['earliest_time'] == '-1d'
        assert search_data['latest_time'] == 'now'
        assert search_data['count'] == 500
    
    @patch('requests.Session.post')
    def test_search_failure(self, mock_post):
        """Test search execution failure"""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_post.return_value = mock_response
        
        with pytest.raises(DataExtractionError) as exc_info:
            self.connector.search("invalid query")
        
        assert "HTTP 400" in str(exc_info.value)
    
    @patch('requests.Session.post')
    def test_search_timeout(self, mock_post):
        """Test search timeout"""
        mock_post.side_effect = requests.exceptions.Timeout()
        
        with pytest.raises(DataExtractionError) as exc_info:
            self.connector.search("index=test")
        
        assert "timed out" in str(exc_info.value).lower()
    
    @patch.object(SplunkConnector, 'search')
    def test_search_to_dataframe_success(self, mock_search):
        """Test successful conversion to DataFrame"""
        mock_search.return_value = [
            {'_time': '2024-01-01T00:00:00', 'count': '100', 'avg_value': '1.5'},
            {'_time': '2024-01-01T01:00:00', 'count': '200', 'avg_value': '2.3'}
        ]
        
        df = self.connector.search_to_dataframe("index=test")
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'count' in df.columns
        assert df['count'].dtype == 'int64'  # Should be converted to numeric
        assert df['avg_value'].dtype == 'float64'  # Should be converted to numeric
    
    @patch.object(SplunkConnector, 'search')
    def test_search_to_dataframe_empty_results(self, mock_search):
        """Test DataFrame creation with empty results"""
        mock_search.return_value = []
        
        df = self.connector.search_to_dataframe("index=test")
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
    
    def test_clean_dataframe_numeric_conversion(self):
        """Test DataFrame cleaning and numeric conversion"""
        df = pd.DataFrame({
            'numeric_col': ['1', '2', '3', '4', '5'],
            'mixed_col': ['1', 'abc', '3', 'def', 'ghi'],  # More non-numeric values (60% non-numeric)
            'text_col': ['abc', 'def', 'ghi', 'jkl', 'mno'],
            '_time': ['2024-01-01T00:00:00', '2024-01-01T01:00:00', '2024-01-01T02:00:00', '2024-01-01T03:00:00', '2024-01-01T04:00:00']
        })
        
        self.connector._clean_dataframe(df)
        
        # numeric_col should be converted to numeric
        assert pd.api.types.is_numeric_dtype(df['numeric_col'])
        
        # mixed_col should stay as object (>50% would be NaN after conversion)  
        assert df['mixed_col'].dtype == 'object'
        
        # text_col should stay as object
        assert df['text_col'].dtype == 'object'
        
        # _time should be converted to datetime
        assert pd.api.types.is_datetime64_any_dtype(df['_time'])
    
    def test_clean_dataframe_empty(self):
        """Test cleaning empty DataFrame"""
        df = pd.DataFrame()
        
        # Should not raise error
        self.connector._clean_dataframe(df)
        assert len(df) == 0
    
    def test_is_metrics_query_detection(self):
        """Test metrics query detection"""
        # Test mstats queries
        assert self.connector._is_metrics_query("| mstats avg(cpu_usage) by host span=5m")
        assert self.connector._is_metrics_query("mstats sum(network_bytes) WHERE host=server1")
        
        # Test other metrics commands
        assert self.connector._is_metrics_query("| mcatalog values(metric_name)")
        
        # Test aggregate functions with time patterns
        assert self.connector._is_metrics_query("index=metrics | timechart avg(value) by host")
        assert self.connector._is_metrics_query("index=metrics | stats avg(value) by _time span=1h")
        
        # Test non-metrics queries
        assert not self.connector._is_metrics_query("index=logs error")
        assert not self.connector._is_metrics_query("search index=events | table host message")
        assert not self.connector._is_metrics_query("| eval new_field=1")
    
    def test_clean_dataframe_metrics_mode(self):
        """Test enhanced DataFrame cleaning for metrics data"""
        df = pd.DataFrame({
            'metric_value': ['1.5', '2.3', 'NaN', '4.1', '5.0'],
            'host': ['server1', 'server2', 'server1', 'server2', 'server1'],
            '_time': ['2024-01-01T00:00:00', '2024-01-01T00:05:00', '2024-01-01T00:10:00', 
                     '2024-01-01T00:15:00', '2024-01-01T00:20:00']
        })
        
        self.connector._clean_dataframe(df, is_metrics=True)
        
        # Should be more aggressive with numeric conversion
        assert pd.api.types.is_numeric_dtype(df['metric_value'])
        
        # Should convert _time to datetime
        assert pd.api.types.is_datetime64_any_dtype(df['_time'])
        
        # Should be sorted by time
        assert df['_time'].is_monotonic_increasing
    
    @patch.object(SplunkConnector, 'search_to_dataframe')
    def test_search_metrics_method(self, mock_search_to_dataframe):
        """Test the search_metrics convenience method"""
        mock_df = pd.DataFrame({'_time': [1, 2, 3], 'cpu_avg': [1.0, 2.0, 3.0]})
        mock_search_to_dataframe.return_value = mock_df
        
        result = self.connector.search_metrics(
            metric_name="cpu_usage",
            earliest_time="-1h",
            latest_time="now",
            span="5m",
            by_fields=["host"],
            aggregation="avg",
            where_clause="host=server1"
        )
        
        # Verify the mstats query was built correctly
        expected_query = "| mstats avg(cpu_usage) as cpu_usage_avg WHERE host=server1 span=5m by host"
        mock_search_to_dataframe.assert_called_once_with(
            query=expected_query,
            earliest_time="-1h",
            latest_time="now",
            max_results=10000,
            optimize_for_metrics=True
        )
        
        assert result is mock_df
    
    @patch.object(SplunkConnector, 'search')
    def test_search_to_dataframe_metrics_optimization(self, mock_search):
        """Test automatic metrics optimization detection"""
        mock_search.return_value = [
            {'_time': '2024-01-01T00:00:00', 'cpu_avg': '85.5'},
            {'_time': '2024-01-01T00:05:00', 'cpu_avg': '90.2'}
        ]
        
        # Test with mstats query (should auto-detect)
        df = self.connector.search_to_dataframe("| mstats avg(cpu_usage) span=5m")
        
        assert len(df) == 2
        assert pd.api.types.is_numeric_dtype(df['cpu_avg'])
        assert pd.api.types.is_datetime64_any_dtype(df['_time'])
    
    def test_context_manager(self):
        """Test SplunkConnector as context manager"""
        with patch.object(self.connector, 'close') as mock_close:
            with self.connector as conn:
                assert conn is self.connector
            
            mock_close.assert_called_once()
    
    def test_close(self):
        """Test connection cleanup"""
        # Mock session and client
        mock_session = Mock()
        mock_client = Mock()
        self.connector._session = mock_session
        self.connector._client = mock_client
        
        self.connector.close()
        
        # Verify methods were called on the original mock objects
        mock_session.close.assert_called_once()
        mock_client.logout.assert_called_once()
        
        # Verify references are cleared
        assert self.connector._session is None
        assert self.connector._client is None
    
    def test_close_with_logout_error(self):
        """Test close handles logout errors gracefully"""
        self.connector._session = Mock()
        self.connector._client = Mock()
        self.connector._client.logout.side_effect = Exception("Logout error")
        
        # Should not raise error
        self.connector.close()
        
        assert self.connector._session is None
        assert self.connector._client is None


class TestCreateSplunkConnector:
    """Test factory function"""
    
    @patch.object(SplunkConnector, 'validate_connection')
    def test_create_splunk_connector_success(self, mock_validate):
        """Test successful connector creation"""
        mock_validate.return_value = True
        
        connector = create_splunk_connector("https://localhost:8089", "token123")
        
        assert isinstance(connector, SplunkConnector)
        assert connector.host == "https://localhost:8089"
        assert connector.token == "token123"
        mock_validate.assert_called_once()
    
    @patch.object(SplunkConnector, 'validate_connection')
    def test_create_splunk_connector_validation_failure(self, mock_validate):
        """Test connector creation with validation failure"""
        mock_validate.side_effect = SplunkConnectionError("Connection failed")
        
        with pytest.raises(SplunkConnectionError):
            create_splunk_connector("https://localhost:8089", "invalid_token")