"""Tests for HEC writer functionality"""

import pytest
import pandas as pd
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import requests

from kepler.connectors.hec import HecWriter, create_hec_writer
from kepler.utils.exceptions import SplunkConnectionError, DataExtractionError


class TestHecWriter:
    """Test HecWriter class"""
    
    def setup_method(self):
        """Set up test environment"""
        self.hec_url = "https://localhost:8088/services/collector"
        self.hec_token = "test_hec_token_123"
        self.writer = HecWriter(
            hec_url=self.hec_url, 
            hec_token=self.hec_token, 
            verify_ssl=False,
            batch_size=2
        )
    
    def test_initialization(self):
        """Test HecWriter initialization"""
        assert self.writer.hec_url == "https://localhost:8088/services/collector"
        assert self.writer.hec_token == "test_hec_token_123"
        assert self.writer.verify_ssl == False
        assert self.writer.batch_size == 2
        assert self.writer.timeout == 30
    
    def test_url_cleanup(self):
        """Test that trailing slashes are removed from HEC URL"""
        writer = HecWriter("https://localhost:8088/services/collector/", "token")
        assert writer.hec_url == "https://localhost:8088/services/collector"
    
    @patch('requests.Session.get')
    def test_health_check_success(self, mock_get):
        """Test successful HEC health check"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        with patch('time.time', side_effect=[0, 0.05]):
            health_info = self.writer.health_check()
        
        assert health_info['connected'] == True
        assert health_info['hec_available'] == True
        assert health_info['response_time_ms'] == 50.0
        assert health_info['error'] is None
    
    @patch('requests.Session.get')
    def test_health_check_failure(self, mock_get):
        """Test HEC health check failure"""
        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.text = "Forbidden"
        mock_get.return_value = mock_response
        
        health_info = self.writer.health_check()
        
        assert health_info['connected'] == False
        assert health_info['error'] == "HTTP 403: Forbidden"
    
    @patch('requests.Session.get')
    def test_health_check_timeout(self, mock_get):
        """Test HEC health check timeout"""
        mock_get.side_effect = requests.exceptions.Timeout()
        
        health_info = self.writer.health_check()
        
        assert health_info['connected'] == False
        assert "timeout" in health_info['error'].lower()
    
    @patch.object(HecWriter, 'health_check')
    def test_validate_connection_success(self, mock_health_check):
        """Test successful HEC connection validation"""
        mock_health_check.return_value = {'connected': True, 'error': None}
        
        result = self.writer.validate_connection()
        assert result == True
    
    @patch.object(HecWriter, 'health_check')
    def test_validate_connection_failure(self, mock_health_check):
        """Test HEC connection validation failure"""
        mock_health_check.return_value = {
            'connected': False, 
            'error': 'HEC token invalid'
        }
        
        with pytest.raises(SplunkConnectionError) as exc_info:
            self.writer.validate_connection()
        
        assert "HEC token invalid" in str(exc_info.value)
    
    @patch('requests.Session.post')
    def test_write_event_success(self, mock_post):
        """Test successful single event write"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'code': 0}
        mock_post.return_value = mock_response
        
        result = self.writer.write_event(
            event_data={'message': 'test event', 'severity': 'info'},
            source='test_source',
            sourcetype='test_sourcetype',
            index='test_index',
            host='test_host'
        )
        
        assert result == True
        
        # Verify request
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert '/event' in call_args[0][0]
        
        # Parse the payload
        payload_lines = call_args[1]['data'].split('\n')
        event_data = json.loads(payload_lines[0])
        
        assert event_data['event'] == {'message': 'test event', 'severity': 'info'}
        assert event_data['source'] == 'test_source'
        assert event_data['sourcetype'] == 'test_sourcetype'
        assert event_data['index'] == 'test_index'
        assert event_data['host'] == 'test_host'
    
    @patch('requests.Session.post')
    def test_write_events_batching(self, mock_post):
        """Test event batching functionality"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'code': 0}
        mock_post.return_value = mock_response
        
        events = [
            {'event': {'msg': f'event_{i}'}} for i in range(5)
        ]
        
        result = self.writer.write_events(events)
        
        assert result == True
        # Should make 3 calls (batch_size=2): 2+2+1 events
        assert mock_post.call_count == 3
    
    @patch('requests.Session.post')
    def test_write_metric_success(self, mock_post):
        """Test successful single metric write"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'code': 0}
        mock_post.return_value = mock_response
        
        result = self.writer.write_metric(
            metric_name='cpu_usage',
            metric_value=85.5,
            dimensions={'host': 'server1', 'env': 'prod'},
            source='monitoring',
            index='metrics'
        )
        
        assert result == True
        
        # Verify request
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert '/metrics' in call_args[0][0]
        
        # Parse the payload
        payload_lines = call_args[1]['data'].split('\n')
        metric_data = json.loads(payload_lines[0])
        
        assert metric_data['metric'] == 'cpu_usage'
        assert metric_data['value'] == 85.5
        assert metric_data['dimensions'] == {'host': 'server1', 'env': 'prod'}
        assert metric_data['source'] == 'monitoring'
        assert metric_data['index'] == 'metrics'
    
    @patch('requests.Session.post')
    def test_write_metrics_batching(self, mock_post):
        """Test metrics batching functionality"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'code': 0}
        mock_post.return_value = mock_response
        
        metrics = [
            {'metric': f'metric_{i}', 'value': i * 10.0} for i in range(3)
        ]
        
        result = self.writer.write_metrics(metrics)
        
        assert result == True
        # Should make 2 calls (batch_size=2): 2+1 metrics
        assert mock_post.call_count == 2
    
    @patch('requests.Session.post')
    def test_write_dataframe_as_events(self, mock_post):
        """Test writing DataFrame as events"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'code': 0}
        mock_post.return_value = mock_response
        
        df = pd.DataFrame({
            '_time': ['2024-01-01T00:00:00', '2024-01-01T01:00:00'],
            'host': ['server1', 'server2'],
            'cpu_usage': [85.5, 92.1]
        })
        
        result = self.writer.write_dataframe_as_events(
            df,
            sourcetype='test:metrics',
            timestamp_column='_time'
        )
        
        assert result == True
        mock_post.assert_called_once()
        
        # Parse payload to verify structure
        call_args = mock_post.call_args
        payload_lines = call_args[1]['data'].split('\n')
        
        event1 = json.loads(payload_lines[0])
        assert event1['event'] == {'host': 'server1', 'cpu_usage': 85.5}
        assert event1['sourcetype'] == 'test:metrics'
        assert 'time' in event1
        
        event2 = json.loads(payload_lines[1])
        assert event2['event'] == {'host': 'server2', 'cpu_usage': 92.1}
    
    @patch('requests.Session.post')
    def test_write_dataframe_as_metrics(self, mock_post):
        """Test writing DataFrame as metrics"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'code': 0}
        mock_post.return_value = mock_response
        
        df = pd.DataFrame({
            '_time': ['2024-01-01T00:00:00', '2024-01-01T01:00:00'],
            'host': ['server1', 'server2'],
            'cpu_usage': [85.5, 92.1],
            'memory_usage': [70.2, 68.9]
        })
        
        result = self.writer.write_dataframe_as_metrics(
            df,
            metric_columns=['cpu_usage', 'memory_usage'],
            dimension_columns=['host'],
            timestamp_column='_time'
        )
        
        assert result == True
        mock_post.assert_called()
        
        # Should create 4 metrics (2 rows × 2 metric columns)
        # Collect all metrics from all batches
        all_metrics = []
        for call in mock_post.call_args_list:
            payload_lines = call[1]['data'].split('\n')
            for line in payload_lines:
                all_metrics.append(json.loads(line))
        
        assert len(all_metrics) == 4
        
        # Find and verify a cpu_usage metric
        cpu_metrics = [m for m in all_metrics if m['metric'] == 'cpu_usage']
        assert len(cpu_metrics) == 2
        
        # Verify one of the cpu_usage metrics has expected structure
        cpu_metric = cpu_metrics[0]
        assert cpu_metric['value'] in [85.5, 92.1]
        assert cpu_metric['dimensions']['host'] in ['server1', 'server2']
    
    @patch('requests.Session.post')
    def test_write_failure_handling(self, mock_post):
        """Test handling of write failures"""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_post.return_value = mock_response
        
        with pytest.raises(DataExtractionError) as exc_info:
            self.writer.write_event({'test': 'data'})
        
        assert "HTTP 400" in str(exc_info.value)
    
    @patch('requests.Session.post')
    def test_hec_error_response_handling(self, mock_post):
        """Test handling of HEC-specific error responses"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'code': 12,
            'text': 'Invalid token'
        }
        mock_post.return_value = mock_response
        
        with pytest.raises(DataExtractionError) as exc_info:
            self.writer.write_event({'test': 'data'})
        
        assert "Invalid token" in str(exc_info.value)
    
    def test_format_timestamp_datetime(self):
        """Test timestamp formatting with datetime"""
        dt = datetime(2024, 1, 1, 12, 0, 0)
        result = self.writer._format_timestamp(dt)
        assert isinstance(result, float)
        assert result == dt.timestamp()
    
    def test_format_timestamp_numeric(self):
        """Test timestamp formatting with numeric values"""
        assert self.writer._format_timestamp(1234567890) == 1234567890.0
        assert self.writer._format_timestamp(1234567890.5) == 1234567890.5
    
    def test_format_timestamp_invalid(self):
        """Test timestamp formatting with invalid type"""
        with pytest.raises(ValueError):
            self.writer._format_timestamp("invalid")
    
    def test_empty_data_handling(self):
        """Test handling of empty data"""
        assert self.writer.write_events([]) == True
        assert self.writer.write_metrics([]) == True
        
        empty_df = pd.DataFrame()
        assert self.writer.write_dataframe_as_events(empty_df) == True
        assert self.writer.write_dataframe_as_metrics(empty_df, ['metric']) == True
    
    def test_context_manager(self):
        """Test HecWriter as context manager"""
        with patch.object(self.writer, 'close') as mock_close:
            with self.writer as writer:
                assert writer is self.writer
            
            mock_close.assert_called_once()
    
    def test_close(self):
        """Test connection cleanup"""
        mock_session = Mock()
        self.writer._session = mock_session
        
        self.writer.close()
        
        mock_session.close.assert_called_once()
        assert self.writer._session is None


class TestCreateHecWriter:
    """Test factory function"""
    
    @patch.object(HecWriter, 'validate_connection')
    def test_create_hec_writer_success(self, mock_validate):
        """Test successful HEC writer creation"""
        mock_validate.return_value = True
        
        writer = create_hec_writer(
            "https://localhost:8088/services/collector", 
            "token123"
        )
        
        assert isinstance(writer, HecWriter)
        assert writer.hec_url == "https://localhost:8088/services/collector"
        assert writer.hec_token == "token123"
        mock_validate.assert_called_once()
    
    @patch.object(HecWriter, 'validate_connection')
    def test_create_hec_writer_validation_failure(self, mock_validate):
        """Test HEC writer creation with validation failure"""
        mock_validate.side_effect = SplunkConnectionError("HEC connection failed")
        
        with pytest.raises(SplunkConnectionError):
            create_hec_writer(
                "https://localhost:8088/services/collector", 
                "invalid_token"
            )