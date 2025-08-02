"""
Integration tests for Splunk connectivity

These tests require a running Splunk instance and will be skipped if not available.
"""

import pytest
import pandas as pd
import os
import time
from datetime import datetime, timedelta

from kepler.connectors.splunk import SplunkConnector
from kepler.connectors.hec import HecWriter
from kepler.utils.exceptions import SplunkConnectionError, DataExtractionError


@pytest.fixture(scope="module")
def splunk_config():
    """Splunk configuration for integration tests"""
    return {
        'host': os.getenv('SPLUNK_HOST', 'https://localhost:8089'),
        'token': os.getenv('SPLUNK_TOKEN', ''),
        'hec_token': os.getenv('SPLUNK_HEC_TOKEN', ''),
        'hec_url': os.getenv('SPLUNK_HEC_URL', 'https://localhost:8088/services/collector'),
        'verify_ssl': False,  # Usually false for dev environments
        'timeout': 30
    }


@pytest.fixture(scope="module")
def splunk_connector(splunk_config):
    """Create SplunkConnector for integration tests"""
    if not splunk_config['token']:
        pytest.skip("SPLUNK_TOKEN not provided - skipping integration tests")
    
    connector = SplunkConnector(
        host=splunk_config['host'],
        token=splunk_config['token'],
        verify_ssl=splunk_config['verify_ssl'],
        timeout=splunk_config['timeout']
    )
    
    # Test connectivity before proceeding
    try:
        connector.validate_connection()
    except SplunkConnectionError:
        pytest.skip("Could not connect to Splunk - skipping integration tests")
    
    yield connector
    connector.close()


@pytest.fixture(scope="module")
def hec_writer(splunk_config):
    """Create HecWriter for integration tests"""
    if not splunk_config['hec_token']:
        pytest.skip("SPLUNK_HEC_TOKEN not provided - skipping HEC integration tests")
    
    writer = HecWriter(
        hec_url=splunk_config['hec_url'],
        hec_token=splunk_config['hec_token'],
        verify_ssl=splunk_config['verify_ssl'],
        timeout=splunk_config['timeout']
    )
    
    # Test connectivity before proceeding
    try:
        writer.validate_connection()
    except SplunkConnectionError:
        pytest.skip("Could not connect to Splunk HEC - skipping HEC integration tests")
    
    yield writer
    writer.close()


class TestSplunkConnectorIntegration:
    """Integration tests for SplunkConnector"""
    
    def test_basic_connectivity(self, splunk_connector):
        """Test basic Splunk connectivity"""
        health_info = splunk_connector.health_check()
        
        assert health_info['connected'] == True
        assert health_info['response_time_ms'] is not None
        assert health_info['response_time_ms'] < 5000  # Less than 5 seconds
    
    def test_authentication(self, splunk_connector):
        """Test Splunk authentication"""
        result = splunk_connector.test_authentication()
        assert result == True
    
    def test_simple_search(self, splunk_connector):
        """Test simple search execution"""
        # Use makeresults to generate test data
        query = "| makeresults count=5 | eval test_field=random() | table _time test_field"
        
        results = splunk_connector.search(query, max_results=10)
        
        assert isinstance(results, list)
        assert len(results) == 5
        assert all('_time' in result for result in results)
        assert all('test_field' in result for result in results)
    
    def test_search_with_time_range(self, splunk_connector):
        """Test search with time range parameters"""
        query = "| makeresults count=3 | eval test_value=1"
        
        results = splunk_connector.search(
            query,
            earliest_time="-1h",
            latest_time="now",
            max_results=5
        )
        
        assert isinstance(results, list)
        assert len(results) == 3
    
    def test_search_to_dataframe(self, splunk_connector):
        """Test search results conversion to DataFrame"""
        query = "| makeresults count=10 | eval number=random() % 100, category=case(random() % 3 == 0, \"A\", random() % 3 == 1, \"B\", 1==1, \"C\")"
        
        df = splunk_connector.search_to_dataframe(query, max_results=15)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10
        assert 'number' in df.columns
        assert 'category' in df.columns
        assert '_time' in df.columns
        
        # Check data type conversion
        assert pd.api.types.is_numeric_dtype(df['number'])
        assert pd.api.types.is_datetime64_any_dtype(df['_time'])
    
    def test_metrics_search(self, splunk_connector):
        """Test metrics search functionality"""
        # Test metrics query detection
        metrics_query = "| mstats avg(cpu_usage) by host span=5m"
        
        # This will likely not return data from a test environment,
        # but should execute without error
        try:
            df = splunk_connector.search_to_dataframe(metrics_query, max_results=10)
            assert isinstance(df, pd.DataFrame)
        except Exception as e:
            # It's OK if no metrics data exists in test environment
            assert "No results found" in str(e) or isinstance(df, pd.DataFrame)
    
    def test_empty_search_results(self, splunk_connector):
        """Test handling of searches that return no results"""
        # Search for something that definitely doesn't exist
        query = "index=nonexistent_index_12345 | head 1"
        
        results = splunk_connector.search(query, max_results=10)
        assert isinstance(results, list)
        assert len(results) == 0
        
        df = splunk_connector.search_to_dataframe(query, max_results=10)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
    
    def test_large_result_handling(self, splunk_connector):
        """Test handling of larger result sets"""
        query = "| makeresults count=1000 | eval id=random()"
        
        # Test with limit
        results = splunk_connector.search(query, max_results=50)
        assert len(results) <= 50
        
        df = splunk_connector.search_to_dataframe(query, max_results=100)
        assert len(df) <= 100


class TestHecWriterIntegration:
    """Integration tests for HecWriter"""
    
    def test_hec_connectivity(self, hec_writer):
        """Test HEC connectivity"""
        health_info = hec_writer.health_check()
        
        assert health_info['connected'] == True or health_info['hec_available'] == True
        assert health_info['response_time_ms'] is not None
    
    def test_write_single_event(self, hec_writer):
        """Test writing a single event to HEC"""
        test_event = {
            'message': f'Integration test event at {datetime.now()}',
            'level': 'INFO',
            'test_run': 'kepler_integration',
            'component': 'test_suite'
        }
        
        result = hec_writer.write_event(
            event_data=test_event,
            source='kepler_integration_test',
            sourcetype='kepler:test',
            index='main'  # Assuming main index exists
        )
        
        assert result == True
    
    def test_write_multiple_events(self, hec_writer):
        """Test writing multiple events to HEC"""
        test_events = []
        for i in range(5):
            test_events.append({
                'event': {
                    'message': f'Batch test event {i}',
                    'event_id': i,
                    'timestamp': datetime.now().isoformat()
                },
                'source': 'kepler_integration_test',
                'sourcetype': 'kepler:test:batch'
            })
        
        result = hec_writer.write_events(test_events)
        assert result == True
    
    def test_write_metric(self, hec_writer):
        """Test writing a single metric to HEC"""
        result = hec_writer.write_metric(
            metric_name='test.integration.cpu_usage',
            metric_value=85.5,
            dimensions={
                'host': 'integration_test_host',
                'environment': 'test',
                'component': 'kepler'
            },
            source='kepler_integration_test'
        )
        
        assert result == True
    
    def test_write_multiple_metrics(self, hec_writer):
        """Test writing multiple metrics to HEC"""
        test_metrics = []
        for i in range(3):
            test_metrics.append({
                'metric': f'test.integration.metric_{i}',
                'value': float(i * 10 + 5),
                'dimensions': {
                    'test_id': str(i),
                    'batch': 'integration_test'
                },
                'source': 'kepler_integration_test'
            })
        
        result = hec_writer.write_metrics(test_metrics)
        assert result == True
    
    def test_write_dataframe_as_events(self, hec_writer):
        """Test writing DataFrame as events"""
        df = pd.DataFrame({
            '_time': pd.date_range('2024-01-01', periods=3, freq='1H'),
            'temperature': [20.5, 21.2, 22.1],
            'humidity': [45, 47, 50],
            'location': ['room_a', 'room_b', 'room_c']
        })
        
        result = hec_writer.write_dataframe_as_events(
            df,
            source='kepler_integration_test',
            sourcetype='kepler:test:dataframe',
            timestamp_column='_time'
        )
        
        assert result == True
    
    def test_write_dataframe_as_metrics(self, hec_writer):
        """Test writing DataFrame as metrics"""
        df = pd.DataFrame({
            '_time': pd.date_range('2024-01-01', periods=3, freq='5min'),
            'cpu_usage': [75.2, 80.1, 85.3],
            'memory_usage': [60.5, 65.2, 70.1],
            'host': ['server1', 'server2', 'server3']
        })
        
        result = hec_writer.write_dataframe_as_metrics(
            df,
            metric_columns=['cpu_usage', 'memory_usage'],
            dimension_columns=['host'],
            timestamp_column='_time',
            source='kepler_integration_test'
        )
        
        assert result == True


class TestEndToEndWorkflow:
    """End-to-end integration tests"""
    
    def test_extract_validate_write_workflow(self, splunk_connector, hec_writer):
        """Test complete extract → validate → write workflow"""
        # Step 1: Extract some test data
        extract_query = "| makeresults count=10 | eval temperature=20+random()%10, humidity=40+random()%20, sensor_id=\"sensor_\".tostring(random()%3)"
        
        df = splunk_connector.search_to_dataframe(extract_query, max_results=15)
        assert len(df) == 10
        
        # Step 2: Validate data quality
        from kepler.utils.data_validator import validate_dataframe_for_ml
        
        report = validate_dataframe_for_ml(df)
        assert report.total_rows == 10
        assert report.ml_ready == True  # Should be good quality synthetic data
        
        # Step 3: Write processed data back to Splunk
        processed_df = df.copy()
        processed_df['processed_timestamp'] = datetime.now()
        processed_df['processing_stage'] = 'integration_test'
        
        result = hec_writer.write_dataframe_as_events(
            processed_df,
            source='kepler_integration_workflow',
            sourcetype='kepler:processed',
            timestamp_column='_time'
        )
        
        assert result == True
        
        # Optional: Wait a bit and try to search for the data we just wrote
        # Note: This might not work immediately due to indexing delays
        time.sleep(2)
        
        search_query = "index=main source=\"kepler_integration_workflow\" | head 5"
        try:
            results = splunk_connector.search(search_query, max_results=10)
            # Results might be empty due to indexing delay, which is OK
            assert isinstance(results, list)
        except Exception:
            # It's OK if this fails - indexing might not be immediate
            pass
    
    def test_metrics_workflow(self, splunk_connector, hec_writer):
        """Test metrics-specific workflow"""
        # Write some test metrics
        test_metrics = []
        for i in range(5):
            test_metrics.append({
                'metric': 'test.integration.workflow.cpu',
                'value': 70.0 + i * 2.5,
                'dimensions': {
                    'host': f'test_host_{i}',
                    'environment': 'integration_test'
                },
                'time': (datetime.now() - timedelta(minutes=i)).timestamp()
            })
        
        result = hec_writer.write_metrics(test_metrics)
        assert result == True
        
        # Try to search for metrics (might not be immediately available)
        time.sleep(1)
        
        # Use mstats if metrics are available
        metrics_query = "| mstats avg(test.integration.workflow.cpu) by host span=1m"
        try:
            df = splunk_connector.search_to_dataframe(metrics_query, max_results=10)
            assert isinstance(df, pd.DataFrame)
            # Might be empty due to indexing delay
        except Exception:
            # It's OK if this fails in test environment
            pass


# Utility functions for test environment setup
def check_splunk_availability():
    """Check if Splunk is available for testing"""
    try:
        connector = SplunkConnector(
            host=os.getenv('SPLUNK_HOST', 'https://localhost:8089'),
            token=os.getenv('SPLUNK_TOKEN', ''),
            verify_ssl=False
        )
        connector.validate_connection()
        return True
    except:
        return False


def check_hec_availability():
    """Check if Splunk HEC is available for testing"""
    try:
        writer = HecWriter(
            hec_url=os.getenv('SPLUNK_HEC_URL', 'https://localhost:8088/services/collector'),
            hec_token=os.getenv('SPLUNK_HEC_TOKEN', ''),
            verify_ssl=False
        )
        writer.validate_connection()
        return True
    except:
        return False


if __name__ == "__main__":
    """Run basic connectivity tests"""
    print("Testing Splunk integration environment...")
    
    print(f"Splunk Host: {os.getenv('SPLUNK_HOST', 'https://localhost:8089')}")
    print(f"Splunk Token: {'✓ Set' if os.getenv('SPLUNK_TOKEN') else '✗ Not set'}")
    print(f"HEC URL: {os.getenv('SPLUNK_HEC_URL', 'https://localhost:8088/services/collector')}")
    print(f"HEC Token: {'✓ Set' if os.getenv('SPLUNK_HEC_TOKEN') else '✗ Not set'}")
    
    print("\nTesting connectivity...")
    splunk_ok = check_splunk_availability()
    hec_ok = check_hec_availability()
    
    print(f"Splunk REST API: {'✓ Available' if splunk_ok else '✗ Not available'}")
    print(f"Splunk HEC: {'✓ Available' if hec_ok else '✗ Not available'}")
    
    if splunk_ok and hec_ok:
        print("\n🎉 Integration test environment is ready!")
    else:
        print("\n⚠️  Integration tests will be skipped due to missing connectivity")
        print("   Set SPLUNK_TOKEN and SPLUNK_HEC_TOKEN environment variables to enable tests")