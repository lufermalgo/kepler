"""
End-to-end realistic testing for Kepler framework

Tests the complete workflow with real data files and actual model training,
avoiding mocks for better validation of real-world usage.
"""

import pytest
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
import os

from kepler.core.global_config import get_global_config_manager
from kepler.trainers.sklearn_trainers import create_trainer
from kepler.trainers.base import TrainingConfig
from kepler.utils.data_validator import validate_dataframe_for_ml


class TestEndToEndWorkflow:
    """
    End-to-end tests using realistic data without mocks
    
    These tests use actual data files, real model training, and file I/O
    to validate that the framework works in realistic scenarios.
    """
    
    @pytest.fixture
    def realistic_sensor_data(self):
        """
        Create realistic industrial sensor data similar to what would come from Splunk
        Based on actual patterns from industrial calcinador sensors
        """
        np.random.seed(42)  # For reproducible tests
        n_samples = 500
        
        # Simulate 30 days of hourly sensor readings
        timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='h')
        
        # Industrial sensor patterns - realistic ranges and correlations
        temperature = np.random.normal(850, 50, n_samples)  # Calcinador temperature
        temperature = np.clip(temperature, 750, 950)  # Physical limits
        
        pressure = np.random.normal(2.5, 0.3, n_samples)  # Bar
        pressure = np.clip(pressure, 2.0, 3.0)
        
        # Flow rate correlated with temperature (realistic industrial relationship)
        flow_rate = 45 + 0.02 * temperature + np.random.normal(0, 3, n_samples)
        flow_rate = np.clip(flow_rate, 40, 65)
        
        # Gas consumption - target variable with realistic relationships
        gas_consumption = (
            20 +  # Base consumption
            0.1 * (temperature - 850) +  # Temperature impact
            15 * (pressure - 2.5) +  # Pressure impact
            0.3 * (flow_rate - 50) +  # Flow impact
            np.random.normal(0, 2, n_samples)  # Noise
        )
        
        # Add some realistic operational issues (maintenance periods, etc.)
        maintenance_periods = np.random.choice(n_samples, size=20, replace=False)
        gas_consumption[maintenance_periods] += np.random.normal(5, 2, 20)  # Higher consumption during maintenance
        
        # Create DataFrame with realistic column names
        df = pd.DataFrame({
            'timestamp': timestamps,
            'temperature_C': temperature,
            'pressure_bar': pressure,
            'flow_rate_m3h': flow_rate,
            'gas_consumption_m3h': gas_consumption,
            'shift': np.random.choice(['A', 'B', 'C'], n_samples),  # Shift information
            'operator_id': np.random.choice(['OP001', 'OP002', 'OP003', 'OP004'], n_samples)
        })
        
        return df
    
    @pytest.fixture 
    def classification_sensor_data(self):
        """
        Create realistic sensor data for classification (maintenance prediction)
        """
        np.random.seed(123)
        n_samples = 300
        
        # Sensor readings
        vibration = np.random.normal(2.5, 0.8, n_samples)
        temperature = np.random.normal(75, 15, n_samples)
        pressure = np.random.normal(100, 20, n_samples)
        runtime_hours = np.random.uniform(0, 8760, n_samples)  # Annual hours
        
        # Maintenance prediction based on realistic thresholds
        maintenance_score = (
            0.3 * (vibration - 2.0) / 2.0 +  # High vibration indicates issues
            0.2 * (temperature - 70) / 30 +  # Overheating
            0.1 * (pressure - 90) / 40 +     # Pressure anomalies
            0.4 * runtime_hours / 8760       # Operating time factor
        )
        
        # Add some noise and create binary classification
        maintenance_score += np.random.normal(0, 0.2, n_samples)
        needs_maintenance = (maintenance_score > 0.6).astype(int)
        
        df = pd.DataFrame({
            'vibration_mm_s': vibration,
            'temperature_C': temperature,
            'pressure_bar': pressure,
            'runtime_hours': runtime_hours,
            'needs_maintenance': needs_maintenance
        })
        
        return df
    
    def test_realistic_regression_workflow(self, realistic_sensor_data):
        """
        Test complete regression workflow with realistic industrial data
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 1. Save realistic data to CSV (simulating data extracted from Splunk)
            data_file = temp_path / "sensor_data.csv"
            realistic_sensor_data.to_csv(data_file, index=False)
            
            # Verify data was saved correctly
            assert data_file.exists()
            loaded_data = pd.read_csv(data_file)
            assert len(loaded_data) == len(realistic_sensor_data)
            
            # 2. Validate data quality for ML (realistic validation)
            validation_report = validate_dataframe_for_ml(
                loaded_data, 
                target_column='gas_consumption_m3h'
            )
            
            # Should pass quality checks with realistic data
            assert validation_report.ml_ready
            assert validation_report.missing_percentage < 1.0  # Minimal missing data
            
            # 3. Train model with realistic configuration
            config = TrainingConfig(
                algorithm='random_forest',
                target_column='gas_consumption_m3h',
                feature_columns=['temperature_C', 'pressure_bar', 'flow_rate_m3h'],
                test_size=0.2,
                random_state=42,
                hyperparameters={
                    'n_estimators': 50,  # Smaller for faster testing
                    'max_depth': 10,
                    'random_state': 42
                }
            )
            
            trainer = create_trainer('random_forest', config)
            training_result = trainer.train(loaded_data)
            
            # 4. Validate training results
            assert training_result.model is not None  # Model was trained
            assert training_result.model_type == 'random_forest'
            assert 'r2' in training_result.metrics  # Note: key is 'r2', not 'r2_score'
            assert 'mse' in training_result.metrics
            assert 'mae' in training_result.metrics
            
            # With realistic data, we should get reasonable performance
            assert training_result.metrics['r2'] > 0.7  # Good R² for industrial data
            assert training_result.metrics['mse'] > 0  # Positive MSE
            
            # 5. Verify model can be loaded and used for prediction
            assert training_result.model_path
            model_file = Path(training_result.model_path)
            assert model_file.exists()
            
            # Load model and make predictions
            import joblib
            model_metadata = joblib.load(model_file)
            loaded_model = model_metadata['model']  # Extract model from metadata
            
            # Test prediction with realistic input
            test_sample = pd.DataFrame({
                'temperature_C': [875.0],
                'pressure_bar': [2.7],
                'flow_rate_m3h': [52.5]
            })
            
            prediction = loaded_model.predict(test_sample)
            assert len(prediction) == 1
            assert 15 < prediction[0] < 35  # Realistic gas consumption range
    
    def test_realistic_classification_workflow(self, classification_sensor_data):
        """
        Test complete classification workflow with realistic maintenance data
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 1. Save data
            data_file = temp_path / "maintenance_data.csv"
            classification_sensor_data.to_csv(data_file, index=False)
            
            # 2. Data quality validation
            validation_report = validate_dataframe_for_ml(
                classification_sensor_data,
                target_column='needs_maintenance'
            )
            
            assert validation_report.ml_ready
            
            # 3. Train classification model
            config = TrainingConfig(
                algorithm='random_forest',
                target_column='needs_maintenance',
                test_size=0.3,
                random_state=123,
                hyperparameters={
                    'n_estimators': 30,
                    'max_depth': 8,
                    'random_state': 123
                }
            )
            
            trainer = create_trainer('random_forest', config)
            training_result = trainer.train(classification_sensor_data)
            
            # 4. Validate results
            assert training_result.model is not None  # Model was trained
            assert training_result.model_type == 'random_forest'
            assert 'accuracy' in training_result.metrics
            assert 'f1' in training_result.metrics  # Note: key is 'f1', not 'f1_score'
            
            # Should achieve decent performance on realistic data
            assert training_result.metrics['accuracy'] > 0.6
            
            # 5. Test prediction capabilities
            import joblib
            model_metadata = joblib.load(training_result.model_path)
            model = model_metadata['model']  # Extract model from metadata
            
            # High-risk scenario (should predict maintenance needed)
            high_risk = pd.DataFrame({
                'vibration_mm_s': [4.5],  # High vibration
                'temperature_C': [95.0],  # High temperature  
                'pressure_bar': [135.0],  # High pressure
                'runtime_hours': [7500]   # High runtime
            })
            
            prediction = model.predict(high_risk)
            prediction_proba = model.predict_proba(high_risk)
            
            assert prediction[0] in [0, 1]  # Valid classification result
            assert len(prediction_proba[0]) == 2  # Binary classification probabilities
            assert 0 <= prediction_proba[0][0] <= 1
            assert 0 <= prediction_proba[0][1] <= 1
            assert abs(sum(prediction_proba[0]) - 1.0) < 0.001  # Probabilities sum to 1
    
    def test_data_quality_with_real_issues(self):
        """
        Test data validation with realistic data quality issues
        """
        # Create data with realistic quality issues
        np.random.seed(999)
        n_samples = 200
        
        df = pd.DataFrame({
            'sensor_1': np.random.normal(50, 10, n_samples),
            'sensor_2': np.random.normal(100, 20, n_samples),
            'target': np.random.normal(25, 5, n_samples)
        })
        
        # Introduce realistic issues that are more severe:
        
        # 1. Missing values (sensor failures) - more aggressive
        missing_indices = np.random.choice(n_samples, size=40, replace=False)
        df.loc[missing_indices, 'sensor_1'] = np.nan
        missing_indices_2 = np.random.choice(n_samples, size=25, replace=False) 
        df.loc[missing_indices_2, 'sensor_2'] = np.nan
        
        # 2. Duplicate readings (sensor stuck) - more duplicates
        df.iloc[50:70] = df.iloc[30:50].values  # Duplicate 20 rows
        df.iloc[120:135] = df.iloc[100:115].values  # Another 15 duplicates
        
        # 3. Constant values (sensor malfunction) - larger section
        df.iloc[160:190, df.columns.get_loc('sensor_2')] = 999.9  # 30 constant values
        
        # 4. Extreme outliers (sensor spikes) - more outliers
        outlier_indices = np.random.choice(n_samples, size=8, replace=False)
        df.loc[outlier_indices, 'sensor_1'] = np.random.choice([-999, 9999], size=8)
        
        # 5. Make target column have high correlation with constant sensor
        df.loc[df['sensor_2'] == 999.9, 'target'] = 999.9  # This creates bias
        
        # Test validation
        validation_report = validate_dataframe_for_ml(df, target_column='target')
        
        # Should detect quality issues - realistic expectation
        # Industrial data often has issues but can still be usable for ML
        assert validation_report.missing_percentage > 10.0  # Should detect significant missing values
        assert validation_report.duplicate_percentage > 15.0  # Should detect significant duplicates
        assert len(validation_report.issues) > 0  # Should identify issues
        
        # Should still be marked as usable since we have enough data (industrial datasets often have issues)
        assert validation_report.ml_ready  # Industrial data with issues can still be usable
        assert validation_report.quality_level.value in ['good', 'fair']  # Not excellent due to issues
        
        # Verify specific issue detection
        issue_descriptions = [issue.description for issue in validation_report.issues]
        assert any('duplicate' in desc.lower() for desc in issue_descriptions)
        
        # Should provide recommendations for improvement
        assert len(validation_report.recommendations) > 0
    
    def test_global_configuration_integration(self):
        """
        Test that global configuration system works with realistic settings
        """
        config_manager = get_global_config_manager()
        
        # Test configuration validation
        credentials = config_manager.validate_credentials_available()
        
        # Should return a dictionary with credential status
        assert isinstance(credentials, dict)
        assert 'splunk_token' in credentials
        assert 'splunk_hec_token' in credentials
        assert 'gcp_project_id' in credentials
        
        # Values should be boolean
        for key, available in credentials.items():
            assert isinstance(available, bool)
        
        # Test configuration loading (should not fail)
        global_config = config_manager.load_global_config()
        assert global_config is not None
        assert hasattr(global_config, 'splunk')
        assert hasattr(global_config, 'gcp')
        assert hasattr(global_config, 'mlflow')


# Additional helper for integration with realistic external data
class TestExternalDataIntegration:
    """
    Tests for integration with external data sources (when available)
    These tests are designed to skip gracefully if external resources aren't available
    """
    
    def test_splunk_connection_realistic_fallback(self):
        """
        Test Splunk connection with realistic fallback behavior
        Only runs if we can attempt a connection (even if it fails)
        """
        from kepler.connectors.splunk import SplunkConnector
        from kepler.utils.exceptions import SplunkConnectionError
        
        # Test with localhost (typical development setup)
        connector = SplunkConnector(
            host="https://localhost:8089",
            token="test_token_for_realistic_testing",
            verify_ssl=False,  # Common in development
            auto_fallback=True
        )
        
        # This should attempt connection and fail gracefully with informative errors
        with pytest.raises(SplunkConnectionError) as exc_info:
            connector.validate_connection()
        
        # Error message should provide helpful suggestions
        error_msg = str(exc_info.value)
        assert "connection attempts failed" in error_msg.lower()
        
        # Should have attempted multiple strategies
        assert any(phrase in error_msg for phrase in ["Primary:", "SSL fallback:", "HTTP fallback:"])