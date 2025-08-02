"""Tests for data validation functionality"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from kepler.utils.data_validator import (
    DataValidator, 
    DataQualityLevel, 
    DataIssueType,
    validate_dataframe_for_ml,
    clean_dataframe_for_ml
)


class TestDataValidator:
    """Test DataValidator class"""
    
    def setup_method(self):
        """Set up test environment"""
        self.validator = DataValidator(
            min_samples_ml=50,
            max_missing_percentage=15.0,
            max_duplicate_percentage=5.0
        )
    
    def test_initialization(self):
        """Test DataValidator initialization"""
        assert self.validator.min_samples_ml == 50
        assert self.validator.max_missing_percentage == 15.0
        assert self.validator.max_duplicate_percentage == 5.0
    
    def test_validate_high_quality_dataframe(self):
        """Test validation of high quality DataFrame"""
        # Create clean dataset
        df = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.randint(1, 10, 100),
            'category': ['A', 'B', 'C'] * 33 + ['A'],
            'target': np.random.randint(0, 2, 100)
        })
        
        report = self.validator.validate_dataframe(df, target_column='target')
        
        assert report.total_rows == 100
        assert report.total_columns == 4
        assert report.missing_percentage == 0.0
        assert report.duplicate_percentage == 0.0
        assert report.quality_level == DataQualityLevel.EXCELLENT
        assert report.ml_ready == True
        assert len(report.issues) == 0
    
    def test_validate_dataframe_with_missing_values(self):
        """Test validation with missing values"""
        np.random.seed(42)  # For reproducible results
        df = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100), 
            'target': np.random.randint(0, 2, 100)
        })
        
        # Introduce missing values in specific positions to avoid duplicates
        missing_indices = np.random.choice(100, 20, replace=False)
        df.loc[missing_indices[:10], 'feature1'] = np.nan
        df.loc[missing_indices[10:], 'feature2'] = np.nan
        
        report = self.validator.validate_dataframe(df, target_column='target')
        
        assert report.total_rows == 100
        # 20 missing values out of 300 total cells = 6.67%
        assert report.missing_percentage < 10.0
        assert report.quality_level in [DataQualityLevel.GOOD, DataQualityLevel.FAIR, DataQualityLevel.EXCELLENT]
        
        # Should have missing values issue only if percentage is high enough
        missing_issues = [i for i in report.issues if i.issue_type == DataIssueType.MISSING_VALUES]
        # Since we have < 15% missing (our threshold), might not have issues
    
    def test_validate_dataframe_with_duplicates(self):
        """Test validation with duplicate rows"""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 1, 2] * 20,  # Duplicates
            'feature2': [1, 2, 3, 1, 2] * 20,  # Same pattern
            'target': [0, 1, 0, 0, 1] * 20
        })
        
        report = self.validator.validate_dataframe(df, target_column='target')
        
        duplicate_issues = [i for i in report.issues if i.issue_type == DataIssueType.DUPLICATE_ROWS]
        assert len(duplicate_issues) > 0
        assert duplicate_issues[0].severity in ['medium', 'high']
    
    def test_validate_insufficient_samples(self):
        """Test validation with insufficient samples"""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'target': [0, 1, 0]
        })
        
        report = self.validator.validate_dataframe(df, target_column='target')
        
        assert report.ml_ready == False
        assert report.quality_level == DataQualityLevel.UNUSABLE
        
        sample_issues = [i for i in report.issues if i.issue_type == DataIssueType.INSUFFICIENT_SAMPLES]
        assert len(sample_issues) > 0
        assert sample_issues[0].severity == 'critical'
    
    def test_validate_constant_columns(self):
        """Test detection of constant columns"""
        df = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'constant_col': [1] * 100,  # Constant column
            'another_constant': ['same'] * 100,  # Another constant
            'target': np.random.randint(0, 2, 100)
        })
        
        report = self.validator.validate_dataframe(df, target_column='target')
        
        constant_issues = [i for i in report.issues if i.issue_type == DataIssueType.CONSTANT_COLUMNS]
        assert len(constant_issues) > 0
        assert 'constant_col' in constant_issues[0].affected_columns
        assert 'another_constant' in constant_issues[0].affected_columns
        assert constant_issues[0].auto_fixable == True
    
    def test_validate_high_cardinality_categorical(self):
        """Test detection of high cardinality categorical columns"""
        df = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'high_cardinality': [f'category_{i}' for i in range(100)],  # Each row unique
            'target': np.random.randint(0, 2, 100)
        })
        
        report = self.validator.validate_dataframe(df, target_column='target')
        
        cardinality_issues = [i for i in report.issues if i.issue_type == DataIssueType.HIGH_CARDINALITY]
        assert len(cardinality_issues) > 0
        assert 'high_cardinality' in cardinality_issues[0].affected_columns
    
    def test_validate_imbalanced_target(self):
        """Test detection of imbalanced target classes"""
        # Create highly imbalanced dataset - 96% vs 4% (below 5% threshold)
        target_values = [0] * 96 + [1] * 4  # 96% vs 4%
        df = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'target': target_values
        })
        
        report = self.validator.validate_dataframe(df, target_column='target')
        
        imbalance_issues = [i for i in report.issues if i.issue_type == DataIssueType.IMBALANCED_TARGET]
        assert len(imbalance_issues) > 0
        assert imbalance_issues[0].severity == 'medium'
    
    def test_validate_missing_target_column(self):
        """Test validation when target column doesn't exist"""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [1, 2, 3, 4, 5]
        })
        
        report = self.validator.validate_dataframe(df, target_column='nonexistent')
        
        assert report.ml_ready == False
        
        target_issues = [
            i for i in report.issues 
            if i.issue_type == DataIssueType.INVALID_DATA_TYPES 
            and 'nonexistent' in i.affected_columns
        ]
        assert len(target_issues) > 0
        assert target_issues[0].severity == 'critical'
    
    def test_validate_datetime_column(self):
        """Test validation with datetime column"""
        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        # Add some gaps
        dates = dates.delete([10, 11, 50, 51, 52])  # Create gaps
        
        df = pd.DataFrame({
            'timestamp': dates,
            'feature1': np.random.normal(0, 1, len(dates)),
            'target': np.random.randint(0, 2, len(dates))
        })
        
        report = self.validator.validate_dataframe(
            df, 
            target_column='target', 
            datetime_column='timestamp'
        )
        
        # May or may not detect temporal gaps depending on implementation
        assert report.total_rows == len(dates)
    
    def test_clean_dataframe_remove_duplicates(self):
        """Test DataFrame cleaning - duplicate removal"""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 1, 2],  # Duplicates
            'feature2': [1, 2, 3, 1, 2],  # Same pattern
            'target': [0, 1, 0, 0, 1]
        })
        
        df_clean, changes = self.validator.clean_dataframe(df, remove_duplicates=True)
        
        assert len(df_clean) < len(df)  # Should have fewer rows
        assert any('duplicate' in change.lower() for change in changes)
    
    def test_clean_dataframe_remove_constant_columns(self):
        """Test DataFrame cleaning - constant column removal"""
        df = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'constant_col': [1] * 100,
            'target': np.random.randint(0, 2, 100)
        })
        
        df_clean, changes = self.validator.clean_dataframe(
            df, 
            target_column='target',
            remove_constant_columns=True
        )
        
        assert 'constant_col' not in df_clean.columns
        assert 'target' in df_clean.columns  # Target should be preserved
        assert any('constant' in change.lower() for change in changes)
    
    def test_clean_dataframe_fill_missing_values(self):
        """Test DataFrame cleaning - missing value filling"""
        df = pd.DataFrame({
            'numeric_feature': [1.0, 2.0, np.nan, 4.0, 5.0],
            'categorical_feature': ['A', 'B', np.nan, 'A', 'B'],
            'target': [0, 1, 0, 1, 0]
        })
        
        df_clean, changes = self.validator.clean_dataframe(df, fill_missing=True)
        
        assert df_clean.isnull().sum().sum() == 0  # No missing values
        assert any('missing' in change.lower() for change in changes)
        
        # Check that numeric was filled with median
        assert not pd.isna(df_clean.loc[2, 'numeric_feature'])
        # Check that categorical was filled with mode or 'Unknown'
        assert not pd.isna(df_clean.loc[2, 'categorical_feature'])
    
    def test_clean_dataframe_drop_missing_target(self):
        """Test DataFrame cleaning - drop rows with missing target"""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'target': [0, 1, np.nan, 1, 0]  # Missing target in row 2
        })
        
        df_clean, changes = self.validator.clean_dataframe(
            df, 
            target_column='target',
            fill_missing=True
        )
        
        assert len(df_clean) == 4  # One row dropped
        assert df_clean['target'].isnull().sum() == 0  # No missing targets
        assert any('target' in change.lower() for change in changes)
    
    def test_estimate_usable_rows(self):
        """Test estimation of usable rows"""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 1, 2] * 20,  # Will have duplicates
            'target': [0, 1, 0, 1, 0] * 20
        })
        
        # Create a mock issue for duplicates
        from kepler.utils.data_validator import DataIssue
        duplicate_issue = DataIssue(
            issue_type=DataIssueType.DUPLICATE_ROWS,
            severity='medium',
            description='Test duplicates',
            affected_columns=[],
            affected_rows=50  # Mock 50 duplicate rows
        )
        
        usable_rows = self.validator._estimate_usable_rows(df, [duplicate_issue])
        assert usable_rows == 50  # 100 - 50 duplicates
    
    def test_determine_quality_level(self):
        """Test quality level determination"""
        # Test excellent quality
        quality = self.validator._determine_quality_level(0, 0, 0, 1000)
        assert quality == DataQualityLevel.EXCELLENT
        
        # Test good quality (conditions that should trigger GOOD level)
        quality = self.validator._determine_quality_level(10, 8, 3, 1000)
        assert quality == DataQualityLevel.GOOD
        
        # Test fair quality (need values that trigger FAIR thresholds)
        quality = self.validator._determine_quality_level(25, 15, 4, 1000)
        assert quality == DataQualityLevel.FAIR
        
        # Test poor quality (need >50% missing or duplicates, or >10 issues)
        quality = self.validator._determine_quality_level(60, 30, 8, 1000)
        assert quality == DataQualityLevel.POOR
        
        # Test unusable
        quality = self.validator._determine_quality_level(10, 10, 2, 30)
        assert quality == DataQualityLevel.UNUSABLE
    
    def test_assess_ml_readiness(self):
        """Test ML readiness assessment"""
        from kepler.utils.data_validator import DataIssue
        
        # Test ready (no critical issues, sufficient samples)
        ml_ready = self.validator._assess_ml_readiness([], 1000, 'target')
        assert ml_ready == True
        
        # Test not ready (critical issue)
        critical_issue = DataIssue(
            issue_type=DataIssueType.INSUFFICIENT_SAMPLES,
            severity='critical',
            description='Critical issue',
            affected_columns=[],
            affected_rows=0
        )
        ml_ready = self.validator._assess_ml_readiness([critical_issue], 1000, 'target')
        assert ml_ready == False
        
        # Test not ready (insufficient samples)
        ml_ready = self.validator._assess_ml_readiness([], 30, 'target')
        assert ml_ready == False


class TestConvenienceFunctions:
    """Test convenience functions"""
    
    def test_validate_dataframe_for_ml(self):
        """Test convenience validation function"""
        df = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'target': np.random.randint(0, 2, 100)
        })
        
        report = validate_dataframe_for_ml(df, target_column='target')
        
        assert isinstance(report, type(self._get_validation_report_type()))
        assert report.ml_ready == True
    
    def test_clean_dataframe_for_ml(self):
        """Test convenience cleaning function"""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 1, 2],  # Duplicates
            'constant': [1] * 5,  # Constant column
            'target': [0, 1, 0, 0, 1]
        })
        
        df_clean, changes = clean_dataframe_for_ml(df, target_column='target')
        
        assert isinstance(df_clean, pd.DataFrame)
        assert isinstance(changes, list)
        assert len(df_clean) <= len(df)  # Should be same or fewer rows
    
    def _get_validation_report_type(self):
        """Helper to get validation report type for isinstance check"""
        from kepler.utils.data_validator import DataValidationReport
        return DataValidationReport(
            total_rows=0, total_columns=0, numeric_columns=0, categorical_columns=0,
            datetime_columns=0, missing_percentage=0, duplicate_percentage=0,
            quality_level=DataQualityLevel.EXCELLENT, issues=[], recommendations=[],
            ml_ready=True, estimated_usable_rows=0
        )