"""
Data validation and cleaning utilities for Kepler framework

Provides comprehensive data quality checks and automatic cleaning for ML readiness.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from kepler.utils.logging import get_logger
from kepler.utils.exceptions import ValidationError


class DataQualityLevel(Enum):
    """Data quality assessment levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNUSABLE = "unusable"


class DataIssueType(Enum):
    """Types of data quality issues"""
    MISSING_VALUES = "missing_values"
    DUPLICATE_ROWS = "duplicate_rows"
    CONSTANT_COLUMNS = "constant_columns"
    HIGH_CARDINALITY = "high_cardinality"
    INSUFFICIENT_SAMPLES = "insufficient_samples"
    IMBALANCED_TARGET = "imbalanced_target"
    INVALID_DATA_TYPES = "invalid_data_types"
    OUTLIERS = "outliers"
    TEMPORAL_GAPS = "temporal_gaps"


@dataclass
class DataIssue:
    """Represents a data quality issue"""
    issue_type: DataIssueType
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    affected_columns: List[str]
    affected_rows: int
    suggestion: Optional[str] = None
    auto_fixable: bool = False


@dataclass
class DataValidationReport:
    """Comprehensive data validation report"""
    total_rows: int
    total_columns: int
    numeric_columns: int
    categorical_columns: int
    datetime_columns: int
    missing_percentage: float
    duplicate_percentage: float
    quality_level: DataQualityLevel
    issues: List[DataIssue]
    recommendations: List[str]
    ml_ready: bool
    estimated_usable_rows: int


class DataValidator:
    """
    Comprehensive data validator for ML readiness assessment
    """
    
    def __init__(self, 
                 min_samples_ml: int = 100,
                 max_missing_percentage: float = 20.0,
                 max_duplicate_percentage: float = 10.0,
                 min_target_class_percentage: float = 5.0,
                 max_cardinality_ratio: float = 0.5):
        """
        Initialize data validator with quality thresholds
        
        Args:
            min_samples_ml: Minimum samples required for ML
            max_missing_percentage: Maximum acceptable missing data percentage
            max_duplicate_percentage: Maximum acceptable duplicate percentage
            min_target_class_percentage: Minimum percentage for target classes
            max_cardinality_ratio: Maximum cardinality ratio for categorical features
        """
        self.min_samples_ml = min_samples_ml
        self.max_missing_percentage = max_missing_percentage
        self.max_duplicate_percentage = max_duplicate_percentage
        self.min_target_class_percentage = min_target_class_percentage
        self.max_cardinality_ratio = max_cardinality_ratio
        self.logger = get_logger(f"{__name__}.DataValidator")
    
    def validate_dataframe(self, 
                          df: pd.DataFrame, 
                          target_column: Optional[str] = None,
                          datetime_column: Optional[str] = None) -> DataValidationReport:
        """
        Perform comprehensive validation of DataFrame
        
        Args:
            df: DataFrame to validate
            target_column: Name of target column for ML
            datetime_column: Name of datetime column if present
            
        Returns:
            DataValidationReport with comprehensive assessment
        """
        self.logger.info(f"Starting data validation for DataFrame with shape {df.shape}")
        
        issues = []
        recommendations = []
        
        # Basic statistics
        total_rows, total_columns = df.shape
        numeric_columns = len(df.select_dtypes(include=[np.number]).columns)
        categorical_columns = len(df.select_dtypes(include=['object', 'category']).columns)
        datetime_columns = len(df.select_dtypes(include=['datetime64[ns]']).columns)
        
        # Missing values analysis
        missing_info = self._analyze_missing_values(df)
        issues.extend(missing_info['issues'])
        missing_percentage = missing_info['overall_percentage']
        
        # Duplicate analysis
        duplicate_info = self._analyze_duplicates(df)
        issues.extend(duplicate_info['issues'])
        duplicate_percentage = duplicate_info['percentage']
        
        # Data type analysis
        dtype_issues = self._analyze_data_types(df)
        issues.extend(dtype_issues)
        
        # Column analysis
        column_issues = self._analyze_columns(df)
        issues.extend(column_issues)
        
        # Target column analysis (if provided)
        if target_column:
            target_issues = self._analyze_target_column(df, target_column)
            issues.extend(target_issues)
        
        # Datetime analysis (if provided)
        if datetime_column:
            datetime_issues = self._analyze_datetime_column(df, datetime_column)
            issues.extend(datetime_issues)
        
        # Sample size analysis
        sample_issues = self._analyze_sample_size(df)
        issues.extend(sample_issues)
        
        # Calculate estimated usable rows
        estimated_usable_rows = self._estimate_usable_rows(df, issues)
        
        # Determine quality level
        quality_level = self._determine_quality_level(
            missing_percentage, duplicate_percentage, len(issues), estimated_usable_rows
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(issues, df)
        
        # Determine ML readiness
        ml_ready = self._assess_ml_readiness(issues, estimated_usable_rows, target_column)
        
        report = DataValidationReport(
            total_rows=total_rows,
            total_columns=total_columns,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            datetime_columns=datetime_columns,
            missing_percentage=missing_percentage,
            duplicate_percentage=duplicate_percentage,
            quality_level=quality_level,
            issues=issues,
            recommendations=recommendations,
            ml_ready=ml_ready,
            estimated_usable_rows=estimated_usable_rows
        )
        
        self.logger.info(f"Data validation completed - Quality: {quality_level.value}, ML Ready: {ml_ready}")
        return report
    
    def clean_dataframe(self, 
                       df: pd.DataFrame, 
                       target_column: Optional[str] = None,
                       auto_fix: bool = True,
                       remove_duplicates: bool = True,
                       fill_missing: bool = True,
                       remove_constant_columns: bool = True) -> Tuple[pd.DataFrame, List[str]]:
        """
        Clean DataFrame based on validation findings
        
        Args:
            df: DataFrame to clean
            target_column: Target column name
            auto_fix: Whether to apply automatic fixes
            remove_duplicates: Whether to remove duplicate rows
            fill_missing: Whether to fill missing values
            remove_constant_columns: Whether to remove constant columns
            
        Returns:
            Tuple of (cleaned_df, list_of_changes_made)
        """
        self.logger.info(f"Starting data cleaning for DataFrame with shape {df.shape}")
        
        df_clean = df.copy()
        changes_made = []
        
        if auto_fix:
            # Remove duplicates
            if remove_duplicates:
                before_rows = len(df_clean)
                df_clean = df_clean.drop_duplicates()
                after_rows = len(df_clean)
                if before_rows != after_rows:
                    removed = before_rows - after_rows
                    changes_made.append(f"Removed {removed} duplicate rows")
                    self.logger.info(f"Removed {removed} duplicate rows")
            
            # Remove constant columns (except target)
            if remove_constant_columns:
                constant_cols = []
                for col in df_clean.columns:
                    if col != target_column and df_clean[col].nunique() <= 1:
                        constant_cols.append(col)
                
                if constant_cols:
                    df_clean = df_clean.drop(columns=constant_cols)
                    changes_made.append(f"Removed {len(constant_cols)} constant columns: {constant_cols}")
                    self.logger.info(f"Removed constant columns: {constant_cols}")
            
            # Fill missing values
            if fill_missing:
                missing_cols = df_clean.columns[df_clean.isnull().any()].tolist()
                if missing_cols:
                    filled_count = 0
                    for col in missing_cols:
                        if col == target_column:
                            # Don't fill target column - drop rows instead
                            before_rows = len(df_clean)
                            df_clean = df_clean.dropna(subset=[col])
                            after_rows = len(df_clean)
                            if before_rows != after_rows:
                                dropped = before_rows - after_rows
                                changes_made.append(f"Dropped {dropped} rows with missing target values")
                        else:
                            # Fill based on data type
                            if df_clean[col].dtype in ['object', 'category']:
                                # Fill categorical with mode
                                mode_value = df_clean[col].mode()
                                if len(mode_value) > 0:
                                    df_clean[col] = df_clean[col].fillna(mode_value[0])
                                else:
                                    df_clean[col] = df_clean[col].fillna('Unknown')
                                filled_count += 1
                            else:
                                # Fill numeric with median
                                median_value = df_clean[col].median()
                                df_clean[col] = df_clean[col].fillna(median_value)
                                filled_count += 1
                    
                    if filled_count > 0:
                        changes_made.append(f"Filled missing values in {filled_count} columns")
                        self.logger.info(f"Filled missing values in {filled_count} columns")
        
        self.logger.info(f"Data cleaning completed - Shape changed from {df.shape} to {df_clean.shape}")
        return df_clean, changes_made
    
    def _analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing values in the DataFrame"""
        missing_info = df.isnull().sum()
        total_missing = missing_info.sum()
        overall_percentage = (total_missing / (len(df) * len(df.columns))) * 100
        
        issues = []
        
        # Check overall missing percentage
        if overall_percentage > self.max_missing_percentage:
            issues.append(DataIssue(
                issue_type=DataIssueType.MISSING_VALUES,
                severity='high' if overall_percentage > 50 else 'medium',
                description=f"High percentage of missing values: {overall_percentage:.1f}%",
                affected_columns=[],
                affected_rows=total_missing,
                suggestion="Consider data imputation or additional data collection",
                auto_fixable=True
            ))
        
        # Check individual columns
        high_missing_cols = []
        for col in df.columns:
            col_missing_pct = (missing_info[col] / len(df)) * 100
            if col_missing_pct > 50:
                high_missing_cols.append(col)
        
        if high_missing_cols:
            issues.append(DataIssue(
                issue_type=DataIssueType.MISSING_VALUES,
                severity='high',
                description=f"Columns with >50% missing values: {high_missing_cols}",
                affected_columns=high_missing_cols,
                affected_rows=0,
                suggestion="Consider removing these columns or finding alternative data sources",
                auto_fixable=False
            ))
        
        return {
            'overall_percentage': overall_percentage,
            'issues': issues,
            'column_details': missing_info.to_dict()
        }
    
    def _analyze_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze duplicate rows"""
        duplicate_count = df.duplicated().sum()
        duplicate_percentage = (duplicate_count / len(df)) * 100
        
        issues = []
        
        if duplicate_percentage > self.max_duplicate_percentage:
            issues.append(DataIssue(
                issue_type=DataIssueType.DUPLICATE_ROWS,
                severity='medium' if duplicate_percentage < 25 else 'high',
                description=f"High percentage of duplicate rows: {duplicate_percentage:.1f}%",
                affected_columns=[],
                affected_rows=duplicate_count,
                suggestion="Remove duplicate rows to improve data quality",
                auto_fixable=True
            ))
        
        return {
            'percentage': duplicate_percentage,
            'count': duplicate_count,
            'issues': issues
        }
    
    def _analyze_data_types(self, df: pd.DataFrame) -> List[DataIssue]:
        """Analyze data types for potential issues"""
        issues = []
        
        # Check for object columns that might be numeric
        object_cols = df.select_dtypes(include=['object']).columns
        for col in object_cols:
            # Try converting to numeric
            numeric_series = pd.to_numeric(df[col], errors='coerce')
            numeric_count = numeric_series.notna().sum()
            if numeric_count > len(df) * 0.8:  # 80% can be converted to numeric
                issues.append(DataIssue(
                    issue_type=DataIssueType.INVALID_DATA_TYPES,
                    severity='low',
                    description=f"Column '{col}' appears to be numeric but stored as object",
                    affected_columns=[col],
                    affected_rows=0,
                    suggestion=f"Convert column '{col}' to numeric type",
                    auto_fixable=True
                ))
        
        return issues
    
    def _analyze_columns(self, df: pd.DataFrame) -> List[DataIssue]:
        """Analyze individual columns for quality issues"""
        issues = []
        
        # Check for constant columns
        constant_cols = []
        for col in df.columns:
            if df[col].nunique() <= 1:
                constant_cols.append(col)
        
        if constant_cols:
            issues.append(DataIssue(
                issue_type=DataIssueType.CONSTANT_COLUMNS,
                severity='low',
                description=f"Constant columns (no variance): {constant_cols}",
                affected_columns=constant_cols,
                affected_rows=0,
                suggestion="Remove constant columns as they don't provide information for ML",
                auto_fixable=True
            ))
        
        # Check for high cardinality categorical columns
        high_cardinality_cols = []
        for col in df.select_dtypes(include=['object', 'category']).columns:
            cardinality_ratio = df[col].nunique() / len(df)
            if cardinality_ratio > self.max_cardinality_ratio:
                high_cardinality_cols.append(col)
        
        if high_cardinality_cols:
            issues.append(DataIssue(
                issue_type=DataIssueType.HIGH_CARDINALITY,
                severity='medium',
                description=f"High cardinality categorical columns: {high_cardinality_cols}",
                affected_columns=high_cardinality_cols,
                affected_rows=0,
                suggestion="Consider feature engineering or dimensionality reduction for high cardinality features",
                auto_fixable=False
            ))
        
        return issues
    
    def _analyze_target_column(self, df: pd.DataFrame, target_column: str) -> List[DataIssue]:
        """Analyze target column for ML suitability"""
        issues = []
        
        if target_column not in df.columns:
            issues.append(DataIssue(
                issue_type=DataIssueType.INVALID_DATA_TYPES,
                severity='critical',
                description=f"Target column '{target_column}' not found in DataFrame",
                affected_columns=[target_column],
                affected_rows=0,
                suggestion="Verify target column name or provide correct column name",
                auto_fixable=False
            ))
            return issues
        
        target_series = df[target_column]
        
        # Check for missing values in target
        missing_target = target_series.isnull().sum()
        if missing_target > 0:
            issues.append(DataIssue(
                issue_type=DataIssueType.MISSING_VALUES,
                severity='high',
                description=f"Target column has {missing_target} missing values",
                affected_columns=[target_column],
                affected_rows=missing_target,
                suggestion="Remove rows with missing target values",
                auto_fixable=True
            ))
        
        # Check for class imbalance (if categorical target or few unique values)
        unique_values = target_series.nunique()
        if target_series.dtype in ['object', 'category'] or unique_values <= 20:
            class_counts = target_series.value_counts(normalize=True) * 100
            min_class_pct = class_counts.min()
            
            # Only check for imbalance if we have classification (2+ classes but not too many)
            if unique_values >= 2 and min_class_pct < self.min_target_class_percentage:
                issues.append(DataIssue(
                    issue_type=DataIssueType.IMBALANCED_TARGET,
                    severity='medium',
                    description=f"Imbalanced target classes - smallest class: {min_class_pct:.1f}%",
                    affected_columns=[target_column],
                    affected_rows=0,
                    suggestion="Consider class balancing techniques or collecting more data for minority classes",
                    auto_fixable=False
                ))
        
        return issues
    
    def _analyze_datetime_column(self, df: pd.DataFrame, datetime_column: str) -> List[DataIssue]:
        """Analyze datetime column for temporal issues"""
        issues = []
        
        if datetime_column not in df.columns:
            return issues
        
        dt_series = pd.to_datetime(df[datetime_column], errors='coerce')
        
        # Check for temporal gaps (basic check)
        if len(dt_series) > 1:
            dt_sorted = dt_series.dropna().sort_values()
            if len(dt_sorted) > 1:
                time_diffs = dt_sorted.diff().dropna()
                if len(time_diffs) > 0:
                    median_diff = time_diffs.median()
                    large_gaps = time_diffs[time_diffs > median_diff * 5]  # Gaps 5x larger than median
                    
                    if len(large_gaps) > 0:
                        issues.append(DataIssue(
                            issue_type=DataIssueType.TEMPORAL_GAPS,
                            severity='low',
                            description=f"Detected {len(large_gaps)} large temporal gaps in data",
                            affected_columns=[datetime_column],
                            affected_rows=len(large_gaps),
                            suggestion="Check for missing time periods in your data collection",
                            auto_fixable=False
                        ))
        
        return issues
    
    def _analyze_sample_size(self, df: pd.DataFrame) -> List[DataIssue]:
        """Analyze if sample size is sufficient for ML"""
        issues = []
        
        if len(df) < self.min_samples_ml:
            issues.append(DataIssue(
                issue_type=DataIssueType.INSUFFICIENT_SAMPLES,
                severity='critical' if len(df) < 50 else 'high',
                description=f"Insufficient samples for ML: {len(df)} (minimum: {self.min_samples_ml})",
                affected_columns=[],
                affected_rows=len(df),
                suggestion="Collect more data or reduce complexity of the ML problem",
                auto_fixable=False
            ))
        
        return issues
    
    def _estimate_usable_rows(self, df: pd.DataFrame, issues: List[DataIssue]) -> int:
        """Estimate number of usable rows after cleaning"""
        usable_rows = len(df)
        
        # Subtract duplicates
        duplicate_issues = [i for i in issues if i.issue_type == DataIssueType.DUPLICATE_ROWS]
        if duplicate_issues:
            usable_rows -= duplicate_issues[0].affected_rows
        
        # Subtract rows with missing target (if any target-related issues)
        target_missing_issues = [
            i for i in issues 
            if i.issue_type == DataIssueType.MISSING_VALUES and len(i.affected_columns) == 1
        ]
        for issue in target_missing_issues:
            usable_rows -= issue.affected_rows
        
        return max(0, usable_rows)
    
    def _determine_quality_level(self, 
                                missing_pct: float, 
                                duplicate_pct: float, 
                                issue_count: int, 
                                usable_rows: int) -> DataQualityLevel:
        """Determine overall data quality level"""
        
        if usable_rows < 50:
            return DataQualityLevel.UNUSABLE
        
        if missing_pct > 50 or duplicate_pct > 50 or issue_count > 10:
            return DataQualityLevel.POOR
        
        if missing_pct > 20 or duplicate_pct > 20 or issue_count > 5:
            return DataQualityLevel.FAIR
        
        if missing_pct > 5 or duplicate_pct > 5 or issue_count > 2:
            return DataQualityLevel.GOOD
        
        return DataQualityLevel.EXCELLENT
    
    def _generate_recommendations(self, issues: List[DataIssue], df: pd.DataFrame) -> List[str]:
        """Generate actionable recommendations based on issues"""
        recommendations = []
        
        # Group issues by type for better recommendations
        issue_types = set(issue.issue_type for issue in issues)
        
        if DataIssueType.MISSING_VALUES in issue_types:
            recommendations.append("Address missing values through imputation or data collection")
        
        if DataIssueType.DUPLICATE_ROWS in issue_types:
            recommendations.append("Remove duplicate rows to improve data quality")
        
        if DataIssueType.CONSTANT_COLUMNS in issue_types:
            recommendations.append("Remove constant columns as they don't contribute to ML models")
        
        if DataIssueType.INSUFFICIENT_SAMPLES in issue_types:
            recommendations.append("Collect more data to meet minimum sample requirements for ML")
        
        if DataIssueType.IMBALANCED_TARGET in issue_types:
            recommendations.append("Consider class balancing techniques for better model performance")
        
        if DataIssueType.HIGH_CARDINALITY in issue_types:
            recommendations.append("Apply feature engineering to reduce high cardinality categorical features")
        
        # Add general recommendations
        if len(df.columns) > 50:
            recommendations.append("Consider feature selection to reduce dimensionality")
        
        if len(recommendations) == 0:
            recommendations.append("Data quality looks good - proceed with model training")
        
        return recommendations
    
    def _assess_ml_readiness(self, 
                           issues: List[DataIssue], 
                           usable_rows: int, 
                           target_column: Optional[str]) -> bool:
        """Assess if data is ready for ML training"""
        
        # Check for critical issues
        critical_issues = [i for i in issues if i.severity == 'critical']
        if critical_issues:
            return False
        
        # Check minimum sample size
        if usable_rows < self.min_samples_ml:
            return False
        
        # Check if target column exists (if specified)
        if target_column:
            target_issues = [
                i for i in issues 
                if i.issue_type == DataIssueType.INVALID_DATA_TYPES 
                and target_column in i.affected_columns
            ]
            if target_issues:
                return False
        
        # If we have high severity issues, might not be ready
        high_severity_issues = [i for i in issues if i.severity == 'high']
        if len(high_severity_issues) > 3:  # More than 3 high severity issues
            return False
        
        return True


def validate_dataframe_for_ml(df: pd.DataFrame, 
                             target_column: Optional[str] = None,
                             datetime_column: Optional[str] = None,
                             **validator_kwargs) -> DataValidationReport:
    """
    Convenience function to validate DataFrame for ML readiness
    
    Args:
        df: DataFrame to validate
        target_column: Target column name
        datetime_column: Datetime column name
        **validator_kwargs: Additional arguments for DataValidator
        
    Returns:
        DataValidationReport
    """
    validator = DataValidator(**validator_kwargs)
    return validator.validate_dataframe(df, target_column, datetime_column)


def clean_dataframe_for_ml(df: pd.DataFrame,
                          target_column: Optional[str] = None,
                          **cleaning_kwargs) -> Tuple[pd.DataFrame, List[str]]:
    """
    Convenience function to clean DataFrame for ML
    
    Args:
        df: DataFrame to clean
        target_column: Target column name
        **cleaning_kwargs: Additional arguments for cleaning
        
    Returns:
        Tuple of (cleaned_df, changes_made)
    """
    validator = DataValidator()
    return validator.clean_dataframe(df, target_column, **cleaning_kwargs)