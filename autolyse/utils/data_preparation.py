"""Data preparation and type detection utilities"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import re


class DataPreparation:
    """Utilities for data preparation and type detection"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize data preparation.
        
        Args:
            df: Input dataframe
        """
        self.df = df
        self.column_types = {}
        self._detect_column_types()
    
    def _detect_column_types(self) -> None:
        """
        Detect and classify column types in the dataframe.
        
        Classifies columns as:
        - numeric: int, float, uint
        - categorical: object, category with few unique values
        - datetime: datetime, timedelta
        - text: object/string with many unique values or text content
        - boolean: bool
        """
        for col in self.df.columns:
            col_type = self.df[col].dtype
            
            if pd.api.types.is_numeric_dtype(col_type):
                self.column_types[col] = 'numeric'
            elif pd.api.types.is_bool_dtype(col_type):
                self.column_types[col] = 'boolean'
            elif pd.api.types.is_datetime64_any_dtype(col_type):
                self.column_types[col] = 'datetime'
            elif col_type == 'object':
                self.column_types[col] = self._infer_object_type(col)
            elif col_type.name == 'category':
                self.column_types[col] = 'categorical'
            else:
                self.column_types[col] = 'other'
    
    def _infer_object_type(self, col: str) -> str:
        """
        Infer the type of an object column (categorical, text, or datetime).
        
        Args:
            col: Column name
        
        Returns:
            Type classification string
        """
        col_data = self.df[col].dropna()
        
        if len(col_data) == 0:
            return 'text'
        
        # Check for datetime
        try:
            pd.to_datetime(col_data.head(10))
            return 'datetime'
        except:
            pass
        
        # Check cardinality: if unique values < 10% of total, likely categorical
        unique_ratio = col_data.nunique() / len(col_data)
        
        if unique_ratio < 0.05:
            return 'categorical'
        elif unique_ratio < 0.5:
            return 'categorical'  # Up to 50% unique allowed
        else:
            return 'text'
    
    def get_column_types(self) -> Dict[str, str]:
        """
        Get dictionary of column names to their detected types.
        
        Returns:
            Dictionary: {column_name: type}
        """
        return self.column_types.copy()
    
    def get_numeric_columns(self) -> List[str]:
        """Get list of numeric column names"""
        return [col for col, dtype in self.column_types.items() if dtype == 'numeric']
    
    def get_categorical_columns(self) -> List[str]:
        """Get list of categorical column names"""
        return [col for col, dtype in self.column_types.items() if dtype == 'categorical']
    
    def get_datetime_columns(self) -> List[str]:
        """Get list of datetime column names"""
        return [col for col, dtype in self.column_types.items() if dtype == 'datetime']
    
    def get_text_columns(self) -> List[str]:
        """Get list of text column names"""
        return [col for col, dtype in self.column_types.items() if dtype == 'text']
    
    def get_boolean_columns(self) -> List[str]:
        """Get list of boolean column names"""
        return [col for col, dtype in self.column_types.items() if dtype == 'boolean']
    
    def get_columns_by_type(self, col_type: str) -> List[str]:
        """
        Get columns of a specific type.
        
        Args:
            col_type: Type classification ('numeric', 'categorical', 'datetime', 'text', 'boolean')
        
        Returns:
            List of column names matching the type
        """
        return [col for col, dtype in self.column_types.items() if dtype == col_type]
    
    def get_type_summary(self) -> Dict[str, int]:
        """
        Get summary of column type counts.
        
        Returns:
            Dictionary: {type: count}
        """
        summary = {}
        for col_type in self.column_types.values():
            summary[col_type] = summary.get(col_type, 0) + 1
        return summary
    
    def get_column_info(self) -> pd.DataFrame:
        """
        Get detailed information about each column.
        
        Returns:
            DataFrame with columns: Name, Type, Non-Null, Null %, Unique, Unique %
        """
        info_list = []
        
        for col in self.df.columns:
            non_null = self.df[col].notna().sum()
            null_count = self.df[col].isna().sum()
            null_pct = (null_count / len(self.df)) * 100
            unique = self.df[col].nunique()
            unique_pct = (unique / len(self.df)) * 100 if len(self.df) > 0 else 0
            
            info_list.append({
                'Column': col,
                'Type': self.column_types[col],
                'Non-Null': non_null,
                'Null %': round(null_pct, 2),
                'Unique': unique,
                'Unique %': round(unique_pct, 2),
            })
        
        return pd.DataFrame(info_list)
    
    def handle_missing_values(self, strategy: str = 'report') -> Dict[str, Any]:
        """
        Handle missing values in the dataframe.
        
        Args:
            strategy: 'report' (default, just report), 'drop', 'mean', 'median', 'forward_fill', 'backward_fill'
        
        Returns:
            Dictionary with action taken and summary
        """
        missing_summary = {
            'total_missing': self.df.isna().sum().sum(),
            'rows_with_missing': len(self.df[self.df.isna().any(axis=1)]),
            'strategy': strategy,
            'action': None
        }
        
        if strategy == 'report':
            missing_summary['action'] = 'No action - reporting only'
        elif strategy == 'drop':
            self.df = self.df.dropna()
            missing_summary['action'] = f"Dropped rows with missing values. New shape: {self.df.shape}"
        elif strategy == 'mean':
            numeric_cols = self.get_numeric_columns()
            for col in numeric_cols:
                self.df[col].fillna(self.df[col].mean(), inplace=True)
            missing_summary['action'] = f"Filled numeric columns with mean"
        elif strategy == 'median':
            numeric_cols = self.get_numeric_columns()
            for col in numeric_cols:
                self.df[col].fillna(self.df[col].median(), inplace=True)
            missing_summary['action'] = f"Filled numeric columns with median"
        elif strategy == 'forward_fill':
            self.df = self.df.fillna(method='ffill')
            missing_summary['action'] = f"Forward filled missing values"
        elif strategy == 'backward_fill':
            self.df = self.df.fillna(method='bfill')
            missing_summary['action'] = f"Backward filled missing values"
        
        return missing_summary
    
    def remove_duplicates(self, subset: List[str] = None, keep: str = 'first') -> Dict[str, Any]:
        """
        Remove duplicate rows from the dataframe.
        
        Args:
            subset: Column names to consider for identifying duplicates. Default: all columns
            keep: Which duplicates to keep ('first', 'last', or False to remove all)
        
        Returns:
            Dictionary with action summary
        """
        duplicates_before = len(self.df[self.df.duplicated(subset=subset, keep=False)])
        
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        
        duplicates_after = len(self.df[self.df.duplicated(subset=subset, keep=False)])
        
        return {
            'duplicates_found': duplicates_before // 2 if duplicates_before > 0 else 0,
            'duplicates_removed': duplicates_before - duplicates_after,
            'new_shape': self.df.shape,
            'action': f'Removed duplicates. New shape: {self.df.shape}'
        }
    
    def remove_low_variance_columns(self, threshold: float = 0.01) -> Dict[str, Any]:
        """
        Remove columns with very low variance (mostly constant values).
        
        Args:
            threshold: Variance threshold ratio (0-1). Default 0.01 = 1% variance
        
        Returns:
            Dictionary with action summary
        """
        cols_removed = []
        
        for col in self.get_numeric_columns():
            col_var = self.df[col].var()
            col_range = self.df[col].max() - self.df[col].min()
            
            if col_range > 0:
                variance_ratio = col_var / (col_range ** 2)
                if variance_ratio < threshold:
                    cols_removed.append(col)
        
        self.df = self.df.drop(columns=cols_removed)
        
        return {
            'columns_removed': cols_removed,
            'new_shape': self.df.shape,
            'action': f'Removed {len(cols_removed)} low-variance columns. New shape: {self.df.shape}'
        }
    
    def normalize_numeric(self, method: str = 'minmax') -> Dict[str, Any]:
        """
        Normalize numeric columns.
        
        Args:
            method: 'minmax' (0-1 scale), 'zscore' (standardize), 'log' (log transformation)
        
        Returns:
            Dictionary with action summary
        """
        numeric_cols = self.get_numeric_columns()
        
        if method == 'minmax':
            for col in numeric_cols:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
            action = "Applied Min-Max normalization (0-1 scale)"
        
        elif method == 'zscore':
            for col in numeric_cols:
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val > 0:
                    self.df[col] = (self.df[col] - mean_val) / std_val
            action = "Applied Z-score normalization (standardization)"
        
        elif method == 'log':
            for col in numeric_cols:
                if (self.df[col] > 0).all():  # Only for positive values
                    self.df[col] = np.log1p(self.df[col])
            action = "Applied log transformation to positive columns"
        
        return {
            'method': method,
            'columns_normalized': numeric_cols,
            'action': action
        }
    
    def encode_categorical(self, method: str = 'label') -> Dict[str, Any]:
        """
        Encode categorical columns.
        
        Args:
            method: 'label' (0, 1, 2...), 'onehot' (one-hot encoding)
        
        Returns:
            Dictionary with action summary and mapping (for label encoding)
        """
        categorical_cols = self.get_categorical_columns()
        mappings = {}
        
        if method == 'label':
            for col in categorical_cols:
                unique_vals = self.df[col].unique()
                mapping = {val: idx for idx, val in enumerate(unique_vals)}
                self.df[col] = self.df[col].map(mapping)
                mappings[col] = mapping
            action = f"Applied label encoding to {len(categorical_cols)} columns"
        
        elif method == 'onehot':
            self.df = pd.get_dummies(self.df, columns=categorical_cols, drop_first=True)
            action = f"Applied one-hot encoding to {len(categorical_cols)} columns"
        
        return {
            'method': method,
            'columns_encoded': categorical_cols,
            'mappings': mappings if method == 'label' else None,
            'action': action,
            'new_shape': self.df.shape
        }
    
    def get_dataframe(self) -> pd.DataFrame:
        """
        Get the current dataframe (may have been modified by preparation methods).
        
        Returns:
            Modified dataframe
        """
        return self.df.copy()
    
    def validate_data(self) -> Dict[str, Any]:
        """
        Validate data quality and return a comprehensive report.
        
        Returns:
            Dictionary with validation checks and results
        """
        validation = {
            'shape': self.df.shape,
            'total_cells': self.df.shape[0] * self.df.shape[1],
            'missing_cells': self.df.isna().sum().sum(),
            'missing_pct': round((self.df.isna().sum().sum() / (self.df.shape[0] * self.df.shape[1])) * 100, 2),
            'duplicate_rows': len(self.df[self.df.duplicated()]),
            'duplicate_pct': round((len(self.df[self.df.duplicated()]) / len(self.df)) * 100, 2),
            'column_types': self.get_type_summary(),
            'numeric_columns': len(self.get_numeric_columns()),
            'categorical_columns': len(self.get_categorical_columns()),
            'datetime_columns': len(self.get_datetime_columns()),
            'text_columns': len(self.get_text_columns()),
            'boolean_columns': len(self.get_boolean_columns()),
        }
        
        # Add data quality score (0-100)
        quality_score = 100
        quality_score -= validation['missing_pct']  # Reduce for missing values
        quality_score -= min(validation['duplicate_pct'], 10)  # Reduce for duplicates (max -10)
        validation['data_quality_score'] = max(0, quality_score)
        
        return validation


class DataInspector:
    """Inspect and explore data characteristics"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize inspector.
        
        Args:
            df: Input dataframe
        """
        self.df = df
    
    def summary_statistics(self) -> pd.DataFrame:
        """Get summary statistics for numeric columns"""
        return self.df.describe().T
    
    def value_counts_summary(self, col: str, top_n: int = 10) -> pd.Series:
        """
        Get value counts for a column.
        
        Args:
            col: Column name
            top_n: Number of top values to return
        
        Returns:
            Series with value counts
        """
        return self.df[col].value_counts().head(top_n)
    
    def correlation_matrix(self, numeric_only: bool = True) -> pd.DataFrame:
        """
        Get correlation matrix.
        
        Args:
            numeric_only: Include only numeric columns
        
        Returns:
            Correlation matrix
        """
        if numeric_only:
            return self.df.select_dtypes(include=[np.number]).corr()
        return self.df.corr(numeric_only=False)
    
    def check_data_types(self) -> pd.DataFrame:
        """Check and display data types"""
        return pd.DataFrame({
            'Column': self.df.columns,
            'Type': self.df.dtypes,
            'Non-Null Count': self.df.notna().sum().values,
            'Null Count': self.df.isna().sum().values,
            'Unique': [self.df[col].nunique() for col in self.df.columns]
        })
