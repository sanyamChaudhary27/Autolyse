"""Missing values analysis module"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List


class MissingValuesAnalyzer:
    """Analyze missing values in the dataset"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.all_cols = df.columns.tolist()
    
    def analyze(self) -> Dict[str, Any]:
        """
        Analyze missing values in the dataset.
        
        Returns:
            Dictionary containing:
            - missing_count: Count of missing values per column
            - missing_percentage: Percentage of missing values per column
            - total_missing: Total missing values in dataset
            - missing_rows: Rows with at least one missing value
            - completely_missing_cols: Columns that are entirely empty
        """
        analysis = {
            "missing_count": {},
            "missing_percentage": {},
            "total_missing": self.df.isna().sum().sum(),
            "missing_rows": len(self.df[self.df.isna().any(axis=1)]),
            "completely_missing_cols": [],
            "no_missing": False
        }
        
        for col in self.all_cols:
            missing = self.df[col].isna().sum()
            missing_pct = (missing / len(self.df)) * 100
            
            analysis["missing_count"][col] = missing
            analysis["missing_percentage"][col] = missing_pct
            
            if missing == len(self.df):
                analysis["completely_missing_cols"].append(col)
        
        # Check if there are no missing values
        if analysis["total_missing"] == 0:
            analysis["no_missing"] = True
        
        # Sort by missing percentage descending
        analysis["missing_count"] = dict(
            sorted(analysis["missing_count"].items(), 
                   key=lambda x: x[1], 
                   reverse=True)
        )
        analysis["missing_percentage"] = dict(
            sorted(analysis["missing_percentage"].items(), 
                   key=lambda x: x[1], 
                   reverse=True)
        )
        
        return analysis
    
    def get_missing_correlations(self) -> pd.DataFrame:
        """
        Calculate correlations between missing values in different columns.
        Useful for understanding if certain columns tend to be missing together.
        
        Returns:
            Correlation matrix of missing values
        """
        missing_df = self.df.isna().astype(int)
        return missing_df.corr()
