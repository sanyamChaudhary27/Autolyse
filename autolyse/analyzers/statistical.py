# autolyse/analyzers/statistical.py

import pandas as pd
import numpy as np
from typing import Dict, Any

class StatisticalAnalyzer:
    """Compute basic statistical metrics for numerical columns"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    def analyze(self) -> Dict[str, Any]:
        """
        Compute comprehensive statistics for all numeric columns.
        
        Returns:
            Dictionary with stats for each numeric column including:
            - mean, median, std, variance
            - min, max, range
            - quartiles (25%, 50%, 75%)
            - skewness, kurtosis
            - count, null_count
        """
        stats = {}
        for col in self.numeric_cols:
            col_data = self.df[col].dropna()
            stats[col] = {
                "count": len(col_data),
                "null_count": self.df[col].isna().sum(),
                "null_percentage": (self.df[col].isna().sum() / len(self.df)) * 100,
                "mean": col_data.mean(),
                "median": col_data.median(),
                "std": col_data.std(),
                "variance": col_data.var(),
                "min": col_data.min(),
                "max": col_data.max(),
                "range": col_data.max() - col_data.min(),
                "q25": col_data.quantile(0.25),
                "q50": col_data.quantile(0.50),
                "q75": col_data.quantile(0.75),
                "iqr": col_data.quantile(0.75) - col_data.quantile(0.25),
                "skewness": col_data.skew(),
                "kurtosis": col_data.kurtosis(),
            }
        return stats