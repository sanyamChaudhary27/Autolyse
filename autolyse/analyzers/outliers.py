"""Outlier detection module"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from sklearn.ensemble import IsolationForest


class OutlierAnalyzer:
    """Detect and analyze outliers in numeric columns"""
    
    def __init__(self, df: pd.DataFrame, contamination: float = 0.1, random_state: int = 42):
        """
        Initialize outlier analyzer.
        
        Args:
            df: Input dataframe
            contamination: Expected proportion of outliers (default 0.1 = 10%)
            random_state: Random seed for reproducibility
        """
        self.df = df
        self.random_state = random_state
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.contamination = contamination
    
    def detect_iqr_outliers(self) -> Dict[str, Any]:
        """
        Detect outliers using Interquartile Range (IQR) method.
        
        Returns:
            Dictionary with outlier info for each numeric column:
            - outlier_count: Number of outliers detected
            - outlier_percentage: Percentage of outliers
            - lower_bound: Lower fence (Q1 - 1.5*IQR)
            - upper_bound: Upper fence (Q3 + 1.5*IQR)
            - outlier_values: List of outlier values
        """
        outliers = {}
        
        for col in self.numeric_cols:
            col_data = self.df[col].dropna()
            
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
            outlier_values = col_data[outlier_mask].values
            
            outliers[col] = {
                "method": "IQR",
                "outlier_count": len(outlier_values),
                "outlier_percentage": (len(outlier_values) / len(col_data)) * 100 if len(col_data) > 0 else 0,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "outlier_values": sorted(outlier_values.tolist()),
            }
        
        return outliers
    
    def detect_isolation_forest_outliers(self) -> Dict[str, Any]:
        """
        Detect outliers using Isolation Forest algorithm.
        More robust for multivariate outlier detection.
        
        Returns:
            Dictionary with outlier info and anomaly scores for each numeric column
        """
        if len(self.numeric_cols) == 0:
            return {}
        
        # Prepare data for Isolation Forest
        numeric_data = self.df[self.numeric_cols].dropna()
        
        if len(numeric_data) < 10:
            return {}  # Not enough data for meaningful detection
        
        # Fit Isolation Forest
        iso_forest = IsolationForest(contamination=self.contamination, random_state=self.random_state)
        outlier_predictions = iso_forest.fit_predict(numeric_data)
        anomaly_scores = iso_forest.score_samples(numeric_data)
        
        outliers = {
            "method": "Isolation Forest",
            "n_outliers": (outlier_predictions == -1).sum(),
            "outlier_percentage": ((outlier_predictions == -1).sum() / len(numeric_data)) * 100,
            "anomaly_scores": anomaly_scores.tolist(),
            "outlier_indices": np.where(outlier_predictions == -1)[0].tolist(),
            "anomaly_threshold": np.percentile(anomaly_scores, self.contamination * 100)
        }
        
        return outliers
    
    def get_outlier_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of outliers using both methods.
        
        Returns:
            Combined outlier analysis from both IQR and Isolation Forest methods
        """
        summary = {
            "iqr_method": self.detect_iqr_outliers(),
            "isolation_forest": self.detect_isolation_forest_outliers(),
        }
        
        return summary
