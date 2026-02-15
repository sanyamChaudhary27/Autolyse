"""Correlation analysis module"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from scipy.stats import spearmanr


class CorrelationAnalyzer:
    """Analyze correlations between variables"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    def analyze_pearson_correlation(self) -> Dict[str, Any]:
        """
        Compute Pearson correlation matrix and identify strong correlations.
        
        Returns:
            Dictionary containing:
            - correlation_matrix: Full correlation matrix
            - strong_correlations: Pairs of columns with |correlation| > 0.7
            - moderate_correlations: Pairs of columns with 0.5 < |correlation| <= 0.7
        """
        if len(self.numeric_cols) < 2:
            return {
                "correlation_matrix": None,
                "strong_correlations": [],
                "moderate_correlations": [],
            }
        
        # Calculate Pearson correlation
        corr_matrix = self.df[self.numeric_cols].corr(method='pearson')
        
        # Find strong and moderate correlations (excluding diagonal)
        strong_corr = []
        moderate_corr = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                col_i = corr_matrix.columns[i]
                col_j = corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                
                if abs(corr_value) > 0.7:
                    strong_corr.append({
                        "col1": col_i,
                        "col2": col_j,
                        "correlation": corr_value
                    })
                elif abs(corr_value) > 0.5:
                    moderate_corr.append({
                        "col1": col_i,
                        "col2": col_j,
                        "correlation": corr_value
                    })
        
        return {
            "correlation_matrix": corr_matrix,
            "strong_correlations": sorted(strong_corr, 
                                         key=lambda x: abs(x['correlation']), 
                                         reverse=True),
            "moderate_correlations": sorted(moderate_corr, 
                                           key=lambda x: abs(x['correlation']), 
                                           reverse=True),
        }
    
    def analyze_spearman_correlation(self) -> Dict[str, Any]:
        """
        Compute Spearman (rank) correlation matrix.
        Useful for non-linear relationships and ordinal data.
        
        Returns:
            Dictionary containing:
            - correlation_matrix: Full Spearman correlation matrix
            - strong_correlations: Pairs with |correlation| > 0.7
        """
        if len(self.numeric_cols) < 2:
            return {
                "correlation_matrix": None,
                "strong_correlations": [],
            }
        
        # Calculate Spearman correlation
        spearman_matrix = self.df[self.numeric_cols].corr(method='spearman')
        
        # Find strong correlations
        strong_corr = []
        for i in range(len(spearman_matrix.columns)):
            for j in range(i + 1, len(spearman_matrix.columns)):
                col_i = spearman_matrix.columns[i]
                col_j = spearman_matrix.columns[j]
                corr_value = spearman_matrix.iloc[i, j]
                
                if abs(corr_value) > 0.7:
                    strong_corr.append({
                        "col1": col_i,
                        "col2": col_j,
                        "correlation": corr_value
                    })
        
        return {
            "correlation_matrix": spearman_matrix,
            "strong_correlations": sorted(strong_corr, 
                                         key=lambda x: abs(x['correlation']), 
                                         reverse=True),
        }
    
    def get_correlation_summary(self) -> Dict[str, Any]:
        """
        Get combined visualization-ready correlation summary.
        
        Returns:
            Dictionary with both Pearson and Spearman correlations
        """
        summary = {
            "pearson": self.analyze_pearson_correlation(),
            "spearman": self.analyze_spearman_correlation(),
        }
        
        return summary
