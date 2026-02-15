"""Distribution analysis module"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from scipy import stats


class DistributionAnalyzer:
    """Analyze distributions of numeric columns"""
    
    def __init__(self, df: pd.DataFrame, random_state: int = 42):
        self.df = df
        self.random_state = random_state
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    def analyze_numeric_distributions(self) -> Dict[str, Any]:
        """
        Analyze distributions of numeric columns.
        
        Returns:
            Dictionary containing distribution characteristics:
            - distribution_type: Normal, Uniform, Bimodal (estimated)
            - normality_test: Shapiro-Wilk test p-value
            - is_normal: Boolean (p-value > 0.05)
            - unique_values: Number of unique values
            - value_counts: Frequency of most common value
        """
        distributions = {}
        
        for col in self.numeric_cols:
            col_data = self.df[col].dropna()
            
            if len(col_data) < 3:
                continue
            
            # Shapiro-Wilk test for normality (works well for up to 5000 samples)
            if len(col_data) <= 5000:
                _, p_value = stats.shapiro(col_data)
            else:
                # For larger samples, use Kolmogorov-Smirnov test
                _, p_value = stats.kstest(col_data, 'norm', 
                                         args=(col_data.mean(), col_data.std()))
            
            # Estimate distribution type based on skewness and kurtosis
            skewness = col_data.skew()
            kurt = col_data.kurtosis()
            
            if abs(skewness) < 0.5 and abs(kurt) < 1:
                dist_type = "Approximately Normal"
            elif skewness > 1:
                dist_type = "Right-skewed"
            elif skewness < -1:
                dist_type = "Left-skewed"
            else:
                dist_type = "Moderately Skewed"
            
            distributions[col] = {
                "distribution_type": dist_type,
                "normality_pvalue": p_value,
                "is_normal": p_value > 0.05,
                "skewness": skewness,
                "kurtosis": kurt,
                "unique_values": col_data.nunique(),
                "mode": col_data.mode()[0] if len(col_data.mode()) > 0 else None,
            }
        
        return distributions
    
    def analyze_categorical_distributions(self) -> Dict[str, Any]:
        """
        Analyze distributions of categorical columns.
        
        Returns:
            Dictionary containing:
            - unique_values: Count of unique categories
            - top_categories: Top 5 most frequent categories
            - diversity: Measure of category diversity (0-1, higher = more diverse)
        """
        distributions = {}
        
        for col in self.categorical_cols:
            col_data = self.df[col].dropna()
            
            value_counts = col_data.value_counts()
            n_unique = len(value_counts)
            
            # Calculate Simpson's Diversity Index (0 = no diversity, 1 = perfect diversity)
            if len(col_data) > 0:
                frequencies = value_counts.values / len(col_data)
                diversity = 1 - np.sum(frequencies ** 2)
            else:
                diversity = 0
            
            distributions[col] = {
                "unique_values": n_unique,
                "top_categories": value_counts.head(5).to_dict(),
                "diversity_index": diversity,
                "missing_count": self.df[col].isna().sum(),
                "total_samples": len(col_data),
            }
        
        return distributions
