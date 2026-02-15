"""Relationships analysis module"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List


class RelationshipsAnalyzer:
    """Analyze relationships between variables"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    def analyze_categorical_numeric_relationships(self) -> Dict[str, Any]:
        """
        Analyze relationships between categorical and numeric variables.
        
        Returns:
            Dictionary containing statistics for each categorical variable's 
            impact on numeric variables (groupby statistics)
        """
        relationships = {}
        
        for cat_col in self.categorical_cols:
            relationships[cat_col] = {}
            
            # For each numeric column, get grouped statistics
            for num_col in self.numeric_cols:
                grouped = self.df.groupby(cat_col)[num_col].agg([
                    'count', 'mean', 'median', 'std', 'min', 'max'
                ]).to_dict('index')
                
                relationships[cat_col][num_col] = grouped
        
        return relationships
    
    def analyze_categorical_relationships(self) -> Dict[str, Any]:
        """
        Analyze relationships between categorical variables using cross-tabulation.
        
        Returns:
            Dictionary with contingency tables for pairs of categorical variables
        """
        relationships = {}
        
        for i, col1 in enumerate(self.categorical_cols):
            for col2 in self.categorical_cols[i+1:]:
                # Create cross-tabulation
                crosstab = pd.crosstab(self.df[col1], self.df[col2])
                
                # Calculate Cramér's V statistic (measure of association)
                chi2 = self._cramers_v(self.df[col1], self.df[col2])
                
                key = f"{col1}_vs_{col2}"
                relationships[key] = {
                    "crosstab": crosstab.to_dict(),
                    "cramers_v": chi2,
                    "shape": crosstab.shape,
                }
        
        return relationships
    
    def analyze_numeric_numeric_relationships(self) -> Dict[str, Any]:
        """
        Analyze pairwise relationships between numeric variables.
        Useful for identifying important feature pairs for visualization.
        
        Returns:
            Dictionary with scatter plot candidates (pairs of numeric columns)
        """
        relationships = []
        
        for i, col1 in enumerate(self.numeric_cols):
            for col2 in self.numeric_cols[i+1:]:
                # Calculate correlation to assess relationship strength
                corr = self.df[col1].corr(self.df[col2])
                
                # Calculate coefficient of determination (R²)
                r_squared = corr ** 2
                
                relationships.append({
                    "col1": col1,
                    "col2": col2,
                    "correlation": corr,
                    "r_squared": r_squared,
                    "relationship_strength": self._get_strength_label(abs(corr))
                })
        
        # Sort by correlation strength (absolute value)
        relationships = sorted(relationships, 
                             key=lambda x: abs(x['correlation']), 
                             reverse=True)
        
        return {"numeric_pairs": relationships}
    
    def get_relationship_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of all relationships in the dataset.
        
        Returns:
            Combined analysis of categorical-numeric, numeric-numeric relationships
        """
        summary = {
            "categorical_numeric": self.analyze_categorical_numeric_relationships(),
            "categorical_categorical": self.analyze_categorical_relationships(),
            "numeric_numeric": self.analyze_numeric_numeric_relationships(),
        }
        
        return summary
    
    @staticmethod
    def _cramers_v(col1: pd.Series, col2: pd.Series) -> float:
        """
        Calculate Cramér's V statistic for categorical association.
        
        Args:
            col1: First categorical column
            col2: Second categorical column
        
        Returns:
            Cramér's V statistic (0 to 1)
        """
        confusion_matrix = pd.crosstab(col1, col2)
        chi2 = ((confusion_matrix.values ** 2) / confusion_matrix.sum().sum()).sum() - 1
        min_dim = min(confusion_matrix.shape) - 1
        
        if min_dim == 0:
            return 0
        
        return np.sqrt(chi2 / (len(col1) * min_dim))
    
    @staticmethod
    def _get_strength_label(correlation: float) -> str:
        """
        Get human-readable label for correlation strength.
        
        Args:
            correlation: Absolute correlation value (0 to 1)
        
        Returns:
            Strength label (Very Weak, Weak, Moderate, Strong, Very Strong)
        """
        if correlation < 0.2:
            return "Very Weak"
        elif correlation < 0.4:
            return "Weak"
        elif correlation < 0.6:
            return "Moderate"
        elif correlation < 0.8:
            return "Strong"
        else:
            return "Very Strong"
