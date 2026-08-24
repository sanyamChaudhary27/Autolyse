"""Relationships analysis module"""

from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


#: Categorical pairs with more levels than this on either side are skipped:
#: their contingency tables are unreadable and chi-square approximations break.
MAX_CATEGORICAL_LEVELS = 30


class RelationshipsAnalyzer:
    """Analyze relationships between variables"""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = [
            c for c in df.columns
            if df[c].dtype.name in ("object", "str", "category")
        ]
    
    def analyze_categorical_numeric_relationships(self) -> Dict[str, Any]:
        """
        Analyze relationships between categorical and numeric variables.
        
        Returns:
            Dictionary containing statistics for each categorical variable's 
            impact on numeric variables (groupby statistics)
        """
        relationships = {}

        for cat_col in self.categorical_cols:
            if self.df[cat_col].nunique() > MAX_CATEGORICAL_LEVELS:
                continue  # group tables would be unreadably large

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
            Dictionary with contingency tables and Cramér's V for each pair
            of categorical variables. Pairs exceeding ``MAX_CATEGORICAL_LEVELS``
            on either side are reported with ``cramers_v: None``.
        """
        relationships = {}

        for i, col1 in enumerate(self.categorical_cols):
            for col2 in self.categorical_cols[i+1:]:
                levels_ok = (
                    self.df[col1].nunique() <= MAX_CATEGORICAL_LEVELS
                    and self.df[col2].nunique() <= MAX_CATEGORICAL_LEVELS
                )
                crosstab = pd.crosstab(self.df[col1], self.df[col2])
                cramers_v = self._cramers_v(self.df[col1], self.df[col2]) if levels_ok else None

                key = f"{col1}_vs_{col2}"
                relationships[key] = {
                    "crosstab": crosstab.to_dict(),
                    "cramers_v": cramers_v,
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

        Uses the standard definition V = sqrt(chi2 / (n * min(r-1, c-1)))
        with chi2 from a proper contingency-table chi-square test.

        Args:
            col1: First categorical column
            col2: Second categorical column

        Returns:
            Cramér's V statistic (0 = independence, 1 = perfect association)
        """
        confusion_matrix = pd.crosstab(col1, col2)
        if min(confusion_matrix.shape) < 2 or len(col1) == 0:
            return 0.0

        try:
            # correction=False keeps the textbook V definition; Yates' default
            # would bias 2x2 tables below 1.0 even under perfect association.
            chi2 = chi2_contingency(confusion_matrix, correction=False)[0]
        except ValueError:
            return 0.0

        min_dim = min(confusion_matrix.shape) - 1
        return float(np.sqrt(chi2 / (len(col1) * min_dim)))
    
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
