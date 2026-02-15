"""Automated feature engineering module"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import warnings

warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Automated feature engineering with optimization.
    
    Creates intelligent feature combinations:
    - Polynomial features (degree 2-3)
    - Interaction terms
    - Ratio features
    - Logarithmic transformations
    - Domain-specific features
    """
    
    def __init__(self, df: pd.DataFrame, random_state: int = 42):
        """
        Initialize feature engineer.
        
        Args:
            df: Input dataframe
            random_state: Random seed for reproducibility
        """
        self.df = df.copy()
        self.random_state = random_state
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.engineered_features = {}
    
    def engineer_features(self, 
                         polynomial_degree: int = 2,
                         include_interactions: bool = True,
                         include_ratios: bool = True,
                         include_logs: bool = True,
                         max_features: int = 20) -> pd.DataFrame:
        """
        Automatically engineer new features from existing ones.
        
        Args:
            polynomial_degree: Degree of polynomial features (2-3 recommended)
            include_interactions: Whether to create interaction terms
            include_ratios: Whether to create ratio features
            include_logs: Whether to apply log transformations
            max_features: Maximum new features to create (for memory efficiency)
        
        Returns:
            DataFrame with original + engineered features
        """
        result_df = self.df.copy()
        feature_count = 0
        
        if len(self.numeric_cols) < 2:
            return result_df
        
        # 1. Polynomial features (limited to top 3 features by variance)
        if polynomial_degree >= 2:
            feature_count += self._add_polynomial_features(
                result_df, polynomial_degree, min(3, len(self.numeric_cols))
            )
        
        # 2. Interaction features (limited)
        if include_interactions and feature_count < max_features:
            feature_count += self._add_interaction_features(
                result_df, max(max_features - feature_count)
            )
        
        # 3. Ratio features (limited)
        if include_ratios and feature_count < max_features:
            feature_count += self._add_ratio_features(
                result_df, max(max_features - feature_count)
            )
        
        # 4. Log transformations (for positive skewed data)
        if include_logs and feature_count < max_features:
            feature_count += self._add_log_features(
                result_df, max(max_features - feature_count)
            )
        
        # 5. Domain-aware features
        if feature_count < max_features:
            feature_count += self._add_domain_features(result_df)
        
        return result_df
    
    def _add_polynomial_features(self, df: pd.DataFrame, degree: int, n_features: int) -> int:
        """Add polynomial features for top N numeric columns"""
        # Select top features by variance
        variances = {col: df[col].var() for col in self.numeric_cols}
        top_cols = sorted(variances.items(), key=lambda x: x[1], reverse=True)[:n_features]
        top_cols = [col for col, _ in top_cols]
        
        if len(top_cols) < 2:
            return 0
        
        try:
            poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
            poly_features = poly.fit_transform(df[top_cols])
            
            # Get feature names
            feature_names = poly.get_feature_names_out(top_cols)
            
            new_cols = 0
            for i, fname in enumerate(feature_names):
                if fname not in df.columns:
                    # Only add polynomial/interaction terms (skip original features)
                    if '^' in fname or ' ' in fname:  # Polynomial or interaction
                        df[fname] = poly_features[:, i]
                        self.engineered_features[fname] = 'polynomial'
                        new_cols += 1
            
            return new_cols
        except:
            return 0
    
    def _add_interaction_features(self, df: pd.DataFrame, max_new: int) -> int:
        """Add carefully selected interaction features"""
        new_cols = 0
        
        # Only interact top correlated features
        corr_matrix = df[self.numeric_cols].corr().fillna(0)
        
        interaction_pairs = []
        for i, col1 in enumerate(self.numeric_cols):
            for col2 in self.numeric_cols[i+1:]:
                corr = abs(corr_matrix.loc[col1, col2])
                if 0.3 < corr < 0.9:  # Moderate correlation good for interactions
                    interaction_pairs.append((col1, col2, corr))
        
        # Sort by correlation strength and take top pairs
        interaction_pairs = sorted(interaction_pairs, key=lambda x: abs(x[2]), reverse=True)[:max_new]
        
        for col1, col2, _ in interaction_pairs:
            fname = f"{col1}_x_{col2}"
            if fname not in df.columns:
                # Normalize before interaction
                v1 = (df[col1] - df[col1].mean()) / (df[col1].std() + 1e-8)
                v2 = (df[col2] - df[col2].mean()) / (df[col2].std() + 1e-8)
                df[fname] = v1 * v2
                self.engineered_features[fname] = 'interaction'
                new_cols += 1
        
        return new_cols
    
    def _add_ratio_features(self, df: pd.DataFrame, max_new: int) -> int:
        """Add ratio/division features for interpretability"""
        new_cols = 0
        
        # Create ratios between pairs (avoid division by zero)
        numeric_data = df[self.numeric_cols]
        
        for col1 in self.numeric_cols[:5]:  # Limit for efficiency
            for col2 in self.numeric_cols[:5]:
                if col1 != col2:
                    # Avoid division by zero
                    if (numeric_data[col2] != 0).sum() / len(numeric_data) > 0.95:  # 95%+ non-zero
                        fname = f"{col1}_div_{col2}"
                        if fname not in df.columns and new_cols < max_new:
                            df[fname] = numeric_data[col1] / (numeric_data[col2] + 1e-8)
                            self.engineered_features[fname] = 'ratio'
                            new_cols += 1
        
        return new_cols
    
    def _add_log_features(self, df: pd.DataFrame, max_new: int) -> int:
        """Add log-transformed features for skewed distributions"""
        new_cols = 0
        
        for col in self.numeric_cols:
            # Only log-transform positive, right-skewed data
            if (df[col] > 0).sum() / len(df) > 0.95 and df[col].skew() > 1:
                fname = f"log_{col}"
                if fname not in df.columns and new_cols < max_new:
                    df[fname] = np.log1p(df[col])
                    self.engineered_features[fname] = 'log_transform'
                    new_cols += 1
        
        return new_cols
    
    def _add_domain_features(self, df: pd.DataFrame) -> int:
        """Add domain-aware features (statistical aggregates)"""
        new_cols = 0
        
        if len(self.numeric_cols) >= 2:
            # Mean of top features
            top_3 = sorted(
                [(col, df[col].var()) for col in self.numeric_cols],
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            if len(top_3) > 1:
                top_cols = [col for col, _ in top_3]
                df['feature_mean'] = df[top_cols].mean(axis=1)
                self.engineered_features['feature_mean'] = 'aggregate'
                new_cols += 1
                
                df['feature_std'] = df[top_cols].std(axis=1)
                self.engineered_features['feature_std'] = 'aggregate'
                new_cols += 1
                
                df['feature_max'] = df[top_cols].max(axis=1)
                self.engineered_features['feature_max'] = 'aggregate'
                new_cols += 1
        
        return new_cols
    
    def get_engineered_features_summary(self) -> Dict[str, any]:
        """Get summary of engineered features"""
        by_type = {}
        for fname, ftype in self.engineered_features.items():
            if ftype not in by_type:
                by_type[ftype] = []
            by_type[ftype].append(fname)
        
        return {
            'total_engineered': len(self.engineered_features),
            'by_type': by_type,
            'all_features': list(self.engineered_features.keys())
        }
    
    def select_best_features(self, target_col: Optional[str] = None, 
                            n_features: int = 10) -> List[str]:
        """
        Select best engineered features by correlation/variance.
        
        Args:
            target_col: Optional target column for supervised selection
            n_features: Number of features to select
        
        Returns:
            List of best feature names
        """
        engineered_only = [col for col in self.df.columns 
                          if col in self.engineered_features]
        
        if not engineered_only:
            return []
        
        scores = {}
        
        if target_col and target_col in self.df.columns:
            # Correlation with target
            for col in engineered_only:
                corr = abs(self.df[col].corr(self.df[target_col]))
                scores[col] = corr
        else:
            # Variance for unsupervised
            for col in engineered_only:
                var = self.df[col].var()
                scores[col] = var
        
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [col for col, _ in ranked[:n_features]]
