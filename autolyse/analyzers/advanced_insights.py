"""Advanced insights analyzer for multivariate and complex relationship analysis"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from itertools import combinations
from scipy.stats import kruskal, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')


class AdvancedInsightsAnalyzer:
    """
    Advanced analyzer for discovering deep patterns and complex relationships.
    
    Analyzes:
    - Multivariate relationships (3+ features)
    - Feature interactions and synergies
    - Hidden patterns and clusters
    - Anomaly patterns
    - Feature importance rankings
    - Deep data meaning and interpretability
    """
    
    def __init__(self, df: pd.DataFrame, random_state: int = 42):
        """
        Initialize advanced analyzer.
        
        Args:
            df: Input dataframe
            random_state: Random seed for reproducibility
        """
        self.df = df
        self.random_state = random_state
        np.random.seed(random_state)
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    def analyze_all(self) -> Dict[str, Any]:
        """Run all advanced analyses and return comprehensive results"""
        return {
            'feature_interactions': self.find_feature_interactions(),
            'feature_clusters': self.detect_feature_clusters(),
            'categorical_influence': self.analyze_categorical_influence(),
            'anomaly_patterns': self.detect_anomaly_patterns(),
            'feature_importance': self.rank_feature_importance(),
            'temporal_patterns': self.detect_temporal_patterns(),
            'multivariate_insights': self.detect_multivariate_patterns()
        }
    
    def find_feature_interactions(self, max_features: int = 5, 
                                 interaction_threshold: float = 0.2) -> Dict[str, Any]:
        """
        Identify important feature interaction effects.
        
        Detects when combination of features has stronger effect than individual features.
        Optimized O(k^3 log k) where k = max_features
        
        Args:
            max_features: Maximum numeric features to consider (for performance)
            interaction_threshold: Minimum interaction strength to report
        
        Returns:
            Dictionary with interaction effects and rankings
        """
        if len(self.numeric_cols) < 3:
            return {
                'interactions_found': [],
                'interpretation': 'Not enough numeric features for interaction analysis'
            }
        
        cols_to_use = self.numeric_cols[:max_features]
        interactions = []
        
        # Standardize for fair comparison
        scaler = StandardScaler()
        numeric_data = self.df[cols_to_use].fillna(self.df[cols_to_use].mean())
        scaled_data = scaler.fit_transform(numeric_data)
        
        # Check all 3-feature combinations (exhaustive but optimized)
        for col_combo in combinations(range(len(cols_to_use)), 3):
            idx1, idx2, idx3 = col_combo
            
            # Create interaction term (multiplicative effect)
            interaction = scaled_data[:, idx1] * scaled_data[:, idx2] * scaled_data[:, idx3]
            
            # Measure interaction strength using variance ratio
            interaction_var = np.var(interaction)
            individual_var = np.var(scaled_data[:, idx1]) + np.var(scaled_data[:, idx2]) + np.var(scaled_data[:, idx3])
            
            if individual_var > 0:
                interaction_strength = interaction_var / (individual_var / 3)  # Normalized
                
                if interaction_strength > interaction_threshold:
                    feature_names = (cols_to_use[idx1], cols_to_use[idx2], cols_to_use[idx3])
                    interactions.append({
                        'features': feature_names,
                        'strength': round(float(interaction_strength), 4),
                        'type': 'multiplicative_synergy'
                    })
        
        # Sort by strength
        interactions = sorted(interactions, key=lambda x: x['strength'], reverse=True)[:10]
        
        return {
            'interactions_found': interactions,
            'total_checked': len(list(combinations(cols_to_use, 3))),
            'interpretation': self._interpret_interactions(interactions)
        }
    
    def detect_feature_clusters(self, n_clusters: int = 3) -> Dict[str, Any]:
        """
        Detect clusters of related features using correlation-based clustering.
        
        Identifies groups of features that move together.
        O(k^2 log k) complexity
        
        Args:
            n_clusters: Number of feature clusters
        
        Returns:
            Dictionary with feature groupings
        """
        if len(self.numeric_cols) < 2:
            return {'clusters': [], 'interpretation': 'Not enough numeric features'}
        
        # Use absolute correlation distance
        corr_matrix = self.df[self.numeric_cols].corr().fillna(0)
        distance_matrix = 1 - np.abs(corr_matrix)
        
        # Cluster features
        n_clusters_actual = min(n_clusters, len(self.numeric_cols), 5)
        try:
            kmeans = KMeans(n_clusters=n_clusters_actual, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(distance_matrix)
        except:
            return {'clusters': {}, 'interpretation': 'Could not perform feature clustering'}
        
        clusters = {}
        for i in range(n_clusters_actual):
            cluster_features = [self.numeric_cols[j] for j, label in enumerate(labels) if label == i]
            if cluster_features:
                # Calculate average correlation within cluster
                cluster_corr_values = []
                for f1, f2 in combinations(cluster_features, 2):
                    cluster_corr_values.append(abs(corr_matrix.loc[f1, f2]))
                
                avg_corr = np.mean(cluster_corr_values) if cluster_corr_values else 0
                
                clusters[f'Cluster_{i+1}'] = {
                    'features': cluster_features,
                    'avg_internal_correlation': round(float(avg_corr), 3),
                    'size': len(cluster_features),
                    'interpretation': self._interpret_cluster_meaning(cluster_features, corr_matrix)
                }
        
        return {
            'clusters': clusters,
            'n_clusters': len(clusters),
            'interpretation': self._interpret_clusters(clusters)
        }
    
    def analyze_categorical_influence(self) -> Dict[str, Any]:
        """
        Analyze how categorical variables influence numeric ones.
        
        Tests for significant differences across categories using Kruskal-Wallis.
        O(n log n) per categorical variable
        
        Returns:
            Dictionary with influence strength per categorical variable
        """
        if not self.categorical_cols or not self.numeric_cols:
            return {'influences': {}, 'interpretation': 'Need both categorical and numeric columns'}
        
        influences = {}
        
        for cat_col in self.categorical_cols[:8]:  # Limit for performance
            cat_data = self.df[cat_col].dropna()
            
            if len(cat_data.unique()) < 2 or len(cat_data.unique()) > 20:
                continue  # Skip if too few or too many categories
            
            influence_details = {
                'numeric_effects': {},
                'overall_strength': 0
            }
            
            for num_col in self.numeric_cols[:8]:  # Limit for performance
                groups = [self.df[self.df[cat_col] == cat][num_col].dropna().values 
                         for cat in cat_data.unique() 
                         if len(self.df[self.df[cat_col] == cat][num_col].dropna()) > 0]
                
                if len(groups) > 1 and all(len(g) > 0 for g in groups):
                    try:
                        stat, p_value = kruskal(*groups)
                        
                        if p_value < 0.05:
                            # Effect size (eta-squared approximation)
                            effect_size = 1 - (stat / (len(self.df) - 1))
                            influence_details['numeric_effects'][num_col] = {
                                'p_value': round(float(p_value), 6),
                                'significant': p_value < 0.05,
                                'effect_size': round(float(abs(effect_size)), 3)
                            }
                            influence_details['overall_strength'] += 1
                    except:
                        pass
            
            if influence_details['numeric_effects']:
                influences[cat_col] = {
                    'significant_effects': len(influence_details['numeric_effects']),
                    'effect_details': influence_details['numeric_effects'],
                    'strength': self._classify_influence_strength(influence_details['overall_strength'], len(self.numeric_cols))
                }
        
        return {
            'influences': influences,
            'interpretation': self._interpret_categorical_influence(influences)
        }
    
    def detect_anomaly_patterns(self, n_anomalies: int = 10, 
                               sensitivity: float = 0.95) -> Dict[str, Any]:
        """
        Detect patterns in anomalous data points.
        O(n log n) using distance calculations
        
        Args:
            n_anomalies: Number of anomalies to analyze
            sensitivity: Percentile threshold for anomalies (0-1)
        
        Returns:
            Dictionary with anomaly patterns
        """
        if len(self.numeric_cols) < 2:
            return {'anomaly_patterns': [], 'interpretation': 'Need at least 2 numeric columns'}
        
        # Calculate anomaly score (distance from centroid)
        numeric_data = self.df[self.numeric_cols].fillna(self.df[self.numeric_cols].mean())
        scaler = StandardScaler()
        scaled = scaler.fit_transform(numeric_data)
        
        # Calculate Mahalanobis distance
        centroid = np.mean(scaled, axis=0)
        distances = np.linalg.norm(scaled - centroid, axis=1)
        
        # Find anomalies
        threshold = np.percentile(distances, sensitivity * 100)
        anomaly_indices = np.argsort(distances)[-n_anomalies:][::-1]
        
        patterns = []
        for idx in anomaly_indices:
            anomaly_values = self.df.iloc[idx][self.numeric_cols].to_dict()
            patterns.append({
                'index': int(idx),
                'anomaly_score': round(float(distances[idx]), 3),
                'severity': 'Critical' if distances[idx] > np.percentile(distances, 99) else 'High',
                'values': {k: round(v, 2) if isinstance(v, (float, int)) else str(v) 
                          for k, v in anomaly_values.items()}
            })
        
        return {
            'anomaly_patterns': patterns,
            'total_anomalies': len(anomaly_indices),
            'threshold': round(float(threshold), 3),
            'interpretation': self._interpret_anomalies(patterns)
        }
    
    def rank_feature_importance(self) -> Dict[str, Any]:
        """
        Rank numeric features by importance using multiple methods.
        O(k^2) where k = number of numeric features
        
        Returns:
            Dictionary with feature rankings
        """
        if not self.numeric_cols:
            return {'feature_ranking': {}, 'interpretation': 'No numeric features'}
        
        importances = {}
        methods = {}
        
        # Method 1: Variance-based (capturing variability)
        variance_scores = {}
        for col in self.numeric_cols:
            var = self.df[col].var()
            variance_scores[col] = var if not np.isnan(var) else 0
        
        max_var = max(variance_scores.values()) if variance_scores else 1
        for col in variance_scores:
            variance_scores[col] = variance_scores[col] / max_var if max_var > 0 else 0
        
        methods['variance'] = {k: round(v, 3) for k, v in variance_scores.items()}
        
        # Method 2: Correlation-based (relationship strength)
        corr_matrix = self.df[self.numeric_cols].corr().fillna(0)
        correlation_scores = {}
        for col in self.numeric_cols:
            corr_vals = np.abs(corr_matrix[col].drop(col)).values
            correlation_scores[col] = np.mean(corr_vals) if len(corr_vals) > 0 else 0
        
        methods['correlation'] = {k: round(v, 3) for k, v in correlation_scores.items()}
        
        # Method 3: Information density (cardinality)
        info_scores = {}
        for col in self.numeric_cols:
            unique_pct = self.df[col].nunique() / len(self.df)
            info_scores[col] = unique_pct
        
        methods['information_density'] = {k: round(v, 3) for k, v in info_scores.items()}
        
        # Method 4: Skewness (deviation from normality)
        skewness_scores = {}
        for col in self.numeric_cols:
            skew = self.df[col].skew()
            skewness_scores[col] = abs(skew) / (1 + abs(skew))  # Normalize
        
        methods['skewness'] = {k: round(v, 3) for k, v in skewness_scores.items()}
        
        # Combine all methods (weighted average)
        weights = {'variance': 0.3, 'correlation': 0.3, 'information': 0.25, 'skewness': 0.15}
        for col in self.numeric_cols:
            combined = (
                variance_scores.get(col, 0) * weights['variance'] +
                correlation_scores.get(col, 0) * weights['correlation'] +
                info_scores.get(col, 0) * weights['information'] +
                skewness_scores.get(col, 0) * weights['skewness']
            )
            importances[col] = round(combined, 4)
        
        # Sort
        ranked = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'feature_ranking': dict(ranked),
            'importance_methods': methods,
            'interpretation': self._interpret_feature_importance(dict(ranked))
        }
    
    def detect_temporal_patterns(self) -> Dict[str, Any]:
        """
        Detect if data has temporal/sequential patterns.
        O(n log n) for trend detection
        
        Returns:
            Dictionary with temporal pattern info
        """
        temporal_patterns = {}
        
        for col in self.numeric_cols[:10]:
            values = self.df[col].dropna().values
            
            if len(values) > 10:
                # Check for trend (correlation with index)
                indices = np.arange(len(values))
                trend_corr = np.corrcoef(indices, values)[0, 1]
                
                # Check for autocorrelation
                if len(values) > 20:
                    autocorr = np.corrcoef(values[:-1], values[1:])[0, 1]
                else:
                    autocorr = 0
                
                # Check for seasonality indicator
                if len(values) > 30:
                    chunk_size = len(values) // 4
                    means = [np.mean(values[i:i+chunk_size]) for i in range(0, len(values)-chunk_size+1, chunk_size)]
                    seasonality = np.std(means) / np.mean(values) if np.mean(values) != 0 else 0
                else:
                    seasonality = 0
                
                if abs(trend_corr) > 0.3 or abs(autocorr) > 0.3 or seasonality > 0.1:
                    temporal_patterns[col] = {
                        'trend_strength': round(float(trend_corr), 4),
                        'autocorrelation': round(float(autocorr), 4),
                        'seasonality_indicator': round(float(seasonality), 4),
                        'has_pattern': True
                    }
        
        return {
            'temporal_patterns': temporal_patterns,
            'interpretation': self._interpret_temporal_patterns(temporal_patterns)
        }
    
    def detect_multivariate_patterns(self, n_patterns: int = 5) -> Dict[str, Any]:
        """
        Detect complex multivariate patterns using PCA-like approach.
        O(n * k^2) where k = numeric features
        
        Args:
            n_patterns: Number of patterns to extract
        
        Returns:
            Dictionary with multivariate pattern descriptions
        """
        if len(self.numeric_cols) < 3:
            return {'patterns': [], 'interpretation': 'Need at least 3 numeric features'}
        
        # Compute correlation-based patterns
        corr_matrix = self.df[self.numeric_cols].corr().fillna(0)
        
        # Find dominant correlation patterns
        patterns = []
        examined = set()
        
        for i, col1 in enumerate(self.numeric_cols):
            for j, col2 in enumerate(self.numeric_cols[i+1:], i+1):
                if (i, j) in examined:
                    continue
                
                corr_val = abs(corr_matrix.loc[col1, col2])
                
                if corr_val > 0.6:  # Strong correlation
                    # Find related third features
                    third_features = []
                    for k, col3 in enumerate(self.numeric_cols):
                        if k != i and k != j:
                            if abs(corr_matrix.loc[col1, col3]) > 0.4 or abs(corr_matrix.loc[col2, col3]) > 0.4:
                                third_features.append(col3)
                    
                    patterns.append({
                        'core_features': [col1, col2],
                        'related_features': third_features[:3],
                        'strength': round(float(corr_val), 3),
                        'pattern_type': 'Strong Correlation Network'
                    })
                    examined.add((i, j))
        
        # Sort by strength
        patterns = sorted(patterns, key=lambda x: x['strength'], reverse=True)[:n_patterns]
        
        return {
            'multivariate_patterns': patterns,
            'n_patterns': len(patterns),
            'interpretation': self._interpret_multivariate_patterns(patterns)
        }
    
    # Helper interpretation methods
    
    @staticmethod
    def _interpret_interactions(interactions: List[Dict]) -> str:
        """Interpret interaction effects"""
        if not interactions:
            return "No significant 3-way feature interactions detected. Features act independently."
        
        strongest = interactions[0]
        return f"Found {len(interactions)} significant feature interactions. " \
               f"Strongest synergy: {strongest['features'][0]}-{strongest['features'][1]}-{strongest['features'][2]} " \
               f"(strength: {strongest['strength']}). These combinations have multiplicative effects."
    
    @staticmethod
    def _interpret_clusters(clusters: Dict) -> str:
        """Interpret feature clusters"""
        if not clusters:
            return "Features are relatively independent with no clear clustering."
        
        return f"Features naturally group into {len(clusters)} clusters based on correlation patterns. " \
               f"This reveals underlying latent factors in your data structure."
    
    @staticmethod
    def _interpret_cluster_meaning(features: List[str], corr_matrix: pd.DataFrame) -> str:
        """Interpret what a specific cluster means"""
        if len(features) <= 1:
            return "Single feature cluster"
        
        avg_corr = np.mean([abs(corr_matrix.loc[f1, f2]) for f1, f2 in combinations(features, 2)])
        
        if avg_corr > 0.7:
            return "Highly cohesive cluster - features are strongly synchronized"
        elif avg_corr > 0.5:
            return "Moderately cohesive - related but distinct concepts"
        else:
            return "Weakly related features"
    
    @staticmethod
    def _classify_influence_strength(significant_count: int, total_numeric: int) -> str:
        """Classify categorical influence strength"""
        if significant_count >= total_numeric * 0.7:
            return "Very Strong"
        elif significant_count >= total_numeric * 0.4:
            return "Strong"
        elif significant_count > 0:
            return "Moderate"
        else:
            return "Weak"
    
    @staticmethod
    def _interpret_categorical_influence(influences: Dict) -> str:
        """Interpret categorical influences"""
        if not influences:
            return "Categorical variables show minimal influence on numeric variables."
        
        strong = sum(1 for inf in influences.values() if inf['strength'] == 'Very Strong')
        total_effects = sum(inf['significant_effects'] for inf in influences.values())
        
        return f"Found {len(influences)} influential categorical variables with {total_effects} significant numeric effects. " \
               f"{strong} show very strong categorical influence. Consider these in your models."
    
    @staticmethod
    def _interpret_anomalies(patterns: List[Dict]) -> str:
        """Interpret anomaly patterns"""
        if not patterns:
            return "No significant anomalies detected. Data appears consistent."
        
        critical = sum(1 for p in patterns if p['severity'] == 'Critical')
        avg_score = np.mean([p['anomaly_score'] for p in patterns])
        
        return f"Detected {len(patterns)} anomalous points (avg score: {avg_score:.2f}, {critical} critical). " \
               f"These outliers may represent measurement errors, data entry issues, or genuinely exceptional cases."
    
    @staticmethod
    def _interpret_feature_importance(rankings: Dict) -> str:
        """Interpret feature importance"""
        if not rankings:
            return "Unable to determine feature importance."
        
        top_features = list(rankings.items())[:3]
        top_str = ", ".join([f"{name} ({score:.3f})" for name, score in top_features])
        
        return f"Top 3 most important features: {top_str}. " \
               f"These capture the most variability and relationships in your dataset."
    
    @staticmethod
    def _interpret_temporal_patterns(patterns: Dict) -> str:
        """Interpret temporal patterns"""
        if not patterns:
            return "No significant temporal patterns detected. Data appears independent of sequence."
        
        trend_cols = [col for col, data in patterns.items() if abs(data['trend_strength']) > 0.3]
        seasonal_cols = [col for col, data in patterns.items() if data['seasonality_indicator'] > 0.1]
        
        msg = f"Detected temporal patterns in {len(patterns)} columns. " \
              f"{len(trend_cols)} show trends, {len(seasonal_cols)} show seasonality-like patterns. "
        
        if trend_cols or seasonal_cols:
            msg += "Data may not be independent - consider time-series analysis approaches."
        
        return msg
    
    @staticmethod
    def _interpret_multivariate_patterns(patterns: List[Dict]) -> str:
        """Interpret multivariate patterns"""
        if not patterns:
            return "No dominant multivariate patterns detected."
        
        strongest = patterns[0]
        return f"Found {len(patterns)} multivariate patterns. Strongest: {strongest['core_features'][0]} and " \
               f"{strongest['core_features'][1]} (correlation: {strongest['strength']}). " \
               f"These patterns reveal the fundamental structure of your data."
