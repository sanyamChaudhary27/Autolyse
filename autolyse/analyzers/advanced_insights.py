"""Advanced multivariate pattern analysis.

Honesty notes baked into the design:
- Feature clustering uses deterministic hierarchical linkage on correlation
  distance (KMeans on 4 points was both slow and meaningless).
- Anomaly detection uses a proper Mahalanobis distance with pseudo-inverse
  covariance fallback.
- Row-order "temporal" heuristics are opt-in: for unordered tabular data they
  detect artifacts, not seasonality.
"""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import kruskal
from sklearn.preprocessing import StandardScaler


class AdvancedInsightsAnalyzer:
    """Discover interactions, feature groups, anomalies and influence effects."""

    def __init__(self, df: pd.DataFrame, random_state: int = 42):
        self.df = df
        self.random_state = random_state
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = [
            c for c in df.columns if df[c].dtype.name in ("object", "str", "category")
        ]

    def analyze_all(self, include_row_order_patterns: bool = False) -> dict:
        """Run all advanced analyses.

        Args:
            include_row_order_patterns: Enable row-order trend/seasonality
                heuristics. Only meaningful when row position encodes time.
        """
        results = {
            "feature_interactions": self.find_feature_interactions(),
            "feature_clusters": self.detect_feature_clusters(),
            "categorical_influence": self.analyze_categorical_influence(),
            "anomaly_patterns": self.detect_anomaly_patterns(),
            "feature_importance": self.rank_feature_importance(),
            "multivariate_patterns": self.detect_multivariate_patterns(),
        }
        if include_row_order_patterns:
            results["temporal_patterns"] = self.detect_temporal_patterns()
        return results

    # ----------------------------------------------------------- interactions

    def find_feature_interactions(self, max_features: int = 5,
                                  interaction_threshold: float = 0.2) -> dict:
        """Rank triplets of standardized features by co-variation of their product.

        This is an unsupervised *co-variation* heuristic, not evidence of a
        statistical interaction effect; confirming those requires a target.
        """
        if len(self.numeric_cols) < 3:
            return {
                "interactions_found": [],
                "interpretation": "Not enough numeric features for interaction analysis",
            }

        cols_to_use = self.numeric_cols[:max_features]
        numeric_data = self.df[cols_to_use].fillna(self.df[cols_to_use].mean())
        scaled_data = StandardScaler().fit_transform(numeric_data)

        interactions = []
        for idx1, idx2, idx3 in combinations(range(len(cols_to_use)), 3):
            product = scaled_data[:, idx1] * scaled_data[:, idx2] * scaled_data[:, idx3]
            individual_var = (
                np.var(scaled_data[:, idx1])
                + np.var(scaled_data[:, idx2])
                + np.var(scaled_data[:, idx3])
            )
            if individual_var > 0:
                strength = float(np.var(product) / (individual_var / 3))
                if strength > interaction_threshold:
                    interactions.append({
                        "features": (cols_to_use[idx1], cols_to_use[idx2], cols_to_use[idx3]),
                        "strength": round(strength, 4),
                        "type": "multiplicative_covariation",
                    })

        interactions = sorted(interactions, key=lambda x: x["strength"], reverse=True)[:10]
        return {
            "interactions_found": interactions,
            "total_checked": len(list(combinations(cols_to_use, 3))),
            "interpretation": self._interpret_interactions(interactions),
        }

    # --------------------------------------------------------------- clusters

    def detect_feature_clusters(self, max_clusters: int = 5) -> dict:
        """Group correlated features via Ward linkage on |1 - corr| distance.

        Deterministic and O(k^3) in the number of features - instant even for
        hundreds of columns, unlike fitting KMeans to degenerate matrices.
        """
        n_num = len(self.numeric_cols)
        if n_num < 2:
            return {"clusters": [], "interpretation": "Not enough numeric features"}

        corr_matrix = self.df[self.numeric_cols].corr().fillna(0.0)
        distance_matrix = np.clip(1.0 - np.abs(corr_matrix.to_numpy()), 0.0, None)
        np.fill_diagonal(distance_matrix, 0.0)

        condensed = squareform(distance_matrix, checks=False)
        link = linkage(condensed, method="average")

        # Cut where gaps between merge distances are largest; fall back to 2.
        distances = link[:, 2]
        if len(distances) >= 2:
            gaps = np.diff(distances)
            n_clusters = int(np.argmax(gaps) + 1) + 1 if gaps.max() > 0 else 2
        else:
            n_clusters = 2
        n_clusters = min(n_clusters, n_num, max_clusters)

        labels = fcluster(link, t=n_clusters, criterion="maxclust")

        clusters = {}
        for label in np.unique(labels):
            members = [self.numeric_cols[j] for j in range(n_num) if labels[j] == label]
            if not members:
                continue
            pair_corrs = [
                abs(corr_matrix.loc[a, b]) for a, b in combinations(members, 2)
            ]
            avg_corr = float(np.mean(pair_corrs)) if pair_corrs else 1.0
            clusters[f"Cluster_{label}"] = {
                "features": members,
                "avg_internal_correlation": round(avg_corr, 3),
                "size": len(members),
                "meaning": self._describe_cluster_cohesion(avg_corr),
            }

        return {
            "clusters": list(clusters.values()),
            "n_clusters": len(clusters),
            "interpretation": self._interpret_clusters(clusters),
        }

    @staticmethod
    def _describe_cluster_cohesion(avg_corr: float) -> str:
        if avg_corr > 0.7:
            return "Highly cohesive - features are strongly synchronized"
        if avg_corr > 0.4:
            return "Moderately cohesive - related but distinct"
        return "Weakly related features grouped by proximity"

    # ---------------------------------------------------------------- influence

    def analyze_categorical_influence(self) -> dict:
        """Kruskal-Wallis tests of numeric distributions across categories."""
        if not self.categorical_cols or not self.numeric_cols:
            return {"influences": {}, "interpretation":
                    "Need both categorical and numeric columns"}

        influences = {}
        for cat_col in self.categorical_cols[:8]:
            cat_series = self.df[cat_col].dropna()
            n_levels = cat_series.nunique()
            if n_levels < 2 or n_levels > 20:
                continue

            effects = {}
            for num_col in self.numeric_cols[:8]:
                groups = [
                    g[num_col].dropna().to_numpy()
                    for _, g in self.df.groupby(cat_col)
                    if g[num_col].notna().any()
                ]
                groups = [g for g in groups if len(g) > 0]
                if len(groups) < 2:
                    continue
                try:
                    stat, p_value = kruskal(*groups)
                except ValueError:
                    continue
                if p_value < 0.05:
                    # Epsilon-squared effect size: H / (n - 1).
                    n_total = sum(len(g) for g in groups)
                    epsilon_squared = stat / max(n_total - 1, 1)
                    effects[num_col] = {
                        "p_value": round(float(p_value), 6),
                        "effect_size_epsilon_sq": round(float(epsilon_squared), 3),
                    }

            if effects:
                influences[cat_col] = {
                    "significant_effects": len(effects),
                    "effect_details": effects,
                    "strength": self._classify_influence_strength(
                        len(effects), min(len(self.numeric_cols), 8)
                    ),
                }

        return {
            "influences": influences,
            "interpretation": self._interpret_categorical_influence(influences),
        }

    # ---------------------------------------------------------------- anomalies

    def detect_anomaly_patterns(self, n_anomalies: int = 10,
                                sensitivity: float = 0.95) -> dict:
        """Flag extreme rows via Mahalanobis distance from the centroid."""
        if len(self.numeric_cols) < 2:
            return {"anomaly_patterns": [], "interpretation":
                    "Need at least 2 numeric columns"}

        numeric_data = self.df[self.numeric_cols].replace(
            [np.inf, -np.inf], np.nan
        ).fillna(self.df[self.numeric_cols].mean())

        # Exclude columns that are constant OR have no computable spread
        # (e.g. entirely missing) - they carry no anomaly information and
        # would push NaNs into the scaler.
        stds = numeric_data.std()
        constant_cols = [c for c in self.numeric_cols
                         if pd.isna(stds.get(c)) or stds[c] == 0]
        usable = [c for c in self.numeric_cols if c not in constant_cols]
        if len(usable) < 2:
            return {"anomaly_patterns": [], "interpretation":
                    "Too many constant columns for anomaly detection"}

        scaled = StandardScaler().fit_transform(numeric_data[usable])

        cov = np.cov(scaled, rowvar=False)
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov)

        centroid = scaled.mean(axis=0)
        delta = scaled - centroid
        distances = np.sqrt(np.einsum("ij,jk,ik->i", delta, cov_inv, delta))

        threshold = float(np.percentile(distances, sensitivity * 100))
        top_idx = np.argsort(distances)[-min(n_anomalies, len(distances)):][::-1]

        patterns = []
        critical_cut = float(np.percentile(distances, 99))
        for idx in top_idx:
            values = self.df.iloc[idx]
            patterns.append({
                "index": int(idx),
                "anomaly_score": round(float(distances[idx]), 3),
                "severity": "Critical" if distances[idx] >= critical_cut else "High",
                "values": {
                    col: (round(float(values[col]), 3)
                          if isinstance(values[col], (int, float, np.number))
                          else str(values[col]))
                    for col in usable
                },
            })

        return {
            "anomaly_patterns": patterns,
            "distance_metric": "mahalanobis",
            "threshold": round(threshold, 3),
            "excluded_constant_columns": constant_cols,
            "interpretation": self._interpret_anomalies(patterns),
        }

    # ---------------------------------------------------------------- importance

    def rank_feature_importance(self) -> dict:
        """Unsupervised importance: variance, connectivity, density, shape."""
        if not self.numeric_cols:
            return {"feature_ranking": {}, "interpretation": "No numeric features"}

        variance_scores = {}
        info_scores = {}
        skew_scores = {}
        for col in self.numeric_cols:
            series = self.df[col].dropna()
            var = series.var()
            variance_scores[col] = var if pd.notna(var) else 0.0
            info_scores[col] = series.nunique() / len(series) if len(series) else 0.0
            skew = series.skew()
            skew_scores[col] = abs(skew) / (1 + abs(skew)) if pd.notna(skew) else 0.0

        max_var = max(variance_scores.values(), default=1.0)
        variance_norm = {
            c: (v / max_var if max_var > 0 else 0.0) for c, v in variance_scores.items()
        }

        corr_matrix = self.df[self.numeric_cols].corr().fillna(0)
        conn_scores = {}
        for col in self.numeric_cols:
            others = np.abs(corr_matrix[col].drop(col)).to_numpy()
            conn_scores[col] = float(np.mean(others)) if others.size else 0.0

        weights = {"variance": 0.35, "connectivity": 0.30,
                   "information_density": 0.20, "shape_deviation": 0.15}
        combined = {}
        for col in self.numeric_cols:
            combined[col] = round(
                weights["variance"] * variance_norm.get(col, 0)
                + weights["connectivity"] * conn_scores[col]
                + weights["information_density"] * info_scores[col]
                + weights["shape_deviation"] * skew_scores[col],
                4,
            )

        ranked = dict(sorted(combined.items(), key=lambda kv: kv[1], reverse=True))
        return {
            "feature_ranking": ranked,
            "importance_methods": {
                "variance": {k: round(v, 3) for k, v in variance_norm.items()},
                "correlation_connectivity": {k: round(v, 3) for k, v in conn_scores.items()},
                "information_density": {k: round(v, 3) for k, v in info_scores.items()},
                "shape_deviation": {k: round(v, 3) for k, v in skew_scores.items()},
            },
            "interpretation": self._interpret_feature_importance(ranked),
        }

    # ------------------------------------------------------------ multivariate

    def detect_multivariate_patterns(self, n_patterns: int = 5) -> dict:
        """Correlation networks: strongly-linked pairs plus their neighbours."""
        if len(self.numeric_cols) < 3:
            return {"patterns": [], "interpretation":
                    "Need at least 3 numeric features"}

        corr_matrix = self.df[self.numeric_cols].corr().fillna(0)
        patterns = []
        seen = set()

        for i, col1 in enumerate(self.numeric_cols):
            for j, col2 in enumerate(self.numeric_cols[i + 1:], start=i + 1):
                if (i, j) in seen:
                    continue
                corr_val = abs(corr_matrix.loc[col1, col2])
                if corr_val > 0.6:
                    related = [
                        col3
                        for k, col3 in enumerate(self.numeric_cols)
                        if k not in (i, j)
                        and (abs(corr_matrix.loc[col1, col3]) > 0.4
                             or abs(corr_matrix.loc[col2, col3]) > 0.4)
                    ]
                    patterns.append({
                        "core_features": [col1, col2],
                        "related_features": related[:3],
                        "strength": round(float(corr_val), 3),
                        "pattern_type": "Strong Correlation Network",
                    })
                    seen.add((i, j))

        patterns = sorted(patterns, key=lambda p: p["strength"], reverse=True)[:n_patterns]
        return {
            "multivariate_patterns": patterns,
            "n_patterns": len(patterns),
            "interpretation": self._interpret_multivariate_patterns(patterns),
        }

    # ------------------------------------------------------- row-order heuristics

    def detect_temporal_patterns(self) -> dict:
        """Trend/autocorrelation heuristics over ROW ORDER.

        Only meaningful when rows are chronologically ordered; kept as an
        explicit opt-in (see :meth:`analyze_all`).
        """
        temporal_patterns = {}
        for col in self.numeric_cols[:10]:
            values = self.df[col].dropna().to_numpy()
            if len(values) <= 10 or np.std(values) == 0:
                continue

            indices = np.arange(len(values))
            trend_corr = float(np.corrcoef(indices, values)[0, 1])
            autocorr = (
                float(np.corrcoef(values[:-1], values[1:])[0, 1])
                if len(values) > 20 else 0.0
            )

            if abs(trend_corr) > 0.3 or abs(autocorr) > 0.3:
                temporal_patterns[col] = {
                    "trend_strength": round(trend_corr, 4),
                    "lag1_autocorrelation": round(autocorr, 4),
                    "has_pattern": True,
                }

        return {
            "temporal_patterns": temporal_patterns,
            "caveat": "Computed over row order; only valid for time-sorted data.",
            "interpretation": self._interpret_temporal_patterns(temporal_patterns),
        }

    # ---------------------------------------------------------- interpretations

    @staticmethod
    def _interpret_interactions(interactions):
        if not interactions:
            return ("No notable multiplicative co-variation among standardized "
                    "feature triplets.")
        strongest = interactions[0]
        names = "-".join(strongest["features"])
        return (f"{len(interactions)} feature triplet(s) show strong co-variation; "
                f"strongest: {names} (score {strongest['strength']}). "
                f"Validate against a target before treating these as true interactions.")

    @staticmethod
    def _interpret_clusters(clusters):
        if not clusters:
            return "Features are relatively independent with no clear clustering."
        sizes = ", ".join(f"{len(c['features'])}" for c in clusters.values())
        return (f"Features group into {len(clusters)} correlation clusters "
                f"(sizes: {sizes}); each cluster behaves like one latent signal.")

    @staticmethod
    def _classify_influence_strength(significant_count, total_numeric):
        ratio = significant_count / total_numeric if total_numeric else 0
        if ratio >= 0.7:
            return "Very Strong"
        if ratio >= 0.4:
            return "Strong"
        if significant_count > 0:
            return "Moderate"
        return "Weak"

    @staticmethod
    def _interpret_categorical_influence(influences):
        if not influences:
            return "Categorical variables show minimal influence on numeric variables."
        total_effects = sum(i["significant_effects"] for i in influences.values())
        strong = sum(1 for i in influences.values() if i["strength"] == "Very Strong")
        return (f"{len(influences)} categorical variable(s) drive {total_effects} "
                f"significant numeric differences ({strong} very strong). "
                f"Worth including in any model of this data.")

    @staticmethod
    def _interpret_anomalies(patterns):
        if not patterns:
            return "No significant anomalies detected."
        critical = sum(1 for p in patterns if p["severity"] == "Critical")
        avg = np.mean([p["anomaly_score"] for p in patterns])
        return (f"Top {len(patterns)} anomalous rows by Mahalanobis distance "
                f"(avg {avg:.2f}, {critical} critical). Check for entry errors "
                f"or genuinely exceptional cases.")

    @staticmethod
    def _interpret_feature_importance(rankings):
        if not rankings:
            return "Unable to determine feature importance."
        top = ", ".join(f"{name} ({score:.3f})"
                        for name, score in list(rankings.items())[:3])
        return f"Top features by unsupervised signal: {top}."

    @staticmethod
    def _interpret_temporal_patterns(patterns):
        if not patterns:
            return ("No trend or autocorrelation over row order "
                    "(only meaningful for sorted data).")
        names = ", ".join(list(patterns)[:3])
        return (f"Row-order structure detected in: {names}. "
                f"If rows are time-ordered, consider time-series methods.")

    @staticmethod
    def _interpret_multivariate_patterns(patterns):
        if not patterns:
            return "No dominant multivariate correlation networks detected."
        strongest = patterns[0]
        a, b = strongest["core_features"]
        return (f"{len(patterns)} correlation network(s) found; densest around "
                f"{a} and {b} (|r|={strongest['strength']}).")
