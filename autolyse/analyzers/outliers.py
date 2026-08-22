"""Outlier detection module."""

import numpy as np
import pandas as pd

try:  # scikit-learn is an optional heavy import; degrade gracefully.
    from sklearn.ensemble import IsolationForest
    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without sklearn
    IsolationForest = None
    _SKLEARN_AVAILABLE = False


class OutlierAnalyzer:
    """Detect outliers via IQR fences and (optionally) Isolation Forest."""

    #: How many of the most anomalous rows to keep in the summary.
    TOP_ANOMALIES_KEPT = 20

    def __init__(self, df: pd.DataFrame, contamination: float = 0.1,
                 random_state: int = 42):
        self.df = df
        self.random_state = random_state
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not 0 < contamination < 0.5:
            raise ValueError("contamination must be between 0 and 0.5")
        self.contamination = contamination

    def detect_iqr_outliers(self) -> dict:
        """Tukey-fence outliers (Q1 - 1.5*IQR, Q3 + 1.5*IQR) per column."""
        outliers = {}
        for col in self.numeric_cols:
            col_data = self.df[col].dropna()
            q1, q3 = col_data.quantile(0.25), col_data.quantile(0.75)
            iqr = q3 - q1
            lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr

            outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
            outlier_values = np.sort(col_data[outlier_mask].to_numpy())

            # Cap stored values: a column that is 40% outliers would otherwise
            # embed millions of floats into every downstream report.
            kept = outlier_values[: self.TOP_ANOMALIES_KEPT * 5]

            outliers[col] = {
                "method": "IQR",
                "outlier_count": int(len(outlier_values)),
                "outlier_percentage":
                    (len(outlier_values) / len(col_data)) * 100 if len(col_data) else 0.0,
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "outlier_values": [float(v) for v in kept],
                "values_truncated": len(kept) < len(outlier_values),
            }
        return outliers

    def detect_isolation_forest_outliers(self) -> dict:
        """Multivariate anomaly detection across all numeric columns.

        Returns aggregate statistics plus at most ``TOP_ANOMALIES_KEPT``
        row indices/scores - never full per-row arrays.
        """
        usable_cols = [
            c for c in self.numeric_cols if np.isfinite(self.df[c].dropna()).all()
        ]
        if len(usable_cols) == 0 or not _SKLEARN_AVAILABLE:
            return {}

        numeric_data = self.df[usable_cols].replace([np.inf, -np.inf], np.nan).dropna()
        if len(numeric_data) < 10:
            return {}

        iso_forest = IsolationForest(
            contamination=self.contamination, random_state=self.random_state
        )
        predictions = iso_forest.fit_predict(numeric_data)
        scores = iso_forest.score_samples(numeric_data)

        anomaly_positions = np.where(predictions == -1)[0]
        order = anomaly_positions[np.argsort(scores[anomaly_positions])][
            : self.TOP_ANOMALIES_KEPT
        ]

        return {
            "method": "Isolation Forest",
            "n_outliers": int(len(anomaly_positions)),
            "outlier_percentage": len(anomaly_positions) / len(numeric_data) * 100,
            "top_anomaly_indices": [int(i) for i in order],
            "top_anomaly_scores": [round(float(scores[i]), 4) for i in order],
            "score_mean": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
            "anomaly_threshold": float(
                np.percentile(scores, self.contamination * 100)
            ),
        }

    def get_outlier_summary(self) -> dict:
        return {
            "iqr_method": self.detect_iqr_outliers(),
            "isolation_forest": self.detect_isolation_forest_outliers(),
        }
