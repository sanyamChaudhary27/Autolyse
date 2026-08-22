"""Distribution analysis module."""

import numpy as np
import pandas as pd
from scipy import stats


class DistributionAnalyzer:
    """Analyze distributions of numeric and categorical columns."""

    # Shapiro-Wilk is only valid up to this many observations.
    SHAPIRO_MAX_N = 5000

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = [
            c for c in df.columns
            if df[c].dtype.name in ("object", "str", "category")
        ]

    def analyze_numeric_distributions(self) -> dict:
        """Classify each numeric column's shape with a normality test.

        ``normality_pvalue`` is ``NaN`` when no valid test exists (e.g.
        constant columns); ``is_normal`` is then ``False`` rather than
        misleading.
        """
        distributions = {}
        for col in self.numeric_cols:
            col_data = self.df[col].dropna()
            if len(col_data) < 3:
                continue

            skewness = float(col_data.skew())
            kurt = float(col_data.kurtosis())
            std = float(col_data.std())

            if col_data.nunique() <= 1 or (std == 0 or np.isnan(std)):
                p_value = float("nan")
                dist_type = "Constant"
            else:
                p_value = self._normality_pvalue(col_data)
                if abs(skewness) < 0.5 and abs(kurt) < 1:
                    dist_type = "Approximately Normal"
                elif skewness > 1:
                    dist_type = "Right-skewed"
                elif skewness < -1:
                    dist_type = "Left-skewed"
                else:
                    dist_type = "Moderately Skewed"

            mode_vals = col_data.mode()
            distributions[col] = {
                "distribution_type": dist_type,
                "normality_pvalue": p_value,
                "is_normal": bool(p_value == p_value and p_value > 0.05),
                "skewness": skewness,
                "kurtosis": kurt,
                "unique_values": int(col_data.nunique()),
                "mode": mode_vals.iloc[0] if len(mode_vals) else None,
            }
        return distributions

    @classmethod
    def _normality_pvalue(cls, col_data: pd.Series) -> float:
        try:
            if len(col_data) <= cls.SHAPIRO_MAX_N:
                return float(stats.shapiro(col_data).pvalue)
            standardized = (col_data - col_data.mean()) / col_data.std()
            return float(stats.kstest(standardized, "norm").pvalue)
        except (ValueError, TypeError):
            # scipy raises on degenerate input it cannot handle; report NaN.
            return float("nan")

    def analyze_categorical_distributions(self) -> dict:
        """Frequency structure of categorical columns incl. Simpson diversity."""
        distributions = {}
        for col in self.categorical_cols:
            col_data = self.df[col].dropna()
            value_counts = col_data.value_counts()

            if len(col_data) > 0:
                frequencies = value_counts.values / len(col_data)
                diversity = float(1 - np.sum(frequencies ** 2))
            else:
                diversity = 0.0

            distributions[col] = {
                "unique_values": int(len(value_counts)),
                "top_categories": value_counts.head(5).to_dict(),
                "diversity_index": diversity,
                "missing_count": int(self.df[col].isna().sum()),
                "total_samples": int(len(col_data)),
            }
        return distributions
