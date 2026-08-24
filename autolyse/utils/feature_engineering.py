"""Automated feature engineering module."""

from typing import Optional

import numpy as np
import pandas as pd


class FeatureEngineer:
    """Create bounded sets of derived numeric features from existing ones.

    Supported transforms:
    - polynomial squares/cubes of the highest-variance columns
    - standardized interaction products of moderately correlated pairs
    - safe ratios (denominator must be almost always non-zero)
    - log1p of strictly-positive right-skewed columns
    - row-wise aggregates over the top-variance columns

    All operations are capped by ``max_features`` so callers cannot explode
    their memory footprint.
    """

    def __init__(self, df: pd.DataFrame, random_state: int = 42):
        self.df = df.copy()
        self.random_state = random_state
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = [
            c for c in df.columns if df[c].dtype.name in ("object", "str", "category")
        ]
        self.engineered_features = {}

    def engineer_features(self,
                          polynomial_degree: int = 2,
                          include_interactions: bool = True,
                          include_ratios: bool = True,
                          include_logs: bool = True,
                          max_features: int = 20) -> pd.DataFrame:
        """Return a copy of the frame plus at most ``max_features`` new columns."""
        result_df = self.df.copy()

        if len(self.numeric_cols) < 2 or not result_df.select_dtypes(include=[np.number]).shape[1]:
            return result_df

        polynomial_degree = int(min(max(polynomial_degree, 2), 3))

        budget = max_features

        budget -= self._add_polynomial_features(
            result_df, degree=polynomial_degree,
            n_features=min(3, len(self.numeric_cols)),
            budget=budget,
        )
        if budget > 0 and include_interactions:
            budget -= self._add_interaction_features(result_df, budget)
        if budget > 0 and include_ratios:
            budget -= self._add_ratio_features(result_df, budget)
        if budget > 0 and include_logs:
            budget -= self._add_log_features(result_df, budget)
        if budget > 0:
            budget -= self._add_aggregate_features(result_df, budget)

        # Keep a reference so select_best_features sees engineered columns.
        self.df = result_df
        return result_df

    def _remaining_budget(self, used: int, budget: int) -> int:
        return min(used, max(budget, 0))

    def _add_polynomial_features(self, df: pd.DataFrame, degree: int,
                                 n_features: int, budget: int) -> int:
        """Square/cube top-variance columns; returns features actually added."""
        variances = {c: pd.to_numeric(df[c], errors="coerce").var() for c in self.numeric_cols}
        ranked = sorted(variances.items(), key=lambda kv: kv[1], reverse=True,
                        )[:n_features]
        top_cols = [c for c, v in ranked if pd.notna(v)]

        added = 0
        for col in top_cols:
            for power in range(2, degree + 1):
                if added >= budget:
                    return added
                fname = f"{col}^{power}"
                values = pd.to_numeric(df[col], errors="coerce") ** power
                df[fname] = values
                self.engineered_features[fname] = "polynomial"
                added += 1
        return added

    def _add_interaction_features(self, df: pd.DataFrame, budget: int) -> int:
        corr_matrix = df[self.numeric_cols].corr().fillna(0)

        pairs = []
        for i, col1 in enumerate(self.numeric_cols):
            for col2 in self.numeric_cols[i + 1:]:
                corr = abs(corr_matrix.loc[col1, col2])
                if 0.3 < corr < 0.9:
                    pairs.append((col1, col2))
                if len(pairs) >= budget * 3:  # small pre-filter pool
                    break

        added = 0
        for col1, col2 in pairs:
            if added >= budget:
                break
            fname = f"{col1}_x_{col2}"
            s1 = pd.to_numeric(df[col1], errors="coerce")
            s2 = pd.to_numeric(df[col2], errors="coerce")
            z1 = (s1 - s1.mean()) / (s1.std() or 1.0)
            z2 = (s2 - s2.mean()) / (s2.std() or 1.0)
            df[fname] = z1 * z2
            self.engineered_features[fname] = "interaction"
            added += 1
        return added

    def _add_ratio_features(self, df: pd.DataFrame, budget: int) -> int:
        added = 0
        candidates = self.numeric_cols[:5]
        for col1 in candidates:
            for col2 in candidates:
                if col1 == col2 or added >= budget:
                    continue
                numerator = pd.to_numeric(df[col1], errors="coerce")
                denominator = pd.to_numeric(df[col2], errors="coerce")
                non_zero_share = (denominator.fillna(0) != 0).mean()
                if non_zero_share < 0.95:
                    continue
                fname = f"{col1}_div_{col2}"
                df[fname] = numerator / denominator.replace(0, np.nan)
                self.engineered_features[fname] = "ratio"
                added += 1
        return added

    def _add_log_features(self, df: pd.DataFrame, budget: int) -> int:
        added = 0
        for col in self.numeric_cols:
            if added >= budget:
                break
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(series) == 0 or not (series > 0).all():
                continue
            skew = series.skew()
            if pd.isna(skew) or skew <= 1:
                continue
            fname = f"log_{col}"
            df[fname] = np.log1p(pd.to_numeric(df[col], errors="coerce"))
            self.engineered_features[fname] = "log_transform"
            added += 1
        return added

    def _add_aggregate_features(self, df: pd.DataFrame, budget: int) -> int:
        if len(self.numeric_cols) < 2 or budget < 1:
            return 0

        variances = {
            c: pd.to_numeric(df[c], errors="coerce").var() for c in self.numeric_cols
        }
        top = [c for c, v in sorted(variances.items(),
                                    key=lambda kv: kv[1],
                                    reverse=True)[:3] if pd.notna(v)]
        if len(top) < 2:
            return 0

        block = df[top].apply(pd.to_numeric, errors="coerce")
        specs = [("feature_mean", block.mean(axis=1)),
                 ("feature_std", block.std(axis=1)),
                 ("feature_max", block.max(axis=1))]

        added = 0
        for fname, values in specs:
            if added >= budget:
                break
            df[fname] = values
            self.engineered_features[fname] = "aggregate"
            added += 1
        return added

    def get_engineered_features_summary(self) -> dict:
        """Summary of created features grouped by transformation type."""
        by_type = {}
        for fname, ftype in self.engineered_features.items():
            by_type.setdefault(ftype, []).append(fname)
        return {
            "total_engineered": len(self.engineered_features),
            "by_type": by_type,
            "all_features": list(self.engineered_features.keys()),
        }

    def select_best_features(self, target_col: Optional[str] = None,
                             n_features: int = 10) -> list:
        """Rank engineered features by target correlation or variance."""
        engineered_only = [c for c in self.df.columns if c in self.engineered_features]
        if not engineered_only:
            return []

        scores = {}
        if target_col and target_col in self.df.columns:
            for col in engineered_only:
                corr = abs(pd.to_numeric(self.df[col], errors="coerce")
                           .corr(pd.to_numeric(self.df[target_col], errors="coerce")))
                scores[col] = corr if pd.notna(corr) else 0.0
        else:
            for col in engineered_only:
                var = pd.to_numeric(self.df[col], errors="coerce").var()
                scores[col] = var if pd.notna(var) else 0.0

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [col for col, _ in ranked[:n_features]]
