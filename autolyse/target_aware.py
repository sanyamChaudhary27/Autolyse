"""Target-aware analysis: signal ranking and leakage detection.

Given an optional target column, ranks every feature by a type-appropriate
association strength and flags features that predict the target too well -
the classic signature of data leakage.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from autolyse.analyzers.relationships import MAX_CATEGORICAL_LEVELS


def _is_categorical(series: pd.Series) -> bool:
    if series.dtype.name in ("object", "str", "category", "bool"):
        return True
    return series.nunique(dropna=True) <= 12


def _correlation_ratio(categories: np.ndarray, values: np.ndarray) -> float:
    """Eta: 0..1 measure of how much of var(values) the groups explain."""
    overall_mean = values.mean()
    groups = pd.Series(values).groupby(pd.Series(categories)).groups
    ss_between = 0.0
    for _, idx in groups.items():
        group_vals = values[np.asarray(list(idx))]
        if len(group_vals):
            ss_between += len(group_vals) * (group_vals.mean() - overall_mean) ** 2
    ss_total = ((values - overall_mean) ** 2).sum()
    return float(np.sqrt(ss_between / ss_total)) if ss_total > 0 else 0.0


def _numeric_numeric(x: pd.Series, y: pd.Series) -> float:
    pearson = abs(x.corr(y))
    spearman = abs(x.corr(y, method="spearman"))
    best = max(v for v in (pearson, spearman) if pd.notna(v))
    return float(best)


def _eta_squared(groups_values: list) -> float:
    """Between-group variance share for cat->num association."""
    all_values = np.concatenate(groups_values)
    grand = all_values.mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups_values)
    ss_total = ((all_values - grand) ** 2).sum()
    return float(ss_between / ss_total) if ss_total > 0 else 0.0


class TargetAnalyzer:
    """Analyze feature-to-target relationships when a target is provided."""

    def __init__(self, df: pd.DataFrame, target: str,
                 column_types: Optional[Dict[str, list]] = None):
        if target not in df.columns:
            raise KeyError(f"Target column '{target}' not found")
        self.df = df
        self.target = target
        self.target_is_categorical = _is_categorical(df[target])

        types = column_types or {}
        self.numeric_cols = [
            c for c in types.get("numeric", df.select_dtypes(include=[np.number]).columns)
            if c != target
        ]
        self.categorical_cols = [
            c for c in types.get("categorical", [])
            if c != target and df[c].nunique() <= MAX_CATEGORICAL_LEVELS
        ]

    # ------------------------------------------------------------------ API

    def summarize_target(self) -> Dict[str, Any]:
        series = self.df[self.target]
        summary = {
            "column": self.target,
            "kind": "categorical" if self.target_is_categorical else "continuous",
            "missing": int(series.isna().sum()),
        }
        counts = series.value_counts()
        if self.target_is_categorical:
            summary["classes"] = {str(k): int(v) for k, v in counts.head(10).items()}
            summary["n_classes"] = int(len(counts))
            minority_pct = counts.min() / counts.sum() * 100 if len(counts) else 0
            summary["minority_pct"] = round(float(minority_pct), 2)
        else:
            summary.update({
                "mean": float(series.mean()) if series.notna().any() else None,
                "std": float(series.std()) if series.notna().any() else None,
                "skewness": float(series.skew()) if series.notna().any() else None,
            })
        return summary

    def rank_predictive_power(self) -> Dict[str, Dict[str, Any]]:
        """Association strength of each feature with the target (0..1)."""
        powers: Dict[str, Dict[str, Any]] = {}

        for col in self.numeric_cols:
            pair = self.df[[col, self.target]].dropna()
            if len(pair) < 20:
                continue
            try:
                if self.target_is_categorical:
                    strength = _correlation_ratio(
                        pair[self.target].to_numpy(),
                        pair[col].to_numpy(dtype=float),
                    )
                    relation = "cat_target_eta"
                else:
                    strength = _numeric_numeric(pair[col], pair[self.target])
                    relation = "monotonic"
            except Exception:
                continue
            powers[col] = {"strength": round(strength, 4), "relation": relation}

        for col in self.categorical_cols:
            pair = self.df[[col, self.target]].dropna()
            if len(pair) < 20 or pair[col].nunique() < 2:
                continue
            try:
                if self.target_is_categorical:
                    from autolyse.analyzers.relationships import RelationshipsAnalyzer
                    strength = RelationshipsAnalyzer._cramers_v(
                        pair[col], pair[self.target])
                    relation = "categorical_cramers_v"
                else:
                    grouped = [g.to_numpy(dtype=float)
                               for _, g in pair.groupby(col)[self.target]
                               if len(g)]
                    strength = _eta_squared(grouped)
                    relation = "num_target_eta_sq"
            except Exception:
                continue
            powers[col] = {"strength": round(strength, 4), "relation": relation}

        return dict(sorted(powers.items(),
                           key=lambda kv: kv[1]["strength"],
                           reverse=True))

    def top_signals(self, n: int = 10) -> list:
        powers = self.rank_predictive_power()
        return [{"feature": name, **info} for name, info in powers[:n].items()]

    def analyze(self) -> Dict[str, Any]:
        powers = self.rank_predictive_power()

        # Leakage candidates are handled by findings.LEAKAGE_RISK; here we
        # simply surface them so reports can show the evidence inline.
        leakage_suspects = [
            {"feature": name, "strength": info["strength"]}
            for name, info in powers.items()
            if info["strength"] >= 0.98
        ]

        return {
            "target_summary": self.summarize_target(),
            "predictive_power": powers,
            "top_features": [
                {"feature": name, **info}
                for name, info in list(powers.items())[:10]
            ],
            "leakage_suspects": leakage_suspects,
        }
