"""LLM-backed insight narration with deterministic fallbacks.

The Gemini path is strictly optional: without an API key every method
returns a locally-computed summary so reports never depend on network access.
"""

import os
import warnings
from typing import Any, Dict, List, Optional

DEFAULT_MODEL = "gemini-2.0-flash"


class GeminiInsights:
    """Generate insights via Google Gemini, or deterministic text offline."""

    def __init__(self, api_key: Optional[str] = None,
                 model_name: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_KEY")
        self.model_name = model_name or os.environ.get(
            "AUTOLYSE_GEMINI_MODEL", DEFAULT_MODEL
        )
        self.model = None
        self.available = False

        if self.api_key:
            self._initialize_client()

    def _initialize_client(self) -> None:
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            self.available = True
        except Exception as error:
            warnings.warn(
                f"Gemini unavailable ({error}); falling back to local summaries."
            )
            self.available = False

    def is_available(self) -> bool:
        return self.available

    # ------------------------------------------------------------- generators

    def generate_statistics_insight(self, stats: Dict[str, Any], column: str) -> str:
        if not self.available:
            return self._fallback_statistics_insight(stats, column)
        prompt = (
            f"Analyze these statistics for column '{column}' in 2-3 sentences:\n"
            f"- Mean: {stats.get('mean')}\n- Median: {stats.get('median')}\n"
            f"- Std Dev: {stats.get('std')}\n- Min: {stats.get('min')}\n"
            f"- Max: {stats.get('max')}\n- Skewness: {stats.get('skewness')}\n"
            f"- Kurtosis: {stats.get('kurtosis')}\n"
            f"- Missing %: {stats.get('null_percentage', 0):.2f}%\n"
            "Be concrete and actionable."
        )
        return self._complete(prompt) or self._fallback_statistics_insight(stats, column)

    def generate_missing_values_insight(self, missing_analysis: Dict[str, Any]) -> str:
        if not self.available:
            return self._fallback_missing_insight(missing_analysis)
        prompt = (
            f"Assess data quality in 2-3 sentences given: "
            f"total missing values = {missing_analysis.get('total_missing', 0)}, "
            f"rows affected = {missing_analysis.get('missing_rows', 0)}, "
            f"empty columns = {len(missing_analysis.get('completely_missing_cols', []))}. "
            f"Advise whether imputation or removal makes sense."
        )
        return self._complete(prompt) or self._fallback_missing_insight(missing_analysis)

    def generate_correlation_insight(self,
                                     strong_correlations: List[Dict],
                                     moderate_correlations: List[Dict]) -> str:
        if not self.available:
            return self._fallback_correlation_insight(strong_correlations,
                                                      moderate_correlations)
        pairs = ", ".join(
            f"{c['col1']}-{c['col2']} ({c['correlation']:.2f})"
            for c in strong_correlations[:5]
        )
        prompt = (
            f"In 2-3 sentences interpret these Pearson correlations: "
            f"{len(strong_correlations)} strong pairs [{pairs}], "
            f"{len(moderate_correlations)} moderate pairs. "
            f"Flag multicollinearity risks if any."
        )
        return (self._complete(prompt)
                or self._fallback_correlation_insight(strong_correlations,
                                                      moderate_correlations))

    def generate_outlier_insight(self, iqr_results: Dict[str, Any],
                                 iso_forest_results: Dict[str, Any]) -> str:
        if not self.available:
            return self._fallback_outlier_insight(iqr_results, iso_forest_results)
        total = sum(d.get("outlier_count", 0) for d in iqr_results.values())
        iso = iso_forest_results.get("n_outliers", 0)
        prompt = (
            f"In 2-3 sentences interpret outlier findings: IQR flagged {total} "
            f"univariate outliers across {len(iqr_results)} columns; Isolation "
            f"Forest flagged {iso} multivariate rows. Advise on next steps."
        )
        return (self._complete(prompt)
                or self._fallback_outlier_insight(iqr_results, iso_forest_results))

    def generate_distribution_insight(self, numeric_distributions: Dict[str, Any],
                                      categorical_distributions: Dict[str, Any]) -> str:
        if not self.available:
            return self._fallback_distribution_insight(numeric_distributions,
                                                       categorical_distributions)
        normal_count = sum(1 for d in numeric_distributions.values()
                           if d.get("is_normal"))
        prompt = (
            f"In 2-3 sentences summarize: {normal_count}/{len(numeric_distributions)} "
            f"numeric columns are approximately normal; "
            f"{len(categorical_distributions)} categorical columns present. "
            f"Mention transformation candidates if skewness is high."
        )
        return (self._complete(prompt)
                or self._fallback_distribution_insight(numeric_distributions,
                                                       categorical_distributions))

    def generate_general_insight(self, df_shape: tuple,
                                 column_types: Dict[str, int],
                                 data_quality_score: float) -> str:
        if not self.available:
            return self._fallback_general_insight(df_shape, column_types,
                                                  data_quality_score)
        prompt = (
            f"Give a 2-3 sentence overall assessment of a dataset with shape "
            f"{df_shape[0]}x{df_shape[1]}, column types {column_types}, and "
            f"data quality score {data_quality_score:.0f}/100. Suggest an "
            f"analysis direction."
        )
        return (self._complete(prompt)
                or self._fallback_general_insight(df_shape, column_types,
                                                  data_quality_score))

    # ---------------------------------------------------------------- plumbing

    def _complete(self, prompt: str) -> Optional[str]:
        try:
            response = self.model.generate_content(prompt)
            text = getattr(response, "text", "")
            return text.strip() or None
        except Exception as error:
            warnings.warn(f"Gemini call failed ({error}); using local summary.")
            return None

    # ------------------------------------------------------- local fallbacks

    @staticmethod
    def _fallback_statistics_insight(stats: Dict[str, Any], column: str) -> str:
        skewness = stats.get("skewness") or 0
        std = stats.get("std") or 0
        mean = stats.get("mean") or 0

        abs_skew = abs(skewness)
        skew_desc = ("highly skewed" if abs_skew > 1
                     else "moderately skewed" if abs_skew > 0.5
                     else "approximately symmetric")
        cv = std / mean if mean else float("inf")
        var_desc = ("very low variability" if cv < 0.1
                    else "moderate to high variability")

        return (f"'{column}' shows a {skew_desc} distribution with {var_desc}; "
                f"{stats.get('null_percentage', 0):.1f}% of values are missing.")

    @staticmethod
    def _fallback_missing_insight(missing_analysis: Dict[str, Any]) -> str:
        total = missing_analysis.get("total_missing", 0)
        if missing_analysis.get("no_missing"):
            return ("No missing values detected - the dataset is complete and "
                    "ready for analysis.")
        empty_cols = missing_analysis.get("completely_missing_cols", [])
        advice = ("consider dropping them" if empty_cols
                  else "investigate whether missingness is random before imputing")
        return (f"{total} missing values found across the dataset; for fully "
                f"empty columns {advice}.")

    @staticmethod
    def _fallback_correlation_insight(strong_corr: List[Dict],
                                      moderate_corr: List[Dict]) -> str:
        if not strong_corr and not moderate_corr:
            return ("No strong or moderate correlations - variables behave "
                    "largely independently.")
        detail = ""
        if strong_corr:
            top = strong_corr[0]
            detail = (f" Strongest pair: {top['col1']} vs {top['col2']} "
                      f"(r={top['correlation']:.2f}).")
        return (f"{len(strong_corr)} strong and {len(moderate_corr)} moderate "
                f"correlations detected.{detail} Watch for redundancy in modeling.")

    @staticmethod
    def _fallback_outlier_insight(iqr_results: Dict[str, Any],
                                  iso_forest: Dict[str, Any]) -> str:
        total = sum(d.get("outlier_count", 0) for d in iqr_results.values())
        worst = max(
            ((col, d["outlier_percentage"]) for col, d in iqr_results.items()),
            key=lambda kv: kv[1], default=(None, 0),
        )
        if total == 0:
            return "No outliers detected under Tukey fences; values look tame."
        base = (f"{total} univariate outliers detected"
                + (f", concentrated in '{worst[0]}' ({worst[1]:.1f}%)"
                   if worst[0] else "") + ". ")
        if iso_forest.get("n_outliers"):
            base += (f"Isolation Forest flags {iso_forest['n_outliers']} rows as "
                     f"multivariate anomalies; verify they are not entry errors.")
        else:
            base += "Verify whether extremes are errors or genuine tail values."
        return base

    @staticmethod
    def _fallback_distribution_insight(numeric_dists: Dict[str, Any],
                                       categorical_dists: Dict[str, Any]) -> str:
        if not numeric_dists:
            return "No numeric columns available for distribution analysis."
        normal = sum(1 for d in numeric_dists.values() if d.get("is_normal"))
        pct = normal / len(numeric_dists) * 100
        skewed = [c for c, d in numeric_dists.items() if abs(d.get("skewness", 0)) > 1]
        extra = f" Highly skewed: {', '.join(list(skewed)[:4])}." if skewed else ""
        return (f"{normal}/{len(numeric_dists)} numeric columns (~{pct:.0f}%) are "
                f"approximately normal.{extra}")

    @staticmethod
    def _fallback_general_insight(df_shape: tuple, column_types: Dict[str, int],
                                  quality_score: float) -> str:
        rows, cols = df_shape
        grade = ("excellent" if quality_score >= 80
                 else "good" if quality_score >= 60
                 else "fair" if quality_score >= 40 else "poor")
        return (f"Dataset of {rows:,} rows x {cols} columns with {grade} quality "
                f"({quality_score:.0f}/100): "
                f"{column_types.get('numeric', 0)} numeric, "
                f"{column_types.get('categorical', 0)} categorical, "
                f"{column_types.get('datetime', 0)} datetime, "
                f"{column_types.get('text', 0)} text columns.")
