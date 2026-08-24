"""Deterministic insight engine with optional LLM narration.

Design principles:
1. The local engine always produces a full, evidence-based narrative - zero
   network dependency.
2. An LLM provider (if configured) receives the computed facts and returns
   polished prose. If it fails or is absent, the deterministic text ships.
3. Providers are pluggable via a tiny interface; Gemini is one implementation,
   not the architecture.
"""

from typing import Any, Dict, List, Optional, Protocol

import numpy as np
import pandas as pd


class LLMProvider(Protocol):
    """Minimal contract any narration backend must satisfy."""

    def complete(self, prompt: str) -> Optional[str]:
        """Return generated text, or None on any failure."""
        ...


class GeminiProvider:
    """Gemini implementation of LLMProvider (lazy import, safe failure)."""

    def __init__(self, api_key: Optional[str] = None,
                 model_name: str = "gemini-2.0-flash"):
        self.api_key = api_key
        self.model_name = model_name
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel(self.model_name)
        return self._model

    def complete(self, prompt: str) -> Optional[str]:
        try:
            response = self._ensure_model().generate_content(prompt)
            return getattr(response, "text", "").strip() or None
        except Exception:
            return None


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:,.3g}"
    return f"{value:,}" if isinstance(value, int) else str(value)


class InsightEngine:
    """Build the narrative from analyses, findings and scores."""

    def __init__(self, df: pd.DataFrame, analyses: Dict[str, Any],
                 findings: List, health_score,
                 validation: Dict[str, Any],
                 target_analysis: Optional[Dict[str, Any]] = None):
        self.df = df
        self.analyses = analyses
        self.findings = findings
        self.health_score = health_score
        self.validation = validation
        self.target_analysis = target_analysis

    # ------------------------------------------------------------- sections

    def executive_summary(self) -> str:
        rows, cols = self.df.shape
        types = self.validation.get("column_types", {})
        critical = [f for f in self.findings if f.severity.value == "critical"]
        high = [f for f in self.findings if f.severity.value == "high"]

        parts = [
            f"{rows:,} rows x {cols} columns "
            f"({types.get('numeric', 0)} numeric, {types.get('categorical', 0)} "
            f"categorical). Health score {self.health_score.overall}/100 "
            f"(grade {self.health_score.grade})."
        ]
        if critical:
            names = ", ".join(f"'{f.columns[0] if f.columns else f.rule_id}'"
                              for f in critical[:3])
            parts.append(f"CRITICAL: {names} must be resolved first.")
        if high:
            parts.append(f"{len(high)} high-severity issue(s) need attention.")
        if not critical and not high:
            parts.append("No blocking issues detected - safe to proceed to modeling.")

        ta = self.target_analysis or {}
        top = (ta.get("top_features") or [])
        if top:
            best = top[0]
            parts.append(f"Strongest signal for the target: "
                         f"'{best['feature']}' (power {best['strength']:.2f}).")
        return " ".join(parts)

    def data_quality_narrative(self) -> str:
        if not self.findings:
            return ("No data-quality issues met reporting thresholds.")
        lines = []
        for finding in self.findings[:8]:
            marker = {"critical": "[CRITICAL]", "high": "[HIGH]",
                      "medium": "[MED]", "low": "[LOW]"}[finding.severity.value]
            lines.append(f"- {marker} {finding.title}. {finding.detail}")
        more = len(self.findings) - 8
        if more > 0:
            lines.append(f"... and {more} more findings below.")
        category_scores = self.health_score.by_category
        weakest = min(category_scores, key=category_scores.get) \
            if category_scores else None
        text = "\n".join(lines)
        if weakest:
            text += (f"\nWeakest dimension: {weakest} "
                     f"({category_scores[weakest]}/100).")
        return text

    def correlation_narrative(self) -> Optional[str]:
        corr = self.analyses.get("correlations", {}).get("pearson", {})
        strong = corr.get("strong_correlations", [])
        if not strong:
            return None
        pairs = ", ".join(
            f"{c['col1']}~{c['col2']} (r={c['correlation']:.2f})"
            for c in strong[:4]
        )
        return (f"{len(strong)} strong correlation pair(s): {pairs}. "
                f"For linear models consider dropping or combining redundant "
                f"features; for trees this matters less.")

    def outlier_narrative(self) -> Optional[str]:
        outliers = self.analyses.get("outliers", {})
        iqr = outliers.get("iqr_method", {})
        worst = sorted(((c, d["outlier_percentage"]) for c, d in iqr.items()),
                       key=lambda kv: kv[1], reverse=True)[:3]
        flagged = [(c, p) for c, p in worst if p > 0]
        if not flagged:
            return None
        detail = ", ".join(f"'{c}' ({p:.1f}%)" for c, p in flagged)
        iso_n = outliers.get("isolation_forest", {}).get("n_outliers")
        text = f"Columns beyond Tukey fences: {detail}."
        if iso_n:
            text += (f" Isolation Forest additionally flags {iso_n} rows "
                     f"multivariately.")
        return text

    def distribution_narrative(self) -> Optional[str]:
        dists = self.analyses.get("distributions", {}) \
            .get("numeric_distributions", {})
        skewed = [(c, d["skewness"]) for c, d in dists.items()
                  if abs(d.get("skewness", 0)) > 1]
        if not skewed:
            return None
        names = ", ".join(f"'{c}' ({s:.1f})" for c, s in skewed[:5])
        return (f"Heavily skewed numeric columns: {names}. Consider log or "
                f"Box-Cox transforms before linear modeling.")

    def target_narrative(self) -> Optional[str]:
        if not self.target_analysis:
            return None
        summary = self.target_analysis.get("target_summary", {})
        if summary.get("kind") == "categorical":
            minority = summary.get("minority_pct")
            classes = summary.get("n_classes")
            text = (f"Target '{summary['column']}' has {classes} class(es); ")
            if minority is not None:
                text += (f"minority share {minority:.1f}%. "
                         if minority < 15 else "reasonably balanced. ")
        else:
            stats = {k: summary.get(k) for k in ("mean", "std")}
            text = (f"Continuous target '{summary['column']}' "
                    f"(mean {_fmt(stats['mean'])}, sd {_fmt(stats['std'])}). ")

        top = self.target_analysis.get("top_features") or []
        if top:
            signals = ", ".join(f"{t['feature']} ({t['strength']:.2f})"
                                for t in top[:3])
            text += f"Top predictive features: {signals}."
        suspects = self.target_analysis.get("leakage_suspects") or []
        if suspects:
            names = ", ".join(s["feature"] for s in suspects[:3])
            text += f" LEAKAGE WARNING: '{names}' predict almost perfectly."
        return text

    # ---------------------------------------------------------------- output

    def build(self) -> Dict[str, str]:
        """All narrative sections keyed by report heading."""
        sections = {"Executive Summary": self.executive_summary()}
        mapping = {
            "Data Quality": self.data_quality_narrative(),
            "Target Analysis": self.target_narrative(),
            "Relationships": self.correlation_narrative(),
            "Outliers": self.outlier_narrative(),
            "Distributions": self.distribution_narrative(),
        }
        sections.update({k: v for k, v in mapping.items() if v})
        return sections


class Narrator:
    """Optionally rewrites deterministic narratives through an LLM."""

    def __init__(self, provider: Optional[LLMProvider]):
        self.provider = provider

    def polish(self, sections: Dict[str, str]) -> Dict[str, str]:
        """Return LLM-refined sections, falling back per-section on failure."""
        if self.provider is None:
            return sections
        refined = {}
        for title, text in sections.items():
            improved = self.provider.complete(
                f"Rewrite this data-analysis finding for a business reader in "
                f"at most 4 sentences. Keep every number exactly as given.\n\n"
                f"SECTION: {title}\n{text}"
            )
            refined[title] = improved if improved else text
        return refined
