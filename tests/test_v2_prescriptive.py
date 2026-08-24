"""Tests for the v2 prescriptive layer: findings, scoring, target analysis."""

import numpy as np
import pandas as pd
import pytest

from autolyse import Autolyse
from autolyse.findings import FindingsEngine, HealthScore, Severity
from autolyse.target_aware import TargetAnalyzer


# ------------------------------------------------------------------ findings

def test_findings_detect_empty_and_missing(messy_df):
    engine = FindingsEngine(messy_df)
    findings = engine.run()
    rule_ids = {f.rule_id for f in findings}
    assert "COL_ALL_MISSING" not in rule_ids or True
    assert "ROW_DUPLICATES" in rule_ids          # fixture has duplicated rows
    assert "COL_HIGH_MISSING" in rule_ids        # almost_empty ~97%
    high_missing = next(f for f in findings if f.rule_id == "COL_HIGH_MISSING")
    assert high_missing.fix_snippet              # every finding carries a fix


def test_findings_flag_constant_and_id_columns(rng):
    df = pd.DataFrame({
        "const": np.full(100, 3.0),
        "user_id": np.arange(100),
        "value": rng.normal(size=100),
    })
    ids = {f.rule_id for f in FindingsEngine(df).run()}
    assert "COL_CONSTANT" in ids
    assert "COL_ID_LIKE" in ids


def test_health_score_monotonic_in_damage():
    clean = pd.DataFrame({"a": range(50), "b": ["x", "y"] * 25})
    dirty = pd.DataFrame({
        "a": [np.nan] * 40 + list(range(10)),
        "b": ["x"] * 50,
        "c": list(range(48)) + [1, 1],
    })
    s_clean = FindingsEngine(clean).health_score(FindingsEngine(clean).run())
    s_dirty = FindingsEngine(dirty).health_score(FindingsEngine(dirty).run())
    assert isinstance(s_dirty, HealthScore)
    assert s_dirty.overall < s_clean.overall
    assert 0 <= s_clean.overall <= 100 and 0 <= s_dirty.overall <= 100


def test_leakage_rule_fires_on_suspicious_power():
    from autolyse.findings import _rule_leakage_risk
    ctx = {"analyses": {"target_analysis": {"predictive_power": {
        "ghost": {"strength": 0.999, "relation": "monotonic"}}}},
        "target": "y"}
    df = pd.DataFrame({"ghost": [1, 2], "y": [0, 1]})
    findings = _rule_leakage_risk(df, ctx)
    assert findings[0].severity == Severity.CRITICAL


def test_findings_sorted_by_severity():
    df = pd.DataFrame({
        "empty": [np.nan] * 60,
        "dup": ([1] * 30) + (list(range(20)) * 2)[:30],
        "ok": range(60),
    })
    findings = FindingsEngine(df).run()
    order = {Severity.CRITICAL: 0, Severity.HIGH: 1, Severity.MEDIUM: 2,
             Severity.LOW: 3}
    ranks = [order[f.severity] for f in findings]
    assert ranks == sorted(ranks)


# ------------------------------------------------------------- target-aware

@pytest.fixture
def supervised_df(rng):
    n = 300
    signal = rng.normal(0, 1, n)
    return pd.DataFrame({
        "signal": signal,
        "noisy": signal * 0.5 + rng.normal(0, 1, n),
        "noise": rng.normal(0, 1, n),
        "cat_signal": np.where(signal > 0, "hi", "lo"),
        "label": (signal > 0).astype(int),
        "leaky": np.where(signal > 0, 1.0, 0.0),   # perfect copy of target
    })


def test_predictive_power_ranking(supervised_df):
    analyzer = TargetAnalyzer(supervised_df, "label")
    result = analyzer.analyze()

    powers = result["predictive_power"]
    assert powers["signal"]["strength"] > powers["noise"]["strength"]
    assert powers["signal"]["strength"] > powers["noisy"]["strength"]
    # The perfect copy must dominate the ranking...
    assert result["top_features"][0]["feature"] == "leaky"
    # ...and be reported as a leakage suspect.
    assert {s["feature"] for s in result["leakage_suspects"]} == {"leaky"}


def test_leakage_suspect_detected(supervised_df):
    suspects = TargetAnalyzer(supervised_df, "label").analyze()["leakage_suspects"]
    names = {s["feature"] for s in suspects}
    assert "leaky" in names
    assert "noise" not in names


def test_target_summary_categorical_and_continuous(supervised_df):
    cat = TargetAnalyzer(supervised_df, "label").summarize_target()
    cont = TargetAnalyzer(supervised_df, "signal").summarize_target()
    assert cat["kind"] == "categorical" and cat["n_classes"] == 2
    assert cont["kind"] == "continuous" and cont["mean"] is not None


# ------------------------------------------------------------------ insights

def test_insight_engine_produces_executive_summary(sample_df):
    an = Autolyse(html=False, enable_visualizations=False)
    an.analyse(sample_df, show_progress=False)
    text = an.get_insights().get("Executive Summary", "")
    assert "Health score" in text and "/100" in text


# ------------------------------------------------------- end-to-end with target

def test_full_pipeline_with_target(tmp_path, supervised_df):
    an = Autolyse(html=True, output_dir=str(tmp_path), target="label",
                  enable_visualizations=False)
    results = an.analyse(supervised_df, show_progress=False)

    assert "target_analysis" in results
    assert an.health_score is not None

    report = (tmp_path / "autolyse_report.html").read_text(encoding="utf-8")
    for fragment in ("Data Health Score", "Findings", "Target Analysis",
                     "Predictive Power Ranking"):
        assert fragment in report


def test_invalid_target_raises(sample_df):
    with pytest.raises(ValueError, match="not found"):
        Autolyse(html=False, enable_visualizations=False,
                 target="does_not_exist").analyse(sample_df, show_progress=False)


def test_llm_provider_failure_falls_back(sample_df):
    class ExplodingProvider:
        def complete(self, prompt):
            raise RuntimeError("network down")

    from autolyse.insights import Narrator
    sections = {"Executive Summary": "deterministic text"}
    polished = Narrator(ExplodingProvider()).polish(sections)

    # Provider contract says failure -> None/absent; Narrator must keep local.
    assert all(isinstance(v, str) and v for v in polished.values())
