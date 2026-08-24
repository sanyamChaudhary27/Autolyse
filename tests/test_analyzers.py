"""Analyzer correctness tests against known statistical ground truth."""

import numpy as np
import pandas as pd
import pytest

from autolyse.analyzers import (
    CorrelationAnalyzer,
    DistributionAnalyzer,
    MissingValuesAnalyzer,
    OutlierAnalyzer,
    RelationshipsAnalyzer,
    StatisticalAnalyzer,
)


# ---------------------------------------------------------------- statistics

def test_statistical_values_match_ground_truth(sample_df):
    result = StatisticalAnalyzer(sample_df).analyze()
    col = sample_df["age"]
    s = result["age"]
    assert s["mean"] == pytest.approx(col.mean())
    assert s["median"] == pytest.approx(col.median())
    assert s["std"] == pytest.approx(col.std())
    assert s["q25"] == pytest.approx(col.quantile(0.25))
    assert s["iqr"] == pytest.approx(col.quantile(0.75) - col.quantile(0.25))
    assert s["skewness"] == pytest.approx(col.skew())
    assert s["null_count"] == 0


def test_statistical_handles_missing(messy_df):
    result = StatisticalAnalyzer(messy_df).analyze()
    assert result["skewed"]["null_count"] >= 30
    assert result["skewed"]["count"] == messy_df["skewed"].notna().sum()


def test_constant_column_statistics():
    df = pd.DataFrame({"const": [5.0] * 50})
    s = StatisticalAnalyzer(df).analyze()["const"]
    assert s["std"] == 0 or pd.isna(s["std"])
    assert s["min"] == s["max"] == 5.0


# ------------------------------------------------------------ missing values

def test_missing_values_counts(sample_df, messy_df):
    clean = MissingValuesAnalyzer(sample_df).analyze()
    assert clean["total_missing"] == 0 and clean["no_missing"] is True

    messy = MissingValuesAnalyzer(messy_df).analyze()
    assert messy["missing_count"]["almost_empty"] >= 146
    assert messy["missing_percentage"]["almost_empty"] == pytest.approx(97.33, abs=0.5)
    assert set(messy["completely_missing_cols"]) == set()


# -------------------------------------------------------------- distribution

def test_normal_vs_exponential_classification(rng):
    df = pd.DataFrame(
        {
            "normalish": rng.normal(0, 1, 2000),
            "exponential": rng.exponential(2.0, 2000),
        }
    )
    dists = DistributionAnalyzer(df).analyze_numeric_distributions()
    assert dists["normalish"]["is_normal"] is True
    assert "skewed" in dists["exponential"]["distribution_type"].lower()
    assert dists["exponential"]["skewness"] > 1


def test_normality_survives_constant_column():
    """Constant columns must not crash the normality test (Shapiro fails there)."""
    df = pd.DataFrame({"const": np.full(500, 3.14), "vary": np.linspace(0, 1, 500)})
    dists = DistributionAnalyzer(df).analyze_numeric_distributions()
    assert set(dists.keys()) == {"const", "vary"}
    assert isinstance(dists["const"]["normality_pvalue"], float)
    assert not pd.isna(dists["vary"]["normality_pvalue"])


def test_categorical_distribution_diversity(sample_df):
    dists = DistributionAnalyzer(sample_df).analyze_categorical_distributions()
    city = dists["city"]
    assert city["unique_values"] == 4
    assert 0 < city["diversity_index"] <= 0.75
    assert sum(city["top_categories"].values()) > 100


# ------------------------------------------------------------------ outliers

def test_iqr_outliers_known_injection(rng):
    data = np.concatenate([rng.normal(0, 1, 300), [1000.0, -1000.0]])
    df = pd.DataFrame({"x": data})
    res = OutlierAnalyzer(df).detect_iqr_outliers()["x"]
    # The injected extremes must always be flagged; ordinary tail points may
    # legitimately cross the fences too.
    assert res["outlier_count"] >= 2
    assert 1000.0 in res["outlier_values"]
    assert -1000.0 in res["outlier_values"]
    # Fences must sit strictly inside the injected extremes.
    assert res["lower_bound"] > -1000.0
    assert res["upper_bound"] < 1000.0
    assert res["lower_bound"] < res["upper_bound"]


def test_isolation_forest_summary_shape(rng):
    df = pd.DataFrame({"a": rng.normal(0, 1, 400), "b": rng.normal(5, 2, 400)})
    iso = OutlierAnalyzer(df).detect_isolation_forest_outliers()
    assert "n_outliers" in iso
    assert 0 <= iso["n_outliers"] <= 400
    # Memory safety: full per-row score arrays must never be stored.
    assert "anomaly_scores" not in iso or len(iso["anomaly_scores"]) < 50


# --------------------------------------------------------------- correlation

def test_correlation_matrix_and_strength_buckets(rng):
    x = rng.normal(0, 1, 500)
    df = pd.DataFrame({"x": x, "y": 3 * x + rng.normal(0, 0.1, 500), "z": rng.normal(0, 1, 500)})
    summary = CorrelationAnalyzer(df).get_correlation_summary()

    pearson = summary["pearson"]["correlation_matrix"]
    assert pearson.loc["x", "y"] == pytest.approx(pearson.loc["y", "x"])
    assert pearson.loc["x", "x"] == pytest.approx(1.0)

    strong_pairs = {(c["col1"], c["col2"]) for c in summary["pearson"]["strong_correlations"]}
    assert ("x", "y") in strong_pairs or ("y", "x") in strong_pairs
    assert all(abs(c["correlation"]) > 0.7 for c in summary["pearson"]["strong_correlations"])
    assert all(
        0.5 < abs(c["correlation"]) <= 0.7 for c in summary["pearson"]["moderate_correlations"]
    )


def test_correlation_single_column_returns_empty_structure():
    summary = CorrelationAnalyzer(pd.DataFrame({"only": [1, 2, 3]})).get_correlation_summary()
    assert summary["pearson"]["correlation_matrix"] is None
    assert summary["pearson"]["strong_correlations"] == []


# ------------------------------------------------------------- relationships

def test_cramers_v_bounded_and_ordered(rng):
    """Perfect association => 1.0; independence => ~0; symmetric."""
    g = rng.choice(["a", "b"], 2000)
    perfect = RelationshipsAnalyzer._cramers_v(pd.Series(g), pd.Series(g))
    indep = RelationshipsAnalyzer._cramers_v(pd.Series(g), rng.choice(["x", "y"], 2000))
    assert perfect == pytest.approx(1.0, abs=1e-9)
    assert 0.0 <= indep <= 0.15
    v1 = RelationshipsAnalyzer._cramers_v(pd.Series(g), pd.Series(g[::-1]))
    assert 0 <= v1 <= 1.0


def test_categorical_numeric_grouping(sample_df):
    rel = RelationshipsAnalyzer(sample_df).analyze_categorical_numeric_relationships()
    group = rel["plan"]["income"]["pro"]
    assert group["count"] > 0
    assert group["max"] >= group["median"] >= group["min"]


def test_numeric_pair_ranking_by_abs_correlation(rng):
    x = rng.normal(0, 1, 300)
    df = pd.DataFrame({"x": x, "twin": -2 * x, "noise": rng.normal(0, 1, 300)})
    pairs = RelationshipsAnalyzer(df).analyze_numeric_numeric_relationships()["numeric_pairs"]
    assert pairs[0]["relationship_strength"] == "Very Strong"
    assert pairs[0]["col1"] in {"x", "twin"} and pairs[0]["col2"] in {"x", "twin"}


def test_strength_labels():
    f = RelationshipsAnalyzer._get_strength_label
    assert f(0.1) == "Very Weak"
    assert f(0.3) == "Weak"
    assert f(0.5) == "Moderate"
    assert f(0.7) == "Strong"
    assert f(0.95) == "Very Strong"
