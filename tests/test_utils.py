"""Tests for utils: type detection, feature engineering, insight fallbacks."""

import numpy as np
import pandas as pd
import pytest

from autolyse.utils import DataPreparation, FeatureEngineer, GeminiInsights


# ---------------------------------------------------------- data preparation

def test_type_detection_mixed_frame(sample_df):
    types = DataPreparation(sample_df).get_column_types()
    assert types["age"] == "numeric"
    assert types["city"] == "categorical"
    assert types["plan"] == "categorical"
    assert types["is_active"] == "boolean"
    assert types["signup_date"] == "datetime"
    assert types["notes"] == "text"


def test_type_detection_pandas3_string_dtype(rng):
    """pandas >= 3.0 uses StringDtype by default; must still detect categorical."""
    values = ["a", "b"] * 15  # 30 rows, low cardinality => categorical evidence
    df = pd.DataFrame({"s": pd.array(values, dtype="string")})
    assert DataPreparation(df).get_type_summary().get("categorical", 0) == 1


def test_validate_data_quality_score_bounds(messy_df):
    v = DataPreparation(messy_df).validate_data()
    assert 0 <= v["data_quality_score"] <= 100
    assert v["duplicate_rows"] > 0
    assert v["missing_pct"] > 0


def test_column_info_shape(sample_df):
    info = DataPreparation(sample_df).get_column_info()
    assert list(info.columns) == ["Column", "Type", "Non-Null", "Null %", "Unique", "Unique %"]
    assert len(info) == sample_df.shape[1]


# ------------------------------------------------------- feature engineering

def test_engineer_features_actually_creates_features(rng):
    """Regression: the old implementation raised TypeError and produced nothing."""
    df = pd.DataFrame(
        {
            "a": rng.normal(10, 2, 300),
            "b": rng.lognormal(0, 1, 300),
            "c": rng.uniform(0, 50, 300),
            "d": rng.normal(0, 5, 300),
        }
    )
    fe = FeatureEngineer(df, random_state=42)
    out = fe.engineer_features(max_features=8)
    new_cols = [c for c in out.columns if c not in df.columns]
    assert 0 < len(new_cols) <= 8, f"expected engineered features, got {new_cols}"
    summary = fe.get_engineered_features_summary()
    assert summary["total_engineered"] == len(new_cols)


def test_engineer_features_respects_max(rng):
    df = pd.DataFrame({f"n{i}": rng.normal(size=200) for i in range(6)})
    out = FeatureEngineer(df).engineer_features(max_features=4)
    assert len(out.columns) - len(df.columns) <= 4


# ------------------------------------------------------------ gemini fallback

def test_insight_fallbacks_without_api_key():
    gi = GeminiInsights(api_key=None)
    assert gi.is_available() is False
    text = gi.generate_statistics_insight(
        {"skewness": 2.5, "std": 3.0, "mean": 10.0, "null_percentage": 4.0}, "revenue"
    )
    assert isinstance(text, str) and "revenue" in text


def test_missing_values_fallback_positive_and_negative():
    gi = GeminiInsights(api_key=None)
    clean = gi.generate_missing_values_insight({"total_missing": 0, "no_missing": True})
    dirty = gi.generate_missing_values_insight({"total_missing": 42, "no_missing": False})
    assert "no missing values" in clean.lower()
    assert "42" in dirty
