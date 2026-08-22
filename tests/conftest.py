"""Shared fixtures for the Autolyse test suite."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def sample_df(rng):
    """Well-behaved mixed-type dataset covering every column role."""
    n = 200
    return pd.DataFrame(
        {
            "age": rng.normal(40, 12, n).round(1),
            "income": rng.exponential(50_000, n).round(2),
            "score": rng.uniform(0, 100, n).round(3),
            "city": rng.choice(["Berlin", "Paris", "Tokyo", "Lima"], n),
            "plan": rng.choice(["free", "pro", "enterprise"], n, p=[0.6, 0.3, 0.1]),
            "is_active": rng.choice([True, False], n),
            "signup_date": pd.date_range("2024-01-01", periods=n, freq="h"),
            "notes": pd.Series([f"note text {i}" for i in range(n)]),
        }
    )


@pytest.fixture
def messy_df(rng):
    """Dataset full of data-quality problems for findings tests."""
    n = 150
    df = pd.DataFrame(
        {
            # Exactly 97.33% missing (146 of 150 rows) - deterministic.
            "almost_empty": np.where(np.arange(n) < 146, np.nan,
                                     rng.normal(size=n)),
            "constant": np.full(n, 7.0),
            "id_like": np.arange(n),
            "skewed": rng.exponential(1.0, n),
        }
    )
    df.loc[df.index[:30], "skewed"] = np.nan  # 20% missing
    df["leaky"] = df["skewed"].shift(-1).fillna(df["skewed"].mean()) * 2
    dup = df.iloc[[0, 1]]
    return pd.concat([df, dup, dup], ignore_index=True)


@pytest.fixture
def tiny_df():
    """Degenerate shapes that must not crash."""
    return pd.DataFrame({"x": [1.0], "y": ["a"]})
