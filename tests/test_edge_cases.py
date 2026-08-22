"""Edge-case hardening: nothing may crash, whatever the input."""

import numpy as np
import pandas as pd
import pytest

from autolyse import Autolyse
from autolyse.analyzers import (
    DistributionAnalyzer,
    MissingValuesAnalyzer,
    OutlierAnalyzer,
    StatisticalAnalyzer,
)
from autolyse.utils import DataPreparation


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame(),                                   # empty frame
        pd.DataFrame({"x": [1.0]}),                       # single row
        pd.DataFrame({"x": pd.Series([np.nan] * 10)}),    # all-NaN column
        pd.DataFrame({"x": [7.0] * 20}),                  # constant column
        pd.DataFrame({"x": [1, 2, 3], "y": ["a", "a", "a"]}),  # degenerate categories
        pd.DataFrame({"x": [np.inf, 1.0, -np.inf, 2.0]}),      # infinities
    ],
)
def test_analyzers_survive_degenerate_input(df):
    StatisticalAnalyzer(df).analyze()
    MissingValuesAnalyzer(df).analyze()
    DistributionAnalyzer(df).analyze_numeric_distributions()
    DistributionAnalyzer(df).analyze_categorical_distributions()
    OutlierAnalyzer(df).get_outlier_summary()


def test_full_pipeline_on_all_nan_column():
    df = pd.DataFrame(
        {"ghost": np.nan, "ok": np.arange(30, dtype=float)}
    )
    analyser = Autolyse(html=False, enable_visualizations=False)
    results = analyser.analyse(df, show_progress=False)
    assert "statistics" in results


def test_duplicate_and_constant_detection(messy_df):
    v = DataPreparation(messy_df).validate_data()
    assert v["duplicate_rows"] >= 4
    types = DataPreparation(messy_df).get_column_types()
    assert types["constant"] == "numeric"
