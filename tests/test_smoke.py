"""End-to-end smoke tests: the public Autolyse() API must keep working."""

import pandas as pd
import pytest

from autolyse import Autolyse


def test_analyse_runs_end_to_end(sample_df):
    analyser = Autolyse(html=False, enable_visualizations=False)
    results = analyser.analyse(sample_df, show_progress=False)

    assert isinstance(results, dict)
    for key in (
        "statistics",
        "missing_values",
        "distributions",
        "outliers",
        "correlations",
        "relationships",
    ):
        assert key in results, f"missing analysis section: {key}"

    # Original input must not be mutated by the analysis.
    assert len(analyser.df_original) == len(sample_df)
    assert analyser.get_analysis_results()["statistics"].keys() == results["statistics"].keys()
    assert isinstance(analyser.get_insights(), dict)
    info = analyser.get_dataframe_info()
    assert isinstance(info, pd.DataFrame) and len(info) == sample_df.shape[1]


def test_analyse_with_visualizations_produces_figures(sample_df):
    analyser = Autolyse(html=False)
    analyser.analyse(sample_df, show_progress=False)
    figures = analyser.figures
    assert "plotly" in figures and figures["plotly"], "expected plotly figures"
    assert figures["plotly"].get("correlation") is not None


def test_granular_flags_skip_sections(sample_df):
    analyser = Autolyse(
        html=False,
        enable_statistics=False,
        enable_missing_values=False,
        enable_distributions=False,
        enable_outliers=False,
        enable_correlations=False,
        enable_relationships=False,
        enable_advanced_insights=False,
        enable_visualizations=False,
    )
    results = analyser.analyse(sample_df, show_progress=False)
    assert results == {}


def test_batch_sampling(sample_df):
    analyser = Autolyse(html=False, batch_size=50, enable_visualizations=False)
    analyser.analyse(sample_df, show_progress=False)
    assert len(analyser.df) == 50
    assert len(analyser.df_original) == len(sample_df)


def test_html_report_written(tmp_path, sample_df):
    analyser = Autolyse(html=True, output_dir=str(tmp_path), enable_visualizations=True)
    path = analyser.analyse(sample_df, show_progress=False)
    report = tmp_path / "autolyse_report.html"
    assert report.exists(), f"expected report at {path}"
    content = report.read_text(encoding="utf-8")
    assert "Autolyse" in content
    assert "<html" in content.lower()


@pytest.mark.parametrize("seed", [7, 123])
def test_reproducible_with_same_seed(sample_df, seed):
    a = Autolyse(html=False, random_seed=seed, enable_visualizations=False).analyse(
        sample_df, show_progress=False
    )
    b = Autolyse(html=False, random_seed=seed, enable_visualizations=False).analyse(
        sample_df, show_progress=False
    )
    assert a["statistics"] == b["statistics"]
