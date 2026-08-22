"""Main Autolyse orchestrator class."""

import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from autolyse.analyzers import (
    AdvancedInsightsAnalyzer,
    CorrelationAnalyzer,
    DistributionAnalyzer,
    MissingValuesAnalyzer,
    OutlierAnalyzer,
    RelationshipsAnalyzer,
    StatisticalAnalyzer,
)
from autolyse.output import HTMLGenerator, JupyterDisplay
from autolyse.output.jupyter_display import _in_kernel
from autolyse.utils import DataPreparation, FeatureEngineer, GeminiInsights
from autolyse.visualizers import MatplotlibVisualizer, PlotlyVisualizer


class Autolyse:
    """Automated EDA with prescriptive findings and optional AI narration.

    Usage:
        >>> import pandas as pd
        >>> from autolyse import Autolyse
        >>> analyser = Autolyse(html=False)
        >>> results = analyser.analyse(df)
    """

    def __init__(self, html: bool = True, api_key: Optional[str] = None,
                 output_dir: str = "./output_reports", random_seed: int = 42,
                 enable_statistics: bool = True, enable_missing_values: bool = True,
                 enable_distributions: bool = True, enable_outliers: bool = True,
                 enable_correlations: bool = True, enable_relationships: bool = True,
                 enable_advanced_insights: bool = True,
                 enable_feature_engineering: bool = False,
                 enable_visualizations: bool = True, enable_html: bool = True,
                 batch_size: Optional[int] = None):
        """
        Args:
            html: Generate an HTML report instead of notebook display.
            api_key: Gemini API key (or set GEMINI_KEY env var). Optional -
                deterministic summaries are used without one.
            output_dir: Directory for HTML reports.
            random_seed: Seed for sampling, outlier models and feature engineering.
            enable_*: Granular switches for each analysis stage.
            batch_size: Analyze a reproducible random sample of N rows.
        """
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        self.html_output = bool(html) and bool(enable_html)
        self.api_key = api_key
        self.output_dir = Path(output_dir)
        self.random_seed = random_seed
        self.insights_generator = GeminiInsights(api_key=api_key)

        self.enable_statistics = enable_statistics
        self.enable_missing_values = enable_missing_values
        self.enable_distributions = enable_distributions
        self.enable_outliers = enable_outliers
        self.enable_correlations = enable_correlations
        self.enable_relationships = enable_relationships
        self.enable_advanced_insights = enable_advanced_insights
        self.enable_feature_engineering = enable_feature_engineering
        self.enable_visualizations = enable_visualizations
        self.batch_size = batch_size

        # Populated by analyse()
        self.df: Optional[pd.DataFrame] = None
        self.df_original: Optional[pd.DataFrame] = None
        self.data_prep: Optional[DataPreparation] = None
        self.validation: Dict[str, Any] = {}
        self.analyses: Dict[str, Any] = {}
        self.insights: Dict[str, str] = {}
        self.figures: Dict[str, Dict] = {}
        self._is_jupyter = _in_kernel()

    # ------------------------------------------------------------------ API

    def analyse(self, df: pd.DataFrame,
                show_progress: bool = True) -> Dict[str, Any]:
        """Run the full pipeline; returns the analyses dictionary."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"analyse() expects a pandas DataFrame, "
                            f"got {type(df).__name__}")
        if df.empty:
            raise ValueError("Cannot analyse an empty DataFrame")

        # One copy doubles as both original reference and working frame;
        # sampling (below) replaces only the working frame.
        self.df_original = df.copy()
        self.df = self.df_original.copy()

        rng = np.random.default_rng(self.random_seed)
        if self.batch_size and len(self.df) > self.batch_size:
            sample_idx = rng.choice(len(self.df), self.batch_size, replace=False)
            self.df = self.df.iloc[np.sort(sample_idx)].reset_index(drop=True)

        steps = self._enabled_steps()
        step_no = 0

        def progress(message: str) -> None:
            nonlocal step_no
            step_no += 1
            self._print(f"[{step_no}/{len(steps)}] {message}", show_progress)

        try:
            if "prepare" in steps:
                progress("Detecting column types and validating data")
                self.data_prep = DataPreparation(self.df)
                self.validation = self.data_prep.validate_data()

            if "engineer" in steps:
                progress("Engineering features")
                try:
                    engineer = FeatureEngineer(self.df, random_state=self.random_seed)
                    before = len(self.df.columns)
                    self.df = engineer.engineer_features(max_features=15)
                    created = len(self.df.columns) - before
                    if created:
                        # Re-run type detection so analyzers see new columns.
                        self.data_prep = DataPreparation(self.df)
                        self.validation = self.data_prep.validate_data()
                except Exception as error:
                    warnings.warn(f"Feature engineering skipped: {error}")

            if "analyze" in steps:
                progress(f"Running {self._count_enabled_analyzers()} analyzers")
                self.analyses = self._run_all_analyzers()

            if "visualize" in steps:
                progress("Generating visualizations")
                self._generate_visualizations()

            if "insights" in steps:
                progress("Generating insights")
                self.insights = self._generate_ai_insights()

            if "output" in steps:
                progress("Preparing output")
                if self.html_output:
                    report_path = self._generate_html_report()
                    self._print(f"HTML report saved to: {report_path}",
                                show_progress)
                else:
                    self._display_jupyter_output()
        finally:
            # Free pyplot's figure registry regardless of outcome; Figure
            # objects stay usable (e.g. savefig) via self.figures.
            self._close_matplotlib_figures()

        return self.analyses

    def get_analysis_results(self) -> Dict[str, Any]:
        """Get the analysis results dictionary."""
        return self.analyses.copy()

    def get_insights(self) -> Dict[str, str]:
        """Get generated insight texts."""
        return self.insights.copy()

    def get_dataframe_info(self) -> Optional[pd.DataFrame]:
        """Detailed per-column information (None before analyse())."""
        if self.data_prep is not None:
            return self.data_prep.get_column_info()
        return None

    # ------------------------------------------------------------- internals

    @staticmethod
    def _print(message: str, enabled: bool) -> None:
        """ASCII-only console output.

        Emoji-rich progress lines crashed hard on Windows consoles using the
        cp1252 codec (UnicodeEncodeError); plain text works everywhere.
        """
        if enabled:
            print(message)

    def _enabled_steps(self):
        steps = ["prepare"]
        if self.enable_feature_engineering:
            steps.append("engineer")
        if any([self.enable_statistics, self.enable_missing_values,
                self.enable_distributions, self.enable_outliers,
                self.enable_correlations, self.enable_relationships,
                self.enable_advanced_insights]):
            steps.append("analyze")
        if self.enable_visualizations:
            steps.append("visualize")
        steps.append("insights")
        steps.append("output")
        return steps

    def _count_enabled_analyzers(self) -> int:
        return sum([
            self.enable_statistics, self.enable_missing_values,
            self.enable_distributions, self.enable_outliers,
            self.enable_correlations, self.enable_relationships,
            self.enable_advanced_insights,
        ])

    def _run_all_analyzers(self) -> Dict[str, Any]:
        analyses = {}

        if self.enable_statistics:
            analyses["statistics"] = StatisticalAnalyzer(
                self.df, random_state=self.random_seed
            ).analyze()

        if self.enable_missing_values:
            analyses["missing_values"] = MissingValuesAnalyzer(self.df).analyze()

        if self.enable_distributions:
            dist = DistributionAnalyzer(self.df)
            analyses["distributions"] = {
                "numeric_distributions": dist.analyze_numeric_distributions(),
                "categorical_distributions": dist.analyze_categorical_distributions(),
            }

        if self.enable_outliers:
            analyses["outliers"] = OutlierAnalyzer(
                self.df, random_state=self.random_seed
            ).get_outlier_summary()

        if self.enable_correlations:
            analyses["correlations"] = CorrelationAnalyzer(
                self.df
            ).get_correlation_summary()

        if self.enable_relationships:
            analyses["relationships"] = RelationshipsAnalyzer(
                self.df
            ).get_relationship_summary()

        if self.enable_advanced_insights:
            try:
                analyses["advanced_insights"] = AdvancedInsightsAnalyzer(
                    self.df, random_state=self.random_seed
                ).analyze_all()
            except Exception as error:
                warnings.warn(f"Advanced insights unavailable: {error}")

        return analyses

    def _generate_visualizations(self) -> None:
        mpl_viz = MatplotlibVisualizer(self.df)
        plotly_viz = PlotlyVisualizer(self.df)
        self.figures = {"matplotlib": {}, "plotly": {}}

        if self.enable_distributions:
            self.figures["matplotlib"]["distributions"] = \
                mpl_viz.plot_distributions()
            self.figures["matplotlib"]["categorical_distributions"] = \
                mpl_viz.plot_categorical_distributions()
            self.figures["plotly"]["distributions"] = plotly_viz.plot_distributions()
            self.figures["plotly"]["categorical_distributions"] = \
                plotly_viz.plot_categorical_distributions()

        if self.enable_missing_values and "missing_values" in self.analyses:
            self.figures["plotly"]["missing_values"] = plotly_viz.plot_missing_values(
                self.analyses["missing_values"]
            )
            self.figures["matplotlib"]["missing_values"] = \
                mpl_viz.plot_missing_values(self.analyses["missing_values"])

        if self.enable_correlations and "correlations" in self.analyses:
            corr_matrix = self.analyses["correlations"].get("pearson", {}) \
                                                     .get("correlation_matrix")
            if corr_matrix is not None:
                self.figures["matplotlib"]["correlation"] = \
                    mpl_viz.plot_correlation_heatmap(corr_matrix)
                self.figures["plotly"]["correlation"] = \
                    plotly_viz.plot_correlation_heatmap(corr_matrix)

        if self.enable_distributions:
            self.figures["matplotlib"]["boxplot"] = mpl_viz.plot_boxplot()
            self.figures["plotly"]["boxplot"] = plotly_viz.plot_boxplot()

        scatter_mpl = mpl_viz.plot_scatter_matrix(max_cols=4)
        if scatter_mpl is not None:
            self.figures["matplotlib"]["scatter_matrix"] = scatter_mpl
        scatter_plotly = plotly_viz.plot_scatter_matrix(max_cols=4)
        if scatter_plotly is not None:
            self.figures["plotly"]["scatter_matrix"] = scatter_plotly

        if self.enable_outliers and "outliers" in self.analyses:
            iqr_results = self.analyses["outliers"].get("iqr_method", {})
            self.figures["matplotlib"]["outliers"] = {}
            self.figures["plotly"]["outliers"] = {}
            for col, bounds in list(iqr_results.items())[:8]:
                fig_mpl = mpl_viz.plot_outliers(col, bounds)
                if fig_mpl is not None:
                    self.figures["matplotlib"]["outliers"][col] = fig_mpl
                fig_plotly = plotly_viz.plot_outliers(col, self.df[col], bounds)
                if fig_plotly is not None:
                    self.figures["plotly"]["outliers"][col] = fig_plotly

    def _generate_ai_insights(self) -> Dict[str, str]:
        """Build insights section-by-section.

        Each block depends only on analyses that actually ran - previously a
        single disabled analyzer raised KeyError and silently discarded ALL
        insights.
        """
        insights = {}
        generator = self.insights_generator
        a = self.analyses

        def add(key, producer):
            try:
                text = producer()
                if text:
                    insights[key] = text
            except Exception as error:
                warnings.warn(f"Could not generate '{key}' insight: {error}")

        if "statistics" in a:
            def stats_text():
                cols = [c for c in self.data_prep.get_numeric_columns()
                        if c in a["statistics"]][:3]
                return "\n\n".join(
                    generator.generate_statistics_insight(a["statistics"][col], col)
                    for col in cols
                )
            add("Statistics", stats_text)

        if "missing_values" in a:
            add("Data Quality",
                lambda: generator.generate_missing_values_insight(
                    a["missing_values"]))

        if "correlations" in a:
            pearson = a["correlations"].get("pearson", {})
            add("Correlations",
                lambda: generator.generate_correlation_insight(
                    pearson.get("strong_correlations", []),
                    pearson.get("moderate_correlations", [])))

        if "outliers" in a:
            add("Outliers",
                lambda: generator.generate_outlier_insight(
                    a["outliers"].get("iqr_method", {}),
                    a["outliers"].get("isolation_forest", {})))

        if "distributions" in a:
            add("Distributions",
                lambda: generator.generate_distribution_insight(
                    a["distributions"].get("numeric_distributions", {}),
                    a["distributions"].get("categorical_distributions", {})))

        add("Dataset Overview",
            lambda: generator.generate_general_insight(
                tuple(self.df.shape),
                self.validation.get("column_types", {}),
                self.validation.get("data_quality_score", 0)))

        return insights

    def _generate_html_report(self) -> str:
        generator = HTMLGenerator(self.df, output_dir=str(self.output_dir))
        return generator.generate_report(
            analyses=self.analyses,
            insights=self.insights,
            filename="autolyse_report.html",
            figures=self.figures.get("plotly", {}),
        )

    def _display_jupyter_output(self) -> None:
        view = JupyterDisplay()

        view.display_header("Automated EDA Analysis - Autolyse", level=1)
        view.display_summary(self.df)

        if self.data_prep is not None:
            view.display_dataframe(self.data_prep.get_column_info(),
                                   "Column Information")

        view.display_statistics(self.analyses.get("statistics", {}))
        view.display_missing_values(self.analyses.get("missing_values", {}))
        view.display_distribution_summary(self.analyses.get("distributions", {}))
        view.display_correlations(self.analyses.get("correlations", {}))
        view.display_outliers(self.analyses.get("outliers", {}))

        if self.insights:
            view.display_insights(self.insights)

        plotly_figs = self.figures.get("plotly", {})
        shown_any = False
        try:
            if plotly_figs.get("distributions"):
                view.display_subheader("Distributions", level=3)
                for fig in list(plotly_figs["distributions"].values())[:3]:
                    view.display_figure(fig)
                    shown_any = True
            for key, label in [("missing_values", "Missing Values"),
                               ("correlation", "Correlation Heatmap"),
                               ("boxplot", "Boxplots")]:
                if plotly_figs.get(key) is not None:
                    view.display_subheader(label, level=3)
                    view.display_figure(plotly_figs[key])
                    shown_any = True
        except Exception as error:
            print(f"Could not display all visualizations: {error}")

        if not shown_any and not self._is_jupyter:
            print("Analysis complete. For interactive charts or an HTML report, "
                  "construct Autolyse(html=True).")

    def _close_matplotlib_figures(self) -> None:
        try:
            import matplotlib.pyplot as plt
            plt.close("all")
        except Exception:
            pass
