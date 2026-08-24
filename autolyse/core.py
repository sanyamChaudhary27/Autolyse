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
from autolyse.findings import FindingsEngine
from autolyse.insights import GeminiProvider, InsightEngine, Narrator
from autolyse.output import HTMLGenerator, JupyterDisplay
from autolyse.output.jupyter_display import _in_kernel
from autolyse.target_aware import TargetAnalyzer
from autolyse.utils import DataPreparation, FeatureEngineer
from autolyse.visualizers import MatplotlibVisualizer, PlotlyVisualizer


class Autolyse:
    """Prescriptive automated EDA: findings, health score and charts.

    Usage:
        >>> import pandas as pd
        >>> from autolyse import Autolyse
        >>> analyser = Autolyse(html=False)          # descriptive EDA
        >>> analyser = Autolyse(target="churn")      # + prescriptive layer
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
                 batch_size: Optional[int] = None,
                 target: Optional[str] = None,
                 llm_provider=None):
        """
        Args:
            html: Generate an HTML report instead of notebook display.
            api_key: Gemini API key for optional LLM narration (or set
                GEMINI_KEY env var). All analysis works fully offline without it.
            output_dir: Directory for HTML reports.
            random_seed: Seed for sampling and stochastic steps.
            enable_*: Granular switches for each analysis stage.
            batch_size: Analyze a reproducible random sample of N rows.
            target: Optional target column - activates predictive-power
                ranking, leakage detection and imbalance findings.
            llm_provider: Optional object with ``complete(prompt)->str``.
                Defaults to GeminiProvider when api_key is given.
        """
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        self.html_output = bool(html) and bool(enable_html)
        self.api_key = api_key
        self.output_dir = Path(output_dir)
        self.random_seed = random_seed
        self.target = target
        self.narrator = Narrator(llm_provider or
                                 (GeminiProvider(api_key=api_key)
                                  if api_key else None))

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
        self.findings: list = []
        self.health_score = None
        self.target_analysis: Optional[Dict[str, Any]] = None
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
        if self.target is not None and self.target not in df.columns:
            raise ValueError(f"Target column '{self.target}' not found in DataFrame")

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

            if "target" in steps:
                progress("Analyzing target relationships")
                try:
                    analyzer = TargetAnalyzer(
                        self.df, self.target,
                        column_types=self._grouped_types(),
                    )
                    self.target_analysis = analyzer.analyze()
                    self.analyses["target_analysis"] = self.target_analysis
                except Exception as error:
                    warnings.warn(f"Target analysis skipped: {error}")

            progress("Scoring data health")
            engine = FindingsEngine(
                self.df, column_types=self._grouped_types(),
                analyses=self.analyses, target=self.target,
                random_seed=self.random_seed,
            )
            self.findings = engine.run()
            self.health_score = engine.health_score(self.findings)

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

    def get_findings(self) -> list:
        """Get ranked prescriptive findings (Finding dataclasses)."""
        return list(self.findings)

    def get_health_score(self):
        """Get the HealthScore (overall, grade, per-category)."""
        return self.health_score

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
        has_analyzers = any([self.enable_statistics, self.enable_missing_values,
                             self.enable_distributions, self.enable_outliers,
                             self.enable_correlations, self.enable_relationships,
                             self.enable_advanced_insights])
        if has_analyzers:
            steps.append("analyze")
        if self.target is not None:
            steps.append("target")
        steps.append("findings")
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
        """Deterministic narratives, optionally polished by the LLM provider."""
        engine = InsightEngine(
            df=self.df,
            analyses=self.analyses,
            findings=self.findings,
            health_score=self.health_score,
            validation=self.validation,
            target_analysis=self.target_analysis,
        )
        return self.narrator.polish(engine.build())

    def _grouped_types(self) -> Dict[str, list]:
        if self.data_prep is None:
            return {}
        return {
            "numeric": self.data_prep.get_numeric_columns(),
            "categorical": self.data_prep.get_categorical_columns(),
            "text": self.data_prep.get_text_columns(),
            "datetime": self.data_prep.get_datetime_columns(),
            "boolean": self.data_prep.get_boolean_columns(),
        }

    def _generate_html_report(self) -> str:
        generator = HTMLGenerator(self.df, output_dir=str(self.output_dir))
        return generator.generate_report(
            analyses=self.analyses,
            insights=self.insights,
            filename="autolyse_report.html",
            figures=self.figures.get("plotly", {}),
            findings=self.findings,
            health_score=self.health_score,
            target_analysis=self.target_analysis,
        )

    def _display_jupyter_output(self) -> None:
        view = JupyterDisplay()

        view.display_header("Automated EDA Analysis - Autolyse", level=1)

        if self.health_score is not None:
            cats = " | ".join(f"{k}: {v}" for k, v in
                              (self.health_score.by_category or {}).items())
            text = (f"**Health Score: {self.health_score.overall}/100 "
                    f"(grade {self.health_score.grade})**"
                    + (f"  \n{cats}" if cats else ""))
            view.display_text(text)

        if self.findings:
            view.display_subheader("Findings", level=2)
            for finding in self.findings[:10]:
                marker = finding.severity.value.upper()
                line = f"**[{marker}]** {finding.title} - {finding.detail}"
                if finding.fix_snippet:
                    line += f"  \n```python\n{finding.fix_snippet}\n```"
                view.display_text(line)
            if len(self.findings) > 10:
                view.display_text(
                    f"... {len(self.findings) - 10} more findings in the report.")

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
