"""Jupyter notebook display module."""

import numpy as np
import pandas as pd

try:
    from IPython.display import HTML, Markdown, clear_output, display
    _IPYTHON_AVAILABLE = True
except ImportError:  # pragma: no cover
    _IPYTHON_AVAILABLE = False


def _in_kernel() -> bool:
    """True only inside a live IPython kernel (notebook/console)."""
    if not _IPYTHON_AVAILABLE:
        return False
    try:
        from IPython import get_ipython
        shell = get_ipython()
        return shell is not None and "zmq" in shell.__class__.__module__
    except Exception:
        return False


class JupyterDisplay:
    """Render analysis results in Jupyter; degrade to plain prints elsewhere."""

    def __init__(self):
        self._rich = _in_kernel()

    # ------------------------------------------------------------ primitives

    def _emit(self, renderable, fallback_text: str) -> None:
        if self._rich:
            display(renderable)
        else:
            print(fallback_text)

    def display_header(self, title: str, level: int = 1) -> None:
        level = min(max(int(level), 1), 6)
        self._emit(Markdown("#" * level + " " + title), "#" * level + " " + title)

    def display_subheader(self, title: str, level: int = 2) -> None:
        self.display_header(title, level=level)

    def display_text(self, text: str) -> None:
        self._emit(Markdown(text), text)

    def display_markdown(self, markdown: str) -> None:
        self._emit(Markdown(markdown), markdown)

    def display_dataframe(self, df: pd.DataFrame, title: str = None) -> None:
        if title:
            self.display_subheader(title)
        if self._rich:
            display(df)
        else:
            print(df.to_string())

    def display_figure(self, fig) -> None:
        """Display a plotly or matplotlib figure appropriately.

        Plotly's ``show()`` is only called inside a live kernel - elsewhere it
        would open a browser tab per chart (plotly's default renderer), which
        is slow, disruptive, and useless in a CLI run.
        """
        module = type(fig).__module__ or ""
        try:
            if module.startswith("plotly"):
                if self._rich:
                    fig.show()
                else:
                    print(f"[interactive chart available: "
                          f"{getattr(fig, 'layout', {}).title.text if getattr(fig, 'layout', None) else 'figure'}]")
            elif module.startswith("matplotlib"):
                display(fig) if self._rich else print("[static figure generated]")
        except Exception as error:
            print(f"Could not display figure: {error}")

    def clear_output(self) -> None:
        if self._rich:
            clear_output(wait=True)

    # --------------------------------------------------------------- sections

    def display_summary(self, df: pd.DataFrame) -> None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        date_cols = [c for c in df.columns if
                     pd.api.types.is_datetime64_any_dtype(df[c].dtype)]
        text = (
            f"**Dataset Shape:** {df.shape[0]} rows x {df.shape[1]} columns\n\n"
            f"- Numeric Columns: {len(numeric_cols)}\n"
            f"- Text/Categorical Columns: "
            f"{sum(1 for c in df.columns if df[c].dtype.name in ('object', 'str', 'category'))}\n"
            f"- Date Columns: {len(date_cols)}\n"
            f"- Missing Values: {df.isna().sum().sum()}"
        )
        self._emit(Markdown(text), text)

    def display_statistics(self, stats: dict,
                           title: str = "Statistical Analysis") -> None:
        if not stats:
            return
        self.display_subheader(title)
        self.display_dataframe(pd.DataFrame(stats).T.round(4))

    def display_missing_values(self, missing_analysis: dict,
                               title: str = "Missing Values Analysis") -> None:
        if not missing_analysis:
            return
        self.display_subheader(title)

        text = (
            f"**Overall Summary:**\n"
            f"- Total Missing Values: {missing_analysis.get('total_missing', 0)}\n"
            f"- Rows with Missing Values: {missing_analysis.get('missing_rows', 0)}\n"
            f"- Completely Empty Columns: "
            f"{len(missing_analysis.get('completely_missing_cols', []))}"
        )
        self._emit(Markdown(text), text)

        if missing_analysis.get("no_missing", True):
            self._emit(Markdown("**No missing values found.**"),
                       "No missing values found.")
            return

        missing_df = pd.DataFrame(
            list(missing_analysis.get("missing_percentage", {}).items()),
            columns=["Column", "Missing %"],
        )
        missing_df = missing_df[missing_df["Missing %"] > 0].sort_values(
            "Missing %", ascending=False
        )
        if len(missing_df) > 0:
            self.display_dataframe(missing_df.round(2),
                                   "Missing Percentage by Column")

    def display_distribution_summary(self, dist_analysis: dict,
                                     title: str = "Distribution Analysis") -> None:
        if not dist_analysis:
            return
        self.display_subheader(title)

        numeric_dists = dist_analysis.get("numeric_distributions", {})
        if numeric_dists:
            rows = [{
                "Column": col,
                "Distribution": d["distribution_type"],
                "Is Normal": "yes" if d["is_normal"] else "no",
                "Skewness": round(d["skewness"], 3),
                "Kurtosis": round(d["kurtosis"], 3),
                "Unique": d["unique_values"],
            } for col, d in numeric_dists.items()]
            self.display_dataframe(pd.DataFrame(rows), "Numeric Columns")

        categorical_dists = dist_analysis.get("categorical_distributions", {})
        if categorical_dists:
            rows = [{
                "Column": col,
                "Unique Values": d["unique_values"],
                "Diversity Index": round(d["diversity_index"], 3),
                "Missing": d["missing_count"],
            } for col, d in categorical_dists.items()]
            self.display_dataframe(pd.DataFrame(rows), "Categorical Columns")

    def display_correlations(self, corr_analysis: dict,
                             title: str = "Correlation Analysis") -> None:
        if not corr_analysis:
            return
        self.display_subheader(title)

        pearson = corr_analysis.get("pearson", {})
        strong = pearson.get("strong_correlations", [])
        moderate = pearson.get("moderate_correlations", [])

        for label, pairs in [
            ("Strong Correlations (|r| > 0.7)", strong),
            ("Moderate Correlations (0.5 < |r| <= 0.7)", moderate),
        ]:
            if pairs:
                frame = pd.DataFrame(pairs).round({"correlation": 4})
                frame.columns = ["Column A", "Column B", "Correlation"]
                self.display_dataframe(frame, label)

        if not strong and not moderate:
            self._emit(Markdown("No strong or moderate correlations found."),
                       "No strong or moderate correlations found.")

    def display_outliers(self, outlier_analysis: dict,
                         title: str = "Outlier Analysis") -> None:
        if not outlier_analysis:
            return
        self.display_subheader(title)

        iqr_results = outlier_analysis.get("iqr_method", {})
        if iqr_results:
            rows = [{
                "Column": col,
                "Outliers": d["outlier_count"],
                "Percentage": f"{d['outlier_percentage']:.2f}%",
                "Lower Bound": round(d["lower_bound"], 3),
                "Upper Bound": round(d["upper_bound"], 3),
            } for col, d in iqr_results.items()]
            self.display_dataframe(pd.DataFrame(rows))

        iso_forest = outlier_analysis.get("isolation_forest", {})
        if iso_forest and "n_outliers" in iso_forest:
            text = (f"\n**Isolation Forest:** {iso_forest['n_outliers']} outliers "
                    f"detected ({iso_forest['outlier_percentage']:.2f}%)")
            self._emit(Markdown(text), text)

    def display_insights(self, insights: dict, title: str = "Insights") -> None:
        if not insights:
            return
        self.display_subheader(title)
        for analysis_type, insight_text in insights.items():
            self.display_subheader(analysis_type, level=3)
            self._emit(Markdown(f"> {insight_text}"), f"> {insight_text}")
