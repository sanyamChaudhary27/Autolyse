"""HTML report generation module."""

import html as html_module
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def _escape(value) -> str:
    """Render any cell value as safe HTML text."""
    if value is None:
        return "&mdash;"
    if isinstance(value, float) and np.isnan(value):
        return "&mdash;"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (float, np.floating)):
        if np.isinf(value):
            return "&#8734;" if value > 0 else "-&#8734;"
        return f"{value:,.4g}"
    text = str(value)
    return html_module.escape(text, quote=False)


def _score_color(score: float) -> str:
    if score >= 90:
        return "#388e3c"
    if score >= 70:
        return "#fbc02d"
    if score >= 50:
        return "#f57c00"
    return "#d32f2f"


class HTMLGenerator:
    """Generate self-contained HTML reports from analysis results."""

    #: Per-column outlier charts can explode report size; cap them.
    MAX_OUTLIER_CHARTS = 8

    def __init__(self, df: pd.DataFrame, output_dir: str = "./output",
                 embed_charts: bool = True, offline_js: bool = True):
        """
        Args:
            df: Input dataframe
            output_dir: Directory to save HTML reports
            embed_charts: Embed interactive Plotly figures into the report
            offline_js: Inline plotly.js so the file works fully offline
                (adds ~3 MB); False loads it from a CDN instead.
        """
        self.df = df
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.embed_charts = embed_charts
        self.offline_js = offline_js
        self._needs_plotly_js = False
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ------------------------------------------------------------------ API

    def generate_report(self, analyses: dict, insights: dict | None = None,
                        filename: str = "autolyse_report.html",
                        figures: dict | None = None,
                        findings: list | None = None,
                        health_score=None,
                        target_analysis: dict | None = None) -> str:
        """Build and save the report; returns the output path."""
        body = [
            self._dataset_summary_section(),
        ]
        if health_score is not None:
            body.append(self._health_score_section(health_score))
        if findings is not None:
            body.append(self._findings_section(findings))
        body.extend([
            self._statistics_section(analyses.get("statistics", {})),
            self._missing_values_section(analyses.get("missing_values", {})),
            self._distribution_section(analyses.get("distributions", {})),
            self._correlation_section(analyses.get("correlations", {})),
            self._outliers_section(analyses.get("outliers", {})),
        ])
        if target_analysis:
            body.append(self._target_section(target_analysis))
        if self.embed_charts and figures:
            body.append(self._visualizations_section(figures))
        body.append(self._insights_section(insights or {}))

        html_content = "\n".join([
            self._html_header(),
            "<body>",
            self._navbar(),
            "<main class='container'>",
            *body,
            "</main>",
            self._footer(),
            "</body>",
            "</html>",
        ])

        output_path = self.output_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        return str(output_path)

    # ------------------------------------------------------------- skeleton

    def _html_header(self) -> str:
        plotly_tag = ""
        if self.embed_charts and self.offline_js:
            try:
                import plotly.offline as po
                plotly_tag = ("<script>"
                              + po.get_plotlyjs()
                              + "</script>")
                self._needs_plotly_js = True
            except Exception:
                plotly_tag = ""
        elif self.embed_charts:
            self._needs_plotly_js = True
            plotly_tag = ('<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" '
                          'charset="utf-8"></script>')

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Autolyse - EDA Report</title>
{plotly_tag}
{self._styles()}
<script>
function copyFix(i) {{
  navigator.clipboard.writeText(document.getElementById('fix' + i).innerText);
}}
</script>
</head>"""

    def _render_figure(self, fig, height: int = 420) -> str:
        """Embed one plotly figure without repeating the JS library."""
        try:
            return fig.to_html(
                full_html=False,
                include_plotlyjs=False,
                default_width="100%",
                default_height=f"{height}px",
                config={"displaylogo": False, "responsive": True},
            )
        except Exception:
            return ""

    def _chart_block(self, title: str, fig, height: int = 420) -> str:
        inner = self._render_figure(fig, height)
        if not inner:
            return ""
        return f"<div class='chart-block'><h3>{_escape(title)}</h3>{inner}</div>"

    # ---------------------------------------------------------------- styles

    def _styles(self) -> str:
        return """<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
       background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
       color: #333; line-height: 1.6; }
.navbar { background: white; padding: 1rem 0; box-shadow: 0 2px 8px rgba(0,0,0,.1);
          position: sticky; top: 0; z-index: 100; }
.navbar .container { display: flex; justify-content: space-between;
                     align-items: center; padding: 0 2rem; }
.navbar h1 { color: #667eea; font-size: 24px; }
.navbar-time { color: #666; font-size: 14px; }
main.container { max-width: 1200px; margin: 2rem auto; padding: 0 2rem 2rem; }
.section { background: white; border-radius: 8px; padding: 2rem;
           margin-bottom: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,.1); }
.section h2 { color: #667eea; margin-bottom: 1.5rem; border-bottom: 3px solid #667eea;
              padding-bottom: .5rem; }
.section h3 { color: #764ba2; margin-top: 1.25rem; margin-bottom: .75rem; font-size: 16px; }
.summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1.5rem; margin-bottom: 1rem; }
.summary-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white; padding: 1.5rem; border-radius: 8px; text-align: center; }
.summary-card h3 { color: white; font-size: 14px; margin-bottom: .5rem;
                   text-transform: uppercase; opacity: .9; }
.summary-card .value { font-size: 28px; font-weight: bold; }
table { width: 100%; border-collapse: collapse; margin: 1rem 0; }
th { background: #667eea; color: white; padding: .75rem; text-align: left; }
td { padding: .75rem; border-bottom: 1px solid #eee; overflow-wrap: anywhere; }
tr:nth-child(even) { background: #f9f9f9; }
tr:hover { background: #f0f0f0; }
.alert { padding: 1rem; border-radius: 4px; margin: 1rem 0; }
.alert-info { background: #e3f2fd; color: #1976d2; border-left: 4px solid #1976d2; }
.alert-warning { background: #fff3e0; color: #f57c00; border-left: 4px solid #f57c00; }
.alert-success { background: #e8f5e9; color: #388e3c; border-left: 4px solid #388e3c; }
.chart-block { background: #fafafa; border: 1px solid #eee; border-radius: 8px;
               padding: 1rem; margin: 1.5rem 0; }
.chart-block h3 { margin-top: 0; }
.insight-box { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
               color: white; padding: 1.5rem; border-radius: 8px; margin: 1rem 0;
               border-left: 5px solid #fff; }
.insight-title { font-weight: bold; margin-bottom: .5rem; font-size: 14px;
                 text-transform: uppercase; opacity: .9; }
.insight-text { font-size: 15px; line-height: 1.6; }
.footer { background: white; padding: 2rem; text-align: center; color: #666;
          border-top: 1px solid #eee; margin-top: 3rem; }
.no-data { text-align: center; padding: 2rem; color: #999; font-style: italic; }
.score-wrap { display: flex; gap: 2rem; align-items: center; flex-wrap: wrap; }
.score-hero { border: 4px solid #667eea; border-radius: 12px; padding: 1rem 2.5rem;
              text-align: center; }
.score-number { font-size: 56px; font-weight: 800; line-height: 1; }
.score-grade { font-size: 18px; color: #555; margin-top: .25rem; }
.score-sub { font-size: 12px; color: #999; }
.score-cats { flex: 1; min-width: 280px; display: grid;
              grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: .75rem; }
.score-cat { display: grid; grid-template-columns: 110px 1fr 40px;
             align-items: center; gap: .5rem; font-size: 14px; }
.bar { background: #eee; border-radius: 6px; height: 10px; overflow: hidden; }
.fill { height: 100%; }
.finding { border-left: 5px solid #999; background: #fafafa; border-radius: 6px;
           padding: 1rem 1.25rem; margin: .9rem 0; }
.finding.sev-critical { border-color: #d32f2f; background: #fdecea; }
.finding.sev-high { border-color: #f57c00; background: #fff3e0; }
.finding.sev-medium { border-color: #fbc02d; background: #fffde7; }
.finding.sev-low { border-color: #1976d2; background: #e3f2fd; }
.finding-head { display: flex; align-items: center; gap: .6rem; margin-bottom: .35rem; }
.sev-badge { font-size: 11px; font-weight: 700; letter-spacing: .05em;
             padding: 2px 8px; border-radius: 10px; color: white; background: #777; }
.sev-critical .sev-badge { background: #d32f2f; }
.sev-high .sev-badge { background: #f57c00; }
.sev-medium .sev-badge { background: #fbc02d; color: #333; }
.sev-low .sev-badge { background: #1976d2; }
.cols { color: #776; font-size: 13px; }
.count-badge { background: #667eea; color: white; border-radius: 12px;
               font-size: 14px; padding: 2px 12px; vertical-align: middle; }
details summary { cursor: pointer; color: #455; margin-top: .4rem; }
pre { background: #263238; color: #eceff1; padding: .8rem; border-radius: 6px;
      overflow-x: auto; font-size: 13px; margin: .5rem 0; }
.copy-btn { background: #667eea; border: none; color: white; padding: 4px 14px;
            border-radius: 4px; cursor: pointer; font-size: 12px; }
.copy-btn:hover { background: #764ba2; }
</style>"""

    # ------------------------------------------------------------- sections

    def _navbar(self) -> str:
        return f"""<nav class="navbar"><div class="container">
<h1>Autolyse</h1><span class="navbar-time">Generated: {self.timestamp}</span>
</div></nav>"""

    def _health_score_section(self, health_score) -> str:
        cats = "".join(
            f"<div class='score-cat'><span>{_escape(name)}</span>"
            f"<div class='bar'><div class='fill' style='width:{value}%;"
            f"background:{_score_color(value)}'></div></div>"
            f"<b>{value}</b></div>"
            for name, value in health_score.by_category.items()
        )
        return f"""<section class='section' id='health'>
<h2>Data Health Score</h2>
<div class='score-wrap'>
  <div class='score-hero' style='border-color:{_score_color(health_score.overall)}'>
    <div class='score-number' style='color:{_score_color(health_score.overall)}'>{health_score.overall}</div>
    <div class='score-grade'>grade {health_score.grade}</div>
    <div class='score-sub'>weighted across {len(health_score.by_category or {}) or 4} dimensions</div>
  </div>
  <div class='score-cats'>{cats}</div>
</div>
</section>"""

    def _findings_section(self, findings: list) -> str:
        if not findings:
            return ("<section class='section'><h2>Findings</h2>"
                    "<div class='alert alert-success'>No issues detected.</div>"
                    "</section>")
        cards = []
        for i, f in enumerate(findings):
            snippet = ""
            if f.fix_snippet:
                snippet = (
                    f"<details><summary>Suggested fix</summary>"
                    f"<pre id='fix{i}'><code>{_escape(f.fix_snippet)}</code></pre>"
                    f"<button class='copy-btn' onclick=\"copyFix({i})\">Copy</button>"
                    f"</details>"
                )
            cols = ", ".join(f"'{_escape(c)}'" for c in f.columns)
            cards.append(f"""
<div class='finding sev-{f.severity.value}'>
  <div class='finding-head'>
    <span class='sev-badge'>{f.severity.value.upper()}</span>
    <b>{_escape(f.title)}</b>
  </div>
  <p>{_escape(f.detail)}</p>
  {f"<p class='cols'>Columns: {cols}</p>" if cols else ''}
  {snippet}
</div>""")
        return (f"<section class='section'><h2>Findings "
                f"<span class='count-badge'>{len(findings)}</span></h2>"
                f"{''.join(cards)}</section>")

    def _target_section(self, target_analysis: dict) -> str:
        summary = target_analysis.get("target_summary", {})
        powers = target_analysis.get("predictive_power", {})
        rows = [{
            "Feature": name,
            "Power": round(info["strength"], 3),
            "Relation": info["relation"],
        } for name, info in list(powers.items())[:15]]
        suspects = target_analysis.get("leakage_suspects") or []
        alert = ""
        if suspects:
            names = ", ".join(s["feature"] for s in suspects[:5])
            alert = (f"<div class='alert alert-warning'><b>Possible leakage:</b> "
                     f"{_escape(names)} predict '{_escape(summary.get('column'))}' "
                     f"near-perfectly. Verify provenance before modeling.</div>")
        return f"""<section class='section'><h2>Target Analysis</h2>
{alert}
<p>Target: <b>{_escape(summary.get('column'))}</b> ({_escape(summary.get('kind'))})</p>
<h3>Predictive Power Ranking</h3>
{self._dataframe_to_html_table(pd.DataFrame(rows)) if rows else '<p class=no-data>No features to rank</p>'}
</section>"""

    def _dataset_summary_section(self) -> str:
        n_rows, n_cols = self.df.shape
        numeric_cols = len(self.df.select_dtypes(include=[np.number]).columns)
        categorical_cols = sum(
            1 for c in self.df.columns if self.df[c].dtype.name in ("object", "str", "category")
        )
        missing_total = int(self.df.isna().sum().sum())

        cards = "".join(
            f"<div class='summary-card'><h3>{label}</h3><div class='value'>{value}</div></div>"
            for label, value in [
                ("Total Rows", f"{n_rows:,}"),
                ("Total Columns", n_cols),
                ("Numeric Columns", numeric_cols),
                ("Categorical Columns", categorical_cols),
                ("Missing Values", f"{missing_total:,}"),
            ]
        )
        return f"<section class='section'><h2>Dataset Overview</h2>"\
               f"<div class='summary-grid'>{cards}</div></section>"

    def _statistics_section(self, stats: dict) -> str:
        if not stats:
            return ("<section class='section'><h2>Statistical Summary</h2>"
                    "<p class='no-data'>No statistical data available</p></section>")
        stats_df = pd.DataFrame(stats).T
        return (f"<section class='section'><h2>Statistical Summary</h2>"
                f"{self._dataframe_to_html_table(stats_df.round(4))}</section>")

    def _missing_values_section(self, missing_analysis: dict) -> str:
        if not missing_analysis:
            return ""
        total_missing = missing_analysis.get("total_missing", 0)

        if missing_analysis.get("no_missing", False):
            alert = "<div class='alert alert-success'>No missing values found.</div>"
        else:
            alert = (f"<div class='alert alert-warning'>Found "
                     f"{total_missing:,} missing values across "
                     f"{missing_analysis.get('missing_rows', 0):,} rows</div>")

        missing_pct = {
            k: v for k, v in missing_analysis.get("missing_percentage", {}).items() if v > 0
        }
        table_section = ""
        if missing_pct:
            missing_df = (
                pd.DataFrame(missing_pct.items(), columns=["Column", "Missing %"])
                .sort_values("Missing %", ascending=False).round(2)
            )
            table_section = ("<h3>Missing Percentage by Column</h3>"
                             + self._dataframe_to_html_table(missing_df))

        return (f"<section class='section'><h2>Missing Values Analysis</h2>"
                f"{alert}{table_section}</section>")

    def _distribution_section(self, dist_analysis: dict) -> str:
        if not dist_analysis:
            return ""

        sections = []
        numeric_dists = dist_analysis.get("numeric_distributions", {})
        if numeric_dists:
            rows = [{
                "Column": col,
                "Type": d["distribution_type"],
                "Normal": "yes" if d["is_normal"] else "no",
                "Normality p": _escape(
                    round(d["normality_pvalue"], 4)
                    if d["normality_pvalue"] == d["normality_pvalue"] else None),
                "Skewness": round(d["skewness"], 3),
                "Kurtosis": round(d["kurtosis"], 3),
                "Unique": d["unique_values"],
            } for col, d in numeric_dists.items()]
            sections.append("<h3>Numeric Columns</h3>"
                            + self._dataframe_to_html_table(pd.DataFrame(rows)))

        categorical_dists = dist_analysis.get("categorical_distributions", {})
        if categorical_dists:
            rows = [{
                "Column": col,
                "Unique Values": d["unique_values"],
                "Diversity Index": round(d["diversity_index"], 3),
                "Missing": d["missing_count"],
            } for col, d in categorical_dists.items()]
            sections.append("<h3>Categorical Columns</h3>"
                            + self._dataframe_to_html_table(pd.DataFrame(rows)))

        return (f"<section class='section'><h2>Distribution Analysis</h2>"
                f"{''.join(sections)}</section>") if sections else ""

    def _correlation_section(self, corr_analysis: dict) -> str:
        if not corr_analysis:
            return ""
        pearson = corr_analysis.get("pearson", {})
        strong = pearson.get("strong_correlations", [])
        moderate = pearson.get("moderate_correlations", [])

        sections = []
        for title, pairs in [
            ("Strong Correlations (|r| &gt; 0.7)", strong),
            ("Moderate Correlations (0.5 &lt; |r| &le; 0.7)", moderate),
        ]:
            if pairs:
                df = pd.DataFrame(pairs).round({"correlation": 4})
                df.columns = ["Column A", "Column B", "Correlation"]
                sections.append(f"<h3>{title}</h3>"
                                + self._dataframe_to_html_table(df))
        if not sections:
            sections.append("<div class='alert alert-info'>"
                            "No strong or moderate correlations found.</div>")

        return (f"<section class='section'><h2>Correlation Analysis</h2>"
                f"{''.join(sections)}</section>")

    def _outliers_section(self, outlier_analysis: dict) -> str:
        if not outlier_analysis:
            return ""
        iqr_results = outlier_analysis.get("iqr_method", {})
        if not iqr_results:
            return ""

        rows = [{
            "Column": col,
            "Outliers": d["outlier_count"],
            "Percentage": f"{d['outlier_percentage']:.2f}%",
            "Lower Bound": round(d["lower_bound"], 3),
            "Upper Bound": round(d["upper_bound"], 3),
        } for col, d in iqr_results.items()]

        iso_forest = outlier_analysis.get("isolation_forest", {})
        iso_text = ""
        if iso_forest and "n_outliers" in iso_forest:
            iso_text = (f"<div class='alert alert-info'>Isolation Forest: "
                        f"{iso_forest['n_outliers']} multivariate outliers "
                        f"({iso_forest['outlier_percentage']:.2f}%)</div>")

        return (f"<section class='section'><h2>Outlier Detection</h2>"
                f"<h3>IQR Method Results</h3>"
                f"{self._dataframe_to_html_table(pd.DataFrame(rows))}{iso_text}</section>")

    def _visualizations_section(self, figures: dict) -> str:
        blocks = []

        if figures.get("missing_values") is not None:
            blocks.append(self._chart_block("Missing Values", figures["missing_values"]))
        if figures.get("correlation") is not None:
            blocks.append(self._chart_block("Correlation Heatmap",
                                            figures["correlation"]))
        if figures.get("boxplot") is not None:
            blocks.append(self._chart_block("Boxplots of Numeric Columns",
                                            figures["boxplot"]))

        for col, fig in (figures.get("distributions") or {}).items():
            blocks.append(self._chart_block(f"Distribution - {col}", fig))
        for col, fig in (figures.get("categorical_distributions") or {}).items():
            blocks.append(self._chart_block(f"Categories - {col}", fig))

        outlier_figs = list((figures.get("outliers") or {}).items())[
            : self.MAX_OUTLIER_CHARTS
        ]
        for col, fig in outlier_figs:
            blocks.append(self._chart_block(f"Outlier Detection - {col}", fig))

        if figures.get("scatter_matrix") is not None:
            blocks.append(self._chart_block("Scatter Matrix",
                                            figures["scatter_matrix"], height=800))

        blocks = [b for b in blocks if b]
        if not blocks:
            return ""
        return (f"<section class='section'><h2>Interactive Charts</h2>"
                f"{''.join(blocks)}</section>")

    def _insights_section(self, insights: dict) -> str:
        if not insights:
            return ""
        boxes = "".join(
            f"<div class='insight-box'><div class='insight-title'>{_escape(k)}</div>"
            f"<div class='insight-text'>{_escape(v)}</div></div>"
            for k, v in insights.items()
        )
        return (f"<section class='section'><h2>Insights</h2>{boxes}</section>")

    def _footer(self) -> str:
        return f"""<footer class="footer">
<p><strong>Autolyse</strong> - Automated Exploratory Data Analysis</p>
<p>Generated on {self.timestamp}</p>
</footer>"""

    # ---------------------------------------------------------------- tables

    @staticmethod
    def _dataframe_to_html_table(df: pd.DataFrame) -> str:
        header = "".join(f"<th>{_escape(c)}</th>" for c in df.columns)
        rows = []
        for _, row in df.iterrows():
            cells = "".join(f"<td>{_escape(v)}</td>" for v in row)
            rows.append(f"<tr>{cells}</tr>")
        return (f"<table>\n<thead>\n<tr>{header}</tr>\n</thead>\n<tbody>\n"
                + "\n".join(rows) + "\n</tbody>\n</table>")
