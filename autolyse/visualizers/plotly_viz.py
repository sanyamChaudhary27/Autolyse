"""Plotly visualization module for interactive plots."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde


def _kde_curve(col_data: np.ndarray):
    """Return (grid, density) or None when KDE is undefined."""
    col_data = np.asarray(col_data, dtype=float)
    col_data = col_data[np.isfinite(col_data)]
    if len(col_data) < 3 or np.std(col_data) == 0:
        return None
    try:
        kde = gaussian_kde(col_data)
        grid = np.linspace(col_data.min(), col_data.max(), 200)
        return grid, kde(grid)
    except np.linalg.LinAlgError:
        return None


class PlotlyVisualizer:
    """Create interactive visualizations using plotly."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = [
            c for c in df.columns if df[c].dtype.name in ("object", "str", "category")
        ]

    def plot_distributions(self) -> dict:
        """Histogram with an honest KDE overlay panel per numeric column."""
        figures = {}
        for col in self.numeric_cols:
            col_data = self.df[col].dropna()
            if len(col_data) == 0:
                continue

            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Histogram", "Density (KDE)"),
                specs=[[{"type": "histogram"}, {"type": "xy"}]],
            )

            fig.add_trace(
                go.Histogram(x=col_data, nbinsx=30, name="Count",
                             marker_color="skyblue", showlegend=False),
                row=1, col=1,
            )

            kde = _kde_curve(col_data.to_numpy())
            if kde is not None:
                grid, density = kde
                fig.add_trace(
                    go.Scatter(x=grid, y=density, name="Density",
                               mode="lines", fill="tozeroy",
                               line=dict(color="darkblue", width=2),
                               fillcolor="rgba(135,206,235,0.3)",
                               showlegend=False),
                    row=1, col=2,
                )
            else:
                fig.add_annotation(
                    text="KDE unavailable (constant or too few values)",
                    xref="x2", yref="y2", x=0.5, y=0.5, showarrow=False,
                    font=dict(size=12, color="gray"),
                )

            fig.update_xaxes(title_text=col, row=1, col=1)
            fig.update_xaxes(title_text=col, row=1, col=2)
            fig.update_yaxes(title_text="Frequency", row=1, col=1)
            fig.update_yaxes(title_text="Density", row=1, col=2)
            fig.update_layout(title_text=f"Distribution of {col}",
                              height=400, hovermode="x unified")

            figures[col] = fig
        return figures

    def plot_categorical_distributions(self) -> dict:
        figures = {}
        for col in self.categorical_cols:
            value_counts = self.df[col].value_counts().head(10)
            if value_counts.empty:
                continue

            fig = go.Figure(data=[go.Bar(
                y=value_counts.index.astype(str).tolist(),
                x=value_counts.values,
                orientation="h",
                marker_color="teal",
                hovertemplate="<b>%{y}</b><br>Count: %{x}<extra></extra>",
            )])
            fig.update_layout(
                title=f"Distribution of {col} (Top 10)",
                xaxis_title="Count", yaxis_title=col,
                height=400, hovermode="closest",
            )
            figures[col] = fig
        return figures

    def plot_missing_values(self, missing_analysis: dict) -> go.Figure:
        missing_pct = missing_analysis.get("missing_percentage", {})
        if not missing_pct or all(v == 0 for v in missing_pct.values()):
            fig = go.Figure()
            fig.add_annotation(text="No Missing Values Found",
                               xref="paper", yref="paper", x=0.5, y=0.5,
                               showarrow=False, font=dict(size=20))
            fig.update_layout(height=400)
            return fig

        missing_cols = {k: v for k, v in missing_pct.items() if v > 0}
        missing_cols = dict(sorted(missing_cols.items(), key=lambda kv: kv[1],
                                   reverse=True))
        colors = ["red" if v > 20 else "orange" if v > 5 else "yellow"
                  for v in missing_cols.values()]

        fig = go.Figure(data=[go.Bar(
            y=list(missing_cols.keys()),
            x=list(missing_cols.values()),
            orientation="h",
            marker_color=colors,
            text=[f"{v:.1f}%" for v in missing_cols.values()],
            textposition="auto",
            hovertemplate="<b>%{y}</b><br>Missing: %{x:.2f}%<extra></extra>",
        )])
        fig.update_layout(
            title="Missing Values Analysis",
            xaxis_title="Missing Percentage (%)",
            yaxis_title="Column", height=400,
        )
        return fig

    def plot_correlation_heatmap(self, corr_matrix: pd.DataFrame) -> go.Figure:
        if corr_matrix is None or len(corr_matrix) < 2:
            fig = go.Figure()
            fig.add_annotation(text="Not enough numeric columns for correlation",
                               xref="paper", yref="paper", x=0.5, y=0.5,
                               showarrow=False, font=dict(size=14))
            return fig

        annot = len(corr_matrix) <= 15
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=[str(c) for c in corr_matrix.columns],
            y=[str(c) for c in corr_matrix.columns],
            colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
            text=corr_matrix.round(3) if annot else None,
            texttemplate=".2f" if annot else None,
            textfont={"size": 10},
            colorbar_title="Correlation",
            hovertemplate="%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>",
        ))
        fig.update_layout(
            title="Pearson Correlation Heatmap",
            height=max(400, len(corr_matrix) * 30),
            width=max(500, len(corr_matrix) * 30),
        )
        return fig

    def plot_outliers(self, col: str, col_data: pd.Series,
                      outlier_bounds: dict) -> go.Figure:
        col_data = col_data.dropna()
        lower, upper = outlier_bounds["lower_bound"], outlier_bounds["upper_bound"]
        is_outlier = (col_data < lower) | (col_data > upper)

        fig = go.Figure()

        normal_data = col_data[~is_outlier]
        fig.add_trace(go.Scatter(
            x=normal_data.index, y=normal_data.values,
            mode="markers", name="Normal",
            marker=dict(color="blue", size=6, opacity=0.6),
            hovertemplate="Index: %{x}<br>Value: %{y:.2f}<extra></extra>",
        ))

        if is_outlier.any():
            flagged = col_data[is_outlier]
            fig.add_trace(go.Scatter(
                x=flagged.index, y=flagged.values,
                mode="markers", name="Outliers",
                marker=dict(color="red", size=10, symbol="x"),
                hovertemplate="Index: %{x}<br>Value: %{y:.2f}<extra></extra>",
            ))

        fig.add_hline(y=lower, line_dash="dash", line_color="orange",
                      annotation_text="Lower Bound", annotation_position="right")
        fig.add_hline(y=upper, line_dash="dash", line_color="orange",
                      annotation_text="Upper Bound", annotation_position="right")

        fig.update_layout(
            title=f"Outlier Detection - {col}",
            xaxis_title="Row index", yaxis_title=col,
            height=400, showlegend=True,
        )
        return fig

    def plot_boxplot(self) -> go.Figure:
        if len(self.numeric_cols) == 0:
            fig = go.Figure()
            fig.add_annotation(text="No numeric columns", xref="paper",
                               yref="paper", x=0.5, y=0.5, showarrow=False,
                               font=dict(size=14))
            return fig

        fig = go.Figure()
        for col in self.numeric_cols:
            fig.add_trace(go.Box(
                y=self.df[col].dropna(), name=str(col), boxmean="sd",
            ))
        fig.update_layout(
            title="Boxplots of Numeric Columns",
            yaxis_title="Value", height=400, showlegend=False,
        )
        return fig

    def plot_scatter_matrix(self, max_cols: int = 4):
        if len(self.numeric_cols) < 2:
            return None

        cols_to_use = self.numeric_cols[:max_cols]
        fig = px.scatter_matrix(
            self.df[cols_to_use].dropna(how="all"),
            dimensions=cols_to_use,
            title="Scatter Plot Matrix",
            height=800,
        )
        fig.update_traces(diagonal_visible=False, showupperhalf=False)
        return fig

    def plot_categorical_numeric_relationships(self, cat_col: str, num_col: str):
        if cat_col not in self.categorical_cols or num_col not in self.numeric_cols:
            return None

        fig = px.box(self.df, x=cat_col, y=num_col,
                     title=f"{num_col} by {cat_col}", height=400)
        fig.update_layout(hovermode="closest")
        return fig

    def plot_pair_histogram(self, col1: str, col2: str):
        if col1 not in self.numeric_cols or col2 not in self.numeric_cols:
            return None

        pair = self.df[[col1, col2]].dropna()
        fig = go.Figure(data=go.Histogram2d(
            x=pair[col1], y=pair[col2], nbinsx=30, nbinsy=30,
            colorscale="Viridis",
            hovertemplate="%{x:.2f}, %{y:.2f}<br>Count: %{z}<extra></extra>",
        ))
        fig.update_layout(
            title=f"2D Distribution: {col1} vs {col2}",
            xaxis_title=col1, yaxis_title=col2, height=400,
        )
        return fig

    def save_figure(self, fig: go.Figure, filepath: str) -> None:
        fig.write_html(filepath)
