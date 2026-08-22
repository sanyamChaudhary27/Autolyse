"""Matplotlib visualization module for static plots."""

from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde


@contextmanager
def _plot_style():
    """Apply seaborn styling without permanently mutating user rcParams."""
    with sns.axes_style("whitegrid"), sns.color_palette("husl"):
        yield


class MatplotlibVisualizer:
    """Create static visualizations using matplotlib and seaborn."""

    #: Above this many columns, heatmap cell annotations become unreadable.
    ANNOT_LIMIT = 15

    def __init__(self, df: pd.DataFrame, figsize=(12, 6)):
        self.df = df
        self.figsize = figsize
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = [
            c for c in df.columns if df[c].dtype.name in ("object", "str", "category")
        ]

    # ------------------------------------------------------------- helpers

    @staticmethod
    def _kde_curve(col_data: np.ndarray):
        """Return (x, y) density curve, or None when KDE is undefined."""
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

    # ---------------------------------------------------------- distributions

    def plot_distributions(self) -> dict:
        """Histogram + KDE panel per numeric column."""
        figures = {}
        for col in self.numeric_cols:
            col_data = self.df[col].dropna()
            if len(col_data) == 0:
                continue

            with _plot_style():
                fig, axes = plt.subplots(1, 2, figsize=self.figsize)
                fig.suptitle(f"Distribution of {col}", fontsize=14, fontweight="bold")

                axes[0].hist(col_data, bins=30, color="skyblue",
                             edgecolor="black", alpha=0.7)
                axes[0].set_xlabel(col, fontsize=11)
                axes[0].set_ylabel("Frequency", fontsize=11)
                axes[0].set_title("Histogram", fontsize=12)

                kde = self._kde_curve(col_data)
                if kde is not None:
                    grid, density = kde
                    axes[1].plot(grid, density, color="darkblue", linewidth=2)
                    axes[1].fill_between(grid, density, alpha=0.3, color="skyblue")
                else:
                    axes[1].text(0.5, 0.5, "KDE unavailable\n(constant or too few values)",
                                 ha="center", va="center", fontsize=10, color="gray")
                axes[1].set_xlabel(col, fontsize=11)
                axes[1].set_ylabel("Density", fontsize=11)
                axes[1].set_title("Density (KDE)", fontsize=12)

                plt.tight_layout()
                figures[col] = fig
        return figures

    def plot_categorical_distributions(self) -> dict:
        """Top-10 frequency bars per categorical column."""
        figures = {}
        for col in self.categorical_cols:
            value_counts = self.df[col].value_counts().head(10)
            if value_counts.empty:
                continue

            with _plot_style():
                fig, ax = plt.subplots(figsize=self.figsize)
                value_counts.plot(kind="barh", ax=ax, color="teal")
                ax.set_title(f"Distribution of {col} (Top 10)",
                             fontsize=14, fontweight="bold")
                ax.set_xlabel("Count", fontsize=11)
                ax.set_ylabel(col, fontsize=11)
                ax.grid(alpha=0.3, axis="x")
                plt.tight_layout()
                figures[col] = fig
        return figures

    # ---------------------------------------------------------------- quality

    def plot_missing_values(self, missing_analysis: dict):
        """Horizontal bars of missing % per affected column."""
        missing_pct = missing_analysis.get("missing_percentage", {})
        if not missing_pct or all(v == 0 for v in missing_pct.values()):
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, "No Missing Values Found", ha="center", va="center",
                    fontsize=16, fontweight="bold")
            ax.axis("off")
            return fig

        missing_cols = {k: v for k, v in missing_pct.items() if v > 0}
        missing_cols = dict(sorted(missing_cols.items(), key=lambda kv: kv[1],
                                   reverse=True))

        with _plot_style():
            fig, ax = plt.subplots(figsize=self.figsize)
            colors = ["red" if v > 20 else "orange" if v > 5 else "yellow"
                      for v in missing_cols.values()]
            ax.barh(list(missing_cols.keys()), list(missing_cols.values()), color=colors)
            ax.set_xlabel("Missing Percentage (%)", fontsize=11)
            ax.set_title("Missing Values Analysis", fontsize=14, fontweight="bold")
            ax.grid(alpha=0.3, axis="x")

            for i, v in enumerate(missing_cols.values()):
                ax.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=10)

            plt.tight_layout()
        return fig

    # ------------------------------------------------------------ correlation

    def plot_correlation_heatmap(self, corr_matrix: pd.DataFrame):
        if corr_matrix is None or len(corr_matrix) < 2:
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, "Not enough numeric columns for correlation",
                    ha="center", va="center", fontsize=14)
            ax.axis("off")
            return fig

        size = min(12, max(len(corr_matrix) + 2, 6))
        with _plot_style():
            fig, ax = plt.subplots(figsize=(size, size * 0.85))
            sns.heatmap(
                corr_matrix,
                annot=len(corr_matrix) <= self.ANNOT_LIMIT,
                fmt=".2f", cmap="coolwarm", center=0, square=True,
                vmin=-1, vmax=1,
                cbar_kws={"label": "Correlation"}, ax=ax,
            )
            ax.set_title("Pearson Correlation Heatmap", fontsize=14,
                         fontweight="bold")
            plt.tight_layout()
        return fig

    # --------------------------------------------------------------- outliers

    def plot_outliers(self, col: str, outlier_bounds: dict):
        if col not in self.numeric_cols:
            return None

        col_data = self.df[col].dropna()
        lower, upper = outlier_bounds["lower_bound"], outlier_bounds["upper_bound"]
        is_outlier = (col_data < lower) | (col_data > upper)

        with _plot_style():
            fig, ax = plt.subplots(figsize=self.figsize)
            normal = col_data[~is_outlier]
            ax.scatter(normal.index, normal.to_numpy(), color="blue",
                       alpha=0.6, label="Normal", s=30)

            if is_outlier.any():
                flagged = col_data[is_outlier]
                ax.scatter(flagged.index, flagged.to_numpy(), color="red",
                           alpha=0.8, label="Outliers", s=80, marker="X")

            ax.axhline(lower, color="orange", linestyle="--", linewidth=2,
                       label="Lower Bound")
            ax.axhline(upper, color="orange", linestyle="--", linewidth=2,
                       label="Upper Bound")
            ax.set_xlabel("Row index", fontsize=11)
            ax.set_ylabel(col, fontsize=11)
            ax.set_title(f"Outlier Detection - {col}", fontsize=14,
                         fontweight="bold")
            ax.legend(loc="best")
            ax.grid(alpha=0.3)
            plt.tight_layout()
        return fig

    # -------------------------------------------------------------- boxplots

    def plot_boxplot(self):
        if len(self.numeric_cols) == 0:
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, "No numeric columns", ha="center", va="center",
                    fontsize=14)
            ax.axis("off")
            return fig

        data_to_plot = [self.df[c].dropna().to_numpy() for c in self.numeric_cols]

        with _plot_style():
            fig, ax = plt.subplots(figsize=self.figsize)
            bp = ax.boxplot(data_to_plot, patch_artist=True)
            for patch in bp["boxes"]:
                patch.set_facecolor("lightblue")
            ax.set_xticklabels(self.numeric_cols, rotation=45, ha="right")
            ax.set_ylabel("Value", fontsize=11)
            ax.set_title("Boxplots of Numeric Columns", fontsize=14,
                         fontweight="bold")
            ax.grid(alpha=0.3, axis="y")
            plt.tight_layout()
        return fig

    def plot_scatter_matrix(self, max_cols: int = 4):
        """Pairwise scatter grid.

        Rows are dropped jointly per pair so x and y always align - dropping
        each axis independently produced length mismatches (ValueError)
        whenever two columns had different missing patterns.
        """
        if len(self.numeric_cols) < 2:
            return None

        cols_to_use = self.numeric_cols[:max_cols]
        n = len(cols_to_use)

        with _plot_style():
            fig, grid_axes = plt.subplots(n, n, figsize=(12, 10))
            axes = np.atleast_2d(grid_axes)

            for i, col_row in enumerate(cols_to_use):
                for j, col_col in enumerate(cols_to_use):
                    ax = axes[i][j]

                    if i == j:
                        ax.hist(self.df[col_row].dropna(), bins=20,
                                color="skyblue", alpha=0.7)
                    else:
                        pair = self.df[[col_row, col_col]].dropna()
                        ax.scatter(pair[col_col], pair[col_row], alpha=0.5, s=20)

                    if i == n - 1:
                        ax.set_xlabel(col_col, fontsize=9)
                    if j == 0:
                        ax.set_ylabel(col_row, fontsize=9)
                    ax.tick_params(labelsize=8)

            fig.suptitle("Scatter Plot Matrix", fontsize=14, fontweight="bold")
            plt.tight_layout()
        return fig

    # ----------------------------------------------------------- relationships

    def plot_categorical_numeric_relationships(self, cat_col: str, num_col: str):
        if cat_col not in self.categorical_cols or num_col not in self.numeric_cols:
            return None

        with _plot_style():
            fig, ax = plt.subplots(figsize=self.figsize)
            grouped = [g[num_col].dropna().to_numpy()
                       for _, g in self.df.groupby(cat_col) if g[num_col].notna().any()]
            labels = [str(k) for k, g in self.df.groupby(cat_col)
                      if g[num_col].notna().any()]

            ax.boxplot(grouped, tick_labels=labels) if grouped else None
            ax.set_title(f"{num_col} by {cat_col}", fontsize=14, fontweight="bold")
            ax.set_xlabel(cat_col, fontsize=11)
            ax.set_ylabel(num_col, fontsize=11)
            plt.tight_layout()
        return fig

    def save_figure(self, fig: plt.Figure, filepath: str) -> None:
        fig.savefig(filepath, dpi=300, bbox_inches="tight")
