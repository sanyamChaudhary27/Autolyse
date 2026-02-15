"""Matplotlib visualization module for static plots"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
sns.set_palette("husl")


class MatplotlibVisualizer:
    """Create static visualizations using matplotlib and seaborn"""
    
    def __init__(self, df: pd.DataFrame, figsize: Tuple[int, int] = (12, 6)):
        """
        Initialize visualizer.
        
        Args:
            df: Input dataframe
            figsize: Default figure size for plots
        """
        self.df = df
        self.figsize = figsize
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    def plot_distributions(self) -> Dict[str, plt.Figure]:
        """
        Create distribution plots for numeric columns.
        
        Returns:
            Dictionary of column names to figure objects
        """
        figures = {}
        
        for col in self.numeric_cols:
            col_data = self.df[col].dropna()
            
            if len(col_data) == 0:
                continue
            
            fig, axes = plt.subplots(1, 2, figsize=self.figsize)
            fig.suptitle(f'Distribution of {col}', fontsize=14, fontweight='bold')
            
            # Histogram
            axes[0].hist(col_data, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
            axes[0].set_xlabel(col, fontsize=11)
            axes[0].set_ylabel('Frequency', fontsize=11)
            axes[0].set_title('Histogram', fontsize=12)
            axes[0].grid(alpha=0.3)
            
            # KDE plot
            col_data.plot(kind='kde', ax=axes[1], color='darkblue', linewidth=2)
            axes[1].fill_between(axes[1].get_lines()[0].get_xdata(), 
                                 axes[1].get_lines()[0].get_ydata(), 
                                 alpha=0.3, color='skyblue')
            axes[1].set_xlabel(col, fontsize=11)
            axes[1].set_ylabel('Density', fontsize=11)
            axes[1].set_title('KDE Plot', fontsize=12)
            axes[1].grid(alpha=0.3)
            
            plt.tight_layout()
            figures[col] = fig
        
        return figures
    
    def plot_categorical_distributions(self) -> Dict[str, plt.Figure]:
        """
        Create bar plots for categorical column distributions.
        
        Returns:
            Dictionary of column names to figure objects
        """
        figures = {}
        
        for col in self.categorical_cols:
            value_counts = self.df[col].value_counts().head(10)  # Top 10 categories
            
            fig, ax = plt.subplots(figsize=self.figsize)
            value_counts.plot(kind='barh', ax=ax, color='teal')
            ax.set_title(f'Distribution of {col} (Top 10)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Count', fontsize=11)
            ax.set_ylabel(col, fontsize=11)
            ax.grid(alpha=0.3, axis='x')
            
            plt.tight_layout()
            figures[col] = fig
        
        return figures
    
    def plot_missing_values(self, missing_analysis: Dict[str, Any]) -> plt.Figure:
        """
        Create visualization of missing values.
        
        Args:
            missing_analysis: Output from MissingValuesAnalyzer.analyze()
        
        Returns:
            Figure object
        """
        missing_pct = missing_analysis['missing_percentage']
        
        if not missing_pct or all(v == 0 for v in missing_pct.values()):
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, 'No Missing Values Found', 
                   ha='center', va='center', fontsize=16, fontweight='bold')
            ax.axis('off')
            return fig
        
        # Filter columns with missing values
        missing_cols = {k: v for k, v in missing_pct.items() if v > 0}
        missing_cols = dict(sorted(missing_cols.items(), key=lambda x: x[1], reverse=True))
        
        fig, ax = plt.subplots(figsize=self.figsize)
        colors = ['red' if v > 20 else 'orange' if v > 5 else 'yellow' for v in missing_cols.values()]
        
        ax.barh(list(missing_cols.keys()), list(missing_cols.values()), color=colors)
        ax.set_xlabel('Missing Percentage (%)', fontsize=11)
        ax.set_title('Missing Values Analysis', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3, axis='x')
        
        # Add percentage labels
        for i, v in enumerate(missing_cols.values()):
            ax.text(v + 1, i, f'{v:.1f}%', va='center', fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def plot_correlation_heatmap(self, corr_matrix: pd.DataFrame) -> plt.Figure:
        """
        Create correlation heatmap.
        
        Args:
            corr_matrix: Correlation matrix from CorrelationAnalyzer
        
        Returns:
            Figure object
        """
        if corr_matrix is None or len(corr_matrix) < 2:
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, 'Not enough numeric columns for correlation', 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        fig, ax = plt.subplots(figsize=(min(12, len(corr_matrix) + 2), 
                                        min(10, len(corr_matrix) + 2)))
        
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, cbar_kws={'label': 'Correlation'},
                   ax=ax, vmin=-1, vmax=1)
        
        ax.set_title('Pearson Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def plot_outliers(self, col: str, outlier_bounds: Dict[str, float]) -> plt.Figure:
        """
        Create scatter plot highlighting outliers.
        
        Args:
            col: Column name
            outlier_bounds: Dictionary with 'lower_bound' and 'upper_bound'
        
        Returns:
            Figure object
        """
        if col not in self.numeric_cols:
            return None
        
        col_data = self.df[col].dropna()
        lower = outlier_bounds['lower_bound']
        upper = outlier_bounds['upper_bound']
        
        is_outlier = (col_data < lower) | (col_data > upper)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot normal values
        ax.scatter(range(len(col_data[~is_outlier])), col_data[~is_outlier], 
                  color='blue', alpha=0.6, label='Normal', s=30)
        
        # Plot outliers
        if is_outlier.sum() > 0:
            outlier_indices = np.where(is_outlier)[0]
            ax.scatter(outlier_indices, col_data[is_outlier], 
                      color='red', alpha=0.8, label='Outliers', s=80, marker='X')
        
        # Add bounds
        ax.axhline(lower, color='orange', linestyle='--', linewidth=2, label='Lower Bound')
        ax.axhline(upper, color='orange', linestyle='--', linewidth=2, label='Upper Bound')
        
        ax.set_xlabel('Index', fontsize=11)
        ax.set_ylabel(col, fontsize=11)
        ax.set_title(f'Outlier Detection - {col}', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_boxplot(self) -> plt.Figure:
        """
        Create boxplots for all numeric columns.
        
        Returns:
            Figure object
        """
        if len(self.numeric_cols) == 0:
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, 'No numeric columns', 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Prepare data for boxplot
        data_to_plot = [self.df[col].dropna() for col in self.numeric_cols]
        
        bp = ax.boxplot(data_to_plot, labels=self.numeric_cols, patch_artist=True)
        
        # Color the boxes
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        
        ax.set_ylabel('Value', fontsize=11)
        ax.set_title('Boxplots of Numeric Columns', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        return fig
    
    def plot_scatter_matrix(self, max_cols: int = 4) -> Optional[plt.Figure]:
        """
        Create scatter plot matrix for numeric columns (limited to avoid too many plots).
        
        Args:
            max_cols: Maximum number of columns to include
        
        Returns:
            Figure object or None if not enough numeric columns
        """
        if len(self.numeric_cols) < 2:
            return None
        
        cols_to_use = self.numeric_cols[:max_cols]
        fig = plt.figure(figsize=(12, 10))
        
        n_cols = len(cols_to_use)
        for i, col1 in enumerate(cols_to_use):
            for j, col2 in enumerate(cols_to_use):
                ax = fig.add_subplot(n_cols, n_cols, i * n_cols + j + 1)
                
                if i == j:
                    # Diagonal: histogram
                    ax.hist(self.df[col1].dropna(), bins=20, color='skyblue', alpha=0.7)
                    ax.set_ylabel(col1, fontsize=9)
                else:
                    # Off-diagonal: scatter plot
                    ax.scatter(self.df[col2].dropna(), self.df[col1].dropna(), 
                              alpha=0.5, s=20)
                    ax.set_ylabel(col1, fontsize=9)
                
                ax.set_xlabel(col2, fontsize=9)
                ax.tick_params(labelsize=8)
        
        fig.suptitle('Scatter Plot Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def plot_categorical_numeric_relationships(self, cat_col: str, num_col: str) -> plt.Figure:
        """
        Create boxplot showing numeric variable distribution across categories.
        
        Args:
            cat_col: Categorical column name
            num_col: Numeric column name
        
        Returns:
            Figure object
        """
        if cat_col not in self.categorical_cols or num_col not in self.numeric_cols:
            return None
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create boxplot
        self.df.boxplot(column=num_col, by=cat_col, ax=ax)
        ax.set_title(f'{num_col} by {cat_col}', fontsize=14, fontweight='bold')
        ax.set_xlabel(cat_col, fontsize=11)
        ax.set_ylabel(num_col, fontsize=11)
        plt.suptitle('')  # Remove the automatic title
        
        plt.tight_layout()
        return fig
    
    def save_figure(self, fig: plt.Figure, filepath: str) -> None:
        """
        Save figure to file.
        
        Args:
            fig: Matplotlib figure object
            filepath: Path to save the figure
        """
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {filepath}")
