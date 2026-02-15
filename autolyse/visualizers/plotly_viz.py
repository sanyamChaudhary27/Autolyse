"""Plotly visualization module for interactive plots"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import warnings

warnings.filterwarnings('ignore')


class PlotlyVisualizer:
    """Create interactive visualizations using plotly"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize visualizer.
        
        Args:
            df: Input dataframe
        """
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    def plot_distributions(self) -> Dict[str, go.Figure]:
        """
        Create interactive distribution plots for numeric columns.
        
        Returns:
            Dictionary of column names to plotly figure objects
        """
        figures = {}
        
        for col in self.numeric_cols:
            col_data = self.df[col].dropna()
            
            if len(col_data) == 0:
                continue
            
            fig = make_subplots(rows=1, cols=2, 
                              subplot_titles=('Histogram', 'Distribution'),
                              specs=[[{"type": "histogram"}, {"type": "histogram"}]])
            
            # Histogram
            fig.add_trace(
                go.Histogram(x=col_data, nbinsx=30, name='Histogram',
                           marker_color='skyblue', showlegend=False),
                row=1, col=1
            )
            
            # Distribution with KDE (using histogram with small bins)
            fig.add_trace(
                go.Histogram(x=col_data, nbinsx=60, name='Distribution',
                           marker_color='lightblue', opacity=0.7,
                           showlegend=False, cumulative_enabled=False),
                row=1, col=2
            )
            
            fig.update_xaxes(title_text=col, row=1, col=1)
            fig.update_xaxes(title_text=col, row=1, col=2)
            fig.update_yaxes(title_text='Frequency', row=1, col=1)
            fig.update_yaxes(title_text='Frequency', row=1, col=2)
            
            fig.update_layout(
                title_text=f'Distribution of {col}',
                height=400,
                showlegend=False,
                hovermode='x unified'
            )
            
            figures[col] = fig
        
        return figures
    
    def plot_categorical_distributions(self) -> Dict[str, go.Figure]:
        """
        Create interactive bar plots for categorical column distributions.
        
        Returns:
            Dictionary of column names to figure objects
        """
        figures = {}
        
        for col in self.categorical_cols:
            value_counts = self.df[col].value_counts().head(10)
            
            fig = go.Figure(data=[
                go.Bar(
                    y=value_counts.index,
                    x=value_counts.values,
                    orientation='h',
                    marker_color='teal',
                    hovertemplate='<b>%{y}</b><br>Count: %{x}<extra></extra>'
                )
            ])
            
            fig.update_layout(
                title=f'Distribution of {col} (Top 10)',
                xaxis_title='Count',
                yaxis_title=col,
                height=400,
                hovermode='closest'
            )
            
            figures[col] = fig
        
        return figures
    
    def plot_missing_values(self, missing_analysis: Dict[str, Any]) -> go.Figure:
        """
        Create interactive visualization of missing values.
        
        Args:
            missing_analysis: Output from MissingValuesAnalyzer.analyze()
        
        Returns:
            Plotly figure object
        """
        missing_pct = missing_analysis['missing_percentage']
        
        if not missing_pct or all(v == 0 for v in missing_pct.values()):
            fig = go.Figure()
            fig.add_annotation(
                text='No Missing Values Found',
                xref='paper', yref='paper',
                x=0.5, y=0.5, showarrow=False, fontsize=20
            )
            fig.update_layout(height=400)
            return fig
        
        # Filter columns with missing values
        missing_cols = {k: v for k, v in missing_pct.items() if v > 0}
        missing_cols = dict(sorted(missing_cols.items(), key=lambda x: x[1], reverse=True))
        
        # Create color gradient based on severity
        colors = []
        for v in missing_cols.values():
            if v > 20:
                colors.append('red')
            elif v > 5:
                colors.append('orange')
            else:
                colors.append('yellow')
        
        fig = go.Figure(data=[
            go.Bar(
                y=list(missing_cols.keys()),
                x=list(missing_cols.values()),
                orientation='h',
                marker_color=colors,
                text=[f'{v:.1f}%' for v in missing_cols.values()],
                textposition='auto',
                hovertemplate='<b>%{y}</b><br>Missing: %{x:.2f}%<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title='Missing Values Analysis',
            xaxis_title='Missing Percentage (%)',
            yaxis_title='Column',
            height=400,
            hovermode='closest'
        )
        
        return fig
    
    def plot_correlation_heatmap(self, corr_matrix: pd.DataFrame) -> go.Figure:
        """
        Create interactive correlation heatmap.
        
        Args:
            corr_matrix: Correlation matrix from CorrelationAnalyzer
        
        Returns:
            Plotly figure object
        """
        if corr_matrix is None or len(corr_matrix) < 2:
            fig = go.Figure()
            fig.add_annotation(
                text='Not enough numeric columns for correlation',
                xref='paper', yref='paper',
                x=0.5, y=0.5, showarrow=False, fontsize=14
            )
            return fig
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            text=corr_matrix.values,
            texttemplate='.2f',
            textfont={"size": 10},
            colorbar_title='Correlation',
            hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Pearson Correlation Heatmap',
            height=max(400, len(corr_matrix) * 30),
            width=max(400, len(corr_matrix) * 30)
        )
        
        return fig
    
    def plot_outliers(self, col: str, col_data: pd.Series, 
                     outlier_bounds: Dict[str, float]) -> go.Figure:
        """
        Create interactive scatter plot highlighting outliers.
        
        Args:
            col: Column name
            col_data: Series of the column data
            outlier_bounds: Dictionary with 'lower_bound' and 'upper_bound'
        
        Returns:
            Plotly figure object
        """
        col_data = col_data.dropna()
        lower = outlier_bounds['lower_bound']
        upper = outlier_bounds['upper_bound']
        
        is_outlier = (col_data < lower) | (col_data > upper)
        
        fig = go.Figure()
        
        # Add normal values
        normal_data = col_data[~is_outlier]
        fig.add_trace(go.Scatter(
            x=normal_data.index,
            y=normal_data.values,
            mode='markers',
            name='Normal',
            marker=dict(color='blue', size=6, opacity=0.6),
            hovertemplate='Index: %{x}<br>Value: %{y:.2f}<extra></extra>'
        ))
        
        # Add outliers
        if is_outlier.sum() > 0:
            outlier_data = col_data[is_outlier]
            fig.add_trace(go.Scatter(
                x=outlier_data.index,
                y=outlier_data.values,
                mode='markers',
                name='Outliers',
                marker=dict(color='red', size=10, symbol='x'),
                hovertemplate='Index: %{x}<br>Value: %{y:.2f}<extra></extra>'
            ))
        
        # Add bounds
        fig.add_hline(y=lower, line_dash='dash', line_color='orange',
                     annotation_text='Lower Bound', annotation_position='right')
        fig.add_hline(y=upper, line_dash='dash', line_color='orange',
                     annotation_text='Upper Bound', annotation_position='right')
        
        fig.update_layout(
            title=f'Outlier Detection - {col}',
            xaxis_title='Index',
            yaxis_title=col,
            height=400,
            hovermode='closest',
            showlegend=True
        )
        
        return fig
    
    def plot_boxplot(self) -> go.Figure:
        """
        Create interactive boxplots for numeric columns.
        
        Returns:
            Plotly figure object
        """
        if len(self.numeric_cols) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text='No numeric columns',
                xref='paper', yref='paper',
                x=0.5, y=0.5, showarrow=False, fontsize=14
            )
            return fig
        
        fig = go.Figure()
        
        for col in self.numeric_cols:
            fig.add_trace(go.Box(
                y=self.df[col].dropna(),
                name=col,
                boxmean='sd'  # Show mean and std dev
            ))
        
        fig.update_layout(
            title='Boxplots of Numeric Columns',
            yaxis_title='Value',
            height=400,
            hovermode='closest',
            showlegend=True
        )
        
        return fig
    
    def plot_scatter_matrix(self, max_cols: int = 4) -> Optional[go.Figure]:
        """
        Create interactive scatter plot matrix for numeric columns.
        
        Args:
            max_cols: Maximum number of columns to include
        
        Returns:
            Plotly figure object or None if not enough numeric columns
        """
        if len(self.numeric_cols) < 2:
            return None
        
        cols_to_use = self.numeric_cols[:max_cols]
        
        fig = px.scatter_matrix(
            self.df[cols_to_use].dropna(),
            dimensions=cols_to_use,
            hover_data={col: ':.2f' for col in cols_to_use},
            title='Scatter Plot Matrix',
            height=800,
            labels={col: col for col in cols_to_use}
        )
        
        fig.update_traces(diagonal_visible=False, showupperhalf=False)
        
        return fig
    
    def plot_categorical_numeric_relationships(self, cat_col: str, num_col: str) -> go.Figure:
        """
        Create interactive boxplot showing numeric variable distribution across categories.
        
        Args:
            cat_col: Categorical column name
            num_col: Numeric column name
        
        Returns:
            Plotly figure object
        """
        if cat_col not in self.categorical_cols or num_col not in self.numeric_cols:
            return None
        
        fig = px.box(
            self.df,
            x=cat_col,
            y=num_col,
            title=f'{num_col} by {cat_col}',
            labels={cat_col: cat_col, num_col: num_col},
            height=400
        )
        
        fig.update_layout(hovermode='closest')
        
        return fig
    
    def plot_pair_histogram(self, col1: str, col2: str) -> go.Figure:
        """
        Create 2D histogram (density plot) for two numeric columns.
        
        Args:
            col1: First numeric column
            col2: Second numeric column
        
        Returns:
            Plotly figure object
        """
        if col1 not in self.numeric_cols or col2 not in self.numeric_cols:
            return None
        
        fig = go.Figure(data=go.Histogram2d(
            x=self.df[col1].dropna(),
            y=self.df[col2].dropna(),
            nbinsx=30,
            nbinsy=30,
            colorscale='Viridis',
            hovertemplate='%{x:.2f}, %{y:.2f}<br>Count: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'2D Distribution: {col1} vs {col2}',
            xaxis_title=col1,
            yaxis_title=col2,
            height=400
        )
        
        return fig
    
    def save_figure(self, fig: go.Figure, filepath: str) -> None:
        """
        Save figure to HTML file.
        
        Args:
            fig: Plotly figure object
            filepath: Path to save the figure (should end with .html)
        """
        fig.write_html(filepath)
        print(f"Figure saved to {filepath}")
