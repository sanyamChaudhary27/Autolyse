"""Jupyter notebook display module"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from IPython.display import display, HTML, Markdown
import plotly.io as pio


class JupyterDisplay:
    """Display analysis results and plots in Jupyter notebooks"""
    
    def __init__(self):
        """Initialize Jupyter display handler"""
        self.renderers = []
    
    def display_header(self, title: str, level: int = 1) -> None:
        """
        Display a formatted header.
        
        Args:
            title: Header text
            level: Markdown header level (1-6)
        """
        markdown = "#" * level + " " + title
        display(Markdown(markdown))
    
    def display_subheader(self, title: str) -> None:
        """Display a subheader"""
        self.display_header(title, level=2)
    
    def display_text(self, text: str) -> None:
        """Display plain text"""
        display(Markdown(text))
    
    def display_dataframe(self, df: pd.DataFrame, title: str = None) -> None:
        """
        Display formatted dataframe.
        
        Args:
            df: Dataframe to display
            title: Optional title
        """
        if title:
            self.display_subheader(title)
        display(df)
    
    def display_statistics(self, stats: Dict[str, Any], title: str = "Statistical Analysis") -> None:
        """
        Display statistical analysis results.
        
        Args:
            stats: Dictionary from StatisticalAnalyzer.analyze()
            title: Section title
        """
        self.display_subheader(title)
        
        stats_df = pd.DataFrame(stats).T
        stats_df = stats_df.round(4)
        display(stats_df)
    
    def display_missing_values(self, missing_analysis: Dict[str, Any], 
                              title: str = "Missing Values Analysis") -> None:
        """
        Display missing values analysis.
        
        Args:
            missing_analysis: Dictionary from MissingValuesAnalyzer.analyze()
            title: Section title
        """
        self.display_subheader(title)
        
        # Overall summary
        total_missing = missing_analysis['total_missing']
        missing_rows = missing_analysis['missing_rows']
        completely_missing = len(missing_analysis['completely_missing_cols'])
        
        summary_text = f"""
**Overall Summary:**
- Total Missing Values: {total_missing}
- Rows with Missing Values: {missing_rows}
- Completely Empty Columns: {completely_missing}
        """
        display(Markdown(summary_text))
        
        # Missing percentages by column
        if not missing_analysis['no_missing']:
            missing_pct_df = pd.DataFrame(
                list(missing_analysis['missing_percentage'].items()),
                columns=['Column', 'Missing %']
            ).sort_values('Missing %', ascending=False)
            missing_pct_df = missing_pct_df[missing_pct_df['Missing %'] > 0]
            
            if len(missing_pct_df) > 0:
                self.display_subheader("Missing Percentage by Column")
                display(missing_pct_df.to_string(index=False))
        else:
            display(Markdown("✅ **No missing values found!**"))
    
    def display_correlations(self, corr_analysis: Dict[str, Any], 
                            title: str = "Correlation Analysis") -> None:
        """
        Display correlation analysis results.
        
        Args:
            corr_analysis: Dictionary from CorrelationAnalyzer.get_correlation_summary()
            title: Section title
        """
        self.display_subheader(title)
        
        pearson = corr_analysis.get('pearson', {})
        
        # Strong correlations
        strong_corr = pearson.get('strong_correlations', [])
        if strong_corr:
            self.display_subheader("Strong Correlations (|r| > 0.7)", level=3)
            strong_df = pd.DataFrame(strong_corr)
            strong_df['correlation'] = strong_df['correlation'].round(4)
            display(strong_df.to_string(index=False))
        
        # Moderate correlations
        moderate_corr = pearson.get('moderate_correlations', [])
        if moderate_corr:
            self.display_subheader("Moderate Correlations (0.5 < |r| ≤ 0.7)", level=3)
            moderate_df = pd.DataFrame(moderate_corr)
            moderate_df['correlation'] = moderate_df['correlation'].round(4)
            display(moderate_df.to_string(index=False))
        
        if not strong_corr and not moderate_corr:
            display(Markdown("ℹ️ No strong or moderate correlations found."))
    
    def display_outliers(self, outlier_analysis: Dict[str, Any], 
                        title: str = "Outlier Analysis") -> None:
        """
        Display outlier analysis results.
        
        Args:
            outlier_analysis: Dictionary from OutlierAnalyzer.get_outlier_summary()
            title: Section title
        """
        self.display_subheader(title)
        
        iqr_results = outlier_analysis.get('iqr_method', {})
        
        if iqr_results:
            outlier_df = pd.DataFrame([
                {
                    'Column': col,
                    'Outliers': data['outlier_count'],
                    'Percentage': f"{data['outlier_percentage']:.2f}%",
                    'Lower Bound': f"{data['lower_bound']:.2f}",
                    'Upper Bound': f"{data['upper_bound']:.2f}"
                }
                for col, data in iqr_results.items()
            ])
            display(outlier_df.to_string(index=False))
        
        # Isolation Forest summary
        iso_forest = outlier_analysis.get('isolation_forest', {})
        if iso_forest and 'n_outliers' in iso_forest:
            summary = f"\n**Isolation Forest:** {iso_forest['n_outliers']} outliers detected ({iso_forest['outlier_percentage']:.2f}%)"
            display(Markdown(summary))
    
    def display_figure(self, fig) -> None:
        """
        Display a matplotlib or plotly figure.
        
        Args:
            fig: Matplotlib figure or Plotly figure object
        """
        try:
            # Try plotly first
            if hasattr(fig, 'show'):
                fig.show()
        except:
            # Fallback to matplotlib
            try:
                display(fig)
            except:
                pass
    
    def display_figures_dict(self, figures_dict: Dict[str, Any], 
                            title: str = "Visualizations") -> None:
        """
        Display a dictionary of figures with titles.
        
        Args:
            figures_dict: Dictionary of {title: figure}
            title: Section title
        """
        self.display_subheader(title)
        
        for fig_title, fig in figures_dict.items():
            self.display_subheader(fig_title, level=3)
            self.display_figure(fig)
    
    def display_distribution_summary(self, dist_analysis: Dict[str, Any],
                                    title: str = "Distribution Analysis") -> None:
        """
        Display distribution analysis summary.
        
        Args:
            dist_analysis: Dictionary from DistributionAnalyzer methods
            title: Section title
        """
        self.display_subheader(title)
        
        # Numeric distributions
        numeric_dists = dist_analysis.get('numeric_distributions', {})
        if numeric_dists:
            self.display_subheader("Numeric Columns", level=3)
            numeric_df = pd.DataFrame([
                {
                    'Column': col,
                    'Distribution': data['distribution_type'],
                    'Is Normal': '✓' if data['is_normal'] else '✗',
                    'Skewness': f"{data['skewness']:.3f}",
                    'Kurtosis': f"{data['kurtosis']:.3f}",
                    'Unique': data['unique_values']
                }
                for col, data in numeric_dists.items()
            ])
            display(numeric_df.to_string(index=False))
        
        # Categorical distributions
        categorical_dists = dist_analysis.get('categorical_distributions', {})
        if categorical_dists:
            self.display_subheader("Categorical Columns", level=3)
            categorical_df = pd.DataFrame([
                {
                    'Column': col,
                    'Unique Values': data['unique_values'],
                    'Diversity Index': f"{data['diversity_index']:.3f}",
                    'Missing': data['missing_count']
                }
                for col, data in categorical_dists.items()
            ])
            display(categorical_df.to_string(index=False))
    
    def display_insights(self, insights: Dict[str, str], title: str = "AI Insights") -> None:
        """
        Display AI-generated insights.
        
        Args:
            insights: Dictionary of {analysis_type: insight_text}
            title: Section title
        """
        self.display_subheader(title)
        
        for analysis_type, insight_text in insights.items():
            self.display_subheader(analysis_type, level=3)
            display(Markdown(f"> {insight_text}"))
    
    def display_summary(self, df: pd.DataFrame) -> None:
        """
        Display basic dataframe summary.
        
        Args:
            df: Input dataframe
        """
        self.display_subheader("Dataset Summary")
        
        summary_text = f"""
**Dataset Shape:** {df.shape[0]} rows × {df.shape[1]} columns

**Column Information:**
- Numeric Columns: {len(df.select_dtypes(include=[np.number]).columns)}
- Categorical Columns: {len(df.select_dtypes(include=['object', 'category']).columns)}
- Date Columns: {len(df.select_dtypes(include=['datetime64']).columns)}
- Missing Values: {df.isna().sum().sum()}
        """
        display(Markdown(summary_text))
    
    def clear_output(self) -> None:
        """Clear all output from the cell"""
        from IPython.display import clear_output
        clear_output(wait=True)
