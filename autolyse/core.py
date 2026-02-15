"""Main Autolyse orchestrator class"""

import pandas as pd
import numpy as np
import warnings
from typing import Optional, Dict, Any
from pathlib import Path
import sys

from autolyse.analyzers import (
    StatisticalAnalyzer,
    MissingValuesAnalyzer,
    DistributionAnalyzer,
    OutlierAnalyzer,
    CorrelationAnalyzer,
    RelationshipsAnalyzer
)

from autolyse.visualizers import MatplotlibVisualizer, PlotlyVisualizer

from autolyse.utils import DataPreparation, GeminiInsights

from autolyse.output import JupyterDisplay, HTMLGenerator

warnings.filterwarnings('ignore')


class Autolyse:
    """
    Automated Exploratory Data Analysis (EDA) with AI insights.
    
    Generates comprehensive analysis with intelligent chart selection,
    missing value analysis, outlier detection, and AI-summarized findings.
    
    Usage:
        >>> import os
        >>> import pandas as pd
        >>> from autolyse import Autolyse
        >>> 
        >>> df = pd.read_csv('data.csv')
        >>> analyser = Autolyse(html=True, api_key=os.environ.get("GEMINI_KEY"))
        >>> analyser.analyse(df)
    """
    
    def __init__(self, html: bool = True, api_key: Optional[str] = None, 
                 output_dir: str = "./output_reports"):
        """
        Initialize Autolyse analyzer.
        
        Args:
            html: If True, generate HTML report. If False, display in Jupyter.
            api_key: Gemini API key for AI insights. Can also be set via GEMINI_KEY env variable.
            output_dir: Directory to save HTML reports (if html=True)
        """
        self.html_output = html
        self.api_key = api_key
        self.output_dir = output_dir
        self.insights_generator = GeminiInsights(api_key=api_key)
        
        # Data and analysis storage
        self.df = None
        self.data_prep = None
        self.analyses = {}
        self.insights = {}
        self.figures = {}
        
        if not self.html_output:
            # Check if we're in Jupyter
            self._is_jupyter = self._check_jupyter()
    
    def analyse(self, df: pd.DataFrame, show_progress: bool = True) -> Dict[str, Any]:
        """
        Run comprehensive automated EDA on the dataframe.
        
        Args:
            df: Input pandas DataFrame
            show_progress: If True, print progress messages
        
        Returns:
            Dictionary containing all analysis results
        """
        self.df = df.copy()
        
        if show_progress:
            print("Starting Autolyse Analysis...")
            print(f"Dataset: {self.df.shape[0]} rows × {self.df.shape[1]} columns\n")
        
        # Phase 1: Data Preparation
        if show_progress:
            print("1️⃣  Analyzing data structure and types...")
        self.data_prep = DataPreparation(self.df)
        data_validation = self.data_prep.validate_data()
        
        # Phase 2: Run all analyzers
        if show_progress:
            print("2️⃣  Running statistical analyzers (6/6)...")
        
        self.analyses = self._run_all_analyzers()
        
        # Phase 3: Generate visualizations
        if show_progress:
            print("3️⃣  Generating visualizations (matplotlib + plotly)...")
        
        self._generate_visualizations()
        
        # Phase 4: Generate AI insights
        if show_progress:
            print("4️⃣  Generating AI insights...")
        
        self.insights = self._generate_ai_insights()
        
        # Phase 5: Output results
        if show_progress:
            print("5️⃣  Preparing output...")
        
        if self.html_output:
            output_path = self._generate_html_report()
            if show_progress:
                print(f"\n✅ Analysis complete! HTML report saved to: {output_path}")
        else:
            if self._is_jupyter:
                self._display_jupyter_output()
            else:
                print("\n✅ Analysis complete! For best results, run this in Jupyter notebook.")
                self._display_jupyter_output()
        
        return self.analyses
    
    def _run_all_analyzers(self) -> Dict[str, Any]:
        """Run all analysis modules"""
        analyses = {}
        
        # Statistical Analysis
        stat_analyzer = StatisticalAnalyzer(self.df)
        analyses['statistics'] = stat_analyzer.analyze()
        
        # Missing Values Analysis
        missing_analyzer = MissingValuesAnalyzer(self.df)
        analyses['missing_values'] = missing_analyzer.analyze()
        
        # Distribution Analysis
        dist_analyzer = DistributionAnalyzer(self.df)
        analyses['distributions'] = {
            'numeric_distributions': dist_analyzer.analyze_numeric_distributions(),
            'categorical_distributions': dist_analyzer.analyze_categorical_distributions()
        }
        
        # Outlier Analysis
        outlier_analyzer = OutlierAnalyzer(self.df)
        analyses['outliers'] = outlier_analyzer.get_outlier_summary()
        
        # Correlation Analysis
        corr_analyzer = CorrelationAnalyzer(self.df)
        analyses['correlations'] = corr_analyzer.get_correlation_summary()
        
        # Relationships Analysis
        rel_analyzer = RelationshipsAnalyzer(self.df)
        analyses['relationships'] = rel_analyzer.get_relationship_summary()
        
        return analyses
    
    def _generate_visualizations(self) -> None:
        """Generate all visualizations using both matplotlib and plotly"""
        # Matplotlib visualizer
        mpl_viz = MatplotlibVisualizer(self.df)
        plotly_viz = PlotlyVisualizer(self.df)
        
        # Initialize figures storage
        self.figures = {
            'matplotlib': {},
            'plotly': {}
        }
        
        # Distribution plots
        self.figures['matplotlib']['distributions'] = mpl_viz.plot_distributions()
        self.figures['matplotlib']['categorical_distributions'] = mpl_viz.plot_categorical_distributions()
        self.figures['plotly']['distributions'] = plotly_viz.plot_distributions()
        self.figures['plotly']['categorical_distributions'] = plotly_viz.plot_categorical_distributions()
        
        # Missing values
        self.figures['matplotlib']['missing_values'] = mpl_viz.plot_missing_values(
            self.analyses['missing_values']
        )
        self.figures['plotly']['missing_values'] = plotly_viz.plot_missing_values(
            self.analyses['missing_values']
        )
        
        # Correlation heatmaps
        corr_matrix = self.analyses['correlations'].get('pearson', {}).get('correlation_matrix')
        if corr_matrix is not None:
            self.figures['matplotlib']['correlation'] = mpl_viz.plot_correlation_heatmap(corr_matrix)
            self.figures['plotly']['correlation'] = plotly_viz.plot_correlation_heatmap(corr_matrix)
        
        # Boxplots
        self.figures['matplotlib']['boxplot'] = mpl_viz.plot_boxplot()
        self.figures['plotly']['boxplot'] = plotly_viz.plot_boxplot()
        
        # Scatter matrix (limited columns to avoid overload)
        scatter_fig_mpl = mpl_viz.plot_scatter_matrix(max_cols=4)
        if scatter_fig_mpl:
            self.figures['matplotlib']['scatter_matrix'] = scatter_fig_mpl
        
        scatter_fig_plotly = plotly_viz.plot_scatter_matrix(max_cols=4)
        if scatter_fig_plotly:
            self.figures['plotly']['scatter_matrix'] = scatter_fig_plotly
        
        # Outlier plots for each numeric column
        iqr_results = self.analyses['outliers'].get('iqr_method', {})
        self.figures['matplotlib']['outliers'] = {}
        self.figures['plotly']['outliers'] = {}
        
        for col, bounds in iqr_results.items():
            fig_mpl = mpl_viz.plot_outliers(col, bounds)
            if fig_mpl:
                self.figures['matplotlib']['outliers'][col] = fig_mpl
            
            fig_plotly = plotly_viz.plot_outliers(col, self.df[col], bounds)
            if fig_plotly:
                self.figures['plotly']['outliers'][col] = fig_plotly
    
    def _generate_ai_insights(self) -> Dict[str, str]:
        """Generate AI-powered insights for each analysis"""
        insights = {}
        
        try:
            # Statistics insights (top 3 numeric columns)
            numeric_cols = self.data_prep.get_numeric_columns()[:3]
            insights['Statistics'] = "\n".join([
                self.insights_generator.generate_statistics_insight(
                    self.analyses['statistics'][col], col
                )
                for col in numeric_cols if col in self.analyses['statistics']
            ])
            
            # Missing values insight
            insights['Data Quality'] = self.insights_generator.generate_missing_values_insight(
                self.analyses['missing_values']
            )
            
            # Correlations insight
            pearson = self.analyses['correlations'].get('pearson', {})
            insights['Correlations'] = self.insights_generator.generate_correlation_insight(
                pearson.get('strong_correlations', []),
                pearson.get('moderate_correlations', [])
            )
            
            # Outliers insight
            insights['Outliers'] = self.insights_generator.generate_outlier_insight(
                self.analyses['outliers'].get('iqr_method', {}),
                self.analyses['outliers'].get('isolation_forest', {})
            )
            
            # Distribution insight
            insights['Distributions'] = self.insights_generator.generate_distribution_insight(
                self.analyses['distributions'].get('numeric_distributions', {}),
                self.analyses['distributions'].get('categorical_distributions', {})
            )
            
            # General dataset insight
            column_types = self.data_prep.get_type_summary()
            data_quality = self.data_prep.validate_data().get('data_quality_score', 0)
            insights['Dataset Overview'] = self.insights_generator.generate_general_insight(
                self.df.shape,
                column_types,
                data_quality
            )
        
        except Exception as e:
            print(f"⚠️  Warning: Could not generate some AI insights: {str(e)}")
            # Continue without AI insights - system is resilient
        
        return insights
    
    def _generate_html_report(self) -> str:
        """Generate and save HTML report"""
        html_gen = HTMLGenerator(self.df, output_dir=self.output_dir)
        
        # Prepare analyses for HTML (remove unpicklable matplotlib figures)
        analyses_for_html = {
            'statistics': self.analyses.get('statistics', {}),
            'missing_values': self.analyses.get('missing_values', {}),
            'distributions': self.analyses.get('distributions', {}),
            'correlations': self.analyses.get('correlations', {}),
            'outliers': self.analyses.get('outliers', {}),
        }
        
        report_path = html_gen.generate_report(
            analyses=analyses_for_html,
            insights=self.insights,
            filename="autolyse_report.html"
        )
        
        return report_path
    
    def _display_jupyter_output(self) -> None:
        """Display analysis results in Jupyter notebook"""
        jupyter = JupyterDisplay()
        
        jupyter.display_header("Automated EDA Analysis - Autolyse", level=1)
        
        # Dataset summary
        jupyter.display_summary(self.df)
        
        # Column information
        column_info = self.data_prep.get_column_info()
        jupyter.display_dataframe(column_info, "Column Information")
        
        # Statistics
        jupyter.display_statistics(self.analyses.get('statistics', {}))
        
        # Missing values
        jupyter.display_missing_values(self.analyses.get('missing_values', {}))
        
        # Distribution analysis
        jupyter.display_distribution_summary(self.analyses.get('distributions', {}))
        
        # Correlation analysis
        jupyter.display_correlations(self.analyses.get('correlations', {}))
        
        # Outliers
        jupyter.display_outliers(self.analyses.get('outliers', {}))
        
        # AI Insights
        if self.insights:
            jupyter.display_insights(self.insights)
        
        # Figures - try to display some key ones
        jupyter.display_subheader("Visualizations", level=2)
        
        try:
            plotly_figs = self.figures.get('plotly', {})
            
            if 'distributions' in plotly_figs:
                jupyter.display_subheader("Distributions", level=3)
                for col, fig in list(plotly_figs['distributions'].items())[:3]:
                    jupyter.display_figure(fig)
            
            if 'missing_values' in plotly_figs:
                jupyter.display_subheader("Missing Values", level=3)
                jupyter.display_figure(plotly_figs['missing_values'])
            
            if 'correlation' in plotly_figs:
                jupyter.display_subheader("Correlation Heatmap", level=3)
                jupyter.display_figure(plotly_figs['correlation'])
            
            if 'boxplot' in plotly_figs:
                jupyter.display_subheader("Boxplots", level=3)
                jupyter.display_figure(plotly_figs['boxplot'])
        
        except Exception as e:
            jupyter.display_text(f"⚠️ Could not display all visualizations: {str(e)}")
    
    def _check_jupyter(self) -> bool:
        """Check if code is running in Jupyter notebook"""
        try:
            from IPython import get_ipython
            if get_ipython() is None:
                return False
            if 'IPKernelApp' not in get_ipython().config:
                return False
            return True
        except ImportError:
            return False
    
    def get_analysis_results(self) -> Dict[str, Any]:
        """Get the analysis results dictionary"""
        return self.analyses.copy()
    
    def get_insights(self) -> Dict[str, str]:
        """Get the generated insights"""
        return self.insights.copy()
    
    def get_dataframe_info(self) -> pd.DataFrame:
        """Get detailed column information"""
        if self.data_prep:
            return self.data_prep.get_column_info()
        return None
