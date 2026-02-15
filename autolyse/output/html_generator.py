"""HTML report generation module"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import base64
from pathlib import Path


class HTMLGenerator:
    """Generate comprehensive HTML reports from analysis results"""
    
    def __init__(self, df: pd.DataFrame, output_dir: str = "./output"):
        """
        Initialize HTML generator.
        
        Args:
            df: Input dataframe
            output_dir: Directory to save HTML reports
        """
        self.df = df
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def generate_report(self, analyses: Dict[str, Any], 
                       insights: Dict[str, str] = None,
                       filename: str = "autolyse_report.html") -> str:
        """
        Generate comprehensive HTML report.
        
        Args:
            analyses: Dictionary containing all analysis results
            insights: Dictionary of AI-generated insights
            filename: Output filename
        
        Returns:
            Path to generated HTML file
        """
        html_content = self._build_html(analyses, insights)
        
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ HTML report saved to {output_path}")
        return str(output_path)
    
    def _build_html(self, analyses: Dict[str, Any], 
                   insights: Dict[str, str] = None) -> str:
        """
        Build complete HTML document.
        
        Args:
            analyses: Analysis results dictionary
            insights: AI insights dictionary
        
        Returns:
            Complete HTML string
        """
        html_parts = [
            self._get_html_header(),
            self._get_styles(),
            "<body>",
            self._get_navbar(),
            "<main class='container'>",
            self._get_dataset_summary(),
            self._get_statistics_section(analyses.get('statistics', {})),
            self._get_missing_values_section(analyses.get('missing_values', {})),
            self._get_distribution_section(analyses.get('distributions', {})),
            self._get_correlation_section(analyses.get('correlations', {})),
            self._get_outliers_section(analyses.get('outliers', {})),
            self._get_insights_section(insights or {}),
            "</main>",
            self._get_footer(),
            "</body>",
            "</html>"
        ]
        
        return "\n".join(html_parts)
    
    def _get_html_header(self) -> str:
        """Get HTML document header"""
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Autolyse - EDA Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>"""
    
    def _get_styles(self) -> str:
        """Get CSS styles"""
        return """<style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #333;
        line-height: 1.6;
    }
    
    .navbar {
        background: white;
        padding: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        position: sticky;
        top: 0;
        z-index: 100;
    }
    
    .navbar .container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0 2rem;
    }
    
    .navbar h1 {
        color: #667eea;
        font-size: 24px;
    }
    
    .navbar-time {
        color: #666;
        font-size: 14px;
    }
    
    main.container {
        max-width: 1200px;
        margin: 2rem auto;
        padding: 0 2rem 2rem;
    }
    
    .section {
        background: white;
        border-radius: 8px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .section h2 {
        color: #667eea;
        margin-bottom: 1.5rem;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
    }
    
    .section h3 {
        color: #764ba2;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-size: 16px;
    }
    
    .summary-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin-bottom: 2rem;
    }
    
    .summary-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .summary-card h3 {
        color: white;
        font-size: 14px;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        opacity: 0.9;
    }
    
    .summary-card .value {
        font-size: 28px;
        font-weight: bold;
    }
    
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
    }
    
    th {
        background: #667eea;
        color: white;
        padding: 0.75rem;
        text-align: left;
    }
    
    td {
        padding: 0.75rem;
        border-bottom: 1px solid #eee;
    }
    
    tr:nth-child(even) {
        background: #f9f9f9;
    }
    
    tr:hover {
        background: #f0f0f0;
    }
    
    .alert {
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    .alert-info {
        background: #e3f2fd;
        color: #1976d2;
        border-left: 4px solid #1976d2;
    }
    
    .alert-warning {
        background: #fff3e0;
        color: #f57c00;
        border-left: 4px solid #f57c00;
    }
    
    .alert-success {
        background: #e8f5e9;
        color: #388e3c;
        border-left: 4px solid #388e3c;
    }
    
    .plot-container {
        margin: 2rem 0;
        padding: 1rem;
        background: #f5f5f5;
        border-radius: 4px;
        display: none;
    }
    
    .plot-container.active {
        display: block;
    }
    
    .insight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 5px solid #fff;
    }
    
    .insight-title {
        font-weight: bold;
        margin-bottom: 0.5rem;
        font-size: 14px;
        text-transform: uppercase;
        opacity: 0.9;
    }
    
    .insight-text {
        font-size: 15px;
        line-height: 1.6;
    }
    
    .footer {
        background: white;
        padding: 2rem;
        text-align: center;
        color: #666;
        border-top: 1px solid #eee;
        margin-top: 3rem;
    }
    
    .footer p {
        margin: 0.25rem 0;
    }
    
    .no-data {
        text-align: center;
        padding: 2rem;
        color: #999;
        font-style: italic;
    }
</style>"""
    
    def _get_navbar(self) -> str:
        """Get navigation bar HTML"""
        return f"""<nav class="navbar">
    <div class="container">
        <h1>Autolyse</h1>
        <span class="navbar-time">Generated: {self.timestamp}</span>
    </div>
</nav>"""
    
    def _get_dataset_summary(self) -> str:
        """Get dataset summary section"""
        n_rows, n_cols = self.df.shape
        numeric_cols = len(self.df.select_dtypes(include=[np.number]).columns)
        categorical_cols = len(self.df.select_dtypes(include=['object', 'category']).columns)
        missing_total = self.df.isna().sum().sum()
        
        return f"""<section class="section">
    <h2>Dataset Overview</h2>
    <div class="summary-grid">
        <div class="summary-card">
            <h3>Total Rows</h3>
            <div class="value">{n_rows:,}</div>
        </div>
        <div class="summary-card">
            <h3>Total Columns</h3>
            <div class="value">{n_cols}</div>
        </div>
        <div class="summary-card">
            <h3>Numeric Columns</h3>
            <div class="value">{numeric_cols}</div>
        </div>
        <div class="summary-card">
            <h3>Categorical Columns</h3>
            <div class="value">{categorical_cols}</div>
        </div>
        <div class="summary-card">
            <h3>Missing Values</h3>
            <div class="value">{missing_total:,}</div>
        </div>
    </div>
</section>"""
    
    def _get_statistics_section(self, stats: Dict[str, Any]) -> str:
        """Get statistics section"""
        if not stats:
            return '<section class="section"><h2>📈 Statistical Summary</h2><p class="no-data">No statistical data available</p></section>'
        
        # Convert to dataframe for table
        stats_df = pd.DataFrame(stats).T.round(4)
        table_html = self._dataframe_to_html_table(stats_df)
        
        return f"""<section class="section">
    <h2>📈 Statistical Summary</h2>
    <p>Comprehensive statistical metrics for all numeric columns.</p>
    {table_html}
</section>"""
    
    def _get_missing_values_section(self, missing_analysis: Dict[str, Any]) -> str:
        """Get missing values section"""
        if not missing_analysis:
            return ""
        
        total_missing = missing_analysis.get('total_missing', 0)
        missing_rows = missing_analysis.get('missing_rows', 0)
        completely_missing = len(missing_analysis.get('completely_missing_cols', []))
        
        if missing_analysis.get('no_missing', False):
            alert = '<div class="alert alert-success">✅ No missing values found in the dataset!</div>'
        else:
            alert = f'<div class="alert alert-warning">⚠️ Found {total_missing} missing values across {missing_rows} rows</div>'
        
        missing_pct = missing_analysis.get('missing_percentage', {})
        missing_pct_filtered = {k: v for k, v in missing_pct.items() if v > 0}
        
        if missing_pct_filtered:
            missing_df = pd.DataFrame(list(missing_pct_filtered.items()), 
                                     columns=['Column', 'Missing %']).sort_values('Missing %', ascending=False)
            table_html = self._dataframe_to_html_table(missing_df.round(2))
            table_section = f"<h3>Missing Percentage by Column</h3>{table_html}"
        else:
            table_section = ""
        
        return f"""<section class="section">
    <h2>🔍 Missing Values Analysis</h2>
    {alert}
    {table_section}
</section>"""
    
    def _get_distribution_section(self, dist_analysis: Dict[str, Any]) -> str:
        """Get distribution section"""
        if not dist_analysis:
            return ""
        
        numeric_dists = dist_analysis.get('numeric_distributions', {})
        categorical_dists = dist_analysis.get('categorical_distributions', {})
        
        sections = []
        
        if numeric_dists:
            numeric_df = pd.DataFrame([
                {
                    'Column': col,
                    'Type': data['distribution_type'],
                    'Normal': '✓' if data['is_normal'] else '✗',
                    'Skewness': f"{data['skewness']:.3f}",
                    'Kurtosis': f"{data['kurtosis']:.3f}",
                    'Unique': data['unique_values']
                }
                for col, data in numeric_dists.items()
            ])
            sections.append(f"<h3>Numeric Columns</h3>{self._dataframe_to_html_table(numeric_df)}")
        
        if categorical_dists:
            categorical_df = pd.DataFrame([
                {
                    'Column': col,
                    'Unique': data['unique_values'],
                    'Diversity': f"{data['diversity_index']:.3f}",
                    'Missing': data['missing_count']
                }
                for col, data in categorical_dists.items()
            ])
            sections.append(f"<h3>Categorical Columns</h3>{self._dataframe_to_html_table(categorical_df)}")
        
        sections_html = "\n".join(sections)
        
        return f"""<section class="section">
    <h2>🔔 Distribution Analysis</h2>
    {sections_html}
</section>"""
    
    def _get_correlation_section(self, corr_analysis: Dict[str, Any]) -> str:
        """Get correlation section"""
        if not corr_analysis:
            return ""
        
        pearson = corr_analysis.get('pearson', {})
        strong_corr = pearson.get('strong_correlations', [])
        moderate_corr = pearson.get('moderate_correlations', [])
        
        sections = []
        
        if strong_corr:
            strong_df = pd.DataFrame(strong_corr)
            strong_df['correlation'] = strong_df['correlation'].round(4)
            sections.append(f"<h3>Strong Correlations (|r| > 0.7)</h3>{self._dataframe_to_html_table(strong_df)}")
        
        if moderate_corr:
            moderate_df = pd.DataFrame(moderate_corr)
            moderate_df['correlation'] = moderate_df['correlation'].round(4)
            sections.append(f"<h3>Moderate Correlations (0.5 < |r| ≤ 0.7)</h3>{self._dataframe_to_html_table(moderate_df)}")
        
        if not strong_corr and not moderate_corr:
            sections.append('<div class="alert alert-info">ℹ️ No strong or moderate correlations found.</div>')
        
        sections_html = "\n".join(sections)
        
        return f"""<section class="section">
    <h2>🔗 Correlation Analysis</h2>
    {sections_html}
</section>"""
    
    def _get_outliers_section(self, outlier_analysis: Dict[str, Any]) -> str:
        """Get outliers section"""
        if not outlier_analysis:
            return ""
        
        iqr_results = outlier_analysis.get('iqr_method', {})
        
        if not iqr_results:
            return ""
        
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
        
        table_html = self._dataframe_to_html_table(outlier_df)
        
        iso_forest = outlier_analysis.get('isolation_forest', {})
        iso_text = ""
        if iso_forest and 'n_outliers' in iso_forest:
            iso_text = f'<div class="alert alert-info">🌲 Isolation Forest: {iso_forest["n_outliers"]} outliers detected ({iso_forest["outlier_percentage"]:.2f}%)</div>'
        
        return f"""<section class="section">
    <h2>🎯 Outlier Detection</h2>
    <h3>IQR Method Results</h3>
    {table_html}
    {iso_text}
</section>"""
    
    def _get_insights_section(self, insights: Dict[str, str]) -> str:
        """Get AI insights section"""
        if not insights:
            return ""
        
        insight_boxes = []
        for analysis_type, insight_text in insights.items():
            insight_boxes.append(f"""<div class="insight-box">
    <div class="insight-title">{analysis_type}</div>
    <div class="insight-text">{insight_text}</div>
</div>""")
        
        insights_html = "\n".join(insight_boxes)
        
        return f"""<section class="section">
    <h2>🤖 AI-Generated Insights</h2>
    {insights_html}
</section>"""
    
    def _get_footer(self) -> str:
        """Get footer HTML"""
        return f"""<footer class="footer">
    <p><strong>Autolyse</strong> - Automated Exploratory Data Analysis</p>
    <p>Generated on {self.timestamp}</p>
    <p>Powered by Pandas, Matplotlib, Plotly, and Gemini AI</p>
</footer>"""
    
    def _dataframe_to_html_table(self, df: pd.DataFrame) -> str:
        """
        Convert dataframe to HTML table.
        
        Args:
            df: Input dataframe
        
        Returns:
            HTML table string
        """
        html = "<table>\n<thead>\n<tr>"
        
        # Header
        for col in df.columns:
            html += f"<th>{col}</th>"
        html += "\n</tr>\n</thead>\n<tbody>\n"
        
        # Rows
        for _, row in df.iterrows():
            html += "<tr>"
            for val in row:
                html += f"<td>{val}</td>"
            html += "</tr>\n"
        
        html += "</tbody>\n</table>"
        
        return html
