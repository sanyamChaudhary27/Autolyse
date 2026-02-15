"""Gemini AI integration for generating insights from analysis results"""

import os
from typing import Dict, Any, Optional, List
import warnings

warnings.filterwarnings('ignore')


class GeminiInsights:
    """Generate AI-powered insights using Google Gemini API"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini insights generator.
        
        Args:
            api_key: Gemini API key. If None, will try to get from environment variable GEMINI_KEY
        """
        self.api_key = api_key or os.environ.get("GEMINI_KEY")
        self.client = None
        self.model = None
        self.available = False
        
        if self.api_key:
            self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize the Gemini client"""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.client = genai
            self.model = genai.GenerativeModel('gemini-pro')
            self.available = True
        except Exception as e:
            warnings.warn(f"Could not initialize Gemini client: {str(e)}. Insights will not be available.")
            self.available = False
    
    def is_available(self) -> bool:
        """Check if Gemini API is available"""
        return self.available
    
    def generate_statistics_insight(self, stats: Dict[str, Any], column: str) -> str:
        """
        Generate insight for statistical analysis of a column.
        
        Args:
            stats: Statistics dictionary for a column from StatisticalAnalyzer
            column: Column name
        
        Returns:
            2-4 line insight text
        """
        if not self.available:
            return self._fallback_statistics_insight(stats, column)
        
        try:
            prompt = f"""Analyze these statistics for the column '{column}' and provide a 2-3 sentence insight:
- Mean: {stats.get('mean', 'N/A')}
- Median: {stats.get('median', 'N/A')}
- Std Dev: {stats.get('std', 'N/A')}
- Min: {stats.get('min', 'N/A')}
- Max: {stats.get('max', 'N/A')}
- Skewness: {stats.get('skewness', 'N/A')}
- Kurtosis: {stats.get('kurtosis', 'N/A')}
- Missing %: {stats.get('null_percentage', 0):.2f}%

Provide actionable insights about this column's distribution and quality. Keep it to 2-4 sentences."""
            
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            warnings.warn(f"Error generating statistics insight: {str(e)}")
            return self._fallback_statistics_insight(stats, column)
    
    def generate_missing_values_insight(self, missing_analysis: Dict[str, Any]) -> str:
        """
        Generate insight for missing values analysis.
        
        Args:
            missing_analysis: Missing values analysis from MissingValuesAnalyzer
        
        Returns:
            2-4 line insight text
        """
        if not self.available:
            return self._fallback_missing_insight(missing_analysis)
        
        try:
            total_missing = missing_analysis.get('total_missing', 0)
            missing_rows = missing_analysis.get('missing_rows', 0)
            no_missing = missing_analysis.get('no_missing', False)
            
            prompt = f"""Analyze these missing values statistics and provide a 2-3 sentence insight:
- Total Missing Values: {total_missing}
- Rows with Missing Values: {missing_rows}
- No Missing Found: {no_missing}
- Completely Empty Columns: {len(missing_analysis.get('completely_missing_cols', []))}

Provide insights about data quality and whether missing values are a concern. Keep it to 2-4 sentences."""
            
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            warnings.warn(f"Error generating missing values insight: {str(e)}")
            return self._fallback_missing_insight(missing_analysis)
    
    def generate_correlation_insight(self, strong_correlations: List[Dict], 
                                    moderate_correlations: List[Dict]) -> str:
        """
        Generate insight for correlation analysis.
        
        Args:
            strong_correlations: List of strong correlation pairs
            moderate_correlations: List of moderate correlation pairs
        
        Returns:
            2-4 line insight text
        """
        if not self.available:
            return self._fallback_correlation_insight(strong_correlations, moderate_correlations)
        
        try:
            strong_str = ""
            if strong_correlations:
                strong_str = "Strong correlations: " + ", ".join([
                    f"{c['col1']}-{c['col2']} ({c['correlation']:.2f})"
                    for c in strong_correlations[:3]
                ])
            
            moderate_str = ""
            if moderate_correlations:
                moderate_str = "Moderate correlations: " + ", ".join([
                    f"{c['col1']}-{c['col2']} ({c['correlation']:.2f})"
                    for c in moderate_correlations[:2]
                ])
            
            context = f"{strong_str}\n{moderate_str}".strip()
            
            prompt = f"""Analyze these correlation findings and provide a 2-3 sentence insight:
{context}

Total strong correlations found: {len(strong_correlations)}
Total moderate correlations found: {len(moderate_correlations)}

Provide insights about relationships between variables and their implications. Keep it to 2-4 sentences."""
            
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            warnings.warn(f"Error generating correlation insight: {str(e)}")
            return self._fallback_correlation_insight(strong_correlations, moderate_correlations)
    
    def generate_outlier_insight(self, iqr_results: Dict[str, Any], 
                                iso_forest_results: Dict[str, Any]) -> str:
        """
        Generate insight for outlier analysis.
        
        Args:
            iqr_results: IQR method results from OutlierAnalyzer
            iso_forest_results: Isolation Forest results
        
        Returns:
            2-4 line insight text
        """
        if not self.available:
            return self._fallback_outlier_insight(iqr_results, iso_forest_results)
        
        try:
            total_outliers = sum(data.get('outlier_count', 0) for data in iqr_results.values())
            avg_outlier_pct = sum(data.get('outlier_percentage', 0) for data in iqr_results.values()) / len(iqr_results) if iqr_results else 0
            
            iso_outliers = iso_forest_results.get('n_outliers', 0)
            
            prompt = f"""Analyze these outlier detection results and provide a 2-3 sentence insight:
- IQR Method Total Outliers: {total_outliers} (avg {avg_outlier_pct:.2f}% per column)
- Isolation Forest Outliers: {iso_outliers}
- Columns with Outliers: {len(iqr_results)}

Provide insights about the prevalence of outliers and whether they're concerning. Keep it to 2-4 sentences."""
            
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            warnings.warn(f"Error generating outlier insight: {str(e)}")
            return self._fallback_outlier_insight(iqr_results, iso_forest_results)
    
    def generate_distribution_insight(self, numeric_distributions: Dict[str, Any],
                                     categorical_distributions: Dict[str, Any]) -> str:
        """
        Generate insight for distribution analysis.
        
        Args:
            numeric_distributions: Distribution analysis for numeric columns
            categorical_distributions: Distribution analysis for categorical columns
        
        Returns:
            2-4 line insight text
        """
        if not self.available:
            return self._fallback_distribution_insight(numeric_distributions, categorical_distributions)
        
        try:
            normal_count = sum(1 for d in numeric_distributions.values() if d.get('is_normal'))
            skewed_count = len(numeric_distributions) - normal_count
            
            prompt = f"""Analyze these distribution characteristics and provide a 2-3 sentence insight:
- Total Numeric Columns: {len(numeric_distributions)}
- Approximately Normal: {normal_count}
- Skewed: {skewed_count}
- Categorical Columns: {len(categorical_distributions)}

Provide insights about the overall distribution patterns in the dataset. Keep it to 2-4 sentences."""
            
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            warnings.warn(f"Error generating distribution insight: {str(e)}")
            return self._fallback_distribution_insight(numeric_distributions, categorical_distributions)
    
    def generate_general_insight(self, df_shape: tuple, column_types: Dict[str, int],
                                data_quality_score: float) -> str:
        """
        Generate general insight about the entire dataset.
        
        Args:
            df_shape: DataFrame shape (rows, columns)
            column_types: Dictionary of column type counts
            data_quality_score: Data quality score (0-100)
        
        Returns:
            2-4 line insight text
        """
        if not self.available:
            return self._fallback_general_insight(df_shape, column_types, data_quality_score)
        
        try:
            prompt = f"""Analyze these dataset characteristics and provide a 2-3 sentence overall insight:
- Dataset Size: {df_shape[0]} rows × {df_shape[1]} columns
- Column Types: {column_types}
- Data Quality Score: {data_quality_score:.1f}/100

Provide an overall assessment of the dataset and suggestions for analysis direction. Keep it to 2-4 sentences."""
            
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            warnings.warn(f"Error generating general insight: {str(e)}")
            return self._fallback_general_insight(df_shape, column_types, data_quality_score)
    
    # Fallback methods for when Gemini is not available
    
    @staticmethod
    def _fallback_statistics_insight(stats: Dict[str, Any], column: str) -> str:
        """Fallback insight for statistics"""
        skewness = stats.get('skewness', 0)
        std = stats.get('std', 0)
        mean = stats.get('mean', 0)
        
        if abs(skewness) > 1:
            skew_desc = "highly skewed"
        elif abs(skewness) > 0.5:
            skew_desc = "moderately skewed"
        else:
            skew_desc = "approximately symmetric"
        
        if std == 0 or (mean != 0 and std / mean < 0.1):
            var_desc = "very low variability"
        else:
            var_desc = "moderate to high variability"
        
        return f"The '{column}' column shows a {skew_desc} distribution with {var_desc}. " \
               f"This indicates {'a non-normal pattern' if abs(skewness) > 1 else 'a relatively normal pattern'} " \
               f"in the data with {stats.get('null_percentage', 0):.1f}% missing values."
    
    @staticmethod
    def _fallback_missing_insight(missing_analysis: Dict[str, Any]) -> str:
        """Fallback insight for missing values"""
        total_missing = missing_analysis.get('total_missing', 0)
        
        if missing_analysis.get('no_missing', False):
            return "Excellent news! The dataset has no missing values, which is ideal for analysis and modeling."
        elif total_missing == 0:
            return "The dataset is complete with no missing values detected."
        else:
            return f"The dataset contains {total_missing} missing values. Consider imputation or removal strategies " \
                   f"depending on the mechanism of missingness and your analysis goals."
    
    @staticmethod
    def _fallback_correlation_insight(strong_corr: List[Dict], moderate_corr: List[Dict]) -> str:
        """Fallback insight for correlations"""
        if not strong_corr and not moderate_corr:
            return "No strong or moderate correlations detected. Variables appear to be relatively independent of each other."
        
        strong_count = len(strong_corr)
        moderate_count = len(moderate_corr)
        
        return f"Found {strong_count} strong correlations and {moderate_count} moderate correlations. " \
               f"This suggests {'significant relationships' if strong_count > 0 else 'weak relationships'} " \
               f"between some variables that may be worth investigating further."
    
    @staticmethod
    def _fallback_outlier_insight(iqr_results: Dict[str, Any], iso_forest: Dict[str, Any]) -> str:
        """Fallback insight for outliers"""
        total_outliers = sum(data.get('outlier_count', 0) for data in iqr_results.values())
        cols_with_outliers = len(iqr_results)
        
        if total_outliers == 0:
            return "No outliers detected in the dataset, suggesting data is consistent and within expected ranges."
        
        return f"Detected {total_outliers} outliers across {cols_with_outliers} columns. " \
               f"Investigate whether these are errors or valid extreme values before removal."
    
    @staticmethod
    def _fallback_distribution_insight(numeric_dists: Dict[str, Any], categorical_dists: Dict[str, Any]) -> str:
        """Fallback insight for distributions"""
        normal_count = sum(1 for d in numeric_dists.values() if d.get('is_normal'))
        total_numeric = len(numeric_dists)
        
        if total_numeric == 0:
            return "No numeric columns in the dataset. Analysis is limited to categorical variables."
        
        normal_pct = (normal_count / total_numeric) * 100 if total_numeric > 0 else 0
        
        return f"Among {total_numeric} numeric columns, {normal_count} ({normal_pct:.0f}%) show approximately normal distributions. " \
               f"The remaining columns exhibit skewed or non-normal patterns that may require transformation."
    
    @staticmethod
    def _fallback_general_insight(df_shape: tuple, column_types: Dict[str, int], quality_score: float) -> str:
        """Fallback insight for general dataset assessment"""
        rows, cols = df_shape
        
        if quality_score >= 80:
            quality_desc = "excellent"
        elif quality_score >= 60:
            quality_desc = "good"
        elif quality_score >= 40:
            quality_desc = "fair"
        else:
            quality_desc = "poor"
        
        return f"You're working with a dataset of {rows:,} rows and {cols} columns with {quality_desc} data quality ({quality_score:.0f}/100). " \
               f"The dataset contains a mix of {column_types.get('numeric', 0)} numeric and {column_types.get('categorical', 0)} categorical variables, " \
               f"which allows for comprehensive analysis and modeling."
