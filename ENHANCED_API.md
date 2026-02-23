# Autolyse v2 - Enhanced API Documentation

## Quick Start

```python
import pandas as pd
from autolyse import Autolyse

df = pd.read_csv('data.csv')
analyser = Autolyse(html=True, api_key="YOUR_GEMINI_KEY")
analyser.analyse(df)
```

## New Features in v2

### 1. Granular Control Flags

Control exactly which analyses run. Perfect for customized workflows.

```python
# Run only statistics and correlations
analyser = Autolyse(
    enable_statistics=True,
    enable_correlations=True,
    enable_visualizations=True,
    # Skip these
    enable_missing_values=False,
    enable_outliers=False,
    enable_distributions=False,
    enable_relationships=False,
    enable_advanced_insights=False
)
analyser.analyse(df)
```

**Available Flags:**

| Flag | Default | Controls |
|------|---------|----------|
| `enable_statistics` | True | Mean, median, std, quartiles, skewness, kurtosis |
| `enable_missing_values` | True | Missing value patterns and data quality |
| `enable_distributions` | True | Distribution type detection (normal, uniform, etc.) |
| `enable_outliers` | True | IQR + Isolation Forest outlier detection |
| `enable_correlations` | True | Pearson & Spearman correlations |
| `enable_relationships` | True | Categorical-numeric-categorical relationships |
| `enable_advanced_insights` | True | Multivariate patterns (3+ features) |
| `enable_feature_engineering` | False | Auto-create polynomial/interaction features |
| `enable_visualizations` | True | Generate matplotlib + plotly charts |
| `enable_html` | True | Generate HTML report (False = Jupyter only) |

---

### 2. Feature Engineering (Automatic)

Automatically create intelligent features from existing ones.

```python
# Enable automatic feature engineering
analyser = Autolyse(
    enable_feature_engineering=True,  # ← NEW
    html=False
)
analyser.analyse(df)
```

**What gets created:**

- **Polynomial Features**: Degree 2-3 polynomials of top variance columns
- **Interaction Features**: Smart combinations of moderately correlated columns
- **Ratio Features**: Safe division features (handles zero division)
- **Log Transformations**: For right-skewed positive data
- **Aggregate Features**: Mean/std/max of top numeric columns

```python
# After analysis, access the enriched dataframe
enriched_df = analyser.df
print(f"Original: {df.shape[1]} columns")
print(f"After engineering: {enriched_df.shape[1]} columns")
```

---

### 3. Random Seeding (Reproducibility)

Set a global random seed for deterministic results.

```python
analyser = Autolyse(
    random_seed=42  # ← NEW: Fixed seed
)
result1 = analyser.analyse(df)

# Run same analysis again - identical results
analyser2 = Autolyse(random_seed=42)
result2 = analyser2.analyse(df)
# result1 == result2 ✓
```

**Applied to:**
- Outlier detection (Isolation Forest)
- Feature engineering randomization
- Batch sampling (if enabled)
- Advanced insights analysis

---

### 4. Batch Sampling (Large Datasets)

Analyze a random sample for 10x faster iteration.

```python
# Analyze only 5000 rows from 50,000 row dataset
analyser = Autolyse(
    batch_size=5000,  # ← NEW: Sample 5000 rows
    random_seed=42
)
analyser.analyse(df)

print(f"Sampled {analyser.df.shape[0]} rows for analysis")
# (Original df stored in analyser.df_original)
```

**Use cases:**
- Fast EDA on new datasets
- Understand patterns before deep dive
- Budget-conscious API calls (fewer rows = fewer Gemini calls)

---

### 5. Advanced Insights (Multivariate Analysis)

Detect complex patterns involving 3+ features.

```python
analyser = Autolyse(
    enable_advanced_insights=True  # ← Enabled by default
)
analyser.analyse(df)

results = analyser.get_analysis_results()
adv = results['advanced_insights']
```

**Detects:**

1. **Feature Interactions** - 3-way synergies
   ```python
   interactions = adv['feature_interactions']
   # [{'features': ('age', 'income', 'years'), 'strength': 2.34}, ...]
   ```

2. **Feature Clusters** - Related feature groups
   ```python
   clusters = adv['feature_clusters']['clusters']
   # [['age', 'years_experience'], ['price', 'quantity', 'revenue'], ...]
   ```

3. **Categorical Influence** - Which categories matter
   ```python
   influence = adv['categorical_influence']
   # Shows significance of categorical variables on numeric columns
   ```

4. **Anomaly Patterns** - Multivariate outliers
   ```python
   anomalies = adv['anomaly_patterns']
   # Severity classification + detailed patterns
   ```

5. **Feature Importance** - 4-method ensemble
   ```python
   importance = adv['feature_importance']
   # {'age': 0.85, 'income': 0.78, ...}  (weighted scores)
   ```

6. **Temporal Patterns** - Trends, autocorrelation
   ```python
   temporal = adv['temporal_patterns']
   # Detects trend, seasonality, autocorrelation
   ```

7. **Multivariate Patterns** - Complex 3+ feature relationships
   ```python
   patterns = adv['multivariate_patterns']
   # Correlation networks, dominant patterns
   ```

---

## Complete Initialization Reference

```python
analyser = Autolyse(
    # Output
    html=True,                              # HTML report (False = Jupyter)
    output_dir="./reports",                 # Where to save HTML
    api_key="YOUR_GEMINI_KEY",             # For AI insights
    
    # Reproducibility
    random_seed=42,                         # Fixed for determinism
    batch_size=None,                        # Sample size (None = all rows)
    
    # Granular Control
    enable_statistics=True,                 # ✓ Stats
    enable_missing_values=True,             # ✓ Missing analysis
    enable_distributions=True,              # ✓ Distribution type
    enable_outliers=True,                   # ✓ Outlier detection
    enable_correlations=True,               # ✓ Correlation analysis
    enable_relationships=True,              # ✓ Variable relationships
    enable_advanced_insights=True,          # ✓ Multivariate patterns
    enable_feature_engineering=False,       # Feature creation (opt-in)
    enable_visualizations=True,             # ✓ Charts
    enable_html=True                        # ✓ HTML generation
)
```

---

## Common Workflows

### Workflow 1: Quick EDA (30 seconds)

```python
from autolyse import Autolyse

analyser = Autolyse(
    html=False,
    enable_statistics=True,
    enable_correlations=True,
    enable_visualizations=True,
    # Skip expensive analyses
    enable_distributions=False,
    enable_outliers=False,
    enable_relationships=False,
    enable_advanced_insights=False,
    api_key=None  # Skip AI insights
)
analyser.analyse(df)
```

### Workflow 2: Deep Analysis (with AI)

```python
analyser = Autolyse(
    html=True,
    api_key="YOUR_GEMINI_KEY",  # Enable all AI insights
    enable_advanced_insights=True
)
analyser.analyse(df)
```

### Workflow 3: Large Dataset Fast Iteration

```python
analyser = Autolyse(
    batch_size=5000,            # Sample 5000 rows
    random_seed=42,
    enable_feature_engineering=True,
    html=False
)
analyser.analyse(df)
```

### Workflow 4: Feature Engineering Focus

```python
analyser = Autolyse(
    enable_feature_engineering=True,  # Create features
    enable_advanced_insights=True,    # Analyze interactions
    html=True
)
analyser.analyse(df)
enriched_df = analyser.df  # Get engineered dataframe
```

### Workflow 5: Custom + Reproducible

```python
analyser = Autolyse(
    random_seed=42,
    batch_size=1000,
    enable_statistics=True,
    enable_outliers=True,
    enable_advanced_insights=True
)
analyser.analyse(df)
```

---

## Performance Considerations

| Dataset Size | Recommended Settings | Est. Time |
|--------------|----------------------|-----------|
| < 10K rows | All features enabled, no sampling | 5-30s |
| 10K-100K rows | Sample 10K, skip visualizations | 10-30s |
| 100K-1M rows | Sample 5K, minimal features | 5-10s |
| > 1M rows | Sample 1K, skip advanced | 1-5s |

**Tips:**
- `enable_advanced_insights=True` adds +5-10 seconds but very insightful
- `enable_feature_engineering=True` adds +2 seconds
- Each visualization adds ~1 second
- Batch sampling can give 10x speedup with minimal insights loss

---

## Accessing Results

```python
# Dictionary of all analyses
results = analyser.get_analysis_results()

# Access specific analyses
stats = results['statistics']
correlations = results['correlations']
outliers = results['outliers']
advanced = results['advanced_insights']

# AI-generated text insights
insights = analyser.get_insights()
for insight_type, text in insights.items():
    print(f"{insight_type}:\n{text}")

# Engineered dataframe (if feature_engineering=True)
enriched_df = analyser.df

# Original dataframe (if batch_sampling used)
original_df = analyser.df_original

# Generated figures
mpl_figs = analyser.figures['matplotlib']
plotly_figs = analyser.figures['plotly']

# Column info
col_info = analyser.get_dataframe_info()
```

---

## Architecture

```
Autolyse
├── Input: DataFrame
├── Phase 1: Data Preparation
│   └── Type detection, validation
├── Phase 1.5: Feature Engineering (optional)
│   └── Polynomial, interactions, ratios, logs
├── Phase 2: Analysis (7 analyzers)
│   ├── StatisticalAnalyzer
│   ├── MissingValuesAnalyzer
│   ├── DistributionAnalyzer
│   ├── OutlierAnalyzer
│   ├── CorrelationAnalyzer
│   ├── RelationshipsAnalyzer
│   └── AdvancedInsightsAnalyzer (3+ features)
├── Phase 3: Visualizations
│   ├── MatplotlibVisualizer (static)
│   └── PlotlyVisualizer (interactive)
├── Phase 4: AI Insights
│   └── GeminiInsights (with fallbacks)
└── Phase 5: Output
    ├── HTML Report (professional)
    └── Jupyter Display (inline)
```

---

## Complexity Analysis

| Analysis | Time | Space | Notes |
|----------|------|-------|-------|
| Statistics | O(n) | O(k) | k = numeric columns |
| Missing Values | O(n) | O(k) | k = all columns |
| Distributions | O(n) | O(k) | Shapiro-Wilk test |
| Outliers (IQR) | O(n log n) | O(k) | Per column sorting |
| Outliers (IF) | O(n log n) | O(k) | Isolation Forest |
| Correlations | O(k²) | O(k²) | Pearson + Spearman |
| Relationships | O(n*k) | O(1) | Variable combinations |
| **Advanced** | | | |
| Interactions | O(min(k³, 1K)) | O(k) | Limited to k=5 |
| Clusters | O(k² log k) | O(k²) | KMeans on correlation |
| Categorical | O(n log n) | O(k) | Per category test |
| Anomaly (Maha) | O(n*k²) | O(k²) | Mahalanobis distance |
| Feature Importance | O(k²) | O(k) | 4-method ensemble |
| Temporal | O(n log n) | O(k) | Trend + autocorr |
| Multivariate | O(n*k²) | O(k²) | PCA-like extraction |

---

## Recent Additions (v2)

✅ **Feature Engineering** - Automatic polynomial/interaction features  
✅ **Granular Controls** - Enable/disable analyses individually  
✅ **Random Seeding** - Reproducible analysis  
✅ **Batch Sampling** - 10x faster iteration on large datasets  
✅ **Advanced Insights** - Multivariate pattern detection (3+ features)  
✅ **Configuration Flexibility** - Fine-grained control over every aspect  

---

## Backward Compatibility

All v2 features are **opt-in** with sensible defaults:

```python
# v1 style still works - defaults run all analyses
analyser = Autolyse(html=True, api_key=None)
analyser.analyse(df)  # Uses all default flags (all = True)
```

New features only activate when explicitly enabled or set.

---

## Next Steps

1. **Try advanced features**: `examples/advanced_features.ipynb`
2. **Read analyzer details**: See individual analyzer docstrings
3. **Explore visualizations**: Run with `enable_visualizations=True`
4. **Generate HTML reports**: Set `html=True, api_key="YOUR_KEY"`

---

## Troubleshooting

**Q: Analysis is slow**  
A: Use `batch_size=5000` and `enable_feature_engineering=False`

**Q: Want exact reproducibility**  
A: Set both `random_seed=42` and avoid randomized analyses

**Q: Too many features created**  
A: `feature_engineering` caps at 20 features max automatically

**Q: Missing AI insights**  
A: Ensure `api_key` is set and GeminiInsights fallbacks activate

---

## License & Citation

See LICENSE file in repository.
