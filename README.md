# Autolyse

Auto Exploratory Data Analysis (EDA) with AI-powered insights.

Generate comprehensive exploratory data analysis with intelligent chart selection, missing value analysis, outlier detection, and AI-summarized findings - all in just **2 lines of code**.

## Features

- **Smart Data Type Detection**: Automatically identifies numerical, categorical, datetime, and text columns
- **Comprehensive Analysis**:
  - Basic statistics (mean, median, std, skewness, kurtosis)
  - Distribution analysis (histograms, KDE plots)
  - Correlation analysis (heatmaps, correlation matrices)
  - Missing value analysis
  - Outlier detection
  - Feature relationships and pair plots
- **Dual Visualization**: Both static (Matplotlib/Seaborn) and interactive (Plotly) plots
- **AI-Powered Insights**: Gemini API integration for 2-4 line summaries of findings
- **Flexible Output**: HTML reports or Jupyter notebook display

## Installation

```bash
pip install -r requirements.txt
```

Or with setup.py:

```bash
python setup.py install
```

## Quick Start

```python
import os
from autolyse import Autolyse
import pandas as pd

# Load your data
df = pd.read_csv('data.csv')

# One-liner analysis (with Gemini API key from environment)
analyser = Autolyse(html=True, api_key=os.environ.get("GEMINI_KEY"))
analyser.analyse(df)
```

## Project Structure

```
autolyse/
├── analyzers/          # Statistical analysis modules
├── visualizers/        # Visualization modules (matplotlib, plotly)
├── utils/              # Utility functions and helpers
├── output/             # Output handlers (HTML, Jupyter)
└── core.py            # Main Autolyse class
```

## Requirements

- Python >= 3.8
- pandas, numpy
- matplotlib, seaborn, plotly
- scikit-learn
- google-generativeai
- scipy

## License

MIT

## Author

Sanyam Chaudhary
