# Autolyse

**Prescriptive automated EDA** - not just a report, a diagnosis.

```python
from autolyse import Autolyse
import pandas as pd

Autolyse(target="churn").analyse(pd.read_csv("customers.csv"))
```

One call produces:

- **Data Health Score** - explainable 0-100 score with per-dimension breakdown
  (completeness, uniqueness, validity, modeling-readiness)
- **Ranked findings** - linter-style diagnostics, each with severity, the exact
  evidence, and a copy-paste pandas fix (missing columns, duplicates, leakage,
  imbalance, skew, high cardinality...)
- **Target-aware analysis** - predictive power ranking for every feature plus
  leakage detection that flags features that know the answer too well
- **Interactive HTML report** - fully self-contained single file (charts work
  offline), or rich Jupyter display
- **Classic EDA** - statistics, distributions, correlations, outliers,
  relationships - all type-safe on pandas 2.x and 3.x

## Install

```bash
pip install -e .
```

Optional AI narration (polishes insights via Gemini - never required):

```bash
pip install -e ".[ai]"
```

## What makes it different

| Typical auto-EDA | Autolyse |
|---|---|
| "Column X has 43% missing" | "[HIGH] X is 43% missing -> imputation injects assumptions; here's when to drop vs impute + code" |
| Pretty charts | Charts **plus** a fix list ordered by what blocks your model first |
| Correlation tables | Predictive-power ranking vs your target + leakage suspects |
| Works only offline by accident | Deterministic engine always works; LLM narration is opt-in polish |

## API tour

```python
from autolyse import Autolyse

an = Autolyse(html=True, target="price")   # prescriptive + report
results = an.analyse(df)

an.get_findings()        # [Finding(rule_id='COL_HIGH_MISSING', ...), ...]
an.get_health_score()    # HealthScore(overall=87, grade='B', by_category={...})
an.get_analysis_results()# raw analyses dict
an.df                    # working frame (sampled / feature-engineered)
```

Large data? Analyze a reproducible sample:

```python
Autolyse(batch_size=5000, random_seed=42).analyse(huge_df)
```

Custom LLM narration (any provider):

```python
class MyProvider:
    def complete(self, prompt: str) -> str | None:
        ...

Autolyse(llm_provider=MyProvider()).analyse(df)
```

Granular control (all stages toggleable): `enable_statistics`,
`enable_missing_values`, `enable_distributions`, `enable_outliers`,
`enable_correlations`, `enable_relationships`, `enable_advanced_insights`,
`enable_feature_engineering`, `enable_visualizations`.

## Development

```bash
pip install -e ".[dev]"
pytest
```

Python >= 3.9. Core deps: pandas, numpy, scipy, scikit-learn, matplotlib,
seaborn, plotly.

## License

MIT - see [LICENSE](LICENSE).
