"""Findings engine: linter-style data-quality diagnostics with fixes.

This module is the core of Autolyse's prescriptive layer. Instead of only
describing a dataset, it produces ranked, evidence-backed *findings* - each
with a severity, the numbers that justify it, and a ready-to-run pandas
snippet that resolves it - plus an explainable weighted health score.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd


class Severity(Enum):
    CRITICAL = "critical"   # likely invalidates downstream work
    HIGH = "high"           # materially skews results if ignored
    MEDIUM = "medium"       # worth fixing soon
    LOW = "low"             # hygiene / informational


@dataclass(frozen=True)
class Finding:
    """One diagnostic observation about the dataset."""
    rule_id: str
    severity: Severity
    title: str
    detail: str                 # includes the concrete evidence
    columns: tuple = ()
    fix_snippet: str = ""       # runnable pandas code, "" when none applies
    metric: Optional[float] = None

    @property
    def severity_weight(self) -> int:
        return _SEVERITY_DEDUCTIONS[self.severity]


_SEVERITY_DEDUCTIONS = {
    Severity.CRITICAL: 30,
    Severity.HIGH: 14,
    Severity.MEDIUM: 6,
    Severity.LOW: 2,
}

_SCORE_CATEGORIES = {
    "completeness": ["COL_ALL_MISSING", "COL_HIGH_MISSING"],
    "uniqueness": ["ROW_DUPLICATES", "COL_ID_LIKE"],
    "validity": ["INF_VALUES", "WHITESPACE_STRINGS", "COL_CONSTANT",
                 "HEAVY_SKEW", "OUTLIER_BURDEN"],
    "modeling": ["HIGH_CARDINALITY", "CLASS_IMBALANCE", "LEAKAGE_RISK",
                 "TARGET_CONSTANT"],
}


@dataclass
class HealthScore:
    overall: int
    grade: str
    by_category: Dict[str, int] = field(default_factory=dict)


def grade_for(score: float) -> str:
    if score >= 97: return "A+"
    if score >= 90: return "A"
    if score >= 80: return "B"
    if score >= 65: return "C"
    if score >= 50: return "D"
    return "F"


# --------------------------------------------------------------------------
# Rule implementations. Each takes (df, ctx) and returns list[Finding].
# --------------------------------------------------------------------------

def _rule_all_missing(df, ctx) -> List[Finding]:
    out = []
    n = len(df)
    for col in df.columns:
        if df[col].isna().all():
            out.append(Finding(
                rule_id="COL_ALL_MISSING",
                severity=Severity.CRITICAL,
                title=f"'{col}' is completely empty",
                detail=f"All {n:,} values in '{col}' are missing.",
                columns=(col,),
                fix_snippet=f"df = df.drop(columns=['{col}'])",
                metric=100.0,
            ))
    return out


def _rule_high_missing(df, ctx) -> List[Finding]:
    out = []
    n = len(df)
    for col in df.columns:
        pct = df[col].isna().mean() * 100
        if 40 <= pct < 100:
            out.append(Finding(
                rule_id="COL_HIGH_MISSING",
                severity=Severity.HIGH if pct >= 70 else Severity.MEDIUM,
                title=f"'{col}' is {pct:.0f}% missing",
                detail=(f"{int(df[col].isna().sum()):,} of {n:,} rows are missing "
                        f"in '{col}'. Imputation at this rate injects strong "
                        f"assumptions; prefer dropping unless the column is "
                        f"business-critical."),
                columns=(col,),
                fix_snippet=(
                    f"# Option A: drop the column\n"
                    f"df = df.drop(columns=['{col}'])\n"
                    f"# Option B: impute (numeric median shown)\n"
                    f"# df['{col}'] = df['{col}'].fillna(df['{col}'].median())"
                ),
                metric=float(pct),
            ))
    return out


def _rule_duplicates(df, ctx) -> List[Finding]:
    dup_count = int(df.duplicated().sum())
    if dup_count == 0:
        return []
    pct = dup_count / len(df) * 100
    return [Finding(
        rule_id="ROW_DUPLICATES",
        severity=Severity.HIGH if pct > 5 else Severity.MEDIUM,
        title=f"{dup_count:,} duplicate rows ({pct:.1f}%)",
        detail=("Exact duplicate rows inflate counts, bias validation splits "
                "and leak between train/test."),
        columns=(),
        fix_snippet="df = df.drop_duplicates()",
        metric=float(pct),
    )]


def _rule_constant_columns(df, ctx) -> List[Finding]:
    out = []
    for col in ctx.get("numeric_cols", []):
        series = df[col].dropna()
        if len(series) == 0:
            continue
        if series.nunique() <= 1:
            out.append(Finding(
                rule_id="COL_CONSTANT",
                severity=Severity.MEDIUM,
                title=f"'{col}' has a single value",
                detail=(f"Every non-null entry equals {series.iloc[0]!r}. "
                        f"Zero variance carries no information."),
                columns=(col,),
                fix_snippet=f"df = df.drop(columns=['{col}'])",
            ))
    return out


def _rule_id_like(df, ctx) -> List[Finding]:
    out = []
    n = len(df)
    keywords = ("id", "uuid", "guid", "key", "index")
    for col in df.columns:
        if df[col].nunique() != n:
            continue
        hinted = any(k in col.lower() for k in keywords)
        out.append(Finding(
            rule_id="COL_ID_LIKE",
            severity=Severity.LOW,
            title=f"'{col}' looks like a row identifier",
            detail=(f"{df[col].nunique():,} distinct values across {n:,} rows."
                    + (" Name suggests an explicit ID." if hinted else "")),
            columns=(col,),
            fix_snippet=(f"id_cols = ['{col}']\n"
                         f"# Exclude identifiers from features before modeling:\n"
                         f"# X = df.drop(columns=id_cols)"),
        ))
    return out


def _rule_high_cardinality(df, ctx) -> List[Finding]:
    out = []
    for col in ctx.get("categorical_cols", []):
        n_unique = df[col].nunique()
        if n_unique > 50:
            ratio = n_unique / max(len(df), 1)
            out.append(Finding(
                rule_id="HIGH_CARDINALITY",
                severity=Severity.MEDIUM,
                title=f"'{col}' has {n_unique:,} categories",
                detail=(f"{ratio:.0%} of rows are distinct categories. One-hot "
                        f"encoding would add thousands of sparse columns; "
                        f"prefer hashing, frequency or target encoding."),
                columns=(col,),
                fix_snippet=(
                    f"# Frequency encoding example\n"
                    f"freq = df['{col}'].value_counts(normalize=True)\n"
                    f"df['{col}_freq'] = df['{col}'].map(freq)"
                ),
                metric=float(n_unique),
            ))
    return out


def _rule_heavy_skew(df, ctx) -> List[Finding]:
    out = []
    for col in ctx.get("numeric_cols", []):
        series = df[col].dropna()
        if len(series) < 20 or series.nunique() <= 2:
            continue
        skew = series.skew()
        if pd.notna(skew) and abs(skew) > 2:
            direction = "right" if skew > 0 else "left"
            out.append(Finding(
                rule_id="HEAVY_SKEW",
                severity=Severity.LOW,
                title=f"'{col}' is heavily skewed (skew={skew:.1f})",
                detail=(f"Strongly {direction}-skewed; linear models and "
                        f"distance metrics treat the long tail poorly."),
                columns=(col,),
                fix_snippet=(
                    f"# If strictly positive:\n"
                    f"df['{col}_log'] = np.log1p(df['{col}'])"
                ) if series.min() > 0 else "",
                metric=float(abs(skew)),
            ))
    return out


def _rule_outlier_burden(df, ctx) -> List[Finding]:
    outlier_summary = (ctx.get("analyses") or {}).get("outliers", {}).get(
        "iqr_method", {})
    out = []
    for col, info in outlier_summary.items():
        pct = info.get("outlier_percentage", 0)
        if pct > 8:
            out.append(Finding(
                rule_id="OUTLIER_BURDEN",
                severity=Severity.MEDIUM,
                title=f"{pct:.0f}% of '{col}' sits beyond Tukey fences",
                detail=(f"{info['outlier_count']:,} points outside "
                        f"[{info['lower_bound']:.2f}, {info['upper_bound']:.2f}]. "
                        f"At this rate 'outliers' may be a genuine subpopulation "
                        f"- inspect before clipping."),
                columns=(col,),
                fix_snippet=(
                    f"# Inspect first, then optionally winsorize:\n"
                    f"# q1, q99 = df['{col}'].quantile([0.01, 0.99])\n"
                    f"# df['{col}_winsor'] = df['{col}'].clip(q1, q99)"
                ),
                metric=float(pct),
            ))
    return out


def _rule_infinite_values(df, ctx) -> List[Finding]:
    out = []
    for col in ctx.get("numeric_cols", []):
        n_inf = int(np.isinf(df[col].to_numpy(dtype=float, na_value=np.nan)).sum()) \
            if len(df) else 0
        if n_inf:
            out.append(Finding(
                rule_id="INF_VALUES",
                severity=Severity.HIGH,
                title=f"'{col}' contains {n_inf:,} infinite values",
                detail="Most sklearn models and scalers reject inf outright.",
                columns=(col,),
                fix_snippet=(f"import numpy as np\n"
                             f"df['{col}'] = df['{col}'].replace("
                             f"[np.inf, -np.inf], np.nan)"),
                metric=float(n_inf),
            ))
    return out


def _rule_whitespace_strings(df, ctx) -> List[Finding]:
    out = []
    sample = df.head(2000)
    for col in ctx.get("categorical_cols", []) + ctx.get("text_cols", []):
        s = sample[col].dropna()
        if len(s) == 0:
            continue
        dirty = s.astype(str).str.match(r"^\s|\s$")
        frac = dirty.mean()
        if frac > 0.02:
            out.append(Finding(
                rule_id="WHITESPACE_STRINGS",
                severity=Severity.LOW,
                title=f"'{col}' values have stray whitespace",
                detail=(f"{frac:.0%} of sampled values start or end with "
                        f"whitespace - silently splits categories like "
                        f"'pro' vs ' pro'."),
                columns=(col,),
                fix_snippet=f"df['{col}'] = df['{col}'].str.strip()",
                metric=float(frac),
            ))
    return out


def _rule_class_imbalance(df, ctx) -> List[Finding]:
    target = ctx.get("target")
    if not target or target not in df.columns:
        return []
    counts = df[target].value_counts(dropna=False)
    if len(counts) != 2:
        return []  # multi-class handled separately
    minority_pct = counts.min() / counts.sum() * 100
    if minority_pct >= 15:
        return []
    return [Finding(
        rule_id="CLASS_IMBALANCE",
        severity=Severity.HIGH if minority_pct < 5 else Severity.MEDIUM,
        title=f"Target '{target}' is imbalanced ({minority_pct:.1f}% minority)",
        detail=(f"Distribution: {counts.to_dict()}. Accuracy becomes meaningless; "
                f"use stratified splits and PR-AUC / class weights."),
        columns=(target,),
        fix_snippet=(
            f"from sklearn.model_selection import train_test_split\n"
            f"train_test_split(X, y, stratify=y, test_size=0.2,\n"
            f"                 random_state={ctx.get('random_seed', 42)})"
        ),
        metric=float(minority_pct),
    )]


def _rule_target_constant(df, ctx) -> List[Finding]:
    target = ctx.get("target")
    if not target or target not in df.columns:
        return []
    if df[target].dropna().nunique() <= 1:
        return [Finding(
            rule_id="TARGET_CONSTANT",
            severity=Severity.CRITICAL,
            title=f"Target '{target}' has a single value",
            detail="A constant target cannot be modeled.",
            columns=(target,),
        )]
    return []


def _rule_leakage_risk(df, ctx) -> List[Finding]:
    powers = (ctx.get("analyses") or {}).get("target_analysis", {}) \
        .get("predictive_power", {})
    target = ctx.get("target")
    out = []
    for col, info in powers.items():
        strength = info.get("strength", 0)
        if strength >= 0.98 and col != target:
            out.append(Finding(
                rule_id="LEAKAGE_RISK",
                severity=Severity.CRITICAL,
                title=f"'{col}' near-perfectly predicts '{target}' "
                      f"(power={strength:.3f})",
                detail=("Features this deterministic usually encode the answer "
                        "- e.g. post-outcome fields, aggregates over the target, "
                        "or duplicated keys. Verify provenance or exclude."),
                columns=(col, target),
                fix_snippet=f"suspicious = ['{col}']\nX = X.drop(columns=suspicious)",
                metric=float(strength),
            ))
    return out


RULES: List[Callable] = [
    _rule_all_missing,
    _rule_high_missing,
    _rule_duplicates,
    _rule_constant_columns,
    _rule_id_like,
    _rule_high_cardinality,
    _rule_heavy_skew,
    _rule_outlier_burden,
    _rule_infinite_values,
    _rule_whitespace_strings,
    _rule_target_constant,
    _rule_class_imbalance,
    _rule_leakage_risk,
]


class FindingsEngine:
    """Run all diagnostic rules and produce ranked findings plus a score."""

    def __init__(self, df: pd.DataFrame, column_types: Optional[Dict] = None,
                 analyses: Optional[Dict] = None, target: Optional[str] = None,
                 random_seed: int = 42):
        self.df = df
        self.analyses = analyses or {}
        self.target = target
        self.random_seed = random_seed

        types = column_types or {}
        self._ctx = {
            "numeric_cols": types.get("numeric", []),
            "categorical_cols": types.get("categorical", []),
            "text_cols": types.get("text", []),
            "datetime_cols": types.get("datetime", []),
            "analyses": self.analyses,
            "target": target,
            "random_seed": random_seed,
        }

    def run(self) -> List[Finding]:
        findings: List[Finding] = []
        for rule in RULES:
            try:
                findings.extend(rule(self.df, self._ctx))
            except Exception as error:  # one broken rule must not sink the rest
                findings.append(Finding(
                    rule_id=f"RULE_ERROR:{getattr(rule, '__name__', '?')}",
                    severity=Severity.LOW,
                    title="Diagnostic rule failed to run",
                    detail=str(error),
                ))

        order = {Severity.CRITICAL: 0, Severity.HIGH: 1,
                 Severity.MEDIUM: 2, Severity.LOW: 3}
        findings.sort(key=lambda f: (order[f.severity],
                                     -(f.metric or 0)))
        return findings

    def health_score(self, findings: List[Finding]) -> HealthScore:
        overall = 100
        per_rule_deduction: Dict[str, float] = {}
        for finding in findings:
            if finding.rule_id.startswith("RULE_ERROR"):
                continue
            # Diminishing returns within a rule id: 1st hit full, then 60%,
            # 36%, ... so 40 medium-missing columns don't zero the score.
            k = sum(1 for r in per_rule_deduction if r == finding.rule_id)
            per_rule_deduction[finding.rule_id] = \
                per_rule_deduction.get(finding.rule_id, 0) + 1
            overall -= finding.severity_weight * (0.6 ** k)

        overall = int(round(max(0.0, min(100.0, overall))))

        by_category = {}
        rule_to_category = {
            rid: cat for cat, rids in _SCORE_CATEGORIES.items() for rid in rids
        }
        for category in _SCORE_CATEGORIES:
            cat_penalty = 0.0
            hits: Dict[str, int] = {}
            for finding in findings:
                cat = rule_to_category.get(finding.rule_id)
                if cat != category:
                    continue
                k = hits.get(finding.rule_id, 0)
                hits[finding.rule_id] = k + 1
                cat_penalty += finding.severity_weight * (0.6 ** k)
            by_category[category] = int(round(max(0.0, min(100.0, 100 - cat_penalty))))

        return HealthScore(overall=overall, grade=grade_for(overall),
                           by_category=by_category)
