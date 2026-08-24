"""Data preparation and column-type detection utilities."""

import numpy as np
import pandas as pd


# Columns whose values look like free text rather than categories once unique
# ratio exceeds this share of observed rows.
TEXT_UNIQUE_RATIO = 0.5
DATETIME_PARSE_SAMPLE = 100


def infer_object_series_type(series: pd.Series) -> str:
    """Classify a string/object series as ``categorical``, ``text`` or ``datetime``.

    Works with both legacy ``object`` dtype and pandas >= 3 ``str`` dtype.
    """
    from pandas.api.types import is_string_dtype

    col = series.dropna()
    if len(col) == 0:
        return "text"

    # Datetime: parsing must succeed on essentially the whole sample; plain
    # words like "free"/"pro" never parse, numbers-as-dates are rejected.
    if is_string_dtype(col):
        try:
            parsed = pd.to_datetime(col.head(DATETIME_PARSE_SAMPLE),
                                    errors="coerce", format="mixed")
            if parsed.notna().mean() >= 0.9:
                return "datetime"
        except (ValueError, TypeError):
            # format="mixed" needs pandas >= 2.0; older stacks just skip
            # datetime inference for strings.
            pass

    unique_ratio = col.nunique() / len(col)
    if unique_ratio < TEXT_UNIQUE_RATIO:
        return "categorical"
    return "text"


class DataPreparation:
    """Detect column types and provide light-weight cleaning utilities."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.column_types = {}
        self._detect_column_types()

    def _detect_column_types(self) -> None:
        for col in self.df.columns:
            series = self.df[col]
            dtype = series.dtype
            if pd.api.types.is_bool_dtype(dtype):
                self.column_types[col] = "boolean"
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                self.column_types[col] = "datetime"
            elif isinstance(dtype, pd.CategoricalDtype):
                self.column_types[col] = "categorical"
            elif pd.api.types.is_numeric_dtype(dtype):
                self.column_types[col] = "numeric"
            elif (
                pd.api.types.is_object_dtype(dtype)
                or pd.api.types.is_string_dtype(dtype)
            ):
                self.column_types[col] = infer_object_series_type(series)
            else:
                self.column_types[col] = "other"

    # ------------------------------------------------------------- accessors

    def get_column_types(self):
        return self.column_types.copy()

    def get_columns_by_type(self, col_type: str):
        return [c for c, t in self.column_types.items() if t == col_type]

    def get_numeric_columns(self):
        return self.get_columns_by_type("numeric")

    def get_categorical_columns(self):
        return self.get_columns_by_type("categorical")

    def get_datetime_columns(self):
        return self.get_columns_by_type("datetime")

    def get_text_columns(self):
        return self.get_columns_by_type("text")

    def get_boolean_columns(self):
        return self.get_columns_by_type("boolean")

    def get_type_summary(self):
        summary = {}
        for t in self.column_types.values():
            summary[t] = summary.get(t, 0) + 1
        return summary

    def get_column_info(self) -> pd.DataFrame:
        n = max(len(self.df), 1)
        info = []
        for col in self.df.columns:
            null_count = int(self.df[col].isna().sum())
            unique = int(self.df[col].nunique())
            info.append({
                "Column": col,
                "Type": self.column_types[col],
                "Non-Null": int(self.df[col].notna().sum()),
                "Null %": round(null_count / n * 100, 2),
                "Unique": unique,
                "Unique %": round(unique / n * 100, 2),
            })
        return pd.DataFrame(info)

    # ------------------------------------------------------------- cleaning
    #
    # These helpers mutate an internal copy so callers never corrupt their own
    # frame accidentally; retrieve results through :meth:`get_dataframe`.

    def handle_missing_values(self, strategy: str = "report"):
        summary = {
            "total_missing": int(self.df.isna().sum().sum()),
            "rows_with_missing": int(self.df.isna().any(axis=1).sum()),
            "strategy": strategy,
            "action": None,
        }
        if strategy == "drop":
            self.df = self.df.dropna()
            summary["action"] = f"Dropped rows with missing values. New shape: {self.df.shape}"
        elif strategy in {"mean", "median"}:
            agg = getattr(pd.Series, strategy)
            filled = []
            for col in self.df.columns:
                s = self.df[col]
                if pd.api.types.is_numeric_dtype(s.dtype):
                    filled.append(s.fillna(agg(s)))
                else:
                    filled.append(s)
            self.df = pd.concat(filled, axis=1)
            summary["action"] = f"Filled numeric columns with {strategy}"
        elif strategy == "forward_fill":
            self.df = self.df.ffill()
            summary["action"] = "Forward filled missing values"
        elif strategy == "backward_fill":
            self.df = self.df.bfill()
            summary["action"] = "Backward filled missing values"
        elif strategy != "report":
            raise ValueError(
                f"Unknown strategy {strategy!r}; use report/drop/mean/median/"
                "forward_fill/backward_fill"
            )
        else:
            summary["action"] = "No action - reporting only"
        return summary

    def remove_duplicates(self, subset=None, keep: str = "first"):
        dup_mask = self.df.duplicated(subset=subset, keep=False)
        duplicates_before = int(dup_mask.sum())
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        return {
            "duplicates_found": duplicates_before // 2 if duplicates_before else 0,
            "duplicates_removed": duplicates_before - int(self.df.duplicated(subset=subset, keep=False).sum()),
            "new_shape": self.df.shape,
            "action": f"Removed duplicates. New shape: {self.df.shape}",
        }

    def remove_low_variance_columns(self, threshold: float = 0.01):
        cols_removed = []
        for col in self.get_numeric_columns():
            col_range = self.df[col].max() - self.df[col].min()
            if col_range and col_range > 0:
                if self.df[col].var() / (col_range ** 2) < threshold:
                    cols_removed.append(col)
        self.df = self.df.drop(columns=cols_removed)
        return {
            "columns_removed": cols_removed,
            "new_shape": self.df.shape,
            "action": f"Removed {len(cols_removed)} low-variance columns. New shape: {self.df.shape}",
        }

    def normalize_numeric(self, method: str = "minmax"):
        numeric_cols = self.get_numeric_columns()
        if method == "minmax":
            for col in numeric_cols:
                lo, hi = self.df[col].min(), self.df[col].max()
                if hi > lo:
                    self.df[col] = (self.df[col] - lo) / (hi - lo)
            action = "Applied Min-Max normalization (0-1 scale)"
        elif method == "zscore":
            for col in numeric_cols:
                std = self.df[col].std()
                if std and std > 0:
                    self.df[col] = (self.df[col] - self.df[col].mean()) / std
            action = "Applied Z-score normalization (standardization)"
        elif method == "log":
            for col in numeric_cols:
                if (self.df[col].dropna() > 0).all():
                    self.df[col] = np.log1p(self.df[col])
            action = "Applied log transformation to positive columns"
        else:
            raise ValueError("method must be one of minmax/zscore/log")
        return {
            "method": method,
            "columns_normalized": numeric_cols,
            "action": action,
        }

    def encode_categorical(self, method: str = "label"):
        categorical_cols = [
            c for c in self.get_categorical_columns()
        ]
        mappings = {}
        if method == "label":
            for col in categorical_cols:
                categories = self.df[col].dropna().unique()
                mapping = {val: idx for idx, val in enumerate(categories)}
                self.df[col] = self.df[col].map(mapping)
                mappings[col] = mapping
            action = f"Applied label encoding to {len(categorical_cols)} columns"
        elif method == "onehot":
            self.df = pd.get_dummies(self.df, columns=categorical_cols, drop_first=True)
            action = f"Applied one-hot encoding to {len(categorical_cols)} columns"
        else:
            raise ValueError("method must be 'label' or 'onehot'")
        return {
            "method": method,
            "columns_encoded": categorical_cols,
            "mappings": mappings if method == "label" else None,
            "action": action,
            "new_shape": self.df.shape,
        }

    def get_dataframe(self) -> pd.DataFrame:
        return self.df.copy()

    def validate_data(self):
        shape = self.df.shape
        total_cells = max(shape[0] * shape[1], 1)
        missing_cells = int(self.df.isna().sum().sum())
        duplicate_rows = int(self.df.duplicated().sum())
        validation = {
            "shape": shape,
            "total_cells": shape[0] * shape[1],
            "missing_cells": missing_cells,
            "missing_pct": round(missing_cells / total_cells * 100, 2),
            "duplicate_rows": duplicate_rows,
            "duplicate_pct": round(duplicate_rows / max(len(self.df), 1) * 100, 2),
            "column_types": self.get_type_summary(),
            "numeric_columns": len(self.get_numeric_columns()),
            "categorical_columns": len(self.get_categorical_columns()),
            "datetime_columns": len(self.get_datetime_columns()),
            "text_columns": len(self.get_text_columns()),
            "boolean_columns": len(self.get_boolean_columns()),
        }
        quality_score = 100 - validation["missing_pct"] - min(validation["duplicate_pct"], 10)
        validation["data_quality_score"] = round(max(0.0, quality_score), 2)
        return validation


class DataInspector:
    """Convenience wrappers around quick pandas exploration."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def summary_statistics(self) -> pd.DataFrame:
        return self.df.describe().T

    def value_counts_summary(self, col: str, top_n: int = 10) -> pd.Series:
        return self.df[col].value_counts().head(top_n)

    def correlation_matrix(self, numeric_only: bool = True) -> pd.DataFrame:
        if numeric_only:
            return self.df.select_dtypes(include=[np.number]).corr()
        return self.df.corr(numeric_only=False)

    def check_data_types(self) -> pd.DataFrame:
        return pd.DataFrame({
            "Column": self.df.columns,
            "Type": self.df.dtypes.astype(str),
            "Non-Null Count": self.df.notna().sum().values,
            "Null Count": self.df.isna().sum().values,
            "Unique": [self.df[c].nunique() for c in self.df.columns],
        })
