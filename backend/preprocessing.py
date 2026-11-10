"""Reading, validating and preparing transaction data."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from .config import MAX_COLUMNS, MAX_ROWS, MAX_UPLOAD_MB

CREDITCARD_FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_CANDIDATES = ["class", "is_fraud", "isfraud", "fraud", "label", "target"]


# --------------------------------------------------------------- reading
def read_csv_safely(source, filename: str | None = None, max_mb: int = MAX_UPLOAD_MB) -> pd.DataFrame:
    """Read a CSV from a path or an uploaded file object with basic safety checks."""
    name = filename or getattr(source, "name", str(source))
    if not str(name).lower().endswith(".csv"):
        raise ValueError("Only .csv files are accepted.")
    size = getattr(source, "size", None)
    if size is None and isinstance(source, (str, Path)):
        size = Path(source).stat().st_size
    if size is not None and size > max_mb * 1024 * 1024:
        raise ValueError(f"File is larger than the {max_mb} MB limit.")
    try:
        df = pd.read_csv(source, low_memory=False)
    except pd.errors.EmptyDataError:
        raise ValueError("The file is empty.") from None
    except Exception as exc:  # malformed CSV, bad encoding, ...
        raise ValueError(f"Could not read the CSV file: {exc}") from None
    if df.empty:
        raise ValueError("The CSV has no rows.")
    if len(df) > MAX_ROWS or df.shape[1] > MAX_COLUMNS:
        raise ValueError(f"File too large to process (max {MAX_ROWS:,} rows, {MAX_COLUMNS} columns).")
    df.columns = [str(c).strip() for c in df.columns]
    return df


# ------------------------------------------------------ column detection
def _is_binary(series: pd.Series) -> bool:
    s = pd.to_numeric(series, errors="coerce").dropna()
    return len(s) > 0 and set(s.unique().tolist()) <= {0, 1, 0.0, 1.0}


def detect_target(df: pd.DataFrame, preferred: str | None = None) -> str | None:
    """Find the 0/1 target column (the credit card dataset uses 'Class')."""
    if preferred and preferred in df.columns and _is_binary(df[preferred]):
        return preferred
    lower = {c.lower(): c for c in df.columns}
    for cand in TARGET_CANDIDATES:
        if cand in lower and _is_binary(df[lower[cand]]):
            return lower[cand]
    return None


def align_columns(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Rename columns that match a required feature ignoring case (e.g. 'v1' -> 'V1')."""
    lower = {c.lower(): c for c in df.columns}
    rename = {lower[f.lower()]: f for f in feature_cols if f not in df.columns and f.lower() in lower}
    return df.rename(columns=rename) if rename else df


def default_feature_cols(df: pd.DataFrame, target: str | None) -> list[str]:
    """Numeric columns except the target. Keeps the standard order for the credit card data."""
    if all(c in df.columns for c in CREDITCARD_FEATURES):
        return list(CREDITCARD_FEATURES)
    numeric = [c for c in df.columns if c != target and pd.api.types.is_numeric_dtype(df[c])]
    return [c for c in numeric if not re.fullmatch(r"(?i)(id|index|unnamed:.*)", c)]


def is_creditcard_format(df: pd.DataFrame) -> bool:
    return all(c in df.columns for c in CREDITCARD_FEATURES)


# ------------------------------------------------------------ validation
def validate_for_model(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Check an uploaded dataframe against the features the trained model needs."""
    df = align_columns(df, feature_cols)
    errors, warnings = [], []
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        shown = ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else "")
        errors.append(f"Missing {len(missing)} required column(s): {shown}")
    present = [c for c in feature_cols if c in df.columns]
    for c in present:
        coerced = pd.to_numeric(df[c], errors="coerce")
        bad = int(coerced.isna().sum() - df[c].isna().sum())
        if bad > 0:
            errors.append(f"Column '{c}' has {bad} non-numeric value(s).")
    na_total = int(df[present].isna().sum().sum()) if present else 0
    if na_total:
        warnings.append(f"{na_total:,} missing value(s) found; they will be filled with the training median.")
    if "Amount" in present and (pd.to_numeric(df["Amount"], errors="coerce") < 0).any():
        warnings.append("Some Amount values are negative.")
    target = detect_target(df)
    return {"ok": not errors, "errors": errors, "warnings": warnings, "target": target,
            "rows": len(df), "df": df}


def coerce_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    return X.replace([np.inf, -np.inf], np.nan)


# -------------------------------------------------------------- pipeline
def split_scale_columns(feature_cols: list[str]) -> tuple[list[str], list[str]]:
    """Credit card data: scale Time and Amount only (V1-V28 are already PCA-scaled).
    Other datasets: scale every feature."""
    if all(c in feature_cols for c in CREDITCARD_FEATURES):
        scale = ["Time", "Amount"]
    else:
        scale = list(feature_cols)
    others = [c for c in feature_cols if c not in scale]
    return scale, others


def build_pipeline(feature_cols: list[str], estimator) -> Pipeline:
    """Imputer + scaler fitted ONLY on training data inside the pipeline (no leakage)."""
    scale_cols, other_cols = split_scale_columns(feature_cols)
    transformers = [("scale", Pipeline([("imp", SimpleImputer(strategy="median")),
                                        ("sc", RobustScaler())]), scale_cols)]
    if other_cols:
        transformers.append(("pass", SimpleImputer(strategy="median"), other_cols))
    return Pipeline([("prep", ColumnTransformer(transformers)), ("clf", estimator)])


def pipeline_feature_order(feature_cols: list[str]) -> list[str]:
    scale_cols, other_cols = split_scale_columns(feature_cols)
    return scale_cols + other_cols
