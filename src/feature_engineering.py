
"""
Feature engineering module for customer churn pipeline.
Creates derived features, drops multicollinear columns, scales, and applies SMOTE.

Pipeline output:
  data/processed/churn_train.csv       — scaled training set (used by modeling.py)
  data/processed/churn_test.csv        — scaled test set  (used by evaluation.py)
  data/processed/churn_train_smote.csv — SMOTE-balanced training set (optional)
  models/scaler.pkl                    — fitted StandardScaler
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

# Support both package import (from src.feature_engineering)
# and direct script execution (python feature_engineering.py)
try:
    from .preprocessing import run_preprocessing_pipeline
except ImportError:
    from preprocessing import run_preprocessing_pipeline


# Charge columns are deterministic linear functions of their corresponding
# minutes columns (charge = minutes x fixed rate), causing perfect
# multicollinearity. Drop charges; keep minutes.
_CHARGE_COLS = [
    "Total day charge",
    "Total eve charge",
    "Total night charge",
    "Total intl charge",
]


# ---------------------------------------------------------------------------
# Feature-creation helpers
# ---------------------------------------------------------------------------

def create_total_minutes(df: pd.DataFrame) -> pd.DataFrame:
    """Total usage minutes across all periods including international."""
    df = df.copy()
    cols = [
        "Total day minutes", "Total eve minutes",
        "Total night minutes", "Total intl minutes",
    ]
    if all(c in df.columns for c in cols):
        df["total_minutes"] = df[cols].sum(axis=1)
    return df


def create_total_calls(df: pd.DataFrame) -> pd.DataFrame:
    """Total call count across all periods."""
    df = df.copy()
    cols = [
        "Total day calls", "Total eve calls",
        "Total night calls", "Total intl calls",
    ]
    if all(c in df.columns for c in cols):
        df["total_calls"] = df[cols].sum(axis=1)
    return df


def create_avg_minutes_per_call(df: pd.DataFrame) -> pd.DataFrame:
    """Average call length — proxy for engagement depth."""
    df = df.copy()
    if "total_minutes" not in df.columns:
        df = create_total_minutes(df)
    if "total_calls" not in df.columns:
        df = create_total_calls(df)
    df["avg_minutes_per_call"] = np.where(
        df["total_calls"] > 0,
        df["total_minutes"] / df["total_calls"],
        0.0,
    )
    return df


def create_calls_per_minute(df: pd.DataFrame) -> pd.DataFrame:
    """Calls per minute — usage intensity / fragmentation proxy."""
    df = df.copy()
    if "total_minutes" not in df.columns:
        df = create_total_minutes(df)
    if "total_calls" not in df.columns:
        df = create_total_calls(df)
    df["calls_per_minute"] = np.where(
        df["total_minutes"] > 0,
        df["total_calls"] / df["total_minutes"],
        0.0,
    )
    return df


def create_high_service_calls(df: pd.DataFrame) -> pd.DataFrame:
    """Binary flag: >=4 customer-service calls is a strong churn signal."""
    df = df.copy()
    if "Customer service calls" in df.columns:
        df["high_service_calls"] = (df["Customer service calls"] >= 4).astype(int)
    return df


def create_intl_active_no_plan(df: pd.DataFrame) -> pd.DataFrame:
    """Binary flag: uses international minutes but has no international plan.
    Unexpected overage charges are a known churn driver."""
    df = df.copy()
    if "Total intl minutes" in df.columns and "International plan" in df.columns:
        df["intl_active_no_plan"] = (
            (df["Total intl minutes"] > 0) & (df["International plan"] == 0)
        ).astype(int)
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_feature_engineering(
    input_path: str = "data/processed/churn_clean.csv",
    train_output_path: str = "data/processed/churn_train.csv",
    test_output_path: str = "data/processed/churn_test.csv",
    smote_output_path: str = "data/processed/churn_train_smote.csv",
    scaler_path: str = "models/scaler.pkl",
    use_raw_fallback: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
    apply_smote: bool = True,
) -> tuple:
    """
    Full feature engineering pipeline:
      1. Load cleaned data (or re-run preprocessing as fallback).
      2. Drop multicollinear charge columns.
      3. Create business-derived and interaction features.
      4. Stratified train/test split — BEFORE scaling (no leakage).
      5. Fit StandardScaler on X_train only; transform both splits.
      6. Optionally apply SMOTE on training split only.
      7. Save scaler, train set, test set, and SMOTE train set.

    Returns:
        (X_train_scaled, X_test_scaled, y_train, y_test)
    """
    # 1. Load data
    path = Path(input_path)
    if path.exists():
        df = pd.read_csv(path)
    elif use_raw_fallback:
        run_preprocessing_pipeline(
            input_path="data/raw/churn-bigml-80_raw.csv",
            output_path="data/processed/churn_clean.csv",
        )
        df = pd.read_csv("data/processed/churn_clean.csv")
    else:
        raise FileNotFoundError(f"Input not found: {input_path}")

    target = "Churn"
    if target not in df.columns:
        raise ValueError("Target column 'Churn' not found in DataFrame.")

    # 2. Drop multicollinear charge columns
    df = df.drop(columns=[c for c in _CHARGE_COLS if c in df.columns])

    # 3. Feature engineering
    df = create_total_minutes(df)
    df = create_total_calls(df)
    df = create_avg_minutes_per_call(df)
    df = create_calls_per_minute(df)
    df = create_high_service_calls(df)
    df = create_intl_active_no_plan(df)

    X = df.drop(columns=[target])
    y = df[target]
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[numeric_cols]

    # 4. Stratified split BEFORE scaling — prevents data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 5. Fit scaler on training data only
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=numeric_cols, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=numeric_cols, index=X_test.index
    )

    # 6. SMOTE on training split only
    if apply_smote:
        smote = SMOTE(random_state=random_state)
        X_smote, y_smote = smote.fit_resample(X_train_scaled, y_train)
        smote_out = pd.DataFrame(X_smote, columns=numeric_cols)
        smote_out[target] = y_smote
        Path(smote_output_path).parent.mkdir(parents=True, exist_ok=True)
        smote_out.to_csv(smote_output_path, index=False)
        print(f"SMOTE train saved : {smote_output_path}  shape={smote_out.shape}")

    # 7. Save scaler and split datasets
    Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)

    Path(train_output_path).parent.mkdir(parents=True, exist_ok=True)
    train_out = X_train_scaled.copy()
    train_out[target] = y_train.values
    train_out.to_csv(train_output_path, index=False)

    test_out = X_test_scaled.copy()
    test_out[target] = y_test.values
    test_out.to_csv(test_output_path, index=False)

    print(f"Train saved       : {train_output_path}  shape={train_out.shape}")
    print(f"Test saved        : {test_output_path}  shape={test_out.shape}")
    print(f"Scaler saved      : {scaler_path}")
    return X_train_scaled, X_test_scaled, y_train, y_test


if __name__ == "__main__":
    run_feature_engineering()
    print("Feature engineering complete.")
