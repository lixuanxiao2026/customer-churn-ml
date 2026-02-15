"""
Feature engineering module for customer churn pipeline.
Feature creation, interaction features, aggregation logic, business-derived features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Import preprocessing for pipeline consistency
from .preprocessing import load_raw_data, validate_types, handle_missing_values, encode_categorical


def create_total_minutes(df: pd.DataFrame) -> pd.DataFrame:
    """Business-derived: total usage minutes across day/eve/night."""
    df = df.copy()
    cols = ["Total day minutes", "Total eve minutes", "Total night minutes"]
    if all(c in df.columns for c in cols):
        df["total_minutes"] = df[cols].sum(axis=1)
    return df


def create_total_charge(df: pd.DataFrame) -> pd.DataFrame:
    """Business-derived: total charge across day/eve/night."""
    df = df.copy()
    cols = ["Total day charge", "Total eve charge", "Total night charge"]
    if all(c in df.columns for c in cols):
        df["total_charge"] = df[cols].sum(axis=1)
    return df


def create_calls_per_minute(df: pd.DataFrame) -> pd.DataFrame:
    """Interaction: calls per minute (usage intensity)."""
    df = df.copy()
    if "total_minutes" not in df.columns:
        df = create_total_minutes(df)
    call_cols = ["Total day calls", "Total eve calls", "Total night calls"]
    if all(c in df.columns for c in call_cols):
        total_calls = df[call_cols].sum(axis=1)
        df["calls_per_minute"] = np.where(df["total_minutes"] > 0, total_calls / df["total_minutes"], 0)
    return df


def run_feature_engineering(
    input_path: str = "data/processed/churn_clean.csv",
    output_path: str = "data/processed/churn_features.csv",
    use_raw_fallback: bool = True,
) -> pd.DataFrame:
    """
    Full feature engineering: load cleaned data, create features, scale.
    If churn_clean.csv missing and use_raw_fallback, runs preprocessing first.
    """
    path = Path(input_path)
    if path.exists():
        df = pd.read_csv(path)
    elif use_raw_fallback:
        from .preprocessing import run_preprocessing_pipeline
        run_preprocessing_pipeline(
            input_path="data/raw/churn-bigml-20_raw.csv",
            output_path="data/processed/churn_clean.csv",
        )
        df = pd.read_csv("data/processed/churn_clean.csv")
    else:
        raise FileNotFoundError(f"Input not found: {input_path}")

    target = "Churn"
    if target not in df.columns:
        raise ValueError("Target 'Churn' not in DataFrame")

    df = create_total_minutes(df)
    df = create_total_charge(df)
    df = create_calls_per_minute(df)

    X = df.drop(columns=[target])
    y = df[target]
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X_numeric = X[numeric_cols]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_numeric), columns=numeric_cols, index=X.index)
    X_scaled[target] = y.values

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    X_scaled.to_csv(output_path, index=False)
    return X_scaled


if __name__ == "__main__":
    run_feature_engineering()
    print("Feature engineering complete. Output: data/processed/churn_features.csv")
