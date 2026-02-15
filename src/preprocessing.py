"""
Preprocessing module for customer churn pipeline.
Handles missing values, type validation, encoding, and scaling.
No plotting code.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_raw_data(data_path: str = "data/raw/churn-bigml-20_raw.csv") -> pd.DataFrame:
    """Load raw churn dataset."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found: {path}")
    return pd.read_csv(path)


def validate_types(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and enforce expected dtypes. Returns validated DataFrame."""
    df = df.copy()
    numeric_cols = [
        "Account length", "Area code", "Number vmail messages",
        "Total day minutes", "Total day calls", "Total day charge",
        "Total eve minutes", "Total eve calls", "Total eve charge",
        "Total night minutes", "Total night calls", "Total night charge",
        "Total intl minutes", "Total intl calls", "Total intl charge",
        "Customer service calls"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def handle_missing_values(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """Handle missing values. strategy: 'median', 'mean', 'drop'."""
    df = df.copy()
    if df.isnull().sum().sum() == 0:
        return df
    numeric = df.select_dtypes(include=[np.number])
    categorical = df.select_dtypes(include=["object", "bool"])
    for col in numeric.columns:
        if df[col].isnull().any():
            if strategy == "median":
                df[col] = df[col].fillna(df[col].median())
            elif strategy == "mean":
                df[col] = df[col].fillna(df[col].mean())
    for col in categorical.columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "Unknown")
    if strategy == "drop":
        df = df.dropna()
    return df


def encode_categorical(df: pd.DataFrame, target: str = "Churn") -> pd.DataFrame:
    """Encode categorical columns. Binary Yes/No -> 0/1, State -> one-hot."""
    df = df.copy()
    if "International plan" in df.columns:
        df["International plan"] = (df["International plan"] == "Yes").astype(int)
    if "Voice mail plan" in df.columns:
        df["Voice mail plan"] = (df["Voice mail plan"] == "Yes").astype(int)
    if "State" in df.columns:
        df = pd.get_dummies(df, columns=["State"], drop_first=True)
    if target in df.columns and df[target].dtype == bool:
        df[target] = df[target].astype(int)
    return df


def scale_features(df: pd.DataFrame, target: str = "Churn", fit: bool = True, scaler=None):
    """Scale numeric features. Returns (scaled_df, scaler)."""
    if target in df.columns:
        y = df[target]
        X = df.drop(columns=[target])
    else:
        y = None
        X = df.copy()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X_numeric = X[numeric_cols]
    if scaler is None:
        scaler = StandardScaler()
    if fit:
        X_scaled = scaler.fit_transform(X_numeric)
    else:
        X_scaled = scaler.transform(X_numeric)
    X_out = pd.DataFrame(X_scaled, columns=numeric_cols, index=X.index)
    if y is not None:
        X_out[target] = y.values
    return X_out, scaler


def run_preprocessing_pipeline(
    input_path: str = "data/raw/churn-bigml-20_raw.csv",
    output_path: str = "data/processed/churn_clean.csv",
) -> pd.DataFrame:
    """Full preprocessing pipeline: load, validate, handle missing, encode."""
    df = load_raw_data(input_path)
    df = validate_types(df)
    df = handle_missing_values(df)
    df = encode_categorical(df)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    run_preprocessing_pipeline()
    print("Preprocessing complete. Output: data/processed/churn_clean.csv")
