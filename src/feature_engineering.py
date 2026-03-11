import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

try:
    from .preprocessing import run_preprocessing_pipeline
except ImportError:
    from preprocessing import run_preprocessing_pipeline


def create_total_minutes(df: pd.DataFrame) -> pd.DataFrame:
    """Business-derived: total usage minutes across day/eve/night/intl."""
    df = df.copy()
    cols = [
        "Total day minutes", "Total eve minutes",
        "Total night minutes", "Total intl minutes",
    ]
    if all(c in df.columns for c in cols):
        df["total_minutes"] = df[cols].sum(axis=1)
    return df


def create_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Business-derived: total charges across day/eve/night/intl."""
    df = df.copy()
    cols = [
        "Total day charge", "Total eve charge",
        "Total night charge", "Total intl charge",
    ]
    if all(c in df.columns for c in cols):
        df["total_charges"] = df[cols].sum(axis=1)
    return df


def create_high_service_calls(df: pd.DataFrame) -> pd.DataFrame:
    """Binary flag: customer service calls >= 3."""
    df = df.copy()
    if "Customer service calls" in df.columns:
        df["high_service_calls"] = (df["Customer service calls"] >= 3).astype(int)
    return df


def create_intl_usage_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Ratio: international minutes / total minutes."""
    df = df.copy()
    if "total_minutes" not in df.columns:
        df = create_total_minutes(df)
    if "Total intl minutes" in df.columns:
        df["intl_usage_ratio"] = (
            df["Total intl minutes"] / df["total_minutes"]
        ).fillna(0)
    return df


def create_account_length_group(df: pd.DataFrame) -> pd.DataFrame:
    """Bin account length into new / mid / long groups, then one-hot encode."""
    df = df.copy()
    if "Account length" in df.columns:
        df["account_length_group"] = pd.cut(
            df["Account length"],
            bins=[0, 50, 150, df["Account length"].max()],
            labels=["new", "mid", "long"],
        )
        df = pd.get_dummies(df, columns=["account_length_group"], drop_first=True)
    return df


def run_feature_engineering(
    input_path: str = "data/processed/churn_clean.csv",
    train_output_path: str = "data/processed/churn_train.csv",
    test_output_path: str = "data/processed/churn_test.csv",
    scaler_path: str = "models/scaler.pkl",
    use_raw_fallback: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Full feature engineering pipeline.

    Features created:
        - total_minutes             : sum of day/eve/night/intl minutes
        - total_charges             : sum of day/eve/night/intl charges
        - high_service_calls        : 1 if customer_service_calls >= 3
        - intl_usage_ratio          : intl minutes / total minutes
        - account_length_group_mid  : one-hot (account length 51-150)
        - account_length_group_long : one-hot (account length > 150)

    Order: split → scale (fit on train only) → SMOTE (train only)

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
        raise ValueError("Target 'Churn' not in DataFrame")

    # 2. Create features
    df = create_total_minutes(df)
    df = create_total_charges(df)
    df = create_high_service_calls(df)
    df = create_intl_usage_ratio(df)
    df = create_account_length_group(df)

    # 3. Select only numeric columns for modelling
    X = df.drop(columns=[target])
    y = df[target]
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[numeric_cols]

    # 4. Split BEFORE scaling — no leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 5. Scale — fit on X_train only
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=numeric_cols
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=numeric_cols
    )

    # 6. SMOTE — after scaling, on train only
    try:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=random_state)
        X_train_scaled, y_train = sm.fit_resample(X_train_scaled, y_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=numeric_cols)
        print(f"SMOTE applied — train distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    except ImportError:
        print("Warning: imbalanced-learn not installed, skipping SMOTE.")
        print("Install with: pip install imbalanced-learn")

    # 7. Save outputs
    Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)

    Path(train_output_path).parent.mkdir(parents=True, exist_ok=True)
    train_out = X_train_scaled.copy()
    train_out[target] = y_train.values if hasattr(y_train, "values") else y_train
    train_out.to_csv(train_output_path, index=False)

    test_out = X_test_scaled.copy()
    test_out[target] = y_test.values
    test_out.to_csv(test_output_path, index=False)

    print(f"Train saved: {train_output_path} | Test saved: {test_output_path}")
    print(f"Scaler saved: {scaler_path}")
    return X_train_scaled, X_test_scaled, y_train, y_test


if __name__ == "__main__":
    run_feature_engineering()
    print("Feature engineering complete.")
