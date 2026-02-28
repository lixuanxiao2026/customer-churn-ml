"""
Modeling module for customer churn pipeline.
Model training, pipeline definition, hyperparameter tuning.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib


def load_features(path: str = "data/processed/churn_features.csv") -> tuple:
    """Load feature-engineered data. Returns (X, y)."""
    df = pd.read_csv(path)
    target = "Churn"
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not in DataFrame")
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def train_test_split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """Stratified train/test split."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def get_models() -> dict:
    """Return dict of model name -> model instance."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
        "Neural_Network_MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(
                hidden_layer_sizes=(32, 16),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                learning_rate_init=1e-3,
                max_iter=300,
                random_state=42
            ))
        ]),
    }


def train_models(X_train, y_train, models: dict = None) -> dict:
    """Train models and return {name: {'model': model, 'cv_auc_mean': float, ...}}."""
    if models is None:
        models = get_models()
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
        results[name] = {
            "model": model,
            "cv_auc_mean": cv_scores.mean(),
            "cv_auc_std": cv_scores.std(),
        }
    return results


def tune_random_forest(X_train, y_train, param_grid: dict = None):
    """Grid search for Random Forest. Returns best model and cv score."""
    if param_grid is None:
        param_grid = {"n_estimators": [50, 100, 200], "max_depth": [5, 10, None]}
    rf = RandomForestClassifier(random_state=42)
    gs = GridSearchCV(rf, param_grid, cv=5, scoring="roc_auc")
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_score_


def save_models(results: dict, output_dir: str = "models"):
    """Save trained models to disk using joblib."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for name, data in results.items():
        fname = name.lower().replace(" ", "_") + ".pkl"
        joblib.dump(data["model"], Path(output_dir) / fname)
    print(f"Models saved to {output_dir}/")


def run_modeling_pipeline(
    features_path: str = "data/processed/churn_features.csv",
    models_dir: str = "models",
) -> dict:
    """Full modeling pipeline: load, split, train, save."""
    X, y = load_features(features_path)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    results = train_models(X_train, y_train)
    save_models(results, models_dir)
    return results


if __name__ == "__main__":
    run_modeling_pipeline()
    print("Modeling complete. Models saved to models/")
