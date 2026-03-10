"""
Evaluation module for customer churn pipeline.
Metric calculation, confusion matrix, ROC curve, model comparison table.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def load_models(models_dir: str = "models") -> dict:
    """Load saved models. Returns {name: model}."""
    models = {}
    path = Path(models_dir)
    if not path.exists():
        return models
    for p in path.glob("*.pkl"):
        name = p.stem.replace("_", " ").title()
        models[name] = joblib.load(p)
    return models


def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Compute accuracy, precision, recall, F1, AUC."""
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        metrics["AUC"] = roc_auc_score(y_true, y_proba)
    return metrics


def get_confusion_matrix(y_true, y_pred) -> np.ndarray:
    """Return confusion matrix."""
    return confusion_matrix(y_true, y_pred)


def get_roc_curve(y_true, y_proba) -> tuple:
    """Return (fpr, tpr, auc) for ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


def model_comparison_table(models: dict, X_test, y_test) -> pd.DataFrame:
    """Generate comparison table for all models."""
    rows = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_proba)
        metrics["Model"] = name
        rows.append(metrics)
    df = pd.DataFrame(rows)
    cols = ["Model", "Accuracy", "Precision", "Recall", "F1", "AUC"]
    return df[[c for c in cols if c in df.columns]]

def run_evaluation(
    models_dir: str = "models",
    test_path: str = "data/processed/churn_test.csv",   
    output_path: str = "reports/model_comparison_table.csv",
) -> pd.DataFrame:
    """Full evaluation: load models, load test data, compute metrics, save table."""
    models = load_models(models_dir)
    if not models:
        print("No models found. Run modeling pipeline first.")
        return pd.DataFrame()

    df = pd.read_csv(test_path)
    X_test = df.drop(columns=["Churn"])
    y_test = df["Churn"]

    table = model_comparison_table(models, X_test, y_test)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    print(f"Comparison table saved to {output_path}")
    return table
    


if __name__ == "__main__":
    run_evaluation()
    print("Evaluation complete.")
