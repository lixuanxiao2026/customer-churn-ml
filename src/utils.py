"""
Utility module for customer churn pipeline.
Logging, save/load model, data validation helpers.
"""

import logging
import joblib
from pathlib import Path
import pandas as pd


def setup_logging(level=logging.INFO, log_file: str = None):
    """Configure logging for the pipeline."""
    handlers = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )
    return logging.getLogger("churn_pipeline")


def save_model(model, path: str):
    """Save model using joblib."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str):
    """Load model from joblib."""
    return joblib.load(path)


def validate_dataframe(df: pd.DataFrame, required_cols: list = None) -> bool:
    """Validate DataFrame has required columns and no critical issues."""
    if df is None or df.empty:
        return False
    if required_cols:
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    return True


def get_project_root() -> Path:
    """Return project root (parent of src/)."""
    return Path(__file__).resolve().parent.parent
