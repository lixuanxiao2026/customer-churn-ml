# Processed Data

This folder contains outputs from the preprocessing and feature engineering pipeline:

- **churn_clean.csv** — Cleaned dataset (missing values handled, types validated)
- **churn_features.csv** — Feature-engineered dataset (encoded, scaled, ready for modeling)

**Generation:** Run `02_feature_engineering.ipynb` or `src/feature_engineering.py` to produce these files.

**Requirements:**
- Reproducible (not manually edited)
- Output of preprocessing pipeline only
