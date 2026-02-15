# Customer Churn Prediction — Machine Learning Pipeline

## Project Title
**Predictive Customer Churn Modeling for Telecommunications Retention Strategy**

---

## Business Problem Statement

Customer churn—the loss of subscribers to competitors—represents a critical revenue risk for telecommunications companies. Acquiring new customers costs significantly more than retaining existing ones, and churn directly impacts profitability, market share, and long-term growth. Without proactive identification of at-risk customers, retention efforts are reactive and inefficient.

This project addresses the business need to **predict which customers are likely to churn** before they leave, enabling targeted retention campaigns, personalized offers, and resource allocation to high-value at-risk accounts. By building a deployable machine learning pipeline, we support executive-level decisions on retention strategy, budget allocation, and customer experience improvements.

---

## Dataset Description

**Source:** BigML Churn Dataset (Churn Modeling Benchmark)

**File:** `data/raw/churn-bigml-20_raw.csv`

**Features:**
- **State** — Customer state (categorical)
- **Account length** — Months with account
- **Area code** — Phone area code
- **International plan** — Yes/No
- **Voice mail plan** — Yes/No
- **Number vmail messages** — Voicemail count
- **Total day/eve/night minutes** — Call usage by time period
- **Total day/eve/night calls** — Call count by time period
- **Total day/eve/night charge** — Billing by time period
- **Total intl minutes/calls/charge** — International usage
- **Customer service calls** — Support contact count

**Target:** `Churn` (True/False) — Whether the customer churned

**Size:** ~3,333 rows, 20 columns

---

## Objectives

1. **Build predictive models** to identify high-risk churn customers
2. **Compare multiple ML algorithms** (Logistic Regression, Random Forest, XGBoost, etc.)
3. **Design a deployable, automated pipeline** for preprocessing, training, and evaluation
4. **Translate model outputs** into actionable retention strategy recommendations

---

## CRISP-DM Alignment

| Phase | Deliverable |
|-------|-------------|
| **1. Business Understanding** | Problem definition, objectives, success criteria |
| **2. Data Understanding** | EDA notebook, data quality report |
| **3. Data Preparation** | Preprocessing pipeline, feature engineering |
| **4. Modeling** | Multiple models, hyperparameter tuning, cross-validation |
| **5. Evaluation** | Metric comparison, ROC/AUC, business interpretation |
| **6. Deployment** | Modular `src/` pipeline, saved models, documentation |

---

## Model List

- Logistic Regression
- Random Forest
- XGBoost
- (Optional) Gradient Boosting / Neural Network

---

## Evaluation Metrics

- **Accuracy** — Overall correctness
- **Precision** — True positives among predicted churners
- **Recall** — Churners correctly identified
- **F1 Score** — Balance of precision and recall
- **ROC-AUC** — Ranking quality across thresholds
- **Business interpretation** — Cost-benefit of retention actions

---

## Deployment Overview

The pipeline is modular and deployable:

- **`src/preprocessing.py`** — Missing value handling, type validation, encoding
- **`src/feature_engineering.py`** — Feature creation, scaling
- **`src/modeling.py`** — Model training, pipelines, tuning
- **`src/evaluation.py`** — Metrics, confusion matrix, ROC curve
- **`models/`** — Saved models (joblib) for inference

---

## Team Members

- **Lixuan Xiao**
- **Michael Dawson**
- **Yang Yu**

*Final Project – G5*

---

## Instructions to Run Project

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/customer-churn-ml.git
cd customer-churn-ml
```

### 2. Create virtual environment (recommended)
```bash
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Place raw data
Ensure `data/raw/churn-bigml-20_raw.csv` exists. If not, download from [BigML](https://www.kaggle.com/datasets/mnassar/telco-customer-churn) or place your dataset in `data/raw/`.

### 5. Run notebooks (in order)
```bash
jupyter notebook notebooks/
```
Execute in sequence:
- `01_eda.ipynb` — Exploratory data analysis
- `02_feature_engineering.ipynb` — Preprocessing and feature creation
- `03_modeling.ipynb` — Model training and comparison
- `04_evaluation.ipynb` — Evaluation and business interpretation

### 6. Run pipeline from `src/` (optional)
```bash
python -m src.preprocessing
python -m src.feature_engineering
python -m src.modeling
python -m src.evaluation
```

---

## Repository Structure

```
customer-churn-ml/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── data/
│   ├── raw/           # Original dataset
│   └── processed/     # Cleaned, feature-engineered data
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   ├── evaluation.py
│   └── utils.py
├── reports/
├── slides/
├── dashboards/
├── models/
└── diagrams/
```
