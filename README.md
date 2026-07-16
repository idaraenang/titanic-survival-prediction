# Titanic Survival Prediction

A machine learning pipeline that predicts passenger survival on the Titanic using a Random Forest classifier.

## Overview

This project builds an end-to-end ML pipeline on the classic Titanic dataset, covering data preprocessing, feature engineering, model training, and evaluation.

## Dataset

- **Source:** [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic)
- **File:** `train.csv`
- **Target:** `Survived` (0 = No, 1 = Yes)

## Features Used

| Feature | Description |
|---|---|
| `Pclass` | Passenger class (1, 2, 3) |
| `Sex` | Gender |
| `Age` | Age in years |
| `SibSp` | Number of siblings/spouses aboard |
| `Parch` | Number of parents/children aboard |
| `Fare` | Ticket fare |

## Pipeline Steps

1. **Load Data** — Read CSV with pandas
2. **Feature Selection** — Select relevant columns, drop unused features
3. **Encoding** — One-hot encode `Sex` using `get_dummies`
4. **Imputation** — Fill missing `Age` and `Fare` values with median
5. **Scaling** — Standardize `Age` and `Fare` with `StandardScaler`
6. **Train/Test Split** — 80/20 split with `random_state=42`
7. **Model Training** — `RandomForestClassifier` (100 estimators, max depth 5)
8. **Evaluation** — Accuracy and F1 Score on validation set

## Model

```python
RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
```

## Evaluation Metrics

- **Accuracy** — Overall proportion of correct predictions
- **F1 Score** — Harmonic mean of precision and recall

## Requirements

```
pandas
scikit-learn
```

## Usage

```bash
# Clone the repo
git clone https://github.com/your-username/titanic-survival-prediction.git
cd titanic-survival-prediction

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook Titanic.ipynb
```
