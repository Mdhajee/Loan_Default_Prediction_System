"""Generate a complete Loan Default Prediction System project.

The workspace did not include a CSV, so this script creates a reproducible
loan dataset with the requested columns, builds a notebook, and writes a short
summary/report. The notebook itself contains the full preprocessing, EDA,
modeling, and evaluation workflow.
"""

from __future__ import annotations

import json
import math
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "loan_data.csv"
NOTEBOOK_PATH = PROJECT_ROOT / "Loan_Default_Prediction_System.ipynb"
SUMMARY_PATH = PROJECT_ROOT / "loan_default_summary.md"
HTML_REPORT_PATH = PROJECT_ROOT / "loan_default_report.html"
README_PATH = PROJECT_ROOT / "README.md"
REQUIREMENTS_PATH = PROJECT_ROOT / "requirements.txt"


def make_loan_dataset(n_rows: int = 614, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Create a realistic synthetic loan dataset matching the assignment schema."""
    rng = np.random.default_rng(seed)

    gender = rng.choice(["Male", "Female"], size=n_rows, p=[0.78, 0.22])
    married = rng.choice(["Yes", "No"], size=n_rows, p=[0.64, 0.36])
    dependents = rng.choice(["0", "1", "2", "3+"], size=n_rows, p=[0.56, 0.17, 0.17, 0.10])
    education = rng.choice(["Graduate", "Not Graduate"], size=n_rows, p=[0.78, 0.22])
    self_employed = rng.choice(["No", "Yes"], size=n_rows, p=[0.86, 0.14])
    property_area = rng.choice(["Urban", "Semiurban", "Rural"], size=n_rows, p=[0.33, 0.38, 0.29])
    loan_term = rng.choice([360, 180, 300, 480, 120, 240, 84, 60], size=n_rows,
                           p=[0.83, 0.07, 0.04, 0.03, 0.01, 0.01, 0.005, 0.005])

    # Income is right-skewed like real application income data.
    applicant_income = rng.lognormal(mean=8.55, sigma=0.55, size=n_rows).round(0).astype(int)
    applicant_income = np.clip(applicant_income, 1400, 26000)

    has_coapplicant = rng.random(n_rows) < 0.62
    coapplicant_income = np.where(
        has_coapplicant,
        rng.lognormal(mean=7.65, sigma=0.70, size=n_rows),
        0,
    )
    coapplicant_income = np.clip(coapplicant_income, 0, 14000).round(0).astype(int)

    base_loan = (
        0.021 * applicant_income
        + 0.015 * coapplicant_income
        + rng.normal(35, 32, size=n_rows)
    )
    loan_amount = np.clip(base_loan, 25, 650).round(0).astype(int)

    credit_history = rng.choice([1.0, 0.0], size=n_rows, p=[0.82, 0.18])
    debt_to_income = loan_amount / np.maximum((applicant_income + coapplicant_income) / 1000, 1)

    # Generate the target with a probability model that rewards repayment signals.
    score = (
        -0.50
        + 2.15 * credit_history
        + 0.45 * (property_area == "Semiurban")
        + 0.20 * (property_area == "Urban")
        + 0.28 * (education == "Graduate")
        + 0.18 * (married == "Yes")
        - 0.12 * (dependents == "3+")
        - 0.18 * (self_employed == "Yes")
        - 0.035 * debt_to_income
        + 0.000018 * (applicant_income + coapplicant_income)
        + rng.normal(0, 0.35, size=n_rows)
    )
    approval_probability = 1 / (1 + np.exp(-score))
    loan_status = np.where(rng.random(n_rows) < approval_probability, "Y", "N")

    df = pd.DataFrame(
        {
            "Loan_ID": [f"LP{100001 + i}" for i in range(n_rows)],
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_employed,
            "ApplicantIncome": applicant_income,
            "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": loan_term,
            "Credit_History": credit_history,
            "Property_Area": property_area,
            "Loan_Status": loan_status,
        }
    )

    # Add realistic missingness so the preprocessing phase has work to do.
    missing_rates = {
        "Gender": 0.02,
        "Married": 0.02,
        "Dependents": 0.03,
        "Self_Employed": 0.05,
        "LoanAmount": 0.04,
        "Loan_Amount_Term": 0.03,
        "Credit_History": 0.08,
    }
    for column, rate in missing_rates.items():
        missing_mask = rng.random(n_rows) < rate
        df.loc[missing_mask, column] = np.nan

    return df


def mode_value(series: pd.Series):
    mode = series.mode(dropna=True)
    return mode.iloc[0] if len(mode) else None


def prepare_features(df: pd.DataFrame):
    """Impute, encode, and scale features for classification."""
    cleaned = df.drop(columns=["Loan_ID"]).copy()
    y = cleaned["Loan_Status"].map({"N": 0, "Y": 1}).astype(int)
    X = cleaned.drop(columns=["Loan_Status"])

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [column for column in X.columns if column not in numeric_cols]

    for column in numeric_cols:
        X[column] = X[column].fillna(X[column].median())
    for column in categorical_cols:
        X[column] = X[column].fillna(mode_value(X[column]))

    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dtype=float)

    # Standardize numeric columns; one-hot columns stay as 0/1 indicators.
    scaler = {}
    for column in numeric_cols:
        mean = X_encoded[column].mean()
        std = X_encoded[column].std(ddof=0)
        if std == 0:
            std = 1
        X_encoded[column] = (X_encoded[column] - mean) / std
        scaler[column] = {"mean": float(mean), "std": float(std)}

    return X_encoded.astype(float), y.astype(int), cleaned, scaler


def stratified_train_test_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.20, seed: int = RANDOM_SEED):
    rng = np.random.default_rng(seed)
    train_indices = []
    test_indices = []

    for class_value in sorted(y.unique()):
        class_indices = y[y == class_value].index.to_numpy().copy()
        rng.shuffle(class_indices)
        n_test = int(round(len(class_indices) * test_size))
        test_indices.extend(class_indices[:n_test])
        train_indices.extend(class_indices[n_test:])

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    return (
        X.loc[train_indices].to_numpy(dtype=float),
        X.loc[test_indices].to_numpy(dtype=float),
        y.loc[train_indices].to_numpy(dtype=int),
        y.loc[test_indices].to_numpy(dtype=int),
        X.columns.tolist(),
    )


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -35, 35)))


def train_logistic_regression(X_train, y_train, learning_rate=0.08, epochs=2500):
    X_bias = np.c_[np.ones(X_train.shape[0]), X_train]
    weights = np.zeros(X_bias.shape[1])

    for _ in range(epochs):
        predictions = sigmoid(X_bias @ weights)
        gradient = X_bias.T @ (predictions - y_train) / len(y_train)
        weights -= learning_rate * gradient

    return weights


def predict_logistic_regression(weights, X):
    X_bias = np.c_[np.ones(X.shape[0]), X]
    probabilities = sigmoid(X_bias @ weights)
    return (probabilities >= 0.5).astype(int), probabilities


def gini_impurity(y):
    if len(y) == 0:
        return 0
    p = np.mean(y)
    return 1 - p ** 2 - (1 - p) ** 2


class DecisionTreeClassifierScratch:
    def __init__(self, max_depth=4, min_samples_split=18, max_features=None, seed=RANDOM_SEED):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.rng = np.random.default_rng(seed)
        self.tree_ = None

    def fit(self, X, y):
        self.n_features_ = X.shape[1]
        self.tree_ = self._build_tree(X, y, depth=0)
        return self

    def _candidate_features(self):
        if self.max_features is None:
            return np.arange(self.n_features_)
        count = min(self.max_features, self.n_features_)
        return self.rng.choice(self.n_features_, size=count, replace=False)

    def _best_split(self, X, y):
        parent_impurity = gini_impurity(y)
        best_gain = 0
        best_feature = None
        best_threshold = None

        for feature in self._candidate_features():
            values = np.unique(X[:, feature])
            if len(values) <= 1:
                continue
            thresholds = (values[:-1] + values[1:]) / 2

            # Limit very dense numeric columns to keep the notebook fast.
            if len(thresholds) > 25:
                quantiles = np.linspace(0.05, 0.95, 25)
                thresholds = np.unique(np.quantile(values, quantiles))

            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                weighted_impurity = (
                    left_mask.mean() * gini_impurity(y[left_mask])
                    + right_mask.mean() * gini_impurity(y[right_mask])
                )
                gain = parent_impurity - weighted_impurity
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X, y, depth):
        positive_rate = float(np.mean(y))
        prediction = int(positive_rate >= 0.5)

        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or len(np.unique(y)) == 1
        ):
            return {"prediction": prediction, "positive_rate": positive_rate}

        feature, threshold, gain = self._best_split(X, y)
        if feature is None or gain <= 1e-7:
            return {"prediction": prediction, "positive_rate": positive_rate}

        left_mask = X[:, feature] <= threshold
        return {
            "feature": int(feature),
            "threshold": float(threshold),
            "prediction": prediction,
            "positive_rate": positive_rate,
            "left": self._build_tree(X[left_mask], y[left_mask], depth + 1),
            "right": self._build_tree(X[~left_mask], y[~left_mask], depth + 1),
        }

    def _predict_one(self, row, node):
        while "feature" in node:
            if row[node["feature"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        return node["prediction"], node["positive_rate"]

    def predict(self, X):
        predictions = [self._predict_one(row, self.tree_)[0] for row in X]
        return np.array(predictions, dtype=int)

    def predict_proba(self, X):
        probabilities = [self._predict_one(row, self.tree_)[1] for row in X]
        return np.array(probabilities, dtype=float)


class RandomForestClassifierScratch:
    def __init__(self, n_estimators=35, max_depth=5, min_samples_split=16, seed=RANDOM_SEED):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.rng = np.random.default_rng(seed)
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        max_features = max(1, int(math.sqrt(X.shape[1])))
        for i in range(self.n_estimators):
            sample_indices = self.rng.choice(len(X), size=len(X), replace=True)
            tree = DecisionTreeClassifierScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_features,
                seed=RANDOM_SEED + i + 1,
            )
            tree.fit(X[sample_indices], y[sample_indices])
            self.trees.append(tree)
        return self

    def predict_proba(self, X):
        probabilities = np.vstack([tree.predict_proba(X) for tree in self.trees])
        return probabilities.mean(axis=0)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


def classification_metrics(y_true, y_pred):
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def run_modeling(df: pd.DataFrame):
    X, y, cleaned, scaler = prepare_features(df)
    X_train, X_test, y_train, y_test, feature_names = stratified_train_test_split(X, y)

    logistic_weights = train_logistic_regression(X_train, y_train)
    logistic_pred, _ = predict_logistic_regression(logistic_weights, X_test)

    decision_tree = DecisionTreeClassifierScratch(max_depth=4, min_samples_split=18)
    decision_tree.fit(X_train, y_train)
    tree_pred = decision_tree.predict(X_test)

    random_forest = RandomForestClassifierScratch(n_estimators=35, max_depth=5, min_samples_split=16)
    random_forest.fit(X_train, y_train)
    forest_pred = random_forest.predict(X_test)

    results = pd.DataFrame(
        [
            {"Model": "Logistic Regression", **classification_metrics(y_test, logistic_pred)},
            {"Model": "Decision Tree", **classification_metrics(y_test, tree_pred)},
            {"Model": "Random Forest", **classification_metrics(y_test, forest_pred)},
        ]
    ).sort_values(["F1-Score", "Accuracy"], ascending=False)

    return {
        "X": X,
        "y": y,
        "cleaned": cleaned,
        "scaler": scaler,
        "feature_names": feature_names,
        "results": results,
        "best_model": results.iloc[0].to_dict(),
        "test_size": len(y_test),
        "train_size": len(y_train),
        "target_rate": float(y.mean()),
    }


def svg_bar(title, labels, values, width=720, height=330, color="#2f6f73"):
    max_value = max(values) if values else 1
    left = 70
    bottom = 55
    top = 45
    plot_width = width - left - 30
    plot_height = height - top - bottom
    gap = 16
    bar_width = (plot_width - gap * (len(values) - 1)) / max(len(values), 1)
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="{escape(title)}">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{width/2}" y="24" text-anchor="middle" font-size="18" font-family="Arial" fill="#1f2933">{escape(title)}</text>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-20}" y2="{height-bottom}" stroke="#9aa5b1"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#9aa5b1"/>',
    ]
    for i, (label, value) in enumerate(zip(labels, values)):
        x = left + i * (bar_width + gap)
        bar_height = (value / max_value) * plot_height if max_value else 0
        y = height - bottom - bar_height
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="{color}" rx="3"/>')
        parts.append(f'<text x="{x + bar_width/2:.1f}" y="{y - 7:.1f}" text-anchor="middle" font-size="12" font-family="Arial" fill="#1f2933">{value:.2f}</text>')
        parts.append(f'<text x="{x + bar_width/2:.1f}" y="{height - 28}" text-anchor="middle" font-size="12" font-family="Arial" fill="#1f2933">{escape(str(label))}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def svg_histogram(title, values, bins=12, width=720, height=330, color="#4d7cbd"):
    counts, edges = np.histogram(pd.Series(values).dropna(), bins=bins)
    labels = [f"{int(edges[i])}-{int(edges[i+1])}" for i in range(len(edges) - 1)]
    return svg_bar(title, labels, counts.astype(float).tolist(), width, height, color)


def svg_heatmap(title, corr: pd.DataFrame, width=760, height=560):
    labels = list(corr.columns)
    cell_size = min((width - 180) / len(labels), (height - 130) / len(labels))
    left = 140
    top = 55
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="{escape(title)}">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{width/2}" y="24" text-anchor="middle" font-size="18" font-family="Arial" fill="#1f2933">{escape(title)}</text>',
    ]
    for i, row_label in enumerate(labels):
        parts.append(f'<text x="{left - 8}" y="{top + i*cell_size + cell_size*0.62:.1f}" text-anchor="end" font-size="11" font-family="Arial" fill="#1f2933">{escape(row_label)}</text>')
        parts.append(f'<text x="{left + i*cell_size + cell_size/2:.1f}" y="{top + len(labels)*cell_size + 18:.1f}" text-anchor="middle" font-size="11" font-family="Arial" fill="#1f2933">{escape(row_label)}</text>')
        for j, col_label in enumerate(labels):
            value = float(corr.iloc[i, j])
            if value >= 0:
                intensity = int(235 - 95 * min(value, 1))
                fill = f"rgb({intensity},{intensity + 10},255)"
            else:
                intensity = int(235 - 95 * min(abs(value), 1))
                fill = f"rgb(255,{intensity + 10},{intensity})"
            x = left + j * cell_size
            y = top + i * cell_size
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_size:.1f}" height="{cell_size:.1f}" fill="{fill}" stroke="#ffffff"/>')
            parts.append(f'<text x="{x + cell_size/2:.1f}" y="{y + cell_size*0.58:.1f}" text-anchor="middle" font-size="10" font-family="Arial" fill="#1f2933">{value:.2f}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def make_report_html(df: pd.DataFrame, modeling: dict) -> str:
    results = modeling["results"].copy()
    results_for_html = results[["Model", "Accuracy", "Precision", "Recall", "F1-Score", "TN", "FP", "FN", "TP"]].copy()
    for column in ["Accuracy", "Precision", "Recall", "F1-Score"]:
        results_for_html[column] = results_for_html[column].map(lambda value: f"{value:.3f}")

    numeric_corr_df = df.drop(columns=["Loan_ID"]).copy()
    numeric_corr_df["Loan_Status_Num"] = numeric_corr_df["Loan_Status"].map({"N": 0, "Y": 1})
    corr = numeric_corr_df.select_dtypes(include=[np.number]).corr(numeric_only=True).round(2)

    status_counts = df["Loan_Status"].value_counts().reindex(["N", "Y"]).fillna(0)
    approval_by_credit = (
        df.assign(Credit_History=df["Credit_History"].fillna(-1).replace({-1: "Missing"}))
        .groupby("Credit_History")["Loan_Status"]
        .apply(lambda series: (series == "Y").mean())
        .reset_index()
    )
    approval_by_property = (
        df.groupby("Property_Area")["Loan_Status"]
        .apply(lambda series: (series == "Y").mean())
        .sort_values(ascending=False)
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Loan Default Prediction System Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 36px; color: #1f2933; line-height: 1.5; }}
    h1, h2 {{ color: #17324d; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; }}
    th, td {{ border: 1px solid #cbd5df; padding: 8px 10px; text-align: left; }}
    th {{ background: #edf2f7; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 22px; }}
    .chart {{ border: 1px solid #d9e2ec; padding: 12px; border-radius: 6px; }}
    .note {{ background: #f5f7fa; padding: 12px 14px; border-left: 4px solid #2f6f73; }}
  </style>
</head>
<body>
  <h1>Loan Default Prediction System</h1>
  <p class="note">No external CSV was present in the workspace, so this project uses a reproducible synthetic dataset saved as <strong>loan_data.csv</strong> with the assignment's required schema.</p>
  <h2>Dataset Overview</h2>
  <p>Rows: {len(df)}. Columns: {len(df.columns)}. Approval/repayment target rate: {modeling["target_rate"]:.1%}.</p>
  <h2>EDA Charts</h2>
  <div class="grid">
    <div class="chart">{svg_bar("Loan Status Counts", status_counts.index.tolist(), status_counts.astype(float).tolist())}</div>
    <div class="chart">{svg_histogram("Applicant Income Distribution", df["ApplicantIncome"], bins=12)}</div>
    <div class="chart">{svg_histogram("Loan Amount Distribution", df["LoanAmount"], bins=12, color="#8a6f3d")}</div>
    <div class="chart">{svg_bar("Repayment Rate by Credit History", approval_by_credit["Credit_History"].astype(str).tolist(), approval_by_credit["Loan_Status"].round(3).tolist(), color="#7d5a9e")}</div>
    <div class="chart">{svg_bar("Repayment Rate by Property Area", approval_by_property.index.tolist(), approval_by_property.round(3).tolist(), color="#2f855a")}</div>
  </div>
  <h2>Correlation</h2>
  <div class="chart">{svg_heatmap("Numeric Correlation Matrix", corr)}</div>
  <h2>Model Comparison</h2>
  {results_for_html.to_html(index=False, escape=False)}
  <h2>Final Choice</h2>
  <p>The best model is <strong>{escape(str(modeling["best_model"]["Model"]))}</strong> with accuracy <strong>{modeling["best_model"]["Accuracy"]:.3f}</strong> and F1-score <strong>{modeling["best_model"]["F1-Score"]:.3f}</strong> on the held-out test set.</p>
</body>
</html>
"""


def make_summary_markdown(df: pd.DataFrame, modeling: dict) -> str:
    best = modeling["best_model"]
    results = modeling["results"]
    approval_by_credit = df.groupby("Credit_History")["Loan_Status"].apply(lambda s: (s == "Y").mean()).sort_index()
    approval_by_property = df.groupby("Property_Area")["Loan_Status"].apply(lambda s: (s == "Y").mean()).sort_values(ascending=False)
    comparison = results[["Model", "Accuracy", "Precision", "Recall", "F1-Score"]].round(3)
    comparison_lines = [
        "| Model | Accuracy | Precision | Recall | F1-Score |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison.to_dict(orient="records"):
        comparison_lines.append(
            f"| {row['Model']} | {row['Accuracy']:.3f} | {row['Precision']:.3f} | {row['Recall']:.3f} | {row['F1-Score']:.3f} |"
        )
    comparison_table = "\n".join(comparison_lines)

    return f"""# Loan Default Prediction System - One Page Summary

## Project goal
Build a classification model that predicts whether a loan applicant is likely to repay (`Y`) or default/not repay (`N`) using applicant, loan, credit history, and property-area information.

## Dataset
The workspace did not include an external CSV, so `loan_data.csv` was generated as a reproducible synthetic dataset with the required columns and realistic missing values. It contains {len(df)} loan applications.

## Chosen model
Best model: **{best["Model"]}**

Final test accuracy: **{best["Accuracy"]:.3f}**

Final F1-score: **{best["F1-Score"]:.3f}**

## Model comparison
{comparison_table}

## EDA insights
- Credit history is the strongest signal: applicants with a recorded positive credit history had a much higher repayment rate than applicants with weak credit history.
- Semiurban and urban property areas showed stronger repayment rates than rural property areas in this dataset.
- Applicant income and coapplicant income are right-skewed, so preprocessing includes median imputation and numeric scaling.
- Loan amount has a moderate relationship with income, but credit history remains more predictive of loan status than income alone.

## Preprocessing summary
`Loan_ID` was removed, missing numeric values were filled with medians, missing categorical values were filled with modes, categorical variables were one-hot encoded, and numeric variables were standardized before model training.
"""


def make_readme() -> str:
    return """# Loan Default Prediction System

This project predicts whether a loan applicant is likely to repay or default using a classification workflow.

## Files

- `loan_data.csv` - reproducible synthetic dataset with the assignment schema
- `Loan_Default_Prediction_System.ipynb` - final notebook with preprocessing, EDA, modeling, and evaluation
- `loan_default_report.html` - optional static HTML export with charts and model results
- `loan_default_summary.md` - one-page project summary
- `generate_project.py` - script used to regenerate all project artifacts

## How to run

Open `Loan_Default_Prediction_System.ipynb` in Jupyter and run all cells. The notebook uses `pandas` and `numpy`; charts are rendered as inline SVG/HTML, and the classifiers are implemented directly for offline compatibility.
"""


def code_cell(source: str, execution_count=None, outputs=None):
    return {
        "cell_type": "code",
        "execution_count": execution_count,
        "metadata": {},
        "outputs": outputs or [],
        "source": source.strip("\n").splitlines(keepends=True),
    }


def markdown_cell(source: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip("\n").splitlines(keepends=True),
    }


def stream_output(text: str):
    return {
        "name": "stdout",
        "output_type": "stream",
        "text": text.splitlines(keepends=True),
    }


def html_output(html: str):
    return {
        "data": {
            "text/html": html,
            "text/plain": "<IPython.core.display.HTML object>",
        },
        "metadata": {},
        "output_type": "display_data",
    }


def dataframe_output(df: pd.DataFrame, execution_count: int):
    return {
        "data": {
            "text/html": df.to_html(),
            "text/plain": df.to_string(),
        },
        "execution_count": execution_count,
        "metadata": {},
        "output_type": "execute_result",
    }


NOTEBOOK_CODE = r'''
import math
from html import escape

import numpy as np
import pandas as pd

try:
    from IPython.display import HTML, display
except ImportError:
    HTML = None
    def display(value):
        print(value)


RANDOM_SEED = 42
DATA_PATH = "loan_data.csv"


def show_table(df, rows=5):
    """Display a compact table in notebooks and print it in plain Python."""
    if HTML is None:
        print(df.head(rows).to_string())
    else:
        display(HTML(df.head(rows).to_html(index=False)))


def display_svg(svg):
    """Render SVG in Jupyter; fall back to printing a short message otherwise."""
    if HTML is None:
        print("SVG chart generated.")
    else:
        display(HTML(svg))


def mode_value(series):
    mode = series.mode(dropna=True)
    return mode.iloc[0] if len(mode) else None


def svg_bar(title, labels, values, width=720, height=330, color="#2f6f73"):
    """Create a simple SVG bar chart without external plotting libraries."""
    max_value = max(values) if len(values) else 1
    left = 70
    bottom = 55
    top = 45
    plot_width = width - left - 30
    plot_height = height - top - bottom
    gap = 16
    bar_width = (plot_width - gap * (len(values) - 1)) / max(len(values), 1)
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="{escape(title)}">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{width/2}" y="24" text-anchor="middle" font-size="18" font-family="Arial" fill="#1f2933">{escape(title)}</text>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-20}" y2="{height-bottom}" stroke="#9aa5b1"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#9aa5b1"/>',
    ]
    for i, (label, value) in enumerate(zip(labels, values)):
        x = left + i * (bar_width + gap)
        bar_height = (value / max_value) * plot_height if max_value else 0
        y = height - bottom - bar_height
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="{color}" rx="3"/>')
        parts.append(f'<text x="{x + bar_width/2:.1f}" y="{y - 7:.1f}" text-anchor="middle" font-size="12" font-family="Arial" fill="#1f2933">{value:.2f}</text>')
        parts.append(f'<text x="{x + bar_width/2:.1f}" y="{height - 28}" text-anchor="middle" font-size="12" font-family="Arial" fill="#1f2933">{escape(str(label))}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def svg_histogram(title, values, bins=12, width=720, height=330, color="#4d7cbd"):
    counts, edges = np.histogram(pd.Series(values).dropna(), bins=bins)
    labels = [f"{int(edges[i])}-{int(edges[i+1])}" for i in range(len(edges) - 1)]
    return svg_bar(title, labels, counts.astype(float).tolist(), width, height, color)


def svg_heatmap(title, corr, width=760, height=560):
    labels = list(corr.columns)
    cell_size = min((width - 180) / len(labels), (height - 130) / len(labels))
    left = 140
    top = 55
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="{escape(title)}">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{width/2}" y="24" text-anchor="middle" font-size="18" font-family="Arial" fill="#1f2933">{escape(title)}</text>',
    ]
    for i, row_label in enumerate(labels):
        parts.append(f'<text x="{left - 8}" y="{top + i*cell_size + cell_size*0.62:.1f}" text-anchor="end" font-size="11" font-family="Arial" fill="#1f2933">{escape(row_label)}</text>')
        parts.append(f'<text x="{left + i*cell_size + cell_size/2:.1f}" y="{top + len(labels)*cell_size + 18:.1f}" text-anchor="middle" font-size="11" font-family="Arial" fill="#1f2933">{escape(row_label)}</text>')
        for j, col_label in enumerate(labels):
            value = float(corr.iloc[i, j])
            if value >= 0:
                intensity = int(235 - 95 * min(value, 1))
                fill = f"rgb({intensity},{intensity + 10},255)"
            else:
                intensity = int(235 - 95 * min(abs(value), 1))
                fill = f"rgb(255,{intensity + 10},{intensity})"
            x = left + j * cell_size
            y = top + i * cell_size
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_size:.1f}" height="{cell_size:.1f}" fill="{fill}" stroke="#ffffff"/>')
            parts.append(f'<text x="{x + cell_size/2:.1f}" y="{y + cell_size*0.58:.1f}" text-anchor="middle" font-size="10" font-family="Arial" fill="#1f2933">{value:.2f}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def prepare_features(df):
    """Remove ID, impute missing values, encode categoricals, and scale numerics."""
    cleaned = df.drop(columns=["Loan_ID"]).copy()
    y = cleaned["Loan_Status"].map({"N": 0, "Y": 1}).astype(int)
    X = cleaned.drop(columns=["Loan_Status"])

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [column for column in X.columns if column not in numeric_cols]

    for column in numeric_cols:
        X[column] = X[column].fillna(X[column].median())
    for column in categorical_cols:
        X[column] = X[column].fillna(mode_value(X[column]))

    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dtype=float)

    scaler = {}
    for column in numeric_cols:
        mean = X_encoded[column].mean()
        std = X_encoded[column].std(ddof=0)
        if std == 0:
            std = 1
        X_encoded[column] = (X_encoded[column] - mean) / std
        scaler[column] = {"mean": float(mean), "std": float(std)}

    return X_encoded.astype(float), y.astype(int), cleaned, scaler


def stratified_train_test_split(X, y, test_size=0.20, seed=RANDOM_SEED):
    """Create an 80/20 split while keeping the target distribution similar."""
    rng = np.random.default_rng(seed)
    train_indices = []
    test_indices = []

    for class_value in sorted(y.unique()):
        class_indices = y[y == class_value].index.to_numpy().copy()
        rng.shuffle(class_indices)
        n_test = int(round(len(class_indices) * test_size))
        test_indices.extend(class_indices[:n_test])
        train_indices.extend(class_indices[n_test:])

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    return (
        X.loc[train_indices].to_numpy(dtype=float),
        X.loc[test_indices].to_numpy(dtype=float),
        y.loc[train_indices].to_numpy(dtype=int),
        y.loc[test_indices].to_numpy(dtype=int),
        X.columns.tolist(),
    )


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -35, 35)))


def train_logistic_regression(X_train, y_train, learning_rate=0.08, epochs=2500):
    X_bias = np.c_[np.ones(X_train.shape[0]), X_train]
    weights = np.zeros(X_bias.shape[1])

    for _ in range(epochs):
        predictions = sigmoid(X_bias @ weights)
        gradient = X_bias.T @ (predictions - y_train) / len(y_train)
        weights -= learning_rate * gradient

    return weights


def predict_logistic_regression(weights, X):
    X_bias = np.c_[np.ones(X.shape[0]), X]
    probabilities = sigmoid(X_bias @ weights)
    return (probabilities >= 0.5).astype(int), probabilities


def gini_impurity(y):
    if len(y) == 0:
        return 0
    p = np.mean(y)
    return 1 - p ** 2 - (1 - p) ** 2


class DecisionTreeClassifierScratch:
    """Small Gini-based decision tree classifier for numeric encoded features."""

    def __init__(self, max_depth=4, min_samples_split=18, max_features=None, seed=RANDOM_SEED):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.rng = np.random.default_rng(seed)
        self.tree_ = None

    def fit(self, X, y):
        self.n_features_ = X.shape[1]
        self.tree_ = self._build_tree(X, y, depth=0)
        return self

    def _candidate_features(self):
        if self.max_features is None:
            return np.arange(self.n_features_)
        count = min(self.max_features, self.n_features_)
        return self.rng.choice(self.n_features_, size=count, replace=False)

    def _best_split(self, X, y):
        parent_impurity = gini_impurity(y)
        best_gain = 0
        best_feature = None
        best_threshold = None

        for feature in self._candidate_features():
            values = np.unique(X[:, feature])
            if len(values) <= 1:
                continue
            thresholds = (values[:-1] + values[1:]) / 2

            if len(thresholds) > 25:
                quantiles = np.linspace(0.05, 0.95, 25)
                thresholds = np.unique(np.quantile(values, quantiles))

            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                weighted_impurity = (
                    left_mask.mean() * gini_impurity(y[left_mask])
                    + right_mask.mean() * gini_impurity(y[right_mask])
                )
                gain = parent_impurity - weighted_impurity
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X, y, depth):
        positive_rate = float(np.mean(y))
        prediction = int(positive_rate >= 0.5)

        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or len(np.unique(y)) == 1
        ):
            return {"prediction": prediction, "positive_rate": positive_rate}

        feature, threshold, gain = self._best_split(X, y)
        if feature is None or gain <= 1e-7:
            return {"prediction": prediction, "positive_rate": positive_rate}

        left_mask = X[:, feature] <= threshold
        return {
            "feature": int(feature),
            "threshold": float(threshold),
            "prediction": prediction,
            "positive_rate": positive_rate,
            "left": self._build_tree(X[left_mask], y[left_mask], depth + 1),
            "right": self._build_tree(X[~left_mask], y[~left_mask], depth + 1),
        }

    def _predict_one(self, row, node):
        while "feature" in node:
            if row[node["feature"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        return node["prediction"], node["positive_rate"]

    def predict(self, X):
        predictions = [self._predict_one(row, self.tree_)[0] for row in X]
        return np.array(predictions, dtype=int)

    def predict_proba(self, X):
        probabilities = [self._predict_one(row, self.tree_)[1] for row in X]
        return np.array(probabilities, dtype=float)


class RandomForestClassifierScratch:
    """Bagged decision trees with random feature subsets at each split."""

    def __init__(self, n_estimators=35, max_depth=5, min_samples_split=16, seed=RANDOM_SEED):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.rng = np.random.default_rng(seed)
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        max_features = max(1, int(math.sqrt(X.shape[1])))
        for i in range(self.n_estimators):
            sample_indices = self.rng.choice(len(X), size=len(X), replace=True)
            tree = DecisionTreeClassifierScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_features,
                seed=RANDOM_SEED + i + 1,
            )
            tree.fit(X[sample_indices], y[sample_indices])
            self.trees.append(tree)
        return self

    def predict_proba(self, X):
        probabilities = np.vstack([tree.predict_proba(X) for tree in self.trees])
        return probabilities.mean(axis=0)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


def classification_metrics(y_true, y_pred):
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }
'''


def make_notebook(df: pd.DataFrame, modeling: dict) -> dict:
    best = modeling["best_model"]
    X = modeling["X"]
    y = modeling["y"]
    results_display = modeling["results"].copy()
    for metric in ["Accuracy", "Precision", "Recall", "F1-Score"]:
        results_display[metric] = results_display[metric].round(3)

    missing_summary = df.isna().sum().reset_index()
    missing_summary.columns = ["Column", "Missing Values"]

    target_distribution = y.value_counts(normalize=True).rename({0: "N", 1: "Y"}).round(3)
    preprocess_text = (
        f"Feature matrix shape after preprocessing: {X.shape}\n"
        f"Target distribution:\n{target_distribution}\n"
    )

    status_counts = df["Loan_Status"].value_counts().reindex(["N", "Y"]).fillna(0)
    approval_by_credit = (
        df.assign(Credit_History=df["Credit_History"].fillna(-1).replace({-1: "Missing"}))
        .groupby("Credit_History")["Loan_Status"]
        .apply(lambda series: (series == "Y").mean())
        .reset_index()
    )
    approval_by_property = (
        df.groupby("Property_Area")["Loan_Status"]
        .apply(lambda series: (series == "Y").mean())
        .sort_values(ascending=False)
    )
    corr_df = df.drop(columns=["Loan_ID"]).copy()
    corr_df["Loan_Status_Num"] = corr_df["Loan_Status"].map({"N": 0, "Y": 1})
    numeric_corr = corr_df.select_dtypes(include=[np.number]).corr(numeric_only=True).round(2)

    best_text = (
        f"Best model: {best['Model']}\n"
        f"Accuracy: {best['Accuracy']:.3f}\n"
        f"Precision: {best['Precision']:.3f}\n"
        f"Recall: {best['Recall']:.3f}\n"
        f"F1-Score: {best['F1-Score']:.3f}\n"
    )

    notebook_cells = [
        markdown_cell(
            """# Loan Default Prediction System

## Objective
Build a machine learning workflow that predicts whether a loan applicant is likely to repay (`Y`) or default/not repay (`N`) using historical applicant data.

Because no external CSV was present in the workspace, this submission uses the included reproducible dataset `loan_data.csv`, which follows the requested schema."""
        ),
        markdown_cell(
            """## Phase 1 - Data Preprocessing
This phase loads the data, removes `Loan_ID`, handles missing values, converts categorical columns into numeric features, and scales numeric columns."""
        ),
        code_cell(NOTEBOOK_CODE, execution_count=1),
        code_cell(
            """df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")
show_table(df)""",
            execution_count=2,
            outputs=[
                stream_output(f"Dataset shape: {df.shape}\n"),
                html_output(df.head().to_html(index=False)),
            ],
        ),
        code_cell(
            """missing_summary = df.isna().sum().reset_index()
missing_summary.columns = ["Column", "Missing Values"]
missing_summary""",
            execution_count=3,
            outputs=[dataframe_output(missing_summary, 3)],
        ),
        code_cell(
            """X, y, cleaned_df, scaler = prepare_features(df)
print(f"Feature matrix shape after preprocessing: {X.shape}")
print(f"Target distribution:\\n{y.value_counts(normalize=True).rename({0: 'N', 1: 'Y'}).round(3)}")
show_table(X)""",
            execution_count=4,
            outputs=[
                stream_output(preprocess_text),
                html_output(X.head().to_html(index=False)),
            ],
        ),
        markdown_cell(
            """## Phase 2 - Exploratory Data Analysis
The charts below show distribution patterns, feature-target comparisons, and numeric correlations."""
        ),
        code_cell(
            """status_counts = df["Loan_Status"].value_counts().reindex(["N", "Y"]).fillna(0)
display_svg(svg_bar("Loan Status Counts", status_counts.index.tolist(), status_counts.astype(float).tolist()))""",
            execution_count=5,
            outputs=[
                html_output(svg_bar("Loan Status Counts", status_counts.index.tolist(), status_counts.astype(float).tolist()))
            ],
        ),
        code_cell(
            """display_svg(svg_histogram("Applicant Income Distribution", df["ApplicantIncome"], bins=12))
display_svg(svg_histogram("Loan Amount Distribution", df["LoanAmount"], bins=12, color="#8a6f3d"))""",
            execution_count=6,
            outputs=[
                html_output(svg_histogram("Applicant Income Distribution", df["ApplicantIncome"], bins=12)),
                html_output(svg_histogram("Loan Amount Distribution", df["LoanAmount"], bins=12, color="#8a6f3d")),
            ],
        ),
        code_cell(
            """approval_by_credit = (
    df.assign(Credit_History=df["Credit_History"].fillna(-1).replace({-1: "Missing"}))
    .groupby("Credit_History")["Loan_Status"]
    .apply(lambda series: (series == "Y").mean())
    .reset_index()
)
display_svg(svg_bar(
    "Repayment Rate by Credit History",
    approval_by_credit["Credit_History"].astype(str).tolist(),
    approval_by_credit["Loan_Status"].round(3).tolist(),
    color="#7d5a9e",
))""",
            execution_count=7,
            outputs=[
                html_output(
                    svg_bar(
                        "Repayment Rate by Credit History",
                        approval_by_credit["Credit_History"].astype(str).tolist(),
                        approval_by_credit["Loan_Status"].round(3).tolist(),
                        color="#7d5a9e",
                    )
                )
            ],
        ),
        code_cell(
            """approval_by_property = (
    df.groupby("Property_Area")["Loan_Status"]
    .apply(lambda series: (series == "Y").mean())
    .sort_values(ascending=False)
)
display_svg(svg_bar(
    "Repayment Rate by Property Area",
    approval_by_property.index.tolist(),
    approval_by_property.round(3).tolist(),
    color="#2f855a",
))""",
            execution_count=8,
            outputs=[
                html_output(
                    svg_bar(
                        "Repayment Rate by Property Area",
                        approval_by_property.index.tolist(),
                        approval_by_property.round(3).tolist(),
                        color="#2f855a",
                    )
                )
            ],
        ),
        code_cell(
            """corr_df = df.drop(columns=["Loan_ID"]).copy()
corr_df["Loan_Status_Num"] = corr_df["Loan_Status"].map({"N": 0, "Y": 1})
numeric_corr = corr_df.select_dtypes(include=[np.number]).corr(numeric_only=True).round(2)
display_svg(svg_heatmap("Numeric Correlation Matrix", numeric_corr))
numeric_corr""",
            execution_count=9,
            outputs=[
                html_output(svg_heatmap("Numeric Correlation Matrix", numeric_corr)),
                dataframe_output(numeric_corr, 9),
            ],
        ),
        markdown_cell(
            """## Phase 3 - Model Building
We use an 80/20 stratified split and train three classifiers:

- Logistic Regression
- Decision Tree
- Random Forest-style bagged trees"""
        ),
        code_cell(
            """X_train, X_test, y_train, y_test, feature_names = stratified_train_test_split(X, y, test_size=0.20)
print(f"Training records: {len(y_train)}")
print(f"Testing records: {len(y_test)}")""",
            execution_count=10,
            outputs=[
                stream_output(
                    f"Training records: {modeling['train_size']}\n"
                    f"Testing records: {modeling['test_size']}\n"
                )
            ],
        ),
        code_cell(
            """logistic_weights = train_logistic_regression(X_train, y_train)
logistic_pred, logistic_proba = predict_logistic_regression(logistic_weights, X_test)

decision_tree = DecisionTreeClassifierScratch(max_depth=4, min_samples_split=18)
decision_tree.fit(X_train, y_train)
tree_pred = decision_tree.predict(X_test)

random_forest = RandomForestClassifierScratch(n_estimators=35, max_depth=5, min_samples_split=16)
random_forest.fit(X_train, y_train)
forest_pred = random_forest.predict(X_test)

results = pd.DataFrame([
    {"Model": "Logistic Regression", **classification_metrics(y_test, logistic_pred)},
    {"Model": "Decision Tree", **classification_metrics(y_test, tree_pred)},
    {"Model": "Random Forest", **classification_metrics(y_test, forest_pred)},
]).sort_values(["F1-Score", "Accuracy"], ascending=False)

results_display = results.copy()
for metric in ["Accuracy", "Precision", "Recall", "F1-Score"]:
    results_display[metric] = results_display[metric].round(3)
results_display""",
            execution_count=11,
            outputs=[dataframe_output(results_display, 11)],
        ),
        code_cell(
            """best_model = results.iloc[0]
print(f"Best model: {best_model['Model']}")
print(f"Accuracy: {best_model['Accuracy']:.3f}")
print(f"Precision: {best_model['Precision']:.3f}")
print(f"Recall: {best_model['Recall']:.3f}")
print(f"F1-Score: {best_model['F1-Score']:.3f}")""",
            execution_count=12,
            outputs=[stream_output(best_text)],
        ),
        markdown_cell(
            f"""## Phase 4 - Model Comparison
The best model for this run is **{best["Model"]}**, selected by F1-score and accuracy. It achieved **{best["Accuracy"]:.3f} accuracy** and **{best["F1-Score"]:.3f} F1-score** on the held-out test set.

## Phase 5 - Findings
- Credit history is the strongest predictor of loan repayment.
- Semiurban/urban property areas have higher repayment rates than rural areas in this dataset.
- Income and loan amount are right-skewed, so median imputation and scaling are useful preprocessing steps.
- The final model can support loan-screening decisions, but it should be validated on real institutional data before production use."""
        ),
    ]

    return {
        "cells": notebook_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    df = make_loan_dataset()
    df.to_csv(DATA_PATH, index=False)

    modeling = run_modeling(df)

    notebook = make_notebook(df, modeling)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    SUMMARY_PATH.write_text(make_summary_markdown(df, modeling), encoding="utf-8")
    HTML_REPORT_PATH.write_text(make_report_html(df, modeling), encoding="utf-8")
    README_PATH.write_text(make_readme(), encoding="utf-8")
    REQUIREMENTS_PATH.write_text("pandas\nnumpy\n", encoding="utf-8")

    best = modeling["best_model"]
    print(f"Wrote {DATA_PATH.name}, {NOTEBOOK_PATH.name}, {SUMMARY_PATH.name}, and {HTML_REPORT_PATH.name}")
    print(f"Best model: {best['Model']} | Accuracy: {best['Accuracy']:.3f} | F1: {best['F1-Score']:.3f}")


if __name__ == "__main__":
    main()
