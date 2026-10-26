"""Train and test a demonstration spam filter using ordinary linear regression.

Run from the repository root:
    python -m spam_filter.train
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "emails.csv"
DEFAULT_MODEL = ROOT / "artifacts" / "spam_linear_regression.joblib"
TEXT_COLUMNS = ["subject", "sender", "body"]
LABELS = {"ham": 0, "spam": 1}
DEFAULT_THRESHOLD = 0.5


def load_emails(csv_path: Path) -> pd.DataFrame:
    """Read the labeled CSV and fail early on missing or invalid examples.

    Labels remain strings in the returned DataFrame for readability.
    """
    df = pd.read_csv(csv_path)
    required = TEXT_COLUMNS + ["label"]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if df.empty or df[required].isna().any().any():
        raise ValueError("Dataset is empty or has missing field values")
    df = df[required].copy()
    for field in required:
        df[field] = df[field].astype(str).str.strip()
        if df[field].eq("").any():
            raise ValueError(f"Dataset contains blank values in {field}")
    df["label"] = df["label"].str.lower()
    unknown = set(df["label"]) - set(LABELS)
    if unknown:
        raise ValueError(f"Unsupported labels: {sorted(unknown)}; use ham or spam")
    if df.duplicated(subset=TEXT_COLUMNS).any():
        raise ValueError("Duplicate emails could leak identical content across splits")
    if df["label"].value_counts().min() < 2 or df["label"].nunique() != 2:
        raise ValueError("At least two examples of each label are required")
    return df


def split_emails(
    df: pd.DataFrame, test_size: float = 0.20, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create stratified train/test sets, using 80%/20% by default.

    Fitting the vectorizers happens *after* this split to avoid data leakage.
    """
    features = df[TEXT_COLUMNS]
    targets = df["label"].map(LABELS)
    return train_test_split(
        features,
        targets,
        test_size=test_size,
        random_state=random_state,
        stratify=targets,
    )


def build_model() -> Pipeline:
    """Build one pipeline that learns vocabulary only from the training set.

    Separate TF-IDF vectorizers preserve subject, sender, and body as distinct
    groups of features. The final estimator is intentionally LinearRegression.
    """
    text_features = ColumnTransformer(
        transformers=[
            ("subject", TfidfVectorizer(ngram_range=(1, 2), max_features=200), "subject"),
            (
                "sender",
                TfidfVectorizer(token_pattern=r"(?u)\b[\w.-]+\b", max_features=120),
                "sender",
            ),
            ("body", TfidfVectorizer(ngram_range=(1, 2), max_features=600), "body"),
        ]
    )
    return Pipeline(
        steps=[("features", text_features), ("regressor", LinearRegression())]
    )


def classify_scores(scores: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    """Turn unconstrained regression scores into binary class predictions.

    A score >= 0.5 means spam (1); otherwise ham (0). Scores are NOT
    probabilities and can be below zero or above one.
    """
    return (np.asarray(scores) >= threshold).astype(int)


def evaluate_model(
    model: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """Evaluate only held-out data using regression and classification metrics."""
    scores = model.predict(x_test)
    predicted = classify_scores(scores, threshold)
    tn, fp, fn, tp = confusion_matrix(y_test, predicted, labels=[0, 1]).ravel()
    return {
        "test_count": len(y_test),
        "accuracy": float(accuracy_score(y_test, predicted)),
        "precision_spam": float(precision_score(y_test, predicted, zero_division=0)),
        "recall_spam": float(recall_score(y_test, predicted, zero_division=0)),
        "f1_spam": float(f1_score(y_test, predicted, zero_division=0)),
        "mean_squared_error": float(mean_squared_error(y_test, scores)),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "classification_report": classification_report(
            y_test, predicted, labels=[0, 1], target_names=["ham", "spam"], zero_division=0
        ),
    }


def train_and_evaluate(
    csv_path: Path = DEFAULT_DATA,
    model_path: Path = DEFAULT_MODEL,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """Load, split, fit, evaluate, and save a reusable model bundle."""
    if not 0 <= threshold <= 1:
        raise ValueError("For this demo choose a threshold from 0 to 1")
    df = load_emails(csv_path)
    x_train, x_test, y_train, y_test = split_emails(df)
    model = build_model()
    model.fit(x_train, y_train)
    metrics = evaluate_model(model, x_test, y_test, threshold)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": model, "threshold": threshold}, model_path)

    print(f"Loaded dataset: {len(df)} emails ({int((df.label == 'spam').sum())} spam, "
          f"{int((df.label == 'ham').sum())} ham)")
    print(f"Training: {len(x_train)} emails | Testing: {len(x_test)} emails")
    print(f"Training classes: {y_train.value_counts().sort_index().to_dict()} (0=ham, 1=spam)")
    print(f"Testing classes:  {y_test.value_counts().sort_index().to_dict()} (0=ham, 1=spam)")
    print(f"Spam threshold: {threshold:.2f} (scores are NOT probabilities)\n")
    for key in ("accuracy", "precision_spam", "recall_spam", "f1_spam", "mean_squared_error"):
        print(f"{key}: {metrics[key]:.3f}")
    print(f"Confusion matrix (TN, FP, FN, TP): {metrics['confusion_matrix']}")
    print("\nClassification report:\n" + metrics["classification_report"])
    print(f"Saved trained model to: {model_path}")
    return metrics


def parse_args() -> argparse.Namespace:
    """Define optional command-line paths and decision threshold."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Labeled CSV path")
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL, help="Model save path")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_args()
    train_and_evaluate(args.data, args.output, args.threshold)


if __name__ == "__main__":
    main()
