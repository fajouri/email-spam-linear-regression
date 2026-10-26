"""Run these automated checks with: python -m pytest -q"""

from pathlib import Path

import numpy as np
import pytest

from spam_filter.predict import predict_email
from spam_filter.train import (
    DEFAULT_DATA,
    build_model,
    classify_scores,
    evaluate_model,
    load_emails,
    split_emails,
    train_and_evaluate,
)


def test_dataset_contains_100_balanced_labeled_emails() -> None:
    df = load_emails(DEFAULT_DATA)
    assert len(df) == 100
    assert df.label.value_counts().to_dict() == {"spam": 50, "ham": 50}
    assert set(["subject", "sender", "body", "label"]).issubset(df.columns)


def test_stratified_80_20_split() -> None:
    x_train, x_test, y_train, y_test = split_emails(load_emails(DEFAULT_DATA))
    assert (len(x_train), len(x_test)) == (80, 20)
    assert y_train.value_counts().sort_index().to_dict() == {0: 40, 1: 40}
    assert y_test.value_counts().sort_index().to_dict() == {0: 10, 1: 10}
    assert not (set(x_train.index) & set(x_test.index))


def test_threshold_converts_raw_scores_without_clamping() -> None:
    assert classify_scores(np.array([-0.4, 0.49, 0.5, 1.4])).tolist() == [0, 0, 1, 1]


def test_fit_and_evaluate_on_holdout_set() -> None:
    x_train, x_test, y_train, y_test = split_emails(load_emails(DEFAULT_DATA))
    model = build_model().fit(x_train, y_train)
    metrics = evaluate_model(model, x_test, y_test)
    assert metrics["test_count"] == 20
    assert set(metrics["confusion_matrix"]) == {"TN", "FP", "FN", "TP"}
    assert sum(metrics["confusion_matrix"].values()) == 20
    assert np.isfinite(metrics["mean_squared_error"])


def test_training_saves_a_model_and_predicts(tmp_path: Path) -> None:
    model_path = tmp_path / "model.joblib"
    train_and_evaluate(model_path=model_path)
    assert model_path.is_file()
    label, score = predict_email(
        model_path,
        subject="URGENT: verify your account",
        sender="offers@bonus.example",
        body="Click now to claim your prize and provide your password",
    )
    assert label in {"ham", "spam"}
    assert np.isfinite(score)


def test_rejects_missing_fields(tmp_path: Path) -> None:
    missing_csv = tmp_path / "bad.csv"
    missing_csv.write_text("subject,sender,label\nHello,a@example,ham\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required columns"):
        load_emails(missing_csv)
