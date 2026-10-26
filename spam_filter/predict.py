"""Classify an example email with a previously trained model.

Run from the repository root after training:
    python -m spam_filter.predict --sender example@news.example \
      --subject "URGENT: verify" --body "Click now and submit your password"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from spam_filter.train import DEFAULT_MODEL, classify_scores


def predict_email(model_path: Path, subject: str, sender: str, body: str) -> tuple[str, float]:
    """Load a trusted local model and predict ham/spam from all three fields.

    Do not load joblib files from untrusted sources: loading uses pickle.
    Returns (class label, raw linear-regression score).
    """
    if not subject.strip() or not sender.strip() or not body.strip():
        raise ValueError("Subject, sender, and body must all be nonempty")
    bundle = joblib.load(model_path)
    email = pd.DataFrame([{"subject": subject, "sender": sender, "body": body}])
    score = float(bundle["pipeline"].predict(email)[0])
    predicted = int(classify_scores([score], bundle["threshold"])[0])
    return ("spam" if predicted == 1 else "ham", score)


def parse_args() -> argparse.Namespace:
    """Parse the example email and optional trained-model path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--sender", required=True)
    parser.add_argument("--body", required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    return parser.parse_args()


def main() -> None:
    """Command-line prediction entry point."""
    args = parse_args()
    label, score = predict_email(args.model, args.subject, args.sender, args.body)
    print(f"Prediction: {label.upper()}")
    print(f"Raw linear-regression score: {score:.3f} (not a probability)")


if __name__ == "__main__":
    main()
