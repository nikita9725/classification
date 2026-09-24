"""Reusable training helpers for the Day 3 sentiment baseline."""

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from finnews_sentiment.features.preprocess import EXPECTED_SENTIMENTS


def validate_training_data(
    df: pd.DataFrame,
    text_col: str = "text_clean",
    target_col: str = "sentiment",
) -> None:
    """Validate the minimal schema required to train the baseline."""
    missing = {text_col, target_col}.difference(df.columns)
    if missing:
        columns = ", ".join(sorted(missing))
        raise ValueError(f"Dataset is missing required columns: {columns}")
    if df.empty:
        raise ValueError("Dataset is empty")

    empty_text = df[text_col].isna() | df[text_col].astype(str).str.strip().eq("")
    if empty_text.any():
        raise ValueError(f"Dataset contains {int(empty_text.sum())} empty text(s)")

    missing_target = df[target_col].isna() | df[target_col].astype(str).str.strip().eq("")
    if missing_target.any():
        raise ValueError(f"Dataset contains {int(missing_target.sum())} missing target(s)")

    actual = set(df[target_col].astype(str))
    unknown = actual.difference(EXPECTED_SENTIMENTS)
    if unknown:
        raise ValueError("Dataset contains unknown sentiments: " + ", ".join(sorted(unknown)))
    missing_classes = set(EXPECTED_SENTIMENTS).difference(actual)
    if missing_classes:
        raise ValueError(
            "Dataset does not contain all expected sentiments: "
            + ", ".join(sorted(missing_classes))
        )


def build_baseline_pipeline(
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
    random_state: int = 42,
) -> Pipeline:
    """Build the fixed TF-IDF and Logistic Regression baseline."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(max_features=max_features, ngram_range=ngram_range),
            ),
            (
                "clf",
                LogisticRegression(max_iter=200, random_state=random_state),
            ),
        ]
    )


def train_baseline(
    df: pd.DataFrame,
    text_col: str = "text_clean",
    target_col: str = "sentiment",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[Pipeline, dict[str, Any]]:
    """Train and evaluate the baseline on a deterministic stratified split."""
    validate_training_data(df, text_col=text_col, target_col=target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col].astype(str),
        df[target_col].astype(str),
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_col],
    )

    model = build_baseline_pipeline(random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    report = classification_report(
        y_test,
        y_pred,
        labels=list(EXPECTED_SENTIMENTS),
        output_dict=True,
        zero_division=0,
    )
    metrics: dict[str, Any] = {
        "model": "TF-IDF (1, 2) + Logistic Regression",
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted")),
        "classification_report": report,
        "split": {
            "test_size": test_size,
            "random_state": random_state,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "train_distribution": {
                label: int(count)
                for label, count in y_train.value_counts().sort_index().items()
            },
            "test_distribution": {
                label: int(count)
                for label, count in y_test.value_counts().sort_index().items()
            },
        },
    }
    return model, metrics


def save_baseline_artifacts(
    model: Pipeline,
    metrics: dict[str, Any],
    model_path: Path,
    metrics_path: Path,
) -> None:
    """Persist the fitted pipeline and its human-readable evaluation results."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "meta": metrics}, model_path)
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
