"""Reproducible holdout error analysis for the Day 4 winning model."""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

from finnews_sentiment.features.preprocess import EXPECTED_SENTIMENTS
from finnews_sentiment.models.train_model import validate_training_data


def split_for_error_analysis(
    df: pd.DataFrame,
    text_col: str = "text_clean",
    target_col: str = "sentiment",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return deterministic stratified train and test frames.

    ``source_index`` preserves the row identity after resetting each split's
    index, which makes exported error examples traceable to the prepared data.
    """
    validate_training_data(df, text_col=text_col, target_col=target_col)
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    prepared = df.copy()
    if "source_index" in prepared.columns:
        raise ValueError("Dataset already contains reserved column: source_index")
    prepared.insert(0, "source_index", prepared.index)
    train_df, test_df = train_test_split(
        prepared,
        test_size=test_size,
        random_state=random_state,
        stratify=prepared[target_col],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _model_parameters(model: Any) -> dict[str, Any]:
    """Extract the small set of pipeline parameters relevant to the report."""
    if not hasattr(model, "named_steps"):
        return {"model_type": type(model).__name__}
    tfidf = model.named_steps.get("tfidf")
    classifier = model.named_steps.get("clf")
    result: dict[str, Any] = {
        "model_type": type(classifier).__name__ if classifier is not None else type(model).__name__
    }
    for name in ("max_features", "min_df", "ngram_range", "sublinear_tf"):
        if tfidf is not None and hasattr(tfidf, name):
            value = getattr(tfidf, name)
            result[f"tfidf__{name}"] = list(value) if isinstance(value, tuple) else value
    for name in ("C", "class_weight", "max_iter", "random_state"):
        if classifier is not None and hasattr(classifier, name):
            result[f"clf__{name}"] = getattr(classifier, name)
    return result


def analyze_predictions(
    model: Any,
    test_df: pd.DataFrame,
    text_col: str = "text_clean",
    target_col: str = "sentiment",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Evaluate a fitted multiclass model and describe every prediction.

    The returned decision margin is the gap between the two largest SVC
    decision scores.  It is a useful ranking signal, but it is not a calibrated
    probability or an absolute confidence estimate.
    """
    validate_training_data(test_df, text_col=text_col, target_col=target_col)
    if not hasattr(model, "predict") or not hasattr(model, "decision_function"):
        raise TypeError("Model must provide predict() and decision_function()")
    if "source_index" not in test_df.columns:
        raise ValueError("test_df must contain source_index; use split_for_error_analysis")

    texts = test_df[text_col].astype(str)
    actual = test_df[target_col].astype(str).to_numpy()
    predicted = np.asarray(model.predict(texts), dtype=str)
    scores = np.asarray(model.decision_function(texts))
    classes = np.asarray(model.classes_, dtype=str)
    if scores.ndim != 2 or scores.shape != (len(test_df), len(classes)):
        raise ValueError("Expected multiclass decision scores with one column per class")
    if set(classes) != set(EXPECTED_SENTIMENTS):
        raise ValueError("Model classes do not match the expected sentiments")

    sorted_positions = np.argsort(scores, axis=1)
    best_positions = sorted_positions[:, -1]
    runner_up_positions = sorted_positions[:, -2]
    row_positions = np.arange(len(test_df))

    predictions = pd.DataFrame(
        {
            "source_index": test_df["source_index"].to_numpy(),
            "text": (
                test_df["text"].astype(str).to_numpy()
                if "text" in test_df.columns
                else texts.to_numpy()
            ),
            "text_clean": texts.to_numpy(),
            "actual": actual,
            "predicted": predicted,
            "correct": actual == predicted,
            "word_count_clean": (
                test_df["word_count_clean"].to_numpy()
                if "word_count_clean" in test_df.columns
                else texts.str.split().str.len().to_numpy()
            ),
            "predicted_score": scores[row_positions, best_positions],
            "runner_up_label": classes[runner_up_positions],
            "runner_up_score": scores[row_positions, runner_up_positions],
            "decision_margin": (
                scores[row_positions, best_positions]
                - scores[row_positions, runner_up_positions]
            ),
        }
    )

    labels = list(EXPECTED_SENTIMENTS)
    counts = confusion_matrix(actual, predicted, labels=labels)
    normalized = confusion_matrix(actual, predicted, labels=labels, normalize="true")
    confusion_counts = pd.DataFrame(counts, index=labels, columns=labels)
    confusion_normalized = pd.DataFrame(normalized, index=labels, columns=labels)

    errors = predictions.loc[~predictions["correct"]].copy()
    errors = errors.sort_values(
        ["actual", "predicted", "decision_margin", "source_index"],
        kind="stable",
    ).reset_index(drop=True)
    transitions = (
        errors.groupby(["actual", "predicted"], sort=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["count", "actual", "predicted"], ascending=[False, True, True])
    )
    comparison = (
        predictions.assign(group=np.where(predictions["correct"], "correct", "error"))
        .groupby("group")
        .agg(
            rows=("correct", "size"),
            word_count_mean=("word_count_clean", "mean"),
            word_count_median=("word_count_clean", "median"),
            margin_mean=("decision_margin", "mean"),
            margin_median=("decision_margin", "median"),
        )
        .reindex(["correct", "error"])
    )
    report = classification_report(
        actual,
        predicted,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    summary: dict[str, Any] = {
        "rows": int(len(predictions)),
        "correct_predictions": int(predictions["correct"].sum()),
        "errors": int(len(errors)),
        "error_rate": float(len(errors) / len(predictions)),
        "metrics": {
            "accuracy": float(accuracy_score(actual, predicted)),
            "macro_f1": float(f1_score(actual, predicted, average="macro")),
            "weighted_f1": float(f1_score(actual, predicted, average="weighted")),
        },
        "classification_report": report,
        "labels": labels,
        "confusion_matrix": counts.tolist(),
        "confusion_matrix_normalized_true": normalized.tolist(),
        "error_transitions": transitions.to_dict("records"),
        "correct_vs_error": {
            index: {key: float(value) for key, value in row.items()}
            for index, row in comparison.dropna(how="all").to_dict("index").items()
        },
        "model_parameters": _model_parameters(model),
    }
    return predictions, confusion_counts, confusion_normalized, summary


def plot_confusion_matrices(
    confusion_counts: pd.DataFrame,
    confusion_normalized: pd.DataFrame,
) -> Figure:
    """Plot absolute and true-class-normalized confusion matrices."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.heatmap(confusion_counts, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0])
    sns.heatmap(
        confusion_normalized,
        annot=True,
        fmt=".1%",
        cmap="Blues",
        vmin=0,
        vmax=1,
        cbar=False,
        ax=axes[1],
    )
    axes[0].set_title("Confusion matrix: количество")
    axes[1].set_title("Confusion matrix: доля истинного класса")
    for ax in axes:
        ax.set_xlabel("Предсказанный класс")
        ax.set_ylabel("Истинный класс")
    fig.tight_layout()
    return fig


def save_error_analysis_artifacts(
    predictions: pd.DataFrame,
    confusion_counts: pd.DataFrame,
    confusion_normalized: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    """Save the complete set of Day 5 reports and return their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "figure": output_dir / "confusion_matrix.png",
        "confusion_csv": output_dir / "confusion_matrix.csv",
        "errors_csv": output_dir / "error_examples.csv",
        "summary_json": output_dir / "error_analysis.json",
    }
    errors = predictions.loc[~predictions["correct"]].sort_values(
        ["actual", "predicted", "decision_margin", "source_index"], kind="stable"
    )
    confusion_counts.rename_axis("actual").to_csv(paths["confusion_csv"])
    errors.to_csv(paths["errors_csv"], index=False)
    paths["summary_json"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    figure = plot_confusion_matrices(confusion_counts, confusion_normalized)
    figure.savefig(paths["figure"], dpi=150, bbox_inches="tight")
    plt.close(figure)
    return paths
