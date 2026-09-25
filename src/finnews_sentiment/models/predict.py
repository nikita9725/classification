"""Load the trained sentiment pipeline and run inference on new texts."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from finnews_sentiment.features.preprocess import clean_text


def load_model(path: str | Path) -> tuple[Any, dict[str, Any]]:
    """Load and validate a trusted joblib model bundle.

    ``joblib`` can execute arbitrary code while loading, so ``path`` must point
    to an artifact produced by this project or another trusted source.
    """
    bundle = joblib.load(Path(path))
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError("Model artifact must be a dictionary containing 'model'")

    model = bundle["model"]
    if not callable(getattr(model, "predict", None)):
        raise TypeError("Loaded model must provide predict()")
    if not hasattr(model, "classes_"):
        raise ValueError("Loaded model is not fitted: classes_ is missing")

    metadata = bundle.get("meta", {})
    if not isinstance(metadata, dict):
        raise ValueError("Model artifact 'meta' must be a dictionary")
    return model, metadata


def _validate_and_clean_texts(texts: str | Sequence[str]) -> tuple[list[str], list[str]]:
    if isinstance(texts, str):
        original = [texts]
    else:
        try:
            original = list(texts)
        except TypeError as exc:
            raise TypeError("texts must be a string or a sequence of strings") from exc

    if not original:
        raise ValueError("At least one text is required for inference")

    non_strings = [index for index, value in enumerate(original) if not isinstance(value, str)]
    if non_strings:
        raise TypeError(f"Texts at positions {non_strings} are not strings")

    cleaned = [clean_text(value) for value in original]
    empty = [index for index, value in enumerate(cleaned) if not value]
    if empty:
        raise ValueError(f"Texts at positions {empty} are empty after preprocessing")
    return original, cleaned


def predict_texts(model: Any, texts: str | Sequence[str]) -> pd.DataFrame:
    """Predict sentiment and return one row per input text.

    LinearSVC decision scores and the gap between its two largest scores are
    useful for ranking ambiguous examples, but they are not probabilities.
    """
    original, cleaned = _validate_and_clean_texts(texts)
    predictions = model.predict(cleaned)
    result = pd.DataFrame(
        {
            "text": original,
            "text_clean": cleaned,
            "pred_sentiment": predictions,
        }
    )

    decision_function = getattr(model, "decision_function", None)
    if not callable(decision_function):
        return result

    classes = [str(label) for label in model.classes_]
    scores = np.asarray(decision_function(cleaned))
    if scores.ndim == 1 and len(classes) == 2:
        scores = np.column_stack((-scores, scores))
    if scores.ndim != 2 or scores.shape != (len(result), len(classes)):
        raise ValueError("Model returned decision scores with an unexpected shape")

    for index, label in enumerate(classes):
        result[f"score_{label}"] = scores[:, index]
    ordered_scores = np.sort(scores, axis=1)
    result["decision_margin"] = ordered_scores[:, -1] - ordered_scores[:, -2]
    return result


def predict_for_file(
    model_path: str | Path,
    input_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """Read a CSV with ``text``, append predictions, and save every row."""
    model, _ = load_model(model_path)
    source = pd.read_csv(input_path)
    if "text" not in source.columns:
        raise ValueError("Input CSV is missing required column: text")

    predictions = predict_texts(model, source["text"].tolist())
    additions = predictions.drop(columns="text")
    result = pd.concat(
        [source.reset_index(drop=True), additions.reset_index(drop=True)],
        axis=1,
    )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    return result
