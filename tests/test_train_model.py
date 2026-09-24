import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from finnews_sentiment.models.train_model import (
    build_baseline_pipeline,
    save_baseline_artifacts,
    train_baseline,
    validate_training_data,
)


def make_training_data() -> pd.DataFrame:
    rows = []
    examples = {
        "negative": "loss declined warning debt",
        "neutral": "company meeting report unchanged",
        "positive": "profit growth gain record",
    }
    for sentiment, phrase in examples.items():
        for index in range(10):
            rows.append({"text_clean": f"{phrase} item {index}", "sentiment": sentiment})
    return pd.DataFrame(rows)


def test_build_baseline_pipeline_uses_fixed_baseline_parameters() -> None:
    pipeline = build_baseline_pipeline()

    assert pipeline.named_steps["tfidf"].max_features == 5000
    assert pipeline.named_steps["tfidf"].ngram_range == (1, 2)
    assert pipeline.named_steps["clf"].max_iter == 200
    assert pipeline.named_steps["clf"].random_state == 42


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (pd.DataFrame(), "required columns"),
        (pd.DataFrame({"text_clean": ["news"], "sentiment": ["positive"]}), "all expected"),
        (
            pd.DataFrame(
                {
                    "text_clean": ["", "flat", "gain"],
                    "sentiment": ["negative", "neutral", "positive"],
                }
            ),
            "empty text",
        ),
    ],
)
def test_validate_training_data_rejects_invalid_input(
    source: pd.DataFrame, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_training_data(source)


def test_train_baseline_returns_reproducible_metrics() -> None:
    source = make_training_data()

    model, metrics = train_baseline(source)
    _, repeated_metrics = train_baseline(source)

    assert model.predict(["profit growth record"])[0] == "positive"
    assert metrics == repeated_metrics
    assert metrics["split"]["train_rows"] == 24
    assert metrics["split"]["test_rows"] == 6
    assert metrics["split"]["test_distribution"] == {
        "negative": 2,
        "neutral": 2,
        "positive": 2,
    }
    assert set(metrics["classification_report"]) >= {
        "negative",
        "neutral",
        "positive",
        "macro avg",
        "weighted avg",
    }


def test_save_baseline_artifacts_round_trips_bundle(tmp_path: Path) -> None:
    model, metrics = train_baseline(make_training_data())
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"

    save_baseline_artifacts(model, metrics, model_path, metrics_path)

    bundle = joblib.load(model_path)
    assert bundle["meta"] == metrics
    assert bundle["model"].predict(["loss debt warning"])[0] == "negative"
    assert json.loads(metrics_path.read_text(encoding="utf-8")) == metrics
