from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from finnews_sentiment.models.predict import (
    load_model,
    predict_for_file,
    predict_texts,
)
from finnews_sentiment.models.train_improved import build_day4_winner_pipeline


@pytest.fixture()
def fitted_model():
    texts = [
        "profit growth gain record improved",
        "strong profit growth improved outlook",
        "revenue increased and profit rose",
        "company meeting report unchanged",
        "board published its regular report",
        "results remained stable unchanged",
        "loss declined warning debt risk",
        "company warned about a major loss",
        "weak demand increased debt risk",
    ]
    labels = ["positive"] * 3 + ["neutral"] * 3 + ["negative"] * 3
    return build_day4_winner_pipeline().fit(texts, labels)


def save_bundle(path: Path, model, meta: dict | None = None) -> None:
    joblib.dump({"model": model, "meta": meta or {}}, path)


def test_model_bundle_round_trip(tmp_path: Path, fitted_model) -> None:
    model_path = tmp_path / "model.joblib"
    metadata = {"training_rows": 9, "fit_scope": "full_clean_dataset"}
    save_bundle(model_path, fitted_model, metadata)

    loaded, loaded_metadata = load_model(model_path)

    assert loaded_metadata == metadata
    assert loaded.predict(["profit growth"])[0] == "positive"


@pytest.mark.parametrize(
    ("bundle", "message"),
    [
        ([], "dictionary"),
        ({"meta": {}}, "containing 'model'"),
        ({"model": object()}, "predict"),
    ],
)
def test_load_model_rejects_invalid_bundles(
    tmp_path: Path, bundle: object, message: str
) -> None:
    model_path = tmp_path / "invalid.joblib"
    joblib.dump(bundle, model_path)

    with pytest.raises((TypeError, ValueError), match=message):
        load_model(model_path)


def test_predict_texts_applies_preprocessing_and_returns_scores(fitted_model) -> None:
    result = predict_texts(
        fitted_model,
        ["  <b>PROFIT&nbsp;growth</b>  ", "company meeting report unchanged"],
    )

    assert result["text_clean"].tolist() == [
        "profit growth",
        "company meeting report unchanged",
    ]
    assert result["pred_sentiment"].tolist() == ["positive", "neutral"]
    assert list(result.columns) == [
        "text",
        "text_clean",
        "pred_sentiment",
        "score_negative",
        "score_neutral",
        "score_positive",
        "decision_margin",
    ]
    scores = result[["score_negative", "score_neutral", "score_positive"]].to_numpy()
    expected_margin = np.sort(scores, axis=1)[:, -1] - np.sort(scores, axis=1)[:, -2]
    assert np.allclose(result["decision_margin"], expected_margin)
    assert (result["decision_margin"] >= 0).all()


def test_predict_texts_accepts_one_string(fitted_model) -> None:
    result = predict_texts(fitted_model, "profit growth improved")

    assert len(result) == 1
    assert result.loc[0, "pred_sentiment"] == "positive"


@pytest.mark.parametrize(
    ("texts", "error", "message"),
    [
        ([], ValueError, "At least one"),
        (["valid", None], TypeError, "positions \\[1\\]"),
        (["<br>   "], ValueError, "empty after preprocessing"),
    ],
)
def test_predict_texts_rejects_invalid_input(
    fitted_model, texts: object, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        predict_texts(fitted_model, texts)


def test_predict_for_file_preserves_rows_duplicates_and_columns(
    tmp_path: Path, fitted_model
) -> None:
    model_path = tmp_path / "model.joblib"
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "nested" / "predictions.csv"
    save_bundle(model_path, fitted_model)
    source = pd.DataFrame(
        {
            "id": [10, 11, 12],
            "text": ["profit growth", "loss debt risk", "profit growth"],
            "source": ["wire", "web", "wire"],
        }
    )
    source.to_csv(input_path, index=False)

    result = predict_for_file(model_path, input_path, output_path)

    assert output_path.exists()
    assert result["id"].tolist() == [10, 11, 12]
    assert result["source"].tolist() == ["wire", "web", "wire"]
    assert result.loc[0, "text_clean"] == result.loc[2, "text_clean"]
    assert result.loc[0, "pred_sentiment"] == result.loc[2, "pred_sentiment"]
    assert pd.read_csv(output_path).shape == result.shape


def test_predict_for_file_requires_text_column(tmp_path: Path, fitted_model) -> None:
    model_path = tmp_path / "model.joblib"
    input_path = tmp_path / "input.csv"
    save_bundle(model_path, fitted_model)
    pd.DataFrame({"headline": ["profit growth"]}).to_csv(input_path, index=False)

    with pytest.raises(ValueError, match="required column: text"):
        predict_for_file(model_path, input_path, tmp_path / "output.csv")
