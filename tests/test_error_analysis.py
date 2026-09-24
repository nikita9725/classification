import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from finnews_sentiment.models.error_analysis import (
    analyze_predictions,
    save_error_analysis_artifacts,
    split_for_error_analysis,
)
from finnews_sentiment.models.train_improved import build_day4_winner_pipeline


def make_data(rows_per_class: int = 18) -> pd.DataFrame:
    examples = {
        "negative": "loss warning decline debt risk",
        "neutral": "company meeting report statement unchanged",
        "positive": "profit growth gain record improved",
    }
    rows = []
    for sentiment, phrase in examples.items():
        for index in range(rows_per_class):
            rows.append(
                {
                    "text": f"Original {phrase} item {index}",
                    "text_clean": f"{phrase} item {index}",
                    "sentiment": sentiment,
                    "word_count_clean": 7,
                }
            )
    return pd.DataFrame(rows)


def fit_test_model() -> tuple[object, pd.DataFrame]:
    train_df, test_df = split_for_error_analysis(make_data(), test_size=1 / 3)
    model = build_day4_winner_pipeline()
    model.fit(train_df["text_clean"], train_df["sentiment"])
    return model, test_df


def test_split_is_reproducible_stratified_and_does_not_mutate_input() -> None:
    source = make_data()
    original = source.copy(deep=True)

    train, test = split_for_error_analysis(source, test_size=1 / 3)
    repeated_train, repeated_test = split_for_error_analysis(source, test_size=1 / 3)

    assert source.equals(original)
    assert train.equals(repeated_train)
    assert test.equals(repeated_test)
    assert train["sentiment"].value_counts().to_dict() == {
        "negative": 12,
        "neutral": 12,
        "positive": 12,
    }
    assert test["sentiment"].value_counts().to_dict() == {
        "negative": 6,
        "neutral": 6,
        "positive": 6,
    }
    assert set(train["source_index"]).isdisjoint(test["source_index"])


def test_analysis_builds_consistent_matrices_scores_and_summary() -> None:
    model, test_df = fit_test_model()
    predictions, counts, normalized, summary = analyze_predictions(model, test_df)

    assert list(counts.index) == ["negative", "neutral", "positive"]
    assert list(counts.columns) == ["negative", "neutral", "positive"]
    assert counts.to_numpy().sum() == len(test_df)
    assert np.allclose(normalized.sum(axis=1), 1)
    assert (predictions["runner_up_label"] != predictions["predicted"]).all()
    assert (predictions["decision_margin"] >= 0).all()
    assert np.allclose(
        predictions["decision_margin"],
        predictions["predicted_score"] - predictions["runner_up_score"],
    )
    assert summary["rows"] == len(test_df)
    assert summary["correct_predictions"] + summary["errors"] == len(test_df)
    json.dumps(summary)


def test_artifacts_contain_only_errors_and_round_trip(tmp_path: Path) -> None:
    model, test_df = fit_test_model()
    first_prediction = model.predict(test_df.loc[[0], "text_clean"])[0]
    alternatives = [label for label in ("negative", "neutral", "positive") if label != first_prediction]
    test_df.loc[0, "sentiment"] = alternatives[0]
    predictions, counts, normalized, summary = analyze_predictions(model, test_df)

    paths = save_error_analysis_artifacts(
        predictions, counts, normalized, summary, tmp_path / "day05"
    )

    assert all(path.exists() for path in paths.values())
    saved_errors = pd.read_csv(paths["errors_csv"])
    assert not saved_errors.empty
    assert (saved_errors["actual"] != saved_errors["predicted"]).all()
    assert pd.read_csv(paths["confusion_csv"], index_col=0).shape == (3, 3)
    assert json.loads(paths["summary_json"].read_text(encoding="utf-8")) == summary
    assert paths["figure"].stat().st_size > 0


@pytest.mark.parametrize("test_size", [0, 1, -0.1])
def test_split_rejects_invalid_test_size(test_size: float) -> None:
    with pytest.raises(ValueError, match="test_size"):
        split_for_error_analysis(make_data(), test_size=test_size)


def test_analysis_requires_split_frame_and_decision_scores() -> None:
    model, test_df = fit_test_model()
    with pytest.raises(ValueError, match="source_index"):
        analyze_predictions(model, test_df.drop(columns="source_index"))

    class PredictOnly:
        def predict(self, values):
            return np.repeat("neutral", len(values))

    with pytest.raises(TypeError, match="decision_function"):
        analyze_predictions(PredictOnly(), test_df)


def test_financial_phrasebank_error_analysis_matches_day4() -> None:
    data_path = Path(__file__).parents[1] / "data/processed/clean_news.csv"
    if not data_path.exists():
        pytest.skip("Prepared Financial PhraseBank data is not available")

    source = pd.read_csv(data_path)
    train_df, test_df = split_for_error_analysis(source)
    model = build_day4_winner_pipeline()
    model.fit(train_df["text_clean"], train_df["sentiment"])
    _, counts, _, summary = analyze_predictions(model, test_df)

    assert summary["rows"] == 690
    assert summary["errors"] == 95
    assert summary["metrics"]["macro_f1"] == pytest.approx(0.8095961672159498)
    assert counts.to_numpy().tolist() == [
        [54, 16, 14],
        [2, 410, 16],
        [6, 41, 131],
    ]
