from pathlib import Path

import pandas as pd
import pytest

from finnews_sentiment.features.preprocess import (
    add_basic_features,
    clean_news_data,
    clean_text,
    prepare_news_data,
)


def test_clean_text_is_conservative_and_deterministic() -> None:
    raw = "  <b>PROFIT&nbsp;rose</b>\n10% to $5.00 — GREAT!  "

    assert clean_text(raw) == "profit rose 10% to $5.00 — great!"


@pytest.mark.parametrize("value", [None, 12, 3.14, pd.NA])
def test_clean_text_returns_empty_string_for_non_strings(value) -> None:
    assert clean_text(value) == ""


def test_clean_news_data_does_not_mutate_input() -> None:
    source = pd.DataFrame({"text": ["  NEWS  "]})

    result = clean_news_data(source)

    assert list(source.columns) == ["text"]
    assert result.loc[0, "text_clean"] == "news"


def test_add_basic_features_counts_normalized_text() -> None:
    source = pd.DataFrame({"text_clean": ["profit rose $5 and $10", ""]})

    result = add_basic_features(source)

    assert result[["word_count_clean", "char_count", "dollar_count"]].to_dict("records") == [
        {"word_count_clean": 5, "char_count": 22, "dollar_count": 2},
        {"word_count_clean": 0, "char_count": 0, "dollar_count": 0},
    ]


def test_prepare_news_data_cleans_filters_deduplicates_and_encodes() -> None:
    source = pd.DataFrame(
        {
            "text": [" Profit rose 10% ", "profit rose 10%", None, "<br>  ", "Loss widened"],
            "sentiment": [" Positive ", "positive", "neutral", "neutral", "NEGATIVE"],
        }
    )

    result = prepare_news_data(source)

    assert result[["text_clean", "sentiment", "label"]].to_dict("records") == [
        {"text_clean": "profit rose 10%", "sentiment": "positive", "label": 2},
        {"text_clean": "loss widened", "sentiment": "negative", "label": 0},
    ]
    assert len(source) == 5


def test_prepare_news_data_rejects_conflicting_labels() -> None:
    source = pd.DataFrame(
        {
            "text": ["Same text", " same   text "],
            "sentiment": ["positive", "negative"],
        }
    )

    with pytest.raises(ValueError, match="conflicting sentiments"):
        prepare_news_data(source)


@pytest.mark.parametrize("sentiment", [None, "", "mixed"])
def test_prepare_news_data_rejects_invalid_labels(sentiment) -> None:
    source = pd.DataFrame({"text": ["News"], "sentiment": [sentiment]})

    with pytest.raises(ValueError, match="missing sentiment|unknown sentiments"):
        prepare_news_data(source)


@pytest.mark.parametrize(
    ("source", "column"),
    [
        (pd.DataFrame({"sentiment": ["neutral"]}), "text"),
        (pd.DataFrame({"text": ["News"]}), "sentiment"),
    ],
)
def test_prepare_news_data_requires_input_columns(source: pd.DataFrame, column: str) -> None:
    with pytest.raises(ValueError, match=column):
        prepare_news_data(source)


def test_financial_phrasebank_preprocessing_result() -> None:
    data_path = Path(__file__).parents[1] / "data/raw/financial_phrasebank_75agree.csv"
    if not data_path.exists():
        pytest.skip("Day 1 dataset has not been generated")

    result = prepare_news_data(pd.read_csv(data_path, dtype="string"))

    assert result.shape == (3448, 7)
    assert result["text_clean"].ne("").all()
    assert not result.duplicated(["text_clean", "sentiment"]).any()
    assert result.groupby(["sentiment", "label"]).size().to_dict() == {
        ("negative", 0): 420,
        ("neutral", 1): 2141,
        ("positive", 2): 887,
    }
