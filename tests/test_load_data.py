from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import pytest

from finnews_sentiment.data import load_data
from finnews_sentiment.data.load_data import (
    load_financial_phrasebank,
    save_processed,
    validate_dataset,
)


def valid_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": ["Loss widened", "The board met", "Profit increased"],
            "sentiment": ["negative", "neutral", "positive"],
        }
    )


def test_validate_dataset_accepts_expected_schema() -> None:
    validate_dataset(valid_dataset())


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (pd.DataFrame(), "missing required columns"),
        (pd.DataFrame(columns=["text", "sentiment"]), "empty"),
        (
            pd.DataFrame(
                {
                    "text": ["loss", "meeting", "gain", "mixed"],
                    "sentiment": ["negative", "neutral", "positive", "mixed"],
                }
            ),
            "unknown sentiments: mixed",
        ),
        (
            pd.DataFrame(
                {"text": ["loss", "gain"], "sentiment": ["negative", "positive"]}
            ),
            "does not contain all expected sentiments: neutral",
        ),
    ],
)
def test_validate_dataset_rejects_invalid_data(
    source: pd.DataFrame, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_dataset(source)


def test_load_financial_phrasebank_parses_local_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_path = tmp_path / "phrasebank.zip"
    member = "FinancialPhraseBank-v1.0/Sentences_75Agree.txt"
    source = (
        "Loss widened@negative\n"
        "The board met at 10@neutral\n"
        "Revenue rose after guidance@positive\n"
    ).encode("iso-8859-1")
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(member, source)

    monkeypatch.setattr(load_data, "hf_hub_download", lambda **_: str(archive_path))

    result = load_financial_phrasebank("sentences_75agree")

    assert result.to_dict("records") == [
        {"text": "Loss widened", "sentiment": "negative"},
        {"text": "The board met at 10", "sentiment": "neutral"},
        {"text": "Revenue rose after guidance", "sentiment": "positive"},
    ]
    assert all(str(dtype) == "string" for dtype in result.dtypes)


def test_load_financial_phrasebank_rejects_unknown_config() -> None:
    with pytest.raises(ValueError, match="Unsupported dataset config"):
        load_financial_phrasebank("sentences_100agree")


def test_save_processed_creates_parent_and_round_trips(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "dataset.csv"

    save_processed(valid_dataset(), destination)

    assert pd.read_csv(destination).to_dict("records") == valid_dataset().to_dict(
        "records"
    )
