"""Download and validate the raw Financial PhraseBank dataset."""

from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from huggingface_hub import hf_hub_download


DATASET_NAME = "takala/financial_phrasebank"
DATA_REPOSITORY = "financial_phrasebank"
ARCHIVE_PATH = "data/FinancialPhraseBank-v1.0.zip"
DEFAULT_CONFIG = "sentences_75agree"
EXPECTED_SENTIMENTS = ("negative", "neutral", "positive")
CONFIG_FILES = {
    "sentences_50agree": "Sentences_50Agree.txt",
    "sentences_66agree": "Sentences_66Agree.txt",
    "sentences_75agree": "Sentences_75Agree.txt",
    "sentences_allagree": "Sentences_AllAgree.txt",
}


def validate_dataset(df: pd.DataFrame) -> None:
    """Validate the stable schema used by the rest of the project."""
    required = {"text", "sentiment"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError("Dataset is missing required columns: " + ", ".join(sorted(missing)))
    if df.empty:
        raise ValueError("Dataset is empty")

    actual = set(df["sentiment"].dropna().astype(str).unique())
    unknown = actual.difference(EXPECTED_SENTIMENTS)
    if unknown:
        raise ValueError("Dataset contains unknown sentiments: " + ", ".join(sorted(unknown)))
    missing_classes = set(EXPECTED_SENTIMENTS).difference(actual)
    if missing_classes:
        raise ValueError(
            "Dataset does not contain all expected sentiments: "
            + ", ".join(sorted(missing_classes))
        )


def load_financial_phrasebank(dataset_config: str = DEFAULT_CONFIG) -> pd.DataFrame:
    """Download one Financial PhraseBank configuration from Hugging Face."""
    if dataset_config not in CONFIG_FILES:
        supported = ", ".join(sorted(CONFIG_FILES))
        raise ValueError(f"Unsupported dataset config '{dataset_config}'. Choose one of: {supported}")

    archive_path = hf_hub_download(
        repo_id=DATA_REPOSITORY,
        repo_type="dataset",
        filename=ARCHIVE_PATH,
    )
    member = f"FinancialPhraseBank-v1.0/{CONFIG_FILES[dataset_config]}"
    with ZipFile(archive_path) as archive, archive.open(member) as source_file:
        lines = (line.decode("iso-8859-1") for line in source_file)
        rows = [line.rstrip("\r\n").rsplit("@", 1) for line in lines]

    malformed = [row for row in rows if len(row) != 2]
    if malformed:
        raise ValueError(f"Source dataset contains {len(malformed)} malformed rows")
    result = pd.DataFrame(rows, columns=["text", "sentiment"], dtype="string")
    validate_dataset(result)
    return result


def save_processed(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame as UTF-8 CSV, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
