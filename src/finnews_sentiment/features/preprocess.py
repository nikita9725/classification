"""Reusable text preprocessing for the sentiment-classification pipeline."""

import html
import re
import unicodedata

import pandas as pd


EXPECTED_SENTIMENTS = ("negative", "neutral", "positive")
LABEL_MAPPING = {"negative": 0, "neutral": 1, "positive": 2}

HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str | None) -> str:
    """Return a conservative, deterministic normalization of one text value.

    Financial tokens such as numbers, currency symbols, percent signs, and
    punctuation are intentionally preserved for downstream vectorization.
    """
    if not isinstance(text, str):
        return ""

    text = html.unescape(text)
    text = HTML_TAG_PATTERN.sub(" ", text)
    text = unicodedata.normalize("NFKC", text).lower()
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def clean_news_data(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Return a copy of ``df`` with the normalized ``text_clean`` column."""
    if text_col not in df.columns:
        raise ValueError(f"Dataset is missing required column: {text_col}")

    result = df.copy()
    result["text_clean"] = result[text_col].map(clean_text)
    return result


def add_basic_features(df: pd.DataFrame, text_col: str = "text_clean") -> pd.DataFrame:
    """Add simple, interpretable features calculated from normalized text."""
    if text_col not in df.columns:
        raise ValueError(f"Dataset is missing required column: {text_col}")

    result = df.copy()
    text = result[text_col].fillna("").astype(str)
    result["word_count_clean"] = text.str.split().str.len()
    result["char_count"] = text.str.len()
    result["dollar_count"] = text.str.count(r"\$")
    return result


def prepare_news_data(
    df: pd.DataFrame,
    text_col: str = "text",
    target_col: str = "sentiment",
) -> pd.DataFrame:
    """Validate, clean, de-duplicate, and enrich a labelled news dataset.

    Rows with missing or empty text are removed. Labels are normalized and
    encoded with :data:`LABEL_MAPPING`. Unknown or missing labels, and
    conflicting labels for the same normalized text, are treated as data
    quality errors rather than silently discarded.
    """
    required = {text_col, target_col}
    missing = required.difference(df.columns)
    if missing:
        columns = ", ".join(sorted(missing))
        raise ValueError(f"Dataset is missing required columns: {columns}")

    result = clean_news_data(df, text_col=text_col)
    result[target_col] = result[target_col].astype("string").str.strip().str.lower()

    missing_labels = result[target_col].isna() | result[target_col].eq("")
    if missing_labels.any():
        raise ValueError(
            f"Dataset contains {int(missing_labels.sum())} missing sentiment label(s)"
        )

    unknown = sorted(set(result.loc[~missing_labels, target_col]) - set(EXPECTED_SENTIMENTS))
    if unknown:
        raise ValueError("Dataset contains unknown sentiments: " + ", ".join(unknown))

    result = result.loc[result["text_clean"].ne("")].copy()

    label_counts = result.groupby("text_clean", sort=False)[target_col].nunique()
    conflicting_texts = label_counts[label_counts > 1].index
    if len(conflicting_texts):
        raise ValueError(
            "Dataset contains conflicting sentiments for "
            f"{len(conflicting_texts)} normalized text(s)"
        )

    result = result.drop_duplicates(subset=["text_clean", target_col], keep="first")
    result["label"] = result[target_col].map(LABEL_MAPPING).astype("int64")
    result = add_basic_features(result, text_col="text_clean")
    return result.reset_index(drop=True)
