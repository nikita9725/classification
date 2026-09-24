"""Reusable plots for exploratory data analysis."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure


def plot_class_distribution(
    class_counts: pd.Series,
    output_path: Path | None = None,
) -> Figure:
    """Build a class-count bar chart and optionally save it as PNG."""
    plot_data = class_counts.rename_axis("sentiment").reset_index(name="count")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=plot_data,
        x="sentiment",
        y="count",
        hue="sentiment",
        legend=False,
        ax=ax,
    )
    ax.set(
        title="Financial PhraseBank: распределение классов",
        xlabel="Тональность",
        ylabel="Количество строк",
    )
    for index, count in enumerate(class_counts):
        ax.text(index, count, str(count), ha="center", va="bottom")
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    return fig


def plot_text_length_distribution(
    word_counts: pd.Series,
    output_path: Path | None = None,
    bins: int = 50,
) -> Figure:
    """Build a word-count histogram and optionally save it as PNG."""
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(word_counts.dropna(), bins=bins, kde=True, ax=ax)
    ax.set(
        title="Financial PhraseBank: распределение длин текстов",
        xlabel="Количество слов в предложении",
        ylabel="Количество строк",
    )
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    return fig
