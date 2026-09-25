from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from finnews_sentiment.visualization.plots import (
    plot_class_distribution,
    plot_text_length_distribution,
)


def test_plot_class_distribution_builds_and_saves_figure(tmp_path: Path) -> None:
    destination = tmp_path / "plots" / "classes.png"

    figure = plot_class_distribution(
        pd.Series([2, 5, 3], index=["negative", "neutral", "positive"]),
        destination,
    )

    try:
        assert destination.stat().st_size > 0
        assert figure.axes[0].get_xlabel() == "Тональность"
        assert [text.get_text() for text in figure.axes[0].texts] == ["2", "5", "3"]
    finally:
        plt.close(figure)


def test_plot_text_length_distribution_builds_and_saves_figure(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "plots" / "lengths.png"

    figure = plot_text_length_distribution(
        pd.Series([4, 7, 7, 10, 12, None]), destination, bins=4
    )

    try:
        assert destination.stat().st_size > 0
        assert figure.axes[0].get_xlabel() == "Количество слов в предложении"
        assert figure.axes[0].patches
    finally:
        plt.close(figure)
