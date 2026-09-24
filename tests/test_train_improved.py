import json
from pathlib import Path

import joblib
import pandas as pd

from finnews_sentiment.models.train_improved import (
    BALANCED_LOGREG,
    BASELINE,
    LINEAR_SVC,
    TUNED_TFIDF,
    ExperimentGrids,
    run_improvement_experiments,
    save_improvement_artifacts,
    select_winner,
)


def make_training_data() -> pd.DataFrame:
    rows = []
    examples = {
        "negative": "loss declined warning debt risk",
        "neutral": "company meeting report unchanged statement",
        "positive": "profit growth gain record improved",
    }
    for sentiment, phrase in examples.items():
        for index in range(15):
            rows.append({"text_clean": f"{phrase} item {index}", "sentiment": sentiment})
    return pd.DataFrame(rows)


def tiny_grids() -> ExperimentGrids:
    return ExperimentGrids(
        vectorizer={
            "tfidf__max_features": [5000],
            "tfidf__ngram_range": [(1, 1)],
            "tfidf__min_df": [1],
            "tfidf__sublinear_tf": [True],
            "clf__C": [1.0],
        },
        balanced_logreg={
            "clf__C": [1.0],
            "clf__class_weight": [None, "balanced"],
        },
        linear_svc={
            "clf__C": [0.5],
            "clf__class_weight": [None, "balanced"],
        },
    )


def test_default_grids_cover_three_improvement_hypotheses() -> None:
    grids = ExperimentGrids()

    assert grids.vectorizer["tfidf__ngram_range"] == [(1, 1), (1, 2), (1, 3)]
    assert grids.vectorizer["tfidf__max_features"] == [5000, None]
    assert grids.balanced_logreg["clf__class_weight"] == [None, "balanced"]
    assert grids.linear_svc["clf__C"] == [0.1, 0.5, 1.0, 2.0, 5.0]


def test_select_winner_uses_cv_and_simple_model_tie_break() -> None:
    experiments = [
        {"method": BASELINE, "cv_macro_f1": 0.70, "test_macro_f1": 0.99},
        {"method": TUNED_TFIDF, "cv_macro_f1": 0.80, "test_macro_f1": 0.75},
        {"method": BALANCED_LOGREG, "cv_macro_f1": 0.80005, "test_macro_f1": 0.90},
        {"method": LINEAR_SVC, "cv_macro_f1": 0.79, "test_macro_f1": 0.95},
    ]

    assert select_winner(experiments) == TUNED_TFIDF


def test_experiments_are_reproducible_and_report_all_models() -> None:
    source = make_training_data()

    model, comparison, metadata = run_improvement_experiments(
        source, cv_splits=3, n_jobs=1, grids=tiny_grids()
    )
    _, repeated_comparison, repeated_metadata = run_improvement_experiments(
        source, cv_splits=3, n_jobs=1, grids=tiny_grids()
    )

    assert list(comparison["method"]) == [
        BASELINE,
        TUNED_TFIDF,
        BALANCED_LOGREG,
        LINEAR_SVC,
    ]
    assert comparison.equals(repeated_comparison)
    assert metadata == repeated_metadata
    assert metadata["winner"] in comparison["method"].tolist()
    assert comparison["selected"].sum() == 1
    assert metadata["target_macro_f1"] == metadata["baseline_macro_f1"] * 1.05
    assert set(metadata["experiments"][0]["classification_report"]) >= {
        "negative",
        "neutral",
        "positive",
        "macro avg",
    }
    assert model.named_steps["tfidf"].vocabulary_


def test_save_improvement_artifacts_round_trips(tmp_path: Path) -> None:
    model, comparison, metadata = run_improvement_experiments(
        make_training_data(), cv_splits=3, n_jobs=1, grids=tiny_grids()
    )
    model_path = tmp_path / "best_model.joblib"
    comparison_path = tmp_path / "comparison.csv"
    metrics_path = tmp_path / "metrics.json"

    save_improvement_artifacts(
        model,
        comparison,
        metadata,
        model_path,
        comparison_path,
        metrics_path,
    )

    bundle = joblib.load(model_path)
    assert bundle["meta"] == metadata
    assert bundle["model"].predict(["profit growth record"])[0] == "positive"
    assert pd.read_csv(comparison_path).shape == comparison.shape
    assert json.loads(metrics_path.read_text(encoding="utf-8")) == metadata
