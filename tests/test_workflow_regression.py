import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from finnews_sentiment.data.load_data import validate_dataset
from finnews_sentiment.features.preprocess import prepare_news_data
from finnews_sentiment.models.error_analysis import (
    analyze_predictions,
    save_error_analysis_artifacts,
    split_for_error_analysis,
)
from finnews_sentiment.models.predict import load_model, predict_for_file, predict_texts
from finnews_sentiment.models.train_improved import (
    BALANCED_LOGREG,
    BASELINE,
    LINEAR_SVC,
    TUNED_TFIDF,
    ExperimentGrids,
    build_day4_winner_pipeline,
    run_improvement_experiments,
)
from finnews_sentiment.models.train_model import (
    save_baseline_artifacts,
    train_baseline,
)


ROOT = Path(__file__).parents[1]


def workflow_fixture(rows_per_class: int = 18) -> pd.DataFrame:
    phrases = {
        "negative": "loss widened after weak demand and debt warning",
        "neutral": "company published the scheduled quarterly business report",
        "positive": "profit and revenue increased after strong sales growth",
    }
    rows = [
        {
            "text": f"<b>{phrase}</b> item {index}",
            "sentiment": sentiment,
        }
        for sentiment, phrase in phrases.items()
        for index in range(rows_per_class)
    ]
    return pd.DataFrame(rows)


def regression_grids() -> ExperimentGrids:
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
            "clf__class_weight": ["balanced"],
        },
        linear_svc={
            "clf__C": [0.5],
            "clf__class_weight": ["balanced"],
        },
    )


@pytest.mark.regression
def test_days_1_to_6_work_as_one_offline_pipeline(tmp_path: Path) -> None:
    raw = workflow_fixture()
    validate_dataset(raw)
    prepared = prepare_news_data(raw)

    assert prepared.shape == (54, 7)
    assert prepared["sentiment"].value_counts().to_dict() == {
        "negative": 18,
        "neutral": 18,
        "positive": 18,
    }

    baseline, baseline_metrics = train_baseline(prepared)
    baseline_model_path = tmp_path / "day03" / "baseline.joblib"
    baseline_metrics_path = tmp_path / "day03" / "metrics.json"
    save_baseline_artifacts(
        baseline, baseline_metrics, baseline_model_path, baseline_metrics_path
    )
    assert load_model(baseline_model_path)[1] == baseline_metrics

    _, comparison, improvement_metadata = run_improvement_experiments(
        prepared,
        cv_splits=3,
        n_jobs=1,
        grids=regression_grids(),
    )
    assert comparison["method"].tolist() == [
        BASELINE,
        TUNED_TFIDF,
        BALANCED_LOGREG,
        LINEAR_SVC,
    ]
    assert improvement_metadata["winner"] in comparison["method"].tolist()

    train, test = split_for_error_analysis(prepared, test_size=1 / 3)
    winner = build_day4_winner_pipeline().fit(
        train["text_clean"], train["sentiment"]
    )
    predictions, counts, normalized, analysis = analyze_predictions(winner, test)
    analysis_paths = save_error_analysis_artifacts(
        predictions, counts, normalized, analysis, tmp_path / "day05"
    )
    assert analysis["correct_predictions"] + analysis["errors"] == len(test)
    assert all(path.exists() for path in analysis_paths.values())

    final_model = build_day4_winner_pipeline().fit(
        prepared["text_clean"], prepared["sentiment"]
    )
    final_path = tmp_path / "day06" / "best_model.joblib"
    final_path.parent.mkdir(parents=True)
    metadata = {
        "training_rows": len(prepared),
        "fit_scope": "full_clean_dataset",
        "classes": list(final_model.classes_),
    }
    joblib.dump({"model": final_model, "meta": metadata}, final_path)

    loaded_model, loaded_metadata = load_model(final_path)
    direct = predict_texts(
        loaded_model,
        [
            "Strong sales growth increased profit.",
            "The company published its scheduled report.",
            "Weak demand widened the loss.",
        ],
    )
    assert loaded_metadata == metadata
    assert direct["pred_sentiment"].tolist() == ["positive", "neutral", "negative"]

    input_path = tmp_path / "inference.csv"
    output_path = tmp_path / "predictions.csv"
    pd.DataFrame(
        {"id": [1, 2, 3], "text": direct["text"], "source": ["a", "b", "c"]}
    ).to_csv(input_path, index=False)
    saved = predict_for_file(final_path, input_path, output_path)
    assert saved["id"].tolist() == [1, 2, 3]
    assert saved["source"].tolist() == ["a", "b", "c"]
    assert saved["pred_sentiment"].tolist() == direct["pred_sentiment"].tolist()


@pytest.mark.regression
def test_published_metrics_and_reports_are_consistent() -> None:
    baseline = json.loads(
        (ROOT / "reports/day03/baseline_metrics.json").read_text(encoding="utf-8")
    )
    improvement = json.loads(
        (ROOT / "reports/day04/improvement_metrics.json").read_text(encoding="utf-8")
    )
    comparison = pd.read_csv(ROOT / "reports/day04/model_comparison.csv")
    analysis = json.loads(
        (ROOT / "reports/day05/error_analysis.json").read_text(encoding="utf-8")
    )
    confusion = pd.read_csv(ROOT / "reports/day05/confusion_matrix.csv", index_col=0)
    errors = pd.read_csv(ROOT / "reports/day05/error_examples.csv")
    baseline_model, baseline_metadata = load_model(
        ROOT / "models/baseline_logreg.joblib"
    )

    assert baseline["split"]["train_rows"] + baseline["split"]["test_rows"] == 3448
    full_distribution = {
        label: baseline["split"]["train_distribution"][label]
        + baseline["split"]["test_distribution"][label]
        for label in ("negative", "neutral", "positive")
    }
    assert full_distribution == {"negative": 420, "neutral": 2141, "positive": 887}
    assert baseline["macro_f1"] == pytest.approx(0.7561150206606854)
    assert baseline_metadata == baseline
    assert set(baseline_model.classes_) == {"negative", "neutral", "positive"}
    assert improvement["baseline_macro_f1"] == pytest.approx(baseline["macro_f1"])
    assert comparison["method"].tolist() == [
        BASELINE,
        TUNED_TFIDF,
        BALANCED_LOGREG,
        LINEAR_SVC,
    ]
    selected = comparison.loc[comparison["selected"]].iloc[0]
    assert improvement["winner"] == selected["method"] == LINEAR_SVC
    assert selected["test_macro_f1"] == pytest.approx(0.8095961672159498)
    assert selected["relative_gain_percent"] > 5
    for experiment in improvement["experiments"]:
        published = comparison.loc[comparison["method"] == experiment["method"]].iloc[0]
        assert published["cv_macro_f1"] == pytest.approx(experiment["cv_macro_f1"])
        assert published["test_macro_f1"] == pytest.approx(
            experiment["test_macro_f1"]
        )
        assert bool(published["selected"]) is experiment["selected"]

    assert confusion.to_numpy().tolist() == analysis["confusion_matrix"]
    assert int(confusion.to_numpy().sum()) == analysis["rows"] == 690
    assert len(errors) == analysis["errors"] == 95
    assert (errors["actual"] != errors["predicted"]).all()
    assert analysis["error_transitions"][0] == {
        "actual": "positive",
        "predicted": "neutral",
        "count": 41,
    }


@pytest.mark.regression
def test_published_model_reproduces_demo_predictions() -> None:
    model, metadata = load_model(ROOT / "models/best_model.joblib")
    demo = pd.read_csv(ROOT / "reports/day06/demo_predictions.csv")

    reproduced = predict_texts(model, demo["text"].tolist())

    assert metadata["training_rows"] == 3448
    assert metadata["fit_scope"] == "full_clean_dataset"
    assert metadata["winner"] == LINEAR_SVC
    assert metadata["classes"] == ["negative", "neutral", "positive"]
    assert reproduced["text_clean"].tolist() == demo["text_clean"].tolist()
    assert reproduced["pred_sentiment"].tolist() == demo["pred_sentiment"].tolist()
    score_columns = [
        "score_negative",
        "score_neutral",
        "score_positive",
        "decision_margin",
    ]
    assert np.allclose(reproduced[score_columns], demo[score_columns])
