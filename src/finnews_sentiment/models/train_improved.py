"""Model-comparison helpers for the Day 4 experiments.

The test split is used only for reporting. All model and hyperparameter
selection happens with stratified cross-validation on the training split.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from finnews_sentiment.features.preprocess import EXPECTED_SENTIMENTS
from finnews_sentiment.models.train_model import (
    build_baseline_pipeline,
    validate_training_data,
)


BASELINE = "baseline_logreg"
TUNED_TFIDF = "tuned_tfidf_logreg"
BALANCED_LOGREG = "balanced_logreg"
LINEAR_SVC = "linear_svc"
MODEL_PRIORITY = (BASELINE, TUNED_TFIDF, BALANCED_LOGREG, LINEAR_SVC)


@dataclass(frozen=True)
class ExperimentGrids:
    """Search spaces used by the three Day 4 hypotheses.

    Supplying smaller grids is useful for fast unit tests. The defaults are
    intentionally compact enough for a laptop while covering meaningful
    text-classification choices.
    """

    vectorizer: dict[str, list[Any]] = field(
        default_factory=lambda: {
            "tfidf__max_features": [5000, None],
            "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
            "tfidf__min_df": [1, 2],
            "tfidf__sublinear_tf": [False, True],
            "clf__C": [0.5, 1.0, 2.0, 5.0],
        }
    )
    balanced_logreg: dict[str, list[Any]] = field(
        default_factory=lambda: {
            "clf__C": [0.5, 1.0, 2.0, 5.0, 10.0],
            "clf__class_weight": [None, "balanced"],
        }
    )
    linear_svc: dict[str, list[Any]] = field(
        default_factory=lambda: {
            "clf__C": [0.1, 0.5, 1.0, 2.0, 5.0],
            "clf__class_weight": [None, "balanced"],
        }
    )


def _logreg_pipeline(random_state: int, vectorizer: TfidfVectorizer | None = None) -> Pipeline:
    return Pipeline(
        [
            ("tfidf", vectorizer or TfidfVectorizer()),
            (
                "clf",
                LogisticRegression(max_iter=1000, random_state=random_state),
            ),
        ]
    )


def _svc_pipeline(random_state: int, vectorizer: TfidfVectorizer) -> Pipeline:
    return Pipeline(
        [
            ("tfidf", vectorizer),
            ("clf", LinearSVC(max_iter=5000, random_state=random_state)),
        ]
    )


def _evaluate(
    name: str,
    model: Pipeline,
    X_test: pd.Series,
    y_test: pd.Series,
    cv_macro_f1: float,
    best_params: dict[str, Any],
) -> dict[str, Any]:
    predictions = model.predict(X_test)
    report = classification_report(
        y_test,
        predictions,
        labels=list(EXPECTED_SENTIMENTS),
        output_dict=True,
        zero_division=0,
    )
    return {
        "method": name,
        "cv_macro_f1": float(cv_macro_f1),
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "test_macro_f1": float(f1_score(y_test, predictions, average="macro")),
        "test_weighted_f1": float(f1_score(y_test, predictions, average="weighted")),
        "best_params": {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in best_params.items()
        },
        "classification_report": report,
    }


def _fit_search(
    pipeline: Pipeline,
    param_grid: dict[str, list[Any]],
    X_train: pd.Series,
    y_train: pd.Series,
    cv: StratifiedKFold,
    n_jobs: int,
) -> GridSearchCV:
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        error_score="raise",
    )
    search.fit(X_train, y_train)
    return search


def select_winner(
    experiments: list[dict[str, Any]],
    tolerance: float = 1e-4,
) -> str:
    """Select by CV score, preferring the simpler model for near-ties."""
    best_score = max(row["cv_macro_f1"] for row in experiments)
    near_best = {
        row["method"]
        for row in experiments
        if best_score - row["cv_macro_f1"] <= tolerance
    }
    return next(name for name in MODEL_PRIORITY if name in near_best)


def run_improvement_experiments(
    df: pd.DataFrame,
    text_col: str = "text_clean",
    target_col: str = "sentiment",
    test_size: float = 0.2,
    random_state: int = 42,
    cv_splits: int = 5,
    n_jobs: int = -1,
    grids: ExperimentGrids | None = None,
    target_relative_improvement: float = 0.05,
) -> tuple[Pipeline, pd.DataFrame, dict[str, Any]]:
    """Run baseline and three leakage-safe improvement experiments.

    Returns the selected fitted pipeline, a compact comparison table and full
    serializable metadata. The winner is selected by mean CV macro F1; test
    metrics never participate in model selection.
    """
    validate_training_data(df, text_col=text_col, target_col=target_col)
    grids = grids or ExperimentGrids()

    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col].astype(str),
        df[target_col].astype(str),
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_col],
    )
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    fitted_models: dict[str, Pipeline] = {}
    experiments: list[dict[str, Any]] = []

    baseline = build_baseline_pipeline(random_state=random_state)
    baseline_cv = cross_val_score(
        baseline,
        X_train,
        y_train,
        scoring="f1_macro",
        cv=cv,
        n_jobs=n_jobs,
    ).mean()
    baseline.fit(X_train, y_train)
    fitted_models[BASELINE] = baseline
    experiments.append(
        _evaluate(
            BASELINE,
            baseline,
            X_test,
            y_test,
            baseline_cv,
            {
                "tfidf__max_features": 5000,
                "tfidf__ngram_range": (1, 2),
                "clf__C": 1.0,
                "clf__class_weight": None,
            },
        )
    )

    vectorizer_search = _fit_search(
        _logreg_pipeline(random_state),
        grids.vectorizer,
        X_train,
        y_train,
        cv,
        n_jobs,
    )
    tuned_model = vectorizer_search.best_estimator_
    fitted_models[TUNED_TFIDF] = tuned_model
    experiments.append(
        _evaluate(
            TUNED_TFIDF,
            tuned_model,
            X_test,
            y_test,
            vectorizer_search.best_score_,
            vectorizer_search.best_params_,
        )
    )

    best_vectorizer = tuned_model.named_steps["tfidf"]
    selected_vectorizer_params = {
        key: value
        for key, value in vectorizer_search.best_params_.items()
        if key.startswith("tfidf__")
    }
    balanced_search = _fit_search(
        _logreg_pipeline(random_state, clone(best_vectorizer)),
        grids.balanced_logreg,
        X_train,
        y_train,
        cv,
        n_jobs,
    )
    balanced_model = balanced_search.best_estimator_
    fitted_models[BALANCED_LOGREG] = balanced_model
    experiments.append(
        _evaluate(
            BALANCED_LOGREG,
            balanced_model,
            X_test,
            y_test,
            balanced_search.best_score_,
            selected_vectorizer_params | balanced_search.best_params_,
        )
    )

    svc_search = _fit_search(
        _svc_pipeline(random_state, clone(best_vectorizer)),
        grids.linear_svc,
        X_train,
        y_train,
        cv,
        n_jobs,
    )
    svc_model = svc_search.best_estimator_
    fitted_models[LINEAR_SVC] = svc_model
    experiments.append(
        _evaluate(
            LINEAR_SVC,
            svc_model,
            X_test,
            y_test,
            svc_search.best_score_,
            selected_vectorizer_params | svc_search.best_params_,
        )
    )

    baseline_f1 = experiments[0]["test_macro_f1"]
    target_macro_f1 = baseline_f1 * (1 + target_relative_improvement)
    for row in experiments:
        row["absolute_gain"] = float(row["test_macro_f1"] - baseline_f1)
        row["relative_gain_percent"] = float(
            (row["test_macro_f1"] / baseline_f1 - 1) * 100
        )

    winner = select_winner(experiments)
    winner_row = next(row for row in experiments if row["method"] == winner)
    for row in experiments:
        row["selected"] = row["method"] == winner

    metadata: dict[str, Any] = {
        "selection_metric": "mean_cv_macro_f1",
        "winner": winner,
        "baseline_macro_f1": baseline_f1,
        "target_relative_improvement": target_relative_improvement,
        "target_macro_f1": target_macro_f1,
        "target_reached": winner_row["test_macro_f1"] >= target_macro_f1,
        "split": {
            "test_size": test_size,
            "random_state": random_state,
            "cv_splits": cv_splits,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "train_distribution": {
                label: int(count)
                for label, count in y_train.value_counts().sort_index().items()
            },
            "test_distribution": {
                label: int(count)
                for label, count in y_test.value_counts().sort_index().items()
            },
        },
        "experiments": experiments,
    }

    table = pd.DataFrame(
        [
            {
                key: row[key]
                for key in (
                    "method",
                    "cv_macro_f1",
                    "test_accuracy",
                    "test_macro_f1",
                    "test_weighted_f1",
                    "absolute_gain",
                    "relative_gain_percent",
                    "selected",
                )
            }
            | {"best_params": json.dumps(row["best_params"], sort_keys=True)}
            for row in experiments
        ]
    )
    return fitted_models[winner], table, metadata


def save_improvement_artifacts(
    model: Pipeline,
    comparison: pd.DataFrame,
    metadata: dict[str, Any],
    model_path: Path,
    comparison_path: Path,
    metrics_path: Path,
) -> None:
    """Persist the winning pipeline and both compact and detailed results."""
    for path in (model_path, comparison_path, metrics_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump({"model": model, "meta": metadata}, model_path)
    comparison.to_csv(comparison_path, index=False)
    metrics_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
