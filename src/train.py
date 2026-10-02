#!/usr/bin/env python
"""Leakage-safe training and model comparison for hospital readmission."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.base import clone

from src.preprocessing import ReadmissionFeatureEngineer


DATA_PATH = Path("data/diabetic_data.csv")
MODEL_PATH = Path("model/model.bin")
METRICS_PATH = Path("model/metrics.json")

RANDOM_STATE = 42
CV_FOLDS = 10


def load_raw_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def prepare_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if "readmitted" not in df.columns:
        raise ValueError("Expected 'readmitted' target column.")

    y = (df["readmitted"] == "<30").astype(int)
    return df.drop(columns=["readmitted"]), y


def patient_level_split(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """Create approximately 60/20/20 patient-disjoint splits."""

    if "patient_nbr" not in X.columns:
        raise ValueError("patient_nbr is required for grouped splitting.")

    groups = X["patient_nbr"]

    # Five stratified groups gives an approximately 20% patient-level test set.
    outer_cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    train_val_idx, test_idx = next(
        outer_cv.split(X, y, groups=groups)
    )

    X_train_val = X.iloc[train_val_idx].copy()
    y_train_val = y.iloc[train_val_idx].copy()

    # Four groups on the remaining 80% gives an approximately 20% validation
    # set overall.
    inner_cv = StratifiedGroupKFold(
        n_splits=4,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    inner_groups = X_train_val["patient_nbr"]

    train_idx, val_idx = next(
        inner_cv.split(
            X_train_val,
            y_train_val,
            groups=inner_groups,
        )
    )

    X_train = X_train_val.iloc[train_idx].copy()
    X_val = X_train_val.iloc[val_idx].copy()
    X_test = X.iloc[test_idx].copy()

    y_train = y_train_val.iloc[train_idx].copy()
    y_val = y_train_val.iloc[val_idx].copy()
    y_test = y.iloc[test_idx].copy()

    train_patients = set(X_train["patient_nbr"])
    val_patients = set(X_val["patient_nbr"])
    test_patients = set(X_test["patient_nbr"])

    assert train_patients.isdisjoint(val_patients)
    assert train_patients.isdisjoint(test_patients)
    assert val_patients.isdisjoint(test_patients)

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )


def _base_steps(*, scale: bool) -> list[tuple[str, object]]:
    steps: list[tuple[str, object]] = [
        ("features", ReadmissionFeatureEngineer()),
        ("vectorizer", DictVectorizer()),
    ]

    if scale:
        # Sparse-safe standardization for logistic regression. Tree models do
        # not need feature scaling.
        steps.append(
            ("scaler", StandardScaler(with_mean=False))
        )

    steps.append(
        (
            "smote",
            SMOTE(
                random_state=RANDOM_STATE,
                k_neighbors=5,
            ),
        )
    )

    return steps


def build_model_searches() -> dict[str, tuple[Pipeline, dict]]:
    """Return notebook-inspired model families and small tuning grids."""

    searches: dict[str, tuple[Pipeline, dict]] = {}

    searches["logistic_regression"] = (
        Pipeline(
            steps=_base_steps(scale=True)
            + [
                (
                    "model",
                    LogisticRegression(
                        penalty="l2",
                        solver="liblinear",
                        max_iter=1000,
                        random_state=RANDOM_STATE,
                    ),
                )
            ]
        ),
        {
            "model__C": [0.1, 1.0],
        },
    )

    searches["decision_tree"] = (
        Pipeline(
            steps=_base_steps(scale=False)
            + [
                (
                    "model",
                    DecisionTreeClassifier(
                        random_state=RANDOM_STATE,
                    ),
                )
            ]
        ),
        {
            "model__criterion": ["gini", "entropy"],
            "model__max_depth": [15, 25],
            "model__min_samples_leaf": [5],
        },
    )

    searches["random_forest"] = (
        Pipeline(
            steps=_base_steps(scale=False)
            + [
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_features="sqrt",
                        n_jobs=1,
                        random_state=RANDOM_STATE,
                    ),
                )
            ]
        ),
        {
            "model__max_depth": [15, 25],
            "model__min_samples_leaf": [3],
        },
    )

    searches["xgboost"] = (
        Pipeline(
            steps=_base_steps(scale=False)
            + [
                (
                    "model",
                    XGBClassifier(
                        objective="binary:logistic",
                        eval_metric="logloss",
                        n_estimators=300,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        n_jobs=1,
                        random_state=RANDOM_STATE,
                    ),
                )
            ]
        ),
        {
            "model__max_depth": [4, 6],
        },
    )

    return searches


def build_cv() -> StratifiedGroupKFold:
    return StratifiedGroupKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


def tune_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[str, Pipeline, dict, dict]:
    """Compare the notebook's model families using leakage-safe grouped CV."""

    results: dict = {}
    best_name = ""
    best_score = -np.inf
    best_estimator = None
    best_params = None

    for name, (pipeline, param_grid) in build_model_searches().items():
        print(f"\n===== {name} =====")

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=build_cv(),
            n_jobs=-1,
            refit=True,
            return_train_score=False,
        )

        search.fit(
            X_train.to_dict(orient="records"),
            y_train.to_numpy(),
            groups=X_train["patient_nbr"].to_numpy(),
        )

        score = float(search.best_score_)

        results[name] = {
            "cv_roc_auc": score,
            "best_params": search.best_params_,
        }

        print(f"Best CV ROC-AUC: {score:.4f}")
        print(f"Best parameters: {search.best_params_}")

        if score > best_score:
            best_score = score
            best_name = name
            best_estimator = search.best_estimator_
            best_params = search.best_params_

    assert best_estimator is not None
    assert best_params is not None

    return (
        best_name,
        best_estimator,
        best_params,
        results,
    )


def choose_threshold(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> float:
    """Choose threshold on validation data only."""

    best_threshold = 0.50
    best_f1 = -1.0

    for threshold in np.arange(0.05, 0.51, 0.01):
        predictions = (probabilities >= threshold).astype(int)
        score = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = float(round(threshold, 2))

    return best_threshold


def evaluate(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> dict:
    predictions = (probabilities >= threshold).astype(int)

    return {
        "threshold": threshold,
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            predictions,
        ).tolist(),
        "positive_rate": float(
            np.mean(predictions)
        ),
    }


def main() -> None:
    df = load_raw_data()
    X, y = prepare_target(df)

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = patient_level_split(X, y)

    print(
        "Split sizes:",
        f"train={len(X_train)}",
        f"val={len(X_val)}",
        f"test={len(X_test)}",
    )
    print(
        "Positive rates:",
        f"train={y_train.mean():.4f}",
        f"val={y_val.mean():.4f}",
        f"test={y_test.mean():.4f}",
    )

    # Model selection happens only inside the training patients.
    (
        best_name,
        best_cv_estimator,
        best_params,
        cv_results,
    ) = tune_models(X_train, y_train)

    # The CV-refit estimator has only seen training patients. Use the separate
    # validation patients solely to choose the probability threshold.
    val_probabilities = best_cv_estimator.predict_proba(
        X_val.to_dict(orient="records")
    )[:, 1]

    threshold = choose_threshold(
        y_val,
        val_probabilities,
    )

    print(
        f"\nSelected model: {best_name}"
    )
    print(
        f"Validation threshold: {threshold:.2f}"
    )

    # Refit a fresh copy on train + validation only, after every modeling
    # decision has been made.
    final_pipeline = clone(best_cv_estimator)

    X_train_final = pd.concat(
        [X_train, X_val],
        axis=0,
    )
    y_train_final = pd.concat(
        [y_train, y_val],
        axis=0,
    )

    final_pipeline.fit(
        X_train_final.to_dict(
            orient="records"
        ),
        y_train_final.to_numpy(),
    )

    # The test patients have not been used for model selection, threshold
    # selection, or hyperparameter tuning.
    test_probabilities = final_pipeline.predict_proba(
        X_test.to_dict(orient="records")
    )[:, 1]

    test_metrics = evaluate(
        y_test,
        test_probabilities,
        threshold,
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with MODEL_PATH.open("wb") as f_out:
        pickle.dump(
            {
                "pipeline": final_pipeline,
                "threshold": threshold,
                "model_name": best_name,
                "target_definition": "readmitted_within_30_days",
            },
            f_out,
        )

    metrics = {
        "methodology": {
            "target": "<30 vs >30/NO",
            "outer_split": "patient-disjoint stratified group split",
            "cv": "10-fold StratifiedGroupKFold",
            "smote": "inside each CV training fold only",
            "model_selection_metric": "roc_auc",
            "threshold_metric": "f1 on validation set",
            "final_test": "untouched until final evaluation",
        },
        "split": {
            "train_rows": len(X_train),
            "validation_rows": len(X_val),
            "test_rows": len(X_test),
            "train_patients": int(
                X_train["patient_nbr"].nunique()
            ),
            "validation_patients": int(
                X_val["patient_nbr"].nunique()
            ),
            "test_patients": int(
                X_test["patient_nbr"].nunique()
            ),
        },
        "model_comparison": cv_results,
        "selected_model": best_name,
        "selected_model_cv_roc_auc": cv_results[
            best_name
        ]["cv_roc_auc"],
        "validation_threshold": threshold,
        "test": test_metrics,
    }

    with METRICS_PATH.open(
        "w",
        encoding="utf-8",
    ) as f_out:
        json.dump(
            metrics,
            f_out,
            indent=2,
        )

    print("\nFinal untouched-test metrics:")
    print(json.dumps(test_metrics, indent=2))
    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
