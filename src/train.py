#!/usr/bin/env python
"""Leakage-safe model comparison and production training."""

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
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    cross_validate,
)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.preprocessing import ReadmissionFeatureEngineer


DATA_PATH = Path("data/diabetic_data.csv")
MODEL_PATH = Path("model/model.bin")
METRICS_PATH = Path("model/metrics.json")

RANDOM_STATE = 42
CV_FOLDS = 10


def load_raw_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def prepare_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    if "readmitted" not in df.columns:
        raise ValueError("Expected 'readmitted' target column.")

    y = (df["readmitted"] == "<30").astype(int)
    return df.drop(columns=["readmitted"]), y


def patient_level_split(
    X: pd.DataFrame,
    y: pd.Series,
):
    """Create patient-disjoint 60/20/20 train/validation/test sets."""

    if "patient_nbr" not in X.columns:
        raise ValueError("patient_nbr is required for grouped splitting.")

    groups = X["patient_nbr"]

    outer = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )
    train_val_idx, test_idx = next(
        outer.split(X, y, groups=groups)
    )

    X_train_val = X.iloc[train_val_idx].copy()
    y_train_val = y.iloc[train_val_idx].copy()

    inner = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=RANDOM_STATE,
    )
    inner_groups = X_train_val["patient_nbr"]

    train_idx, val_idx = next(
        inner.split(
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


def _base_steps() -> list[tuple[str, object]]:
    """Common preprocessing and fold-local SMOTE."""

    return [
        ("features", ReadmissionFeatureEngineer()),
        ("vectorizer", DictVectorizer()),
        (
            "smote",
            SMOTE(
                random_state=RANDOM_STATE,
                k_neighbors=5,
            ),
        ),
    ]


def build_model_pipelines() -> dict[str, Pipeline]:
    """
    Candidate model families from the original notebook.

    SMOTE is inside every pipeline, so cross-validation generates synthetic
    observations only from each fold's training portion.
    """

    return {
        "logistic_regression": Pipeline(
            steps=[
                *_base_steps(),
                ("scaler", StandardScaler(with_mean=False)),
                (
                    "model",
                    LogisticRegression(
                        penalty="l1",
                        solver="liblinear",
                        C=1.0,
                        max_iter=2000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "decision_tree": Pipeline(
            steps=[
                *_base_steps(),
                (
                    "model",
                    DecisionTreeClassifier(
                        max_depth=28,
                        min_samples_split=10,
                        criterion="gini",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                *_base_steps(),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=25,
                        min_samples_split=10,
                        criterion="gini",
                        max_features="sqrt",
                        n_jobs=1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "xgboost": Pipeline(
            steps=[
                *_base_steps(),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=500,
                        max_depth=8,
                        learning_rate=0.05,
                        colsample_bytree=0.9,
                        subsample=0.8,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        tree_method="hist",
                        n_jobs=1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def evaluate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[dict[str, Pipeline], dict[str, dict]]:
    """Compare models using patient-grouped 10-fold CV."""

    cv = StratifiedGroupKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "roc_auc": "roc_auc",
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }

    pipelines = build_model_pipelines()
    results: dict[str, dict] = {}

    records = X_train.to_dict(orient="records")
    groups = X_train["patient_nbr"].to_numpy()
    y_array = y_train.to_numpy()

    for name, pipeline in pipelines.items():
        print(f"\nCross-validating {name}...")

        scores = cross_validate(
            pipeline,
            records,
            y_array,
            groups=groups,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )

        results[name] = {
            metric: {
                "mean": float(np.mean(scores[f"test_{metric}"])),
                "std": float(np.std(scores[f"test_{metric}"])),
            }
            for metric in scoring
        }

        print(
            f"{name}: "
            f"ROC-AUC={results[name]['roc_auc']['mean']:.4f} "
            f"+/- {results[name]['roc_auc']['std']:.4f}"
        )

    return pipelines, results


def choose_model(
    cv_results: dict[str, dict],
) -> str:
    """Select the candidate with the highest mean CV ROC-AUC."""

    return max(
        cv_results,
        key=lambda name: cv_results[name]["roc_auc"]["mean"],
    )


def choose_threshold(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> float:
    """Select threshold using validation data only."""

    best_threshold = 0.50
    best_f1 = -1.0

    for threshold in np.arange(0.05, 0.51, 0.01):
        predictions = (
            probabilities >= threshold
        ).astype(int)

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
    predictions = (
        probabilities >= threshold
    ).astype(int)

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
        "positive_rate": float(np.mean(predictions)),
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
        "Patient-disjoint split:",
        f"train={len(X_train)}",
        f"val={len(X_val)}",
        f"test={len(X_test)}",
    )

    # CV happens only on training patients.
    pipelines, cv_results = evaluate_models(
        X_train,
        y_train,
    )

    selected_model_name = choose_model(cv_results)

    print(
        f"\nSelected model by CV ROC-AUC: "
        f"{selected_model_name}"
    )

    # Fit selected model on training patients only.
    selected_pipeline = pipelines[selected_model_name]
    selected_pipeline.fit(
        X_train.to_dict(orient="records"),
        y_train.to_numpy(),
    )

    # Validation is used only for threshold selection.
    val_probabilities = selected_pipeline.predict_proba(
        X_val.to_dict(orient="records")
    )[:, 1]

    threshold = choose_threshold(
        y_val,
        val_probabilities,
    )

    print(
        f"Selected validation threshold: {threshold:.2f}"
    )

    # Refit only after model and threshold decisions are complete.
    X_train_final = pd.concat(
        [X_train, X_val],
        axis=0,
    )
    y_train_final = pd.concat(
        [y_train, y_val],
        axis=0,
    )

    final_pipeline = build_model_pipelines()[
        selected_model_name
    ]

    final_pipeline.fit(
        X_train_final.to_dict(orient="records"),
        y_train_final.to_numpy(),
    )

    # The test set is touched exactly once.
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
                "model_name": selected_model_name,
                "target_definition": (
                    "readmitted_within_30_days"
                ),
            },
            f_out,
        )

    metrics = {
        "methodology": {
            "target": "readmitted == '<30'",
            "outer_split": "patient-level 60/20/20",
            "cv": "10-fold StratifiedGroupKFold",
            "group_column": "patient_nbr",
            "smote_inside_pipeline": True,
            "test_set_used_for_tuning": False,
            "test_set_used_for_threshold_selection": False,
            "auc_uses_probabilities": True,
        },
        "selected_model": selected_model_name,
        "cv_results": cv_results,
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
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
