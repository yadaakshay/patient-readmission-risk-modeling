#!/usr/bin/env python
"""Train the production readmission model with patient-level evaluation."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

from preprocessing import ReadmissionFeatureEngineer


DATA_PATH = Path("data/diabetic_data.csv")
MODEL_PATH = Path("model/model.bin")
METRICS_PATH = Path("model/metrics.json")
RANDOM_STATE = 42


def load_raw_data() -> pd.DataFrame:
    """Load the raw encounter data without fitting any preprocessing on it."""
    return pd.read_csv(DATA_PATH)


def prepare_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if "readmitted" not in df.columns:
        raise ValueError("Expected 'readmitted' target column.")

    # <30 is the positive class: readmitted within 30 days.
    y = (df["readmitted"] == "<30").astype(int)

    # patient_nbr is retained temporarily for grouped splitting and is removed
    # from the model features by ReadmissionFeatureEngineer.
    return df.drop(columns=["readmitted"]), y


def patient_level_split(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Create 60/20/20 train/validation/test splits with no patient overlap."""

    if "patient_nbr" not in X.columns:
        raise ValueError("patient_nbr is required for patient-level splitting.")

    groups = X["patient_nbr"]

    outer = GroupShuffleSplit(
        n_splits=1, test_size=0.20, random_state=RANDOM_STATE
    )
    train_val_idx, test_idx = next(outer.split(X, y, groups=groups))

    X_train_val = X.iloc[train_val_idx].copy()
    y_train_val = y.iloc[train_val_idx].copy()

    inner = GroupShuffleSplit(
        n_splits=1, test_size=0.25, random_state=RANDOM_STATE
    )
    inner_groups = X_train_val["patient_nbr"]
    train_idx, val_idx = next(
        inner.split(X_train_val, y_train_val, groups=inner_groups)
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

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_pipeline() -> Pipeline:
    """Build the complete preprocessing -> encoding -> model pipeline."""

    return Pipeline(
        steps=[
            ("features", ReadmissionFeatureEngineer()),
            ("vectorizer", DictVectorizer()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    max_features="sqrt",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def choose_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    """Choose a classification threshold using validation data only.

    The test set is never used to choose this threshold.
    """
    best_threshold = 0.50
    best_f1 = -1.0

    for threshold in np.arange(0.05, 0.51, 0.01):
        predictions = (probabilities >= threshold).astype(int)
        score = f1_score(y_true, predictions, zero_division=0)

        if score > best_f1:
            best_f1 = score
            best_threshold = float(round(threshold, 2))

    return best_threshold


def evaluate(
    y_true: pd.Series, probabilities: np.ndarray, threshold: float
) -> dict:
    predictions = (probabilities >= threshold).astype(int)

    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
        "positive_rate": float(np.mean(predictions)),
    }


def main() -> None:
    df = load_raw_data()
    X, y = prepare_target(df)

    X_train, X_val, X_test, y_train, y_val, y_test = patient_level_split(X, y)

    print(
        "Split sizes:",
        f"train={len(X_train)}, val={len(X_val)}, test={len(X_test)}",
    )
    print(
        "Positive rates:",
        f"train={y_train.mean():.4f}, val={y_val.mean():.4f}, test={y_test.mean():.4f}",
    )

    # First fit only on training data. Validation is used only for threshold
    # selection; the test set remains untouched.
    validation_pipeline = build_pipeline()
    validation_pipeline.fit(
        X_train.to_dict(orient="records"),
        y_train.to_numpy(),
    )

    val_probabilities = validation_pipeline.predict_proba(
        X_val.to_dict(orient="records")
    )[:, 1]
    threshold = choose_threshold(y_val, val_probabilities)

    # After model/threshold selection, refit the model on train + validation.
    # The test set remains completely untouched until final evaluation.
    X_train_final = pd.concat([X_train, X_val], axis=0)
    y_train_final = pd.concat([y_train, y_val], axis=0)

    final_pipeline = build_pipeline()
    final_pipeline.fit(
        X_train_final.to_dict(orient="records"),
        y_train_final.to_numpy(),
    )

    test_probabilities = final_pipeline.predict_proba(
        X_test.to_dict(orient="records")
    )[:, 1]
    test_metrics = evaluate(y_test, test_probabilities, threshold)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as f_out:
        pickle.dump(
            {
                "pipeline": final_pipeline,
                "threshold": threshold,
                "target_definition": "readmitted_within_30_days",
            },
            f_out,
        )

    metrics = {
        "split": {
            "train_rows": len(X_train),
            "validation_rows": len(X_val),
            "test_rows": len(X_test),
            "train_patients": int(X_train["patient_nbr"].nunique()),
            "validation_patients": int(X_val["patient_nbr"].nunique()),
            "test_patients": int(X_test["patient_nbr"].nunique()),
        },
        "validation_threshold": threshold,
        "test": test_metrics,
    }

    with METRICS_PATH.open("w", encoding="utf-8") as f_out:
        json.dump(metrics, f_out, indent=2)

    print(json.dumps(test_metrics, indent=2))
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
