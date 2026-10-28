"""
Unified machine learning pipeline module.
Constructs scikit-learn compatible pipelines combining domain feature engineering,
preprocessing transformations, gradient boosted classifiers, and probability calibration.
"""
from typing import Dict, Any, Optional, Tuple
import pickle
import json
import joblib
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from src.config import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    RANDOM_SEED,
    MODEL_PATH,
    METADATA_PATH
)
from src.features import ClinicalFeatureEngineer


def build_clinical_preprocessor() -> ColumnTransformer:
    """Creates column transformer applying standard scaling to numerics and one-hot encoding to categoricals."""
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )


def build_base_classifier(
    model_type: str = 'lightgbm',
    scale_pos_weight: Optional[float] = None,
    random_state: int = RANDOM_SEED
):
    """Initializes the requested classification algorithm with cost-sensitive parameters."""
    m_type = model_type.lower()
    if m_type == 'lightgbm':
        weight = scale_pos_weight if scale_pos_weight else 5.0
        return LGBMClassifier(
            n_estimators=180,
            max_depth=5,
            num_leaves=24,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=weight,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1
        )
    elif m_type == 'xgboost':
        weight = scale_pos_weight if scale_pos_weight else 5.0
        return XGBClassifier(
            n_estimators=180,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=weight,
            eval_metric='logloss',
            random_state=random_state,
            n_jobs=-1
        )
    elif m_type == 'random_forest':
        return RandomForestClassifier(
            n_estimators=180,
            max_depth=12,
            min_samples_leaf=8,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        )
    elif m_type == 'logistic_regression':
        return LogisticRegression(
            max_iter=1000,
            C=0.1,
            class_weight='balanced',
            random_state=random_state
        )
    else:
        raise ValueError(f"Unsupported model type '{model_type}'. Choose from: lightgbm, xgboost, random_forest, logistic_regression")


def build_full_pipeline(
    model_type: str = 'lightgbm',
    calibrate: bool = True,
    calibration_method: str = 'sigmoid',
    scale_pos_weight: Optional[float] = None,
    random_state: int = RANDOM_SEED
) -> Pipeline:
    """
    Builds the end-to-end pipeline:
    ClinicalFeatureEngineer -> ColumnTransformer -> Classifier (optionally calibrated)
    """
    feature_engineer = ClinicalFeatureEngineer()
    preprocessor = build_clinical_preprocessor()
    base_clf = build_base_classifier(model_type, scale_pos_weight, random_state)

    if calibrate:
        clf = CalibratedClassifierCV(estimator=base_clf, method=calibration_method, cv=3)
    else:
        clf = base_clf

    pipeline = Pipeline(steps=[
        ('features', feature_engineer),
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])
    return pipeline


def save_full_pipeline(
    pipeline: Pipeline,
    metadata: Dict[str, Any],
    model_path: Path = MODEL_PATH,
    metadata_path: Path = METADATA_PATH
):
    """Serializes the entire fitted pipeline and metadata JSON."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, 'wb') as f:
        joblib.dump(pipeline, f)

    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print(f"Pipeline successfully saved to {model_path}")
    print(f"Metadata saved to {metadata_path}")


def load_full_pipeline(
    model_path: Path = MODEL_PATH,
    metadata_path: Path = METADATA_PATH
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Loads the serialized pipeline and associated metadata."""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    with open(model_path, 'rb') as f:
        pipeline = joblib.load(f)

    metadata = {}
    if Path(metadata_path).exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

    return pipeline, metadata
