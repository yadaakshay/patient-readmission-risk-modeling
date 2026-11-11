"""
Hyperparameter tuning module for clinical risk prediction models.
Uses RandomizedSearchCV with StratifiedKFold to optimize Average Precision (PR-AUC).
"""
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from lightgbm import LGBMClassifier

from src.config import RANDOM_SEED
from src.features import ClinicalFeatureEngineer
from src.models.smote_pipeline import build_clinical_preprocessor
from imblearn.pipeline import Pipeline as ImbPipeline


def run_hyperparameter_search(
    df_train: pd.DataFrame,
    y_train: np.ndarray,
    n_iter: int = 15,
    cv_splits: int = 3,
    random_state: int = RANDOM_SEED
) -> Tuple[Any, Dict[str, Any]]:
    """
    Runs hyperparameter optimization on LightGBM using StratifiedKFold cross-validation.
    Objective Metric: 'average_precision' (PR-AUC) - strongly preferred for imbalanced clinical outcomes.
    """
    print(f"\n========================================================")
    print(f"Hyperparameter Tuning (RandomizedSearchCV, n_iter={n_iter}, cv={cv_splits})")
    print(f"Scoring: average_precision (PR-AUC)")
    print(f"========================================================")

    # Pipeline definition
    pipeline = ImbPipeline(steps=[
        ('features', ClinicalFeatureEngineer()),
        ('preprocessor', build_clinical_preprocessor()),
        ('classifier', LGBMClassifier(
            random_state=random_state,
            n_jobs=-1,
            verbose=-1
        ))
    ])

    param_distributions = {
        'classifier__n_estimators': [100, 150, 200, 250],
        'classifier__max_depth': [3, 4, 5, 6],
        'classifier__num_leaves': [15, 24, 31, 45],
        'classifier__learning_rate': [0.015, 0.03, 0.05, 0.08],
        'classifier__subsample': [0.7, 0.8, 0.9, 1.0],
        'classifier__colsample_bytree': [0.6, 0.7, 0.8, 0.9],
        'classifier__scale_pos_weight': [3.0, 5.0, 7.0, 10.0],
        'classifier__min_child_samples': [20, 50, 100]
    }

    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring='average_precision',
        cv=skf,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
        refit=True
    )

    search.fit(df_train, y_train)

    best_params = {k.replace('classifier__', ''): v for k, v in search.best_params_.items()}
    best_score = float(search.best_score_)
    print(f"\nBest Cross-Validated PR-AUC: {best_score:.4f}")
    print(f"Best Parameters: {best_params}")

    results = {
        'best_score_pr_auc': best_score,
        'best_params': best_params,
        'cv_splits': cv_splits,
        'n_iter': n_iter,
        'scoring_metric': 'average_precision'
    }

    return search.best_estimator_, results
