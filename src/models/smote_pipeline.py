"""
Unified imbalanced-learning pipeline construction module.
Supports three imbalance handling strategies:
1. 'none': No resampling, standard loss
2. 'class_weight': Cost-sensitive learning via scale_pos_weight or class_weight='balanced'
3. 'smote': Synthetic Minority Over-sampling Technique (SMOTE) strictly within imblearn.pipeline.Pipeline
"""
from typing import Optional, Literal
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from src.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, RANDOM_SEED
from src.features import ClinicalFeatureEngineer


def build_clinical_preprocessor() -> ColumnTransformer:
    """Creates standard ColumnTransformer: numeric scaling + categorical one-hot encoding."""
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )


def build_model_classifier(
    model_type: str = 'lightgbm',
    imbalance_strategy: Literal['none', 'class_weight', 'smote'] = 'class_weight',
    random_state: int = RANDOM_SEED
):
    """
    Instantiates classifier configured for the specified imbalance strategy.
    """
    m_type = model_type.lower()
    use_weight = (imbalance_strategy == 'class_weight')

    if m_type == 'logistic_regression':
        cw = 'balanced' if use_weight else None
        return LogisticRegression(
            C=0.1,
            max_iter=1000,
            class_weight=cw,
            random_state=random_state
        )

    elif m_type == 'random_forest':
        cw = 'balanced' if use_weight else None
        return RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_leaf=10,
            class_weight=cw,
            random_state=random_state,
            n_jobs=-1
        )

    elif m_type == 'lightgbm':
        scale_pos = 5.0 if use_weight else 1.0
        return LGBMClassifier(
            n_estimators=150,
            max_depth=5,
            num_leaves=24,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1
        )

    elif m_type == 'xgboost':
        scale_pos = 5.0 if use_weight else 1.0
        return XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            eval_metric='logloss',
            random_state=random_state,
            n_jobs=-1
        )

    else:
        raise ValueError(f"Unsupported model_type '{model_type}'. Choose from: logistic_regression, random_forest, lightgbm, xgboost")


def build_pipeline_with_strategy(
    model_type: str = 'lightgbm',
    imbalance_strategy: Literal['none', 'class_weight', 'smote'] = 'class_weight',
    calibrate: bool = False,
    random_state: int = RANDOM_SEED
) -> ImbPipeline:
    """
    Builds the end-to-end pipeline using imblearn.pipeline.Pipeline:
    - Step 1: Feature Engineering (ClinicalFeatureEngineer)
    - Step 2: Preprocessor (StandardScaler + OneHotEncoder)
    - Step 3 (Optional): SMOTE sampler (applied ONLY during fit, never during predict)
    - Step 4: Classifier (optionally calibrated)
    """
    feature_engineer = ClinicalFeatureEngineer()
    preprocessor = build_clinical_preprocessor()
    base_clf = build_model_classifier(model_type, imbalance_strategy, random_state)

    if calibrate:
        clf = CalibratedClassifierCV(estimator=base_clf, method='sigmoid', cv=3)
    else:
        clf = base_clf

    steps = [
        ('features', feature_engineer),
        ('preprocessor', preprocessor)
    ]

    if imbalance_strategy == 'smote':
        # SMOTE applied strictly on the preprocessed numeric feature matrix inside the pipeline
        smote = SMOTE(random_state=random_state, sampling_strategy=0.35)
        steps.append(('sampler', smote))

    steps.append(('classifier', clf))

    return ImbPipeline(steps=steps)
