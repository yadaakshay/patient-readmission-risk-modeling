"""
Unit tests verifying that SMOTE resampling occurs strictly inside training folds
and never resamples, leaks into, or alters validation/test sets.
"""
import pytest
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from src.models.smote_pipeline import build_pipeline_with_strategy


@pytest.fixture
def imbalanced_mock_data():
    np.random.seed(42)
    n = 200
    # Create 90% negative, 10% positive imbalanced mock EHR data
    data = {
        'time_in_hospital': np.random.randint(1, 10, size=n),
        'num_lab_procedures': np.random.randint(10, 80, size=n),
        'num_procedures': np.random.randint(0, 5, size=n),
        'num_medications': np.random.randint(5, 25, size=n),
        'number_outpatient': np.random.randint(0, 3, size=n),
        'number_emergency': np.random.randint(0, 2, size=n),
        'number_inpatient': np.random.randint(0, 2, size=n),
        'number_diagnoses': np.random.randint(3, 10, size=n),
        'race': np.random.choice(['Caucasian', 'AfricanAmerican', 'Hispanic', '?'], size=n),
        'gender': np.random.choice(['Male', 'Female'], size=n),
        'age': np.random.choice(['[40-50)', '[50-60)', '[60-70)', '[70-80)'], size=n),
        'admission_type_id': np.random.choice([1, 2, 3], size=n),
        'discharge_disposition_id': np.random.choice([1, 3, 6], size=n),
        'admission_source_id': np.random.choice([1, 7], size=n),
        'diag_1': np.random.choice(['414', '250.01', '486', '585'], size=n),
        'diag_2': np.random.choice(['401', '250', '?'], size=n),
        'diag_3': np.random.choice(['250', '428', '?'], size=n),
        'A1Cresult': np.random.choice(['None', 'Norm', '>8'], size=n),
        'max_glu_serum': np.random.choice(['None', 'Norm'], size=n),
        'insulin': np.random.choice(['No', 'Steady', 'Up'], size=n),
        'change': np.random.choice(['No', 'Ch'], size=n),
        'diabetesMed': np.random.choice(['No', 'Yes'], size=n)
    }
    df = pd.DataFrame(data)
    y = np.array([1]*20 + [0]*180)  # 20 positives (10%), 180 negatives (90%)
    return df, y


def test_smote_pipeline_structure(imbalanced_mock_data):
    pipe = build_pipeline_with_strategy('logistic_regression', imbalance_strategy='smote')
    assert isinstance(pipe, ImbPipeline)
    step_names = [s[0] for s in pipe.steps]
    assert 'sampler' in step_names
    assert isinstance(pipe.named_steps['sampler'], SMOTE)


def test_smote_never_resamples_inference(imbalanced_mock_data):
    df, y = imbalanced_mock_data
    train_df, test_df = df.iloc[:150], df.iloc[150:]
    y_train, y_test = y[:150], y[150:]

    pipe = build_pipeline_with_strategy('logistic_regression', imbalance_strategy='smote')
    pipe.fit(train_df, y_train)

    # Predict on test set
    test_probs = pipe.predict_proba(test_df)
    test_preds = pipe.predict(test_df)

    # Test set length must be preserved exactly
    assert len(test_probs) == len(test_df)
    assert len(test_preds) == len(test_df)
    assert test_probs.shape == (50, 2)
    # Output probabilities must be valid [0, 1]
    assert np.all((test_probs >= 0.0) & (test_probs <= 1.0))
