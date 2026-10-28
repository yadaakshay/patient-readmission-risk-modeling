"""
Unit tests for pipeline construction, calibration, and inference.
"""
import pytest
import numpy as np
import pandas as pd

from src.pipeline import build_full_pipeline
from src.config import RANDOM_SEED


@pytest.fixture
def mock_ehr_data():
    np.random.seed(RANDOM_SEED)
    n = 100
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
    y = np.random.choice([0, 1], p=[0.85, 0.15], size=n)
    return df, y


@pytest.mark.parametrize("model_name", ["logistic_regression", "lightgbm"])
def test_pipeline_fit_predict(mock_ehr_data, model_name):
    df, y = mock_ehr_data
    pipeline = build_full_pipeline(model_type=model_name, calibrate=False, random_state=42)
    pipeline.fit(df, y)

    # Predict proba on DataFrame
    probs = pipeline.predict_proba(df)
    assert probs.shape == (len(df), 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)

    # Predict proba on list of single raw dictionary
    single_record = [df.iloc[0].to_dict()]
    single_prob = pipeline.predict_proba(single_record)
    assert single_prob.shape == (1, 2)
    assert 0.0 <= single_prob[0, 1] <= 1.0
