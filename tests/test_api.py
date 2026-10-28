"""
Integration tests for the FastAPI clinical risk prediction service.
"""
import pytest
from fastapi.testclient import TestClient
import numpy as np
import pandas as pd

from src.predict import app, init_model
from src.pipeline import build_full_pipeline, save_full_pipeline
from src.config import MODEL_PATH, METADATA_PATH


@pytest.fixture(scope="module", autouse=True)
def setup_dummy_model():
    """Ensure a trained pipeline exists for API testing."""
    if not MODEL_PATH.exists():
        df_dummy = pd.DataFrame([{
            'time_in_hospital': 3,
            'num_lab_procedures': 40,
            'num_procedures': 1,
            'num_medications': 12,
            'number_outpatient': 0,
            'number_emergency': 0,
            'number_inpatient': 0,
            'number_diagnoses': 5,
            'race': 'Caucasian',
            'gender': 'Female',
            'age': '[60-70)',
            'admission_type_id': 1,
            'discharge_disposition_id': 1,
            'admission_source_id': 7,
            'diag_1': '414',
            'diag_2': '250',
            'diag_3': '401',
            'A1Cresult': '>8',
            'max_glu_serum': 'None',
            'insulin': 'Up',
            'change': 'Ch',
            'diabetesMed': 'Yes'
        }] * 10)
        y_dummy = np.array([0, 1, 0, 0, 0, 1, 0, 0, 0, 0])
        pipe = build_full_pipeline('lightgbm', calibrate=False)
        pipe.fit(df_dummy, y_dummy)
        meta = {
            'model_name': 'lightgbm',
            'optimal_threshold': 0.15,
            'clinical_risk_tiers': {'low_risk_max': 0.10, 'moderate_risk_max': 0.20}
        }
        save_full_pipeline(pipe, meta, MODEL_PATH, METADATA_PATH)
    
    # Trigger model load
    init_model()


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_single_prediction_endpoint():
    client = TestClient(app)
    payload = {
        "time_in_hospital": 5,
        "num_lab_procedures": 50,
        "num_procedures": 2,
        "num_medications": 20,
        "number_outpatient": 1,
        "number_emergency": 2,
        "number_inpatient": 2,
        "number_diagnoses": 9,
        "race": "Caucasian",
        "gender": "Female",
        "age": "[70-80)",
        "admission_type_id": 1,
        "discharge_disposition_id": 3,
        "admission_source_id": 7,
        "diag_1": "428",
        "diag_2": "250.02",
        "diag_3": "585",
        "max_glu_serum": "None",
        "A1Cresult": ">8",
        "insulin": "Up",
        "change": "Ch",
        "diabetesMed": "Yes"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "readmission_probability" in data
    assert 0.0 <= data["readmission_probability"] <= 1.0
    assert data["risk_tier"] in ["Low", "Moderate", "High"]
    assert isinstance(data["readmitted_flag"], bool)
    assert len(data["key_risk_drivers"]) > 0
    assert "clinical_action_plan" in data


def test_batch_prediction_endpoint():
    client = TestClient(app)
    patient_a = {
        "time_in_hospital": 2,
        "num_lab_procedures": 30,
        "num_procedures": 0,
        "num_medications": 8,
        "number_outpatient": 0,
        "number_emergency": 0,
        "number_inpatient": 0,
        "number_diagnoses": 4
    }
    patient_b = {
        "time_in_hospital": 8,
        "num_lab_procedures": 65,
        "num_procedures": 3,
        "num_medications": 22,
        "number_outpatient": 2,
        "number_emergency": 3,
        "number_inpatient": 3,
        "number_diagnoses": 9,
        "A1Cresult": ">8",
        "insulin": "Up"
    }
    payload = {"patients": [patient_a, patient_b]}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_processed"] == 2
    assert len(data["predictions"]) == 2
