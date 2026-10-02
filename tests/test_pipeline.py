import numpy as np
import pandas as pd

from src.preprocessing import ReadmissionFeatureEngineer
from src.train import build_pipeline, patient_level_split


def _sample_rows():
    rows = []
    for patient in range(12):
        for encounter in range(2):
            rows.append(
                {
                    "patient_nbr": patient,
                    "time_in_hospital": 3 + encounter,
                    "num_lab_procedures": 40,
                    "num_procedures": 1,
                    "num_medications": 5,
                    "number_outpatient": encounter,
                    "number_emergency": 0,
                    "number_inpatient": patient % 2,
                    "number_diagnoses": 5,
                    "race": "Caucasian",
                    "gender": "Male",
                    "age": "[60-70)",
                    "diag_1": "250",
                    "diag_2": "401",
                    "diag_3": "585",
                    "insulin": "Steady",
                    "metformin": "No",
                    "readmitted": "<30" if patient % 3 == 0 else "NO",
                }
            )
    return pd.DataFrame(rows)


def test_feature_engineering_is_deterministic():
    rows = _sample_rows().drop(columns=["readmitted"])
    first = ReadmissionFeatureEngineer().fit_transform(rows.to_dict("records"))
    second = ReadmissionFeatureEngineer().fit_transform(rows.to_dict("records"))

    assert first == second
    assert first[0]["age"] == 65
    assert first[0]["had_previous_inpatient"] == 0
    assert "total_previous_visits" in first[0]
    assert "num_medications_used" in first[0]


def test_patient_split_has_no_patient_overlap():
    df = _sample_rows()
    y = (df["readmitted"] == "<30").astype(int)
    X = df.drop(columns=["readmitted"])

    X_train, X_val, X_test, *_ = patient_level_split(X, y)

    train_patients = set(X_train["patient_nbr"])
    val_patients = set(X_val["patient_nbr"])
    test_patients = set(X_test["patient_nbr"])

    assert train_patients.isdisjoint(val_patients)
    assert train_patients.isdisjoint(test_patients)
    assert val_patients.isdisjoint(test_patients)


def test_production_pipeline_accepts_raw_records():
    df = _sample_rows()
    y = (df["readmitted"] == "<30").astype(int)
    X = df.drop(columns=["readmitted"])

    pipeline = build_pipeline()
    pipeline.fit(X.to_dict("records"), y.to_numpy())

    probabilities = pipeline.predict_proba(
        X.head(2).to_dict("records")
    )[:, 1]

    assert probabilities.shape == (2,)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
