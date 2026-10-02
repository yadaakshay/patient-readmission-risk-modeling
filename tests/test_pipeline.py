import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import StratifiedGroupKFold

from src.preprocessing import ReadmissionFeatureEngineer
from src.train import (
    build_cv,
    build_model_searches,
    patient_level_split,
)


def _sample_rows():
    rows = []

    for patient in range(30):
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
                    "admission_type_id": 1,
                    "discharge_disposition_id": 1,
                    "admission_source_id": 1,
                    "max_glu_serum": "Norm",
                    "A1Cresult": "Norm",
                    "diag_1": "250.01",
                    "diag_2": "401.9",
                    "diag_3": "585.9",
                    "insulin": "Steady",
                    "metformin": "No",
                    "change": "No",
                    "diabetesMed": "Yes",
                    "readmitted": "<30"
                    if patient % 3 == 0
                    else "NO",
                }
            )

    return pd.DataFrame(rows)


def test_feature_engineering_is_deterministic():
    rows = _sample_rows().drop(
        columns=["readmitted"]
    )

    first = ReadmissionFeatureEngineer().fit_transform(
        rows.to_dict("records")
    )
    second = ReadmissionFeatureEngineer().fit_transform(
        rows.to_dict("records")
    )

    assert first == second
    assert first[0]["age"] == 65
    assert first[0]["patient_service"] == 0
    assert "num_med" in first[0]
    assert "med_change_log" in first[0]
    assert "primary_diag" in first[0]


def test_patient_split_has_no_patient_overlap():
    df = _sample_rows()

    y = (
        df["readmitted"] == "<30"
    ).astype(int)

    X = df.drop(
        columns=["readmitted"]
    )

    (
        X_train,
        X_val,
        X_test,
        *_,
    ) = patient_level_split(X, y)

    train_patients = set(
        X_train["patient_nbr"]
    )
    val_patients = set(
        X_val["patient_nbr"]
    )
    test_patients = set(
        X_test["patient_nbr"]
    )

    assert train_patients.isdisjoint(
        val_patients
    )
    assert train_patients.isdisjoint(
        test_patients
    )
    assert val_patients.isdisjoint(
        test_patients
    )


def test_cv_is_stratified_group_kfold():
    cv = build_cv()

    assert isinstance(
        cv,
        StratifiedGroupKFold,
    )
    assert cv.n_splits == 10


def test_all_model_pipelines_apply_smote():
    searches = build_model_searches()

    assert {
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "xgboost",
    } == set(searches)

    for _, (pipeline, _) in searches.items():
        assert isinstance(
            pipeline,
            ImbPipeline,
        )

        names = list(
            pipeline.named_steps
        )

        assert "features" in names
        assert "vectorizer" in names
        assert "smote" in names
        assert "model" in names

        assert names.index(
            "smote"
        ) < names.index("model")


def test_random_forest_pipeline_can_fit_small_data():
    df = _sample_rows()

    y = (
        df["readmitted"] == "<30"
    ).astype(int)

    X = df.drop(
        columns=["readmitted"]
    )

    pipeline, params = build_model_searches()[
        "random_forest"
    ]

    pipeline.set_params(
        model__max_depth=10,
        model__min_samples_leaf=2,
    )

    pipeline.fit(
        X.to_dict("records"),
        y.to_numpy(),
    )

    probabilities = pipeline.predict_proba(
        X.head(2).to_dict("records")
    )[:, 1]

    assert probabilities.shape == (2,)
    assert np.all(
        (probabilities >= 0)
        & (probabilities <= 1)
    )
