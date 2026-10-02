"""Shared deterministic preprocessing and feature engineering.

This module adapts the strongest feature-engineering ideas from the original
Hospital Readmission notebook while keeping them safe for a reusable
train/inference pipeline.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List

from sklearn.base import BaseEstimator, TransformerMixin


DROP_HIGH_MISSING_COLUMNS = {
    "weight",
    "medical_specialty",
    "payer_code",
}

DROP_COLUMNS = {
    "encounter_id",
    "patient_nbr",
    "readmitted",
    "citoglipton",
    "examide",
}

AGE_MAP = {
    "[0-10)": 5,
    "[10-20)": 15,
    "[20-30)": 25,
    "[30-40)": 35,
    "[40-50)": 45,
    "[50-60)": 55,
    "[60-70)": 65,
    "[70-80)": 75,
    "[80-90)": 85,
    "[90-100)": 95,
}

MEDICATION_COLUMNS = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "glipizide",
    "glyburide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "insulin",
    "glyburide-metformin",
    "tolazamide",
    "metformin-pioglitazone",
    "metformin-rosiglitazone",
    "glimepiride-pioglitazone",
    "glipizide-metformin",
    "troglitazone",
    "tolbutamide",
    "acetohexamide",
]

NUMERIC_COLUMNS = {
    "age",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
}

CATEGORICAL_COLUMNS = {
    "race",
    "gender",
    "diag_1",
    "diag_2",
    "diag_3",
    "A1Cresult",
    "max_glu_serum",
    "change",
    "diabetesMed",
    *MEDICATION_COLUMNS,
}

# These are the skewed log features retained by the original notebook's
# final feature set. The transformation is deterministic and does not learn
# anything from the target.
LOG_COLUMNS = {
    "number_emergency",
    "patient_service",
    "time_in_hospital",
    "med_change",
    "num_procedures",
    "number_outpatient",
    "num_medications",
    "number_inpatient",
}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return False


def _normalise_string(value: Any) -> Any:
    if isinstance(value, str):
        return value.lower().replace(" ", "_")
    return value


def _diagnosis_group(value: Any) -> str:
    """Map ICD-9-style diagnosis codes into the notebook's 8 groups."""

    if _is_missing(value) or value == "?":
        return "group_0"

    try:
        code = float(value)
    except (TypeError, ValueError):
        return "group_0"

    if 390 <= code < 460 or math.floor(code) == 785:
        group = 1
    elif 460 <= code < 520 or math.floor(code) == 786:
        group = 2
    elif 520 <= code < 580 or math.floor(code) == 787:
        group = 3
    elif math.floor(code) == 250:
        group = 4
    elif 800 <= code < 1000:
        group = 5
    elif 710 <= code < 740:
        group = 6
    elif 580 <= code < 630 or math.floor(code) == 788:
        group = 7
    elif 140 <= code < 240:
        group = 8
    else:
        group = 0

    return f"group_{group}"


def _categorical(value: Any, default: str = "NA") -> str:
    if _is_missing(value) or value == "?":
        return default
    return str(_normalise_string(value))


def _numeric(value: Any, default: float = 0.0) -> float:
    if _is_missing(value) or value == "?":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _recode_admission_type(value: Any) -> str:
    mapping = {2: 1, 7: 1, 6: 5, 8: 5}
    value = int(_numeric(value))
    return f"category_{mapping.get(value, value)}"


def _recode_discharge(value: Any) -> str:
    mapping = {
        6: 1, 8: 1, 9: 1, 13: 1,
        3: 2, 4: 2, 5: 2, 14: 2, 22: 2, 23: 2, 24: 2,
        12: 10, 15: 10, 16: 10, 17: 10,
        25: 18, 26: 18,
    }
    value = int(_numeric(value))
    return f"category_{mapping.get(value, value)}"


def _recode_source(value: Any) -> str:
    mapping = {
        2: 1, 3: 1,
        5: 4, 6: 4, 10: 4, 22: 4, 25: 4,
        7: 9, 17: 9, 20: 9, 21: 9,
        13: 11, 14: 11,
    }
    value = int(_numeric(value))
    return f"category_{mapping.get(value, value)}"


class ReadmissionFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create the final model features from raw hospital encounters."""

    def fit(self, X: Iterable[Dict[str, Any]], y=None):
        return self

    def transform(self, X: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        transformed: List[Dict[str, Any]] = []

        for raw in X:
            row = dict(raw)

            # Remove high-missing/irrelevant raw columns.
            for column in DROP_HIGH_MISSING_COLUMNS | DROP_COLUMNS:
                row.pop(column, None)

            age_value = row.get("age")
            age = AGE_MAP.get(age_value, 0.0)

            # Medication summary features from the original notebook.
            medication_values = [
                _categorical(row.get(column), default="No")
                for column in MEDICATION_COLUMNS
            ]

            med_change = sum(
                value in {"up", "down"}
                for value in medication_values
            )

            num_med = sum(
                value != "no"
                for value in medication_values
            )

            outpatient = _numeric(row.get("number_outpatient"))
            emergency = _numeric(row.get("number_emergency"))
            inpatient = _numeric(row.get("number_inpatient"))
            time_in_hospital = _numeric(row.get("time_in_hospital"))
            num_procedures = _numeric(row.get("num_procedures"))
            num_medications = _numeric(row.get("num_medications"))

            patient_service = (
                outpatient + emergency + inpatient
            )

            features: Dict[str, Any] = {
                # Notebook-style categorical features.
                "race": _categorical(row.get("race")),
                "gender": _categorical(row.get("gender")),
                "admission_type_id": _recode_admission_type(
                    row.get("admission_type_id")
                ),
                "discharge_disposition_id": _recode_discharge(
                    row.get("discharge_disposition_id")
                ),
                "admission_source_id": _recode_source(
                    row.get("admission_source_id")
                ),
                "max_glu_serum": _categorical(
                    row.get("max_glu_serum")
                ),
                "A1Cresult": _categorical(
                    row.get("A1Cresult")
                ),
                "primary_diag": _diagnosis_group(
                    row.get("diag_1")
                ),
                "secondary_diag": _diagnosis_group(
                    row.get("diag_2")
                ),
                "additional_diag": _diagnosis_group(
                    row.get("diag_3")
                ),
                "change": _categorical(
                    row.get("change")
                ),
                "diabetesMed": _categorical(
                    row.get("diabetesMed")
                ),

                # Continuous features.
                "age": age,
                "num_lab_procedures": _numeric(
                    row.get("num_lab_procedures")
                ),
                "number_diagnoses": _numeric(
                    row.get("number_diagnoses")
                ),
                "num_med": float(num_med),

                # Medication features retained by the notebook.
                **{
                    column: (
                        0.0
                        if value == "no"
                        else 1.0
                    )
                    for column, value in zip(
                        MEDICATION_COLUMNS,
                        medication_values,
                    )
                    if column in {
                        "metformin",
                        "repaglinide",
                        "nateglinide",
                        "chlorpropamide",
                        "glimepiride",
                        "glipizide",
                        "glyburide",
                        "tolbutamide",
                        "pioglitazone",
                        "rosiglitazone",
                        "acarbose",
                        "miglitol",
                        "troglitazone",
                        "tolazamide",
                        "insulin",
                        "glyburide-metformin",
                        "glipizide-metformin",
                        "glimepiride-pioglitazone",
                        "metformin-rosiglitazone",
                        "metformin-pioglitazone",
                        "acetohexamide",
                    }
                },
                "med_change": float(med_change),
                "patient_service": patient_service,
                "number_emergency": emergency,
                "time_in_hospital": time_in_hospital,
                "num_procedures": num_procedures,
                "number_outpatient": outpatient,
                "num_medications": num_medications,
                "number_inpatient": inpatient,
            }

            # Log1p features used in the notebook's final feature set.
            raw_log_values = {
                "number_emergency": emergency,
                "patient_service": patient_service,
                "time_in_hospital": time_in_hospital,
                "med_change": float(med_change),
                "num_procedures": num_procedures,
                "number_outpatient": outpatient,
                "num_medications": num_medications,
                "number_inpatient": inpatient,
            }

            for column in LOG_COLUMNS:
                features[f"{column}_log"] = math.log1p(
                    max(raw_log_values[column], 0.0)
                )

            transformed.append(features)

        return transformed
