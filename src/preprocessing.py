"""Shared preprocessing and feature engineering for training and inference."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List

from sklearn.base import BaseEstimator, TransformerMixin


DROP_COLUMNS = {
    "encounter_id",
    "patient_nbr",
    "readmitted",
}

DROP_HIGH_MISSING_COLUMNS = {
    "weight",
    "max_glu_serum",
    "A1Cresult",
    "medical_specialty",
    "payer_code",
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

CATEGORICAL_COLUMNS = [
    "race", "gender", "age", "diag_1", "diag_2", "diag_3",
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone", "change", "diabetesMed",
]

MEDICATION_COLUMNS = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
]


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return False


def _normalise_value(column: str, value: Any) -> Any:
    if _is_missing(value) or value == "?":
        return "NA" if column not in NUMERIC_COLUMNS else 0.0

    if isinstance(value, str):
        value = value.lower().replace(" ", "_")

    if column == "age" and value in AGE_MAP:
        return AGE_MAP[value]

    return value


NUMERIC_COLUMNS = {
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
}


class ReadmissionFeatureEngineer(BaseEstimator, TransformerMixin):
    """Normalize raw encounter records and create the project's engineered features.

    The transformer deliberately contains only deterministic transformations.
    This means the exact same code is used for training and API inference.
    """

    def fit(self, X: Iterable[Dict[str, Any]], y=None):
        return self

    def transform(self, X: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        transformed: List[Dict[str, Any]] = []

        for raw in X:
            row = {
                key: _normalise_value(key, value)
                for key, value in dict(raw).items()
                if key not in DROP_HIGH_MISSING_COLUMNS
            }

            row.pop("readmitted", None)
            row.pop("encounter_id", None)
            row.pop("patient_nbr", None)

            # Make missing schema fields explicit so training and inference have
            # identical semantics. Missing categorical values become the learned
            # "NA" category; missing numeric values become 0.
            for column in NUMERIC_COLUMNS:
                row.setdefault(column, 0.0)
            for column in CATEGORICAL_COLUMNS:
                row.setdefault(column, "NA")

            # Visit-history features.
            outpatient = float(row.get("number_outpatient", 0) or 0)
            emergency = float(row.get("number_emergency", 0) or 0)
            inpatient = float(row.get("number_inpatient", 0) or 0)
            time_in_hospital = float(row.get("time_in_hospital", 0) or 0)
            num_medications = float(row.get("num_medications", 0) or 0)
            num_procedures = float(row.get("num_procedures", 0) or 0)
            num_lab_procedures = float(row.get("num_lab_procedures", 0) or 0)

            row["total_previous_visits"] = outpatient + emergency + inpatient
            row["had_previous_inpatient"] = int(inpatient > 0)
            row["avg_medications_per_day"] = num_medications / (time_in_hospital + 1)
            row["procedure_to_lab_ratio"] = (
                num_procedures / (num_lab_procedures + 1)
            )

            # Medication summary features.
            medication_values = [
                row.get(column, "NA") for column in MEDICATION_COLUMNS
            ]
            row["num_medications_used"] = sum(
                value not in {"no", "NA"} for value in medication_values
            )
            row["num_adjusted_medications"] = sum(
                value in {"up", "down"} for value in medication_values
            )
            row["any_medication_change"] = int(
                row["num_adjusted_medications"] > 0
            )
            row["on_insulin"] = int(row.get("insulin", "NA") != "no")

            transformed.append(row)

        return transformed
