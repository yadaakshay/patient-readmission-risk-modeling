"""Shared, deterministic feature engineering for training and inference.

The transformations mirror the useful feature-engineering ideas from the
original notebook, but are implemented as a reusable sklearn transformer so
training and API inference cannot silently diverge.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


DROP_COLUMNS = {"encounter_id", "patient_nbr", "readmitted"}
DROP_HIGH_MISSING_COLUMNS = {"weight", "medical_specialty", "payer_code"}

AGE_MAP = {
    "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
    "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
    "[80-90)": 85, "[90-100)": 95,
}

MEDICATION_COLUMNS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide", "citoglipton",
    "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

NUMERIC_COLUMNS = {
    "age", "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses", "admission_type_id",
    "discharge_disposition_id", "admission_source_id",
}

CATEGORICAL_COLUMNS = [
    "race", "gender", "max_glu_serum", "A1Cresult",
    *MEDICATION_COLUMNS,
    "change", "diabetesMed",
    "primary_diag", "secondary_diag", "additional_diag",
]


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
    """Map ICD-9 diagnosis codes into the notebook's clinical groups."""

    if _is_missing(value) or value == "?":
        return "0"

    try:
        text = str(value).upper()
        if text.startswith("V") or text.startswith("E"):
            return "0"
        code = float(value)
    except (TypeError, ValueError):
        return "0"

    if (390 <= code < 460) or math.floor(code) == 785:
        return "1"
    if (460 <= code < 520) or math.floor(code) == 786:
        return "2"
    if (520 <= code < 580) or math.floor(code) == 787:
        return "3"
    if math.floor(code) == 250:
        return "4"
    if 800 <= code < 1000:
        return "5"
    if 710 <= code < 740:
        return "6"
    if (580 <= code < 630) or math.floor(code) == 788:
        return "7"
    if 140 <= code < 240:
        return "8"
    return "0"


def _recode_admission_type(value: Any) -> int:
    mapping = {2: 1, 7: 1, 6: 5, 8: 5}
    try:
        value = int(float(value))
        return mapping.get(value, value)
    except (TypeError, ValueError):
        return 0


def _recode_discharge(value: Any) -> int:
    mapping = {
        6: 1, 8: 1, 9: 1, 13: 1,
        3: 2, 4: 2, 5: 2, 14: 2, 22: 2, 23: 2, 24: 2,
        12: 10, 15: 10, 16: 10, 17: 10,
        25: 18, 26: 18,
    }
    try:
        value = int(float(value))
        return mapping.get(value, value)
    except (TypeError, ValueError):
        return 0


def _recode_admission_source(value: Any) -> int:
    mapping = {
        2: 1, 3: 1,
        5: 4, 6: 4, 10: 4, 22: 4, 25: 4,
        7: 9, 17: 9, 20: 9, 21: 9,
        13: 11, 14: 11,
    }
    try:
        value = int(float(value))
        return mapping.get(value, value)
    except (TypeError, ValueError):
        return 0


def _recode_measure(value: Any) -> str:
    if _is_missing(value) or value == "?":
        return "none"
    value = str(value)
    if value in {">7", ">8", ">200", ">300"}:
        return "high"
    if value == "Norm":
        return "normal"
    if value == "None":
        return "none"
    return value.lower().replace(" ", "_")


class ReadmissionFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create deterministic features without fitting on the target."""

    def fit(self, X: Iterable[Dict[str, Any]], y=None):
        return self

    def transform(self, X: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        transformed: List[Dict[str, Any]] = []

        for raw in X:
            raw = dict(raw)
            row: Dict[str, Any] = {}

            for key, value in raw.items():
                if key in DROP_COLUMNS or key in DROP_HIGH_MISSING_COLUMNS:
                    continue

                if _is_missing(value) or value == "?":
                    row[key] = 0.0 if key in NUMERIC_COLUMNS else "NA"
                    continue

                if key == "age":
                    row[key] = AGE_MAP.get(
                        value,
                        float(value) if str(value).isdigit() else 0.0,
                    )
                elif key == "admission_type_id":
                    row[key] = _recode_admission_type(value)
                elif key == "discharge_disposition_id":
                    row[key] = _recode_discharge(value)
                elif key == "admission_source_id":
                    row[key] = _recode_admission_source(value)
                elif key in {"max_glu_serum", "A1Cresult"}:
                    row[key] = _recode_measure(value)
                elif key in {"change", "diabetesMed"}:
                    row[key] = str(value).lower().replace(" ", "_")
                else:
                    row[key] = _normalise_string(value)

            for column in NUMERIC_COLUMNS:
                row.setdefault(column, 0.0)
            for column in CATEGORICAL_COLUMNS:
                row.setdefault(column, "NA")

            outpatient = float(row.get("number_outpatient", 0) or 0)
            emergency = float(row.get("number_emergency", 0) or 0)
            inpatient = float(row.get("number_inpatient", 0) or 0)
            time_in_hospital = float(row.get("time_in_hospital", 0) or 0)
            num_medications = float(row.get("num_medications", 0) or 0)
            num_procedures = float(row.get("num_procedures", 0) or 0)
            num_lab_procedures = float(row.get("num_lab_procedures", 0) or 0)

            # Features explicitly engineered in the original notebook.
            row["patient_service"] = outpatient + emergency + inpatient

            medication_values = [
                row.get(column, "NA")
                for column in MEDICATION_COLUMNS
            ]
            row["num_med"] = sum(
                value not in {"no", "NA"}
                for value in medication_values
            )
            row["med_change"] = sum(
                value in {"up", "down"}
                for value in medication_values
            )

            # Additional deterministic production features.
            row["total_previous_visits"] = row["patient_service"]
            row["had_previous_inpatient"] = int(inpatient > 0)
            row["avg_medications_per_day"] = (
                num_medications / (time_in_hospital + 1)
            )
            row["procedure_to_lab_ratio"] = (
                num_procedures / (num_lab_procedures + 1)
            )
            row["num_medications_used"] = row["num_med"]
            row["num_adjusted_medications"] = row["med_change"]
            row["any_medication_change"] = int(row["med_change"] > 0)
            row["on_insulin"] = int(
                row.get("insulin", "NA") in {"steady", "up", "down"}
            )

            # Group raw ICD-9 diagnoses as done in the notebook.
            row["primary_diag"] = _diagnosis_group(raw.get("diag_1"))
            row["secondary_diag"] = _diagnosis_group(raw.get("diag_2"))
            row["additional_diag"] = _diagnosis_group(raw.get("diag_3"))
            row.pop("diag_1", None)
            row.pop("diag_2", None)
            row.pop("diag_3", None)

            # Fixed log transforms for the skewed count-like variables
            # identified in the notebook. The list is fixed so skewness is
            # not estimated using the full dataset.
            for column in [
                "number_emergency",
                "patient_service",
                "time_in_hospital",
                "med_change",
                "num_procedures",
                "number_outpatient",
                "num_medications",
                "number_inpatient",
            ]:
                value = float(row.get(column, 0) or 0)
                row[f"{column}_log"] = float(np.log1p(max(value, 0.0)))

            transformed.append(row)

        return transformed
