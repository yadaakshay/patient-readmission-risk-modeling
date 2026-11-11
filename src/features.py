"""
Backwards-compatibility proxy for src.features.
"""
from src.features.engineering import (
    map_icd9_to_clinical_category,
    categorize_admission_type,
    categorize_discharge_disposition,
    categorize_admission_source,
    ClinicalFeatureEngineer
)

__all__ = [
    "map_icd9_to_clinical_category",
    "categorize_admission_type",
    "categorize_discharge_disposition",
    "categorize_admission_source",
    "ClinicalFeatureEngineer"
]
