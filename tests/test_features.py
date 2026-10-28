"""
Unit tests for clinical feature engineering and ICD-9 mapping logic.
"""
import pytest
import pandas as pd
import numpy as np

from src.features import (
    map_icd9_to_clinical_category,
    ClinicalFeatureEngineer,
    categorize_admission_type,
    categorize_discharge_disposition
)
from src.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES


def test_icd9_category_mapping():
    # Circulatory
    assert map_icd9_to_clinical_category("414") == "Circulatory"
    assert map_icd9_to_clinical_category("428") == "Circulatory"
    assert map_icd9_to_clinical_category("785") == "Circulatory"

    # Diabetes
    assert map_icd9_to_clinical_category("250.02") == "Diabetes"
    assert map_icd9_to_clinical_category("250") == "Diabetes"

    # Respiratory
    assert map_icd9_to_clinical_category("486") == "Respiratory"
    assert map_icd9_to_clinical_category("491") == "Respiratory"

    # Digestive
    assert map_icd9_to_clinical_category("530") == "Digestive"
    assert map_icd9_to_clinical_category("571") == "Digestive"

    # Genitourinary
    assert map_icd9_to_clinical_category("585") == "Genitourinary"
    assert map_icd9_to_clinical_category("599") == "Genitourinary"

    # Neoplasms
    assert map_icd9_to_clinical_category("199") == "Neoplasms"

    # Injury
    assert map_icd9_to_clinical_category("820") == "Injury"

    # Supplementary V and External E
    assert map_icd9_to_clinical_category("V45") == "Supplementary_V"
    assert map_icd9_to_clinical_category("E880") == "External_Injury_E"

    # Missing / None
    assert map_icd9_to_clinical_category("?") == "Missing"
    assert map_icd9_to_clinical_category(None) == "Missing"
    assert map_icd9_to_clinical_category(np.nan) == "Missing"


def test_feature_engineer_single_dict():
    engineer = ClinicalFeatureEngineer()
    sample_patient = {
        'time_in_hospital': 4,
        'num_lab_procedures': 40,
        'num_procedures': 1,
        'num_medications': 15,
        'number_outpatient': 0,
        'number_emergency': 1,
        'number_inpatient': 1,
        'number_diagnoses': 8,
        'race': 'Caucasian',
        'gender': 'Male',
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
    }

    df_trans = engineer.transform(sample_patient)
    assert isinstance(df_trans, pd.DataFrame)
    assert len(df_trans) == 1

    # Check that all configured columns are present
    expected_cols = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    for col in expected_cols:
        assert col in df_trans.columns, f"Missing expected column: {col}"

    # Verify calculated values
    assert df_trans['age_num'].iloc[0] == 65.0
    assert df_trans['age_over_65'].iloc[0] == 1.0
    assert df_trans['total_prior_visits'].iloc[0] == 2.0
    assert df_trans['had_prior_inpatient'].iloc[0] == 1.0
    assert df_trans['on_insulin'].iloc[0] == 1.0
    assert df_trans['insulin_adjusted'].iloc[0] == 1.0
    assert df_trans['a1c_tested'].iloc[0] == 1.0
    assert df_trans['a1c_high'].iloc[0] == 1.0
    assert df_trans['diag_1_group'].iloc[0] == 'Circulatory'
    assert df_trans['has_diabetes_diag'].iloc[0] == 1.0


def test_feature_engineer_missing_fields():
    engineer = ClinicalFeatureEngineer()
    # Minimal sparse dictionary
    sparse_patient = {
        'time_in_hospital': 2,
        'num_medications': 5
    }
    df_trans = engineer.transform(sparse_patient)
    assert len(df_trans) == 1
    assert df_trans['age_num'].iloc[0] == 55.0  # default midpoint
    assert df_trans['total_prior_visits'].iloc[0] == 0.0
    assert df_trans['on_insulin'].iloc[0] == 0.0
    assert df_trans['diag_1_group'].iloc[0] == 'Missing'
