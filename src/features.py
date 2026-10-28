"""
Clinical feature engineering transformers and domain helper functions.
Implements ICD-9 categorization, comorbidity grouping, medication complexity,
and utilization intensity indicators conforming to the scikit-learn Transformer API.
"""
from typing import Union, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import (
    AGE_MIDPOINT_MAP,
    DIABETES_MEDICATIONS,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES
)


def map_icd9_to_clinical_category(code: Any) -> str:
    """
    Maps an ICD-9 diagnosis code into standard clinical categories based on
    the Clinical Classifications Software (CCS) logic:
    - Circulatory (390–459, 785)
    - Respiratory (460–519, 786)
    - Digestive (520–579, 787)
    - Diabetes (250.xx)
    - Injury & Poisoning (800–999)
    - Musculoskeletal (710–739)
    - Genitourinary (580–629, 788)
    - Neoplasms (140–239)
    - Supplementary / External (V/E codes)
    - Other
    """
    if pd.isna(code) or str(code).strip() in ['?', '', 'None', 'nan', 'NA']:
        return 'Missing'
    
    code_str = str(code).strip()
    if code_str.upper().startswith('V'):
        return 'Supplementary_V'
    if code_str.upper().startswith('E'):
        return 'External_Injury_E'
    
    try:
        val = float(code_str)
        if (390 <= val <= 459) or val == 785:
            return 'Circulatory'
        elif (460 <= val <= 519) or val == 786:
            return 'Respiratory'
        elif (520 <= val <= 579) or val == 787:
            return 'Digestive'
        elif 250 <= val < 251:
            return 'Diabetes'
        elif 800 <= val <= 999:
            return 'Injury'
        elif 710 <= val <= 739:
            return 'Musculoskeletal'
        elif (580 <= val <= 629) or val == 788:
            return 'Genitourinary'
        elif 140 <= val <= 239:
            return 'Neoplasms'
        else:
            return 'Other'
    except ValueError:
        return 'Other'


def categorize_admission_type(val: Any) -> str:
    try:
        v = int(val)
        if v in [1, 2]:
            return 'Emergency_Urgent'
        elif v == 3:
            return 'Elective'
        elif v == 4:
            return 'Newborn'
        elif v == 7:
            return 'Trauma'
        else:
            return 'Other'
    except (ValueError, TypeError):
        return 'Other'


def categorize_discharge_disposition(val: Any) -> str:
    try:
        v = int(val)
        if v == 1:
            return 'Home'
        elif v in [3, 6]:
            return 'Home_Health_or_SNF'
        elif v in [2, 4, 5, 22]:
            return 'Transferred_Facility'
        elif v == 7:
            return 'Left_AMA'
        else:
            return 'Other'
    except (ValueError, TypeError):
        return 'Other'


def categorize_admission_source(val: Any) -> str:
    try:
        v = int(val)
        if v == 7:
            return 'Emergency_Room'
        elif v in [1, 2]:
            return 'Physician_Clinic_Referral'
        elif v in [4, 5, 6]:
            return 'Transfer_Facility'
        else:
            return 'Other'
    except (ValueError, TypeError):
        return 'Other'


class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that transforms raw EHR encounter data
    into rich clinical predictive features.
    """

    def __init__(self):
        self.categorical_cols = CATEGORICAL_FEATURES
        self.numerical_cols = NUMERICAL_FEATURES

    def fit(self, X: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]], y=None):
        return self

    def transform(self, X: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]) -> pd.DataFrame:
        """
        Transforms input dictionary or DataFrame into engineered feature DataFrame.
        """
        if isinstance(X, dict):
            df = pd.DataFrame([X])
        elif isinstance(X, list):
            df = pd.DataFrame(X)
        else:
            df = X.copy()

        # Fill potential missing columns with safe defaults
        default_fields = {
            'race': 'Unknown',
            'gender': 'Unknown',
            'age': '[50-60)',
            'admission_type_id': 1,
            'discharge_disposition_id': 1,
            'admission_source_id': 7,
            'time_in_hospital': 1,
            'num_lab_procedures': 0,
            'num_procedures': 0,
            'num_medications': 0,
            'number_outpatient': 0,
            'number_emergency': 0,
            'number_inpatient': 0,
            'diag_1': '?',
            'diag_2': '?',
            'diag_3': '?',
            'number_diagnoses': 1,
            'max_glu_serum': 'None',
            'A1Cresult': 'None',
            'change': 'No',
            'diabetesMed': 'No',
            'insulin': 'No'
        }
        for col, default_val in default_fields.items():
            if col not in df.columns:
                df[col] = default_val

        for med in DIABETES_MEDICATIONS:
            if med not in df.columns:
                df[med] = 'No'

        # 1. Demographics & Age
        mapped_age = df['age'].astype(str).map(AGE_MIDPOINT_MAP)
        num_age = pd.to_numeric(df['age'], errors='coerce')
        df['age_num'] = mapped_age.fillna(num_age).fillna(55.0).astype(float)
        df['age_over_65'] = (df['age_num'] >= 65.0).astype(float)

        race_clean = df['race'].astype(str).replace({'?': 'Unknown', 'nan': 'Unknown'})
        df['race_clean'] = race_clean
        gender_clean = df['gender'].astype(str).replace({'Unknown/Invalid': 'Unknown', '?': 'Unknown', 'nan': 'Unknown'})
        df['gender_clean'] = gender_clean

        # 2. Clinical ICD-9 Categorization
        df['diag_1_group'] = df['diag_1'].apply(map_icd9_to_clinical_category)
        df['diag_2_group'] = df['diag_2'].apply(map_icd9_to_clinical_category)
        df['diag_3_group'] = df['diag_3'].apply(map_icd9_to_clinical_category)

        for cat in ['Circulatory', 'Respiratory', 'Digestive', 'Diabetes', 'Genitourinary']:
            df[f'has_{cat.lower()}_diag'] = (
                (df['diag_1_group'] == cat) |
                (df['diag_2_group'] == cat) |
                (df['diag_3_group'] == cat)
            ).astype(float)

        # 3. Clinical HbA1c & Blood Glucose Testing
        df['A1Cresult'] = df['A1Cresult'].fillna('None').astype(str)
        df['max_glu_serum'] = df['max_glu_serum'].fillna('None').astype(str)

        df['a1c_tested'] = (df['A1Cresult'] != 'None').astype(float)
        df['a1c_elevated'] = df['A1Cresult'].isin(['>7', '>8']).astype(float)
        df['a1c_high'] = (df['A1Cresult'] == '>8').astype(float)

        df['glucose_tested'] = (df['max_glu_serum'] != 'None').astype(float)
        df['glucose_high'] = df['max_glu_serum'].isin(['>200', '>300']).astype(float)

        # 4. Healthcare Utilization
        for c in ['number_outpatient', 'number_emergency', 'number_inpatient']:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(float)

        df['total_prior_visits'] = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']
        df['had_prior_inpatient'] = (df['number_inpatient'] > 0).astype(float)
        df['had_prior_emergency'] = (df['number_emergency'] > 0).astype(float)
        df['high_utilizer'] = (df['total_prior_visits'] >= 3).astype(float)

        # 5. Medication Regimen Complexity
        for med in DIABETES_MEDICATIONS:
            df[med] = df[med].fillna('No').astype(str)

        df['num_active_diabetes_meds'] = df[DIABETES_MEDICATIONS].apply(
            lambda row: (row != 'No').sum(), axis=1
        ).astype(float)

        df['num_adjusted_diabetes_meds'] = df[DIABETES_MEDICATIONS].apply(
            lambda row: row.isin(['Up', 'Down']).sum(), axis=1
        ).astype(float)

        df['on_insulin'] = (df['insulin'] != 'No').astype(float)
        df['insulin_adjusted'] = df['insulin'].isin(['Up', 'Down']).astype(float)
        df['med_change_binary'] = (df['change'].isin(['Ch', 'yes', 'Yes'])).astype(float)
        df['diabetes_med_prescribed'] = (df['diabetesMed'].isin(['YES', 'Yes', 'yes'])).astype(float)

        # 6. Length of Stay & Intensity Metrics
        for c in ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_diagnoses']:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(float)

        df['meds_per_day'] = df['num_medications'] / (df['time_in_hospital'] + 1.0)
        df['lab_per_day'] = df['num_lab_procedures'] / (df['time_in_hospital'] + 1.0)
        df['procedures_per_day'] = df['num_procedures'] / (df['time_in_hospital'] + 1.0)
        df['high_risk_complexity'] = ((df['number_diagnoses'] >= 8) & (df['num_medications'] >= 15)).astype(float)

        # 7. Admission & Discharge Groups
        df['admission_type_grp'] = df['admission_type_id'].apply(categorize_admission_type)
        df['discharge_disp_grp'] = df['discharge_disposition_id'].apply(categorize_discharge_disposition)
        df['admission_src_grp'] = df['admission_source_id'].apply(categorize_admission_source)

        return df[self.categorical_cols + self.numerical_cols]
