"""
Configuration constants, paths, and feature definitions for Patient Readmission Risk Modeling.
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "diabetic_data.csv"
INTERIM_DATA_DIR = DATA_DIR / "interim"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "best_model.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
OUTPUT_DIR = BASE_DIR / "output_images"

# Ensure directories exist
INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Random seed for reproducibility
RANDOM_SEED = 42

# CMS & Clinical Exclusion Criteria:
# Discharge disposition codes indicating in-hospital mortality or hospice discharge.
# 11: Expired
# 13: Hospice / home
# 14: Hospice / medical facility
# 19: Expired at home, Medicaid/Medicare hospice
# 20: Expired in a medical facility, Medicaid/Medicare hospice
# 21: Expired, place unknown
EXCLUDED_DISCHARGE_IDS = [11, 13, 14, 19, 20, 21]

# Medications tracked in the EHR dataset (23 medications)
DIABETES_MEDICATIONS = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
    'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
    'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
    'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
    'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
]

# Age group to numeric midpoint mapping
AGE_MIDPOINT_MAP = {
    '[0-10)': 5,
    '[10-20)': 15,
    '[20-30)': 25,
    '[30-40)': 35,
    '[40-50)': 45,
    '[50-60)': 55,
    '[60-70)': 65,
    '[70-80)': 75,
    '[80-90)': 85,
    '[90-100)': 95
}

# Categorical features for one-hot encoding
CATEGORICAL_FEATURES = [
    'race_clean',
    'gender_clean',
    'diag_1_group',
    'diag_2_group',
    'diag_3_group',
    'admission_type_grp',
    'discharge_disp_grp',
    'admission_src_grp',
    'A1Cresult',
    'max_glu_serum',
    'insulin'
]

# Numerical features for scaling
NUMERICAL_FEATURES = [
    'age_num',
    'age_over_65',
    'time_in_hospital',
    'num_lab_procedures',
    'num_procedures',
    'num_medications',
    'number_outpatient',
    'number_emergency',
    'number_inpatient',
    'number_diagnoses',
    'total_prior_visits',
    'had_prior_inpatient',
    'had_prior_emergency',
    'high_utilizer',
    'num_active_diabetes_meds',
    'num_adjusted_diabetes_meds',
    'on_insulin',
    'insulin_adjusted',
    'med_change_binary',
    'diabetes_med_prescribed',
    'meds_per_day',
    'lab_per_day',
    'procedures_per_day',
    'high_risk_complexity',
    'a1c_tested',
    'a1c_elevated',
    'a1c_high',
    'glucose_tested',
    'glucose_high',
    'has_circulatory_diag',
    'has_respiratory_diag',
    'has_digestive_diag',
    'has_diabetes_diag',
    'has_genitourinary_diag'
]
