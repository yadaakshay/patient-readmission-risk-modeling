"""
Target definition and clinical cohort construction module.

Target Definition:
- Positive class (1): Readmitted within 30 days ('<30')
- Negative class (0): Not readmitted ('NO') or readmitted after 30 days ('>30')

CMS Exclusion Criteria:
- Discharges resulting in in-hospital mortality or hospice transfer (codes 11, 13, 14, 19, 20, 21).
  Deceased patients cannot physiologically be readmitted to acute care; retaining them introduces target distortion.

Limitations:
- Incomplete follow-up: Out-of-network readmissions to non-participating hospitals are unobserved.
- Post-discharge mortality: Deaths occurring after discharge outside the hospital are not recorded in this EHR.
- Planned vs. unplanned: The historical dataset does not distinguish planned chemotherapy/surgical follow-ups from emergency readmissions.
"""
from typing import Tuple, Optional
import pandas as pd
import numpy as np

from src.config import EXCLUDED_DISCHARGE_IDS, RANDOM_SEED


def create_readmission_target(df: pd.DataFrame) -> pd.Series:
    """
    Creates the binary readmission target:
    1: Readmission within 30 days ('<30')
    0: Not readmitted ('NO') or readmitted after 30 days ('>30')
    """
    if 'readmitted' not in df.columns:
        raise KeyError("Column 'readmitted' not found in DataFrame.")
    return (df['readmitted'].astype(str).str.strip() == '<30').astype(int)


def filter_clinical_exclusions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Excludes patients who expired in hospital or were discharged to hospice.
    """
    if 'discharge_disposition_id' not in df.columns:
        return df.copy()
    return df[~df['discharge_disposition_id'].isin(EXCLUDED_DISCHARGE_IDS)].copy()


def prepare_cohort(
    df: pd.DataFrame,
    deduplicate_patients: bool = True
) -> pd.DataFrame:
    """
    Prepares the clinical analysis cohort:
    1. Filters hospice and mortality exclusions.
    2. Deduplicates to first encounter per patient if deduplicate_patients=True (zero patient leakage).
    3. Adds binary target column.
    """
    cohort = filter_clinical_exclusions(df)
    if deduplicate_patients and 'patient_nbr' in cohort.columns:
        cohort = cohort.drop_duplicates(subset=['patient_nbr'], keep='first').copy()
    
    cohort['target'] = create_readmission_target(cohort)
    return cohort.reset_index(drop=True)
