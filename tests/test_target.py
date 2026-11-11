"""
Unit tests for target definition, CMS exclusion criteria, and patient cohort preparation.
"""
import pytest
import pandas as pd
import numpy as np

from src.data.target import create_readmission_target, filter_clinical_exclusions, prepare_cohort


def test_target_creation_binary():
    df = pd.DataFrame({
        'readmitted': ['<30', '>30', 'NO', '<30', 'NO', '>30']
    })
    target = create_readmission_target(df)
    assert list(target) == [1, 0, 0, 1, 0, 0]


def test_target_creation_missing_col():
    df = pd.DataFrame({'other_col': [1, 2]})
    with pytest.raises(KeyError):
        create_readmission_target(df)


def test_cms_exclusion_criteria():
    df = pd.DataFrame({
        'patient_nbr': [1, 2, 3, 4, 5, 6, 7],
        'discharge_disposition_id': [1, 11, 3, 13, 6, 14, 19],
        'readmitted': ['NO', '<30', 'NO', '<30', '>30', '<30', 'NO']
    })
    # 11, 13, 14, 19 are excluded (mortality/hospice)
    filtered = filter_clinical_exclusions(df)
    assert len(filtered) == 3
    assert set(filtered['discharge_disposition_id'].tolist()) == {1, 3, 6}


def test_patient_deduplication_zero_leakage():
    df = pd.DataFrame({
        'patient_nbr': [101, 102, 101, 103, 102, 104],
        'discharge_disposition_id': [1, 1, 1, 1, 1, 1],
        'readmitted': ['<30', 'NO', '>30', 'NO', '<30', '<30']
    })
    cohort = prepare_cohort(df, deduplicate_patients=True)
    # Must have 4 unique patients
    assert len(cohort) == 4
    assert cohort['patient_nbr'].tolist() == [101, 102, 103, 104]
    # First encounter of 101 had '<30' -> 1
    assert cohort.loc[cohort['patient_nbr'] == 101, 'target'].iloc[0] == 1
    # First encounter of 102 had 'NO' -> 0
    assert cohort.loc[cohort['patient_nbr'] == 102, 'target'].iloc[0] == 0
