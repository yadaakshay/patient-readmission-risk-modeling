"""
Data loading and cohort preparation module.
Handles CMS exclusion criteria, patient deduplication to prevent data leakage,
and stratified patient-level dataset splitting.
"""
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.config import RAW_DATA_PATH, INTERIM_DATA_DIR, EXCLUDED_DISCHARGE_IDS, RANDOM_SEED


def load_raw_data(data_path=RAW_DATA_PATH) -> pd.DataFrame:
    """Loads the raw EHR CSV dataset."""
    df = pd.read_csv(data_path)
    return df


def prepare_clinical_cohort(df: pd.DataFrame, keep_first_encounter: bool = True) -> pd.DataFrame:
    """
    Applies clinical exclusion criteria and handles patient encounter grouping:
    1. Removes deceased and hospice-discharged patients (disposition codes 11, 13, 14, 19, 20, 21).
    2. Keeps only the first encounter per patient to eliminate repeat-visit correlation bias.
    3. Defines binary target: readmitted within 30 days ('<30' -> 1, otherwise -> 0).
    """
    # Exclude in-hospital mortality and hospice
    df_clean = df[~df['discharge_disposition_id'].isin(EXCLUDED_DISCHARGE_IDS)].copy()

    if keep_first_encounter and 'patient_nbr' in df_clean.columns:
        df_clean = df_clean.drop_duplicates(subset=['patient_nbr'], keep='first').copy()

    # Define binary target for 30-day hospital readmission
    df_clean['target'] = (df_clean['readmitted'] == '<30').astype(int)

    df_clean = df_clean.reset_index(drop=True)
    return df_clean


def split_patient_cohort(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the patient cohort into Train, Validation, and Test sets stratified by target.
    Since repeat encounters were deduplicated per patient, patient leakage is strictly zero.
    """
    # Split train+val vs test
    df_train_val, df_test = train_test_split(
        df,
        test_size=test_size,
        stratify=df['target'],
        random_state=random_state
    )

    # Relative validation size
    val_relative = val_size / (1.0 - test_size)
    df_train, df_val = train_test_split(
        df_train_val,
        test_size=val_relative,
        stratify=df_train_val['target'],
        random_state=random_state
    )

    return (
        df_train.reset_index(drop=True),
        df_val.reset_index(drop=True),
        df_test.reset_index(drop=True)
    )


def save_splits(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame):
    """Saves interim parquet splits for fast notebook experimentation."""
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_train.to_parquet(INTERIM_DATA_DIR / "df_train.parquet", index=False)
    df_val.to_parquet(INTERIM_DATA_DIR / "df_val.parquet", index=False)
    df_test.to_parquet(INTERIM_DATA_DIR / "df_test.parquet", index=False)
    print(f"Saved interim splits to {INTERIM_DATA_DIR}")
