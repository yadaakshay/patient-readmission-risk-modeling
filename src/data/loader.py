"""
Data loader and dataset splitting module.
Implements reproducible stratified splitting and patient-grouped splitting strategies.
"""
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupKFold, StratifiedKFold

from src.config import RAW_DATA_PATH, INTERIM_DATA_DIR, RANDOM_SEED
from src.data.target import prepare_cohort

PROCESSED_DATA_DIR = RAW_DATA_PATH.parent / "processed"


def load_raw_data(data_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Loads raw EHR dataset from CSV."""
    if not Path(data_path).exists():
        # Fallback to parent data dir if needed
        alt_path = Path(__file__).resolve().parent.parent.parent / "data" / "diabetic_data.csv"
        if alt_path.exists():
            data_path = alt_path
        else:
            raise FileNotFoundError(f"Raw data file not found at {data_path}")
    return pd.read_csv(data_path)


def split_stratified_cohort(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs reproducible stratified splitting into Train (70%), Validation (15%), Test (15%).
    Ensures identical class prevalence across splits.
    """
    # 1. Split Train+Val vs Test
    df_train_val, df_test = train_test_split(
        df,
        test_size=test_size,
        stratify=df['target'],
        random_state=random_state
    )

    # 2. Split Train vs Val
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


def save_dataset_splits(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    output_dir: Path = INTERIM_DATA_DIR
):
    """Persists train, val, test datasets to Parquet."""
    output_dir.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df_train.to_parquet(output_dir / "df_train.parquet", index=False)
    df_val.to_parquet(output_dir / "df_val.parquet", index=False)
    df_test.to_parquet(output_dir / "df_test.parquet", index=False)

    # Also save to processed
    df_train.to_parquet(PROCESSED_DATA_DIR / "train.parquet", index=False)
    df_val.to_parquet(PROCESSED_DATA_DIR / "val.parquet", index=False)
    df_test.to_parquet(PROCESSED_DATA_DIR / "test.parquet", index=False)
