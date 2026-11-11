"""
Analytical Database initialization module (SQLite / DuckDB).
Persists cleaned clinical encounters, model predictions, and evaluation metrics.
"""
import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "processed" / "readmission_analytics.db"


def initialize_analytical_database(
    df_cohort: pd.DataFrame,
    df_test_predictions: pd.DataFrame,
    metrics_df: pd.DataFrame,
    db_path: Path = DB_PATH
):
    """
    Creates and loads the analytical database tables:
    - encounters: Cohort-level clinical encounter records
    - predictions: Scored test patient encounters with predicted probability & risk tiers
    - model_benchmark: Cross-validation and test benchmark metrics
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. Store encounters table
    # Select clinically relevant columns
    encounter_cols = [
        'encounter_id', 'patient_nbr', 'race', 'gender', 'age',
        'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
        'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
        'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses',
        'diag_1', 'diag_2', 'diag_3', 'max_glu_serum', 'A1Cresult', 'insulin',
        'change', 'diabetesMed', 'readmitted', 'target'
    ]
    avail_cols = [c for c in encounter_cols if c in df_cohort.columns]
    df_enc = df_cohort[avail_cols].copy()
    df_enc.to_sql('encounters', conn, if_exists='replace', index=False)

    # 2. Store predictions table
    pred_cols = [
        'encounter_id', 'patient_nbr', 'race', 'gender', 'age',
        'admission_type_id', 'discharge_disposition_id', 'time_in_hospital',
        'number_inpatient', 'number_emergency', 'number_outpatient',
        'num_medications', 'number_diagnoses', 'A1Cresult', 'insulin',
        'target', 'readmission_prob', 'predicted_flag', 'prediction_tier'
    ]
    avail_pred_cols = [c for c in pred_cols if c in df_test_predictions.columns]
    df_pred = df_test_predictions[avail_pred_cols].copy()
    
    # Calculate error category (TP, FP, TN, FN)
    df_pred['error_type'] = np.where(
        (df_pred['target'] == 1) & (df_pred['predicted_flag'] == 1), 'True Positive',
        np.where(
            (df_pred['target'] == 0) & (df_pred['predicted_flag'] == 0), 'True Negative',
            np.where(
                (df_pred['target'] == 0) & (df_pred['predicted_flag'] == 1), 'False Positive',
                'False Negative'
            )
        )
    )
    df_pred.to_sql('predictions', conn, if_exists='replace', index=False)

    # 3. Store model benchmark table
    metrics_df.to_sql('model_benchmark', conn, if_exists='replace', index=False)

    # Create helpful indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_enc_patient ON encounters(patient_nbr);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_enc_target ON encounters(target);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_tier ON predictions(prediction_tier);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_error ON predictions(error_type);")

    conn.commit()
    conn.close()
    print(f"Analytical database successfully initialized at {db_path}")
    return db_path
