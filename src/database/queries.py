"""
SQL Analytics queries module.
Defines and executes 8 business-driven SQL queries for clinical healthcare intelligence.
"""
from pathlib import Path
from typing import Dict, List, Any
import sqlite3
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "processed" / "readmission_analytics.db"


CLINICAL_QUERIES = {
    # 1. Overall Readmission Prevalence
    "1_readmission_prevalence": {
        "title": "Overall Readmission Prevalence & CMS Baseline",
        "question": "What is the baseline 30-day readmission prevalence across the clean diabetic inpatient cohort?",
        "sql": """
        SELECT 
            COUNT(*) AS total_encounters,
            SUM(target) AS readmitted_within_30_days,
            ROUND(AVG(target) * 100.0, 2) AS readmission_rate_pct,
            COUNT(DISTINCT patient_nbr) AS unique_patients
        FROM encounters;
        """
    },

    # 2. Readmission Rates by Age Group
    "2_readmission_by_age": {
        "title": "Readmission Rates by Age Bracket",
        "question": "How does 30-day readmission risk vary across age brackets?",
        "sql": """
        SELECT 
            age,
            COUNT(*) AS encounter_count,
            SUM(target) AS readmissions_30d,
            ROUND(AVG(target) * 100.0, 2) AS readmission_rate_pct,
            CASE WHEN COUNT(*) < 100 THEN 'Small Sample (<100)' ELSE 'Adequate Sample' END AS sample_size_flag
        FROM encounters
        GROUP BY age
        ORDER BY age;
        """
    },

    # 3. Readmission Rates by Length of Stay
    "3_readmission_by_los": {
        "title": "Readmission Rates by Hospital Length of Stay",
        "question": "Is prolonged acute hospitalization (>7 days) associated with higher readmission risk?",
        "sql": """
        SELECT 
            CASE 
                WHEN time_in_hospital BETWEEN 1 AND 2 THEN '1-2 Days (Short)'
                WHEN time_in_hospital BETWEEN 3 AND 5 THEN '3-5 Days (Medium)'
                WHEN time_in_hospital BETWEEN 6 AND 8 THEN '6-8 Days (Long)'
                ELSE '9+ Days (Extended)'
            END AS los_category,
            COUNT(*) AS total_patients,
            SUM(target) AS readmitted_patients,
            ROUND(AVG(target) * 100.0, 2) AS readmission_rate_pct,
            ROUND(AVG(num_medications), 1) AS avg_medications_prescribed
        FROM encounters
        GROUP BY los_category
        ORDER BY MIN(time_in_hospital);
        """
    },

    # 4. Readmission Rates by Prior Healthcare Utilization
    "4_readmission_by_prior_utilization": {
        "title": "Readmission Risk by Prior Inpatient Admissions",
        "question": "What is the relationship between prior inpatient visits in the preceding year and 30-day readmission?",
        "sql": """
        SELECT 
            CASE 
                WHEN number_inpatient = 0 THEN '0 Prior Admissions'
                WHEN number_inpatient = 1 THEN '1 Prior Admission'
                WHEN number_inpatient = 2 THEN '2 Prior Admissions'
                ELSE '3+ Prior Admissions (High Utilizer)'
            END AS prior_inpatient_tier,
            COUNT(*) AS total_encounters,
            SUM(target) AS readmissions,
            ROUND(AVG(target) * 100.0, 2) AS readmission_rate_pct,
            ROUND(AVG(number_diagnoses), 1) AS avg_comorbidities
        FROM encounters
        GROUP BY prior_inpatient_tier
        ORDER BY MIN(number_inpatient);
        """
    },

    # 5. Readmission Rates by Admission Type & Discharge Disposition
    "5_readmission_by_admission_discharge": {
        "title": "Readmission Rates by Discharge Disposition",
        "question": "How do post-acute discharge pathways (Home vs Home Health / SNF vs Transfer) correlate with readmission?",
        "sql": """
        SELECT 
            CASE 
                WHEN discharge_disposition_id = 1 THEN 'Discharged to Home'
                WHEN discharge_disposition_id IN (3, 6) THEN 'Home Health Care or SNF'
                WHEN discharge_disposition_id IN (2, 4, 5, 22) THEN 'Transferred to Acute/Rehab'
                WHEN discharge_disposition_id = 7 THEN 'Left Against Medical Advice (AMA)'
                ELSE 'Other Disposition'
            END AS discharge_destination,
            COUNT(*) AS encounter_count,
            SUM(target) AS readmissions,
            ROUND(AVG(target) * 100.0, 2) AS readmission_rate_pct,
            CASE WHEN COUNT(*) < 100 THEN 'Small Subgroup' ELSE 'Reliable' END AS reliability_flag
        FROM encounters
        GROUP BY discharge_destination
        ORDER BY readmission_rate_pct DESC;
        """
    },

    # 6. Model Prediction Distribution across Risk Tiers
    "6_prediction_distribution": {
        "title": "Model Prediction Distribution across Risk Tiers",
        "question": "How are test cohort patients distributed across Low, Moderate, and High risk tiers, and what is the empirical readmission rate in each?",
        "sql": """
        SELECT 
            prediction_tier,
            COUNT(*) AS patient_volume,
            ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM predictions), 1) AS volume_share_pct,
            SUM(target) AS observed_readmissions,
            ROUND(AVG(target) * 100.0, 2) AS observed_readmission_rate_pct,
            ROUND(AVG(readmission_prob) * 100.0, 2) AS mean_predicted_risk_pct,
            ROUND(MIN(readmission_prob) * 100.0, 2) AS min_predicted_risk_pct,
            ROUND(MAX(readmission_prob) * 100.0, 2) AS max_predicted_risk_pct
        FROM predictions
        GROUP BY prediction_tier
        ORDER BY mean_predicted_risk_pct ASC;
        """
    },

    # 7. False Positives & False Negatives Error Profile
    "7_error_analysis": {
        "title": "Confusion Matrix & Prediction Error Characteristics",
        "question": "What are the clinical characteristics of False Positives vs False Negatives in the test set?",
        "sql": """
        SELECT 
            error_type,
            COUNT(*) AS patient_count,
            ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM predictions), 1) AS share_of_test_cohort_pct,
            ROUND(AVG(readmission_prob), 3) AS mean_predicted_prob,
            ROUND(AVG(time_in_hospital), 1) AS avg_length_of_stay,
            ROUND(AVG(number_inpatient), 2) AS avg_prior_inpatient,
            ROUND(AVG(num_medications), 1) AS avg_medications,
            ROUND(AVG(number_diagnoses), 1) AS avg_diagnoses
        FROM predictions
        GROUP BY error_type
        ORDER BY patient_count DESC;
        """
    },

    # 8. Model Subgroup Performance
    "8_subgroup_performance": {
        "title": "Model Performance Across Clinical Subgroups",
        "question": "How does model sensitivity and precision perform across key vulnerable subgroups (Elderly 65+, Prior Inpatient History, Insulin Users)?",
        "sql": """
        SELECT 
            'Elderly (Age 65+)' AS subgroup,
            COUNT(*) AS total_patients,
            SUM(target) AS actual_readmissions,
            SUM(predicted_flag) AS flagged_by_model,
            SUM(CASE WHEN target = 1 AND predicted_flag = 1 THEN 1 ELSE 0 END) AS true_positives,
            ROUND(SUM(CASE WHEN target = 1 AND predicted_flag = 1 THEN 1.0 ELSE 0.0 END) / NULLIF(SUM(target), 0) * 100.0, 1) AS sensitivity_pct,
            ROUND(SUM(CASE WHEN target = 1 AND predicted_flag = 1 THEN 1.0 ELSE 0.0 END) / NULLIF(SUM(predicted_flag), 0) * 100.0, 1) AS precision_pct
        FROM predictions
        WHERE age IN ('[60-70)', '[70-80)', '[80-90)', '[90-100)')
        UNION ALL
        SELECT 
            'Prior Inpatient Admission',
            COUNT(*),
            SUM(target),
            SUM(predicted_flag),
            SUM(CASE WHEN target = 1 AND predicted_flag = 1 THEN 1 ELSE 0 END),
            ROUND(SUM(CASE WHEN target = 1 AND predicted_flag = 1 THEN 1.0 ELSE 0.0 END) / NULLIF(SUM(target), 0) * 100.0, 1),
            ROUND(SUM(CASE WHEN target = 1 AND predicted_flag = 1 THEN 1.0 ELSE 0.0 END) / NULLIF(SUM(predicted_flag), 0) * 100.0, 1)
        FROM predictions
        WHERE number_inpatient > 0
        UNION ALL
        SELECT 
            'Insulin Therapy Active',
            COUNT(*),
            SUM(target),
            SUM(predicted_flag),
            SUM(CASE WHEN target = 1 AND predicted_flag = 1 THEN 1 ELSE 0 END),
            ROUND(SUM(CASE WHEN target = 1 AND predicted_flag = 1 THEN 1.0 ELSE 0.0 END) / NULLIF(SUM(target), 0) * 100.0, 1),
            ROUND(SUM(CASE WHEN target = 1 AND predicted_flag = 1 THEN 1.0 ELSE 0.0 END) / NULLIF(SUM(predicted_flag), 0) * 100.0, 1)
        FROM predictions
        WHERE insulin IN ('Steady', 'Up', 'Down');
        """
    }
}


def execute_sql_query(query_key: str, db_path: Path = DB_PATH) -> pd.DataFrame:
    """Executes one of the documented clinical queries and returns a DataFrame."""
    if query_key not in CLINICAL_QUERIES:
        raise KeyError(f"Query key '{query_key}' not found in CLINICAL_QUERIES.")

    sql = CLINICAL_QUERIES[query_key]['sql']
    conn = sqlite3.connect(db_path)
    try:
        df_res = pd.read_sql_query(sql, conn)
    finally:
        conn.close()
    return df_res


def execute_all_queries(db_path: Path = DB_PATH) -> Dict[str, pd.DataFrame]:
    """Runs all 8 queries and returns dictionary of DataFrames."""
    results = {}
    for key in CLINICAL_QUERIES.keys():
        results[key] = execute_sql_query(key, db_path)
    return results
