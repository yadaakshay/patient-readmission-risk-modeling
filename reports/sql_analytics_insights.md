# Clinical SQL Analytics & Subgroup Insights Report

## Overall Readmission Prevalence & CMS Baseline
**Business / Clinical Question**: *What is the baseline 30-day readmission prevalence across the clean diabetic inpatient cohort?*

```sql
SELECT 
            COUNT(*) AS total_encounters,
            SUM(target) AS readmitted_within_30_days,
            ROUND(AVG(target) * 100.0, 2) AS readmission_rate_pct,
            COUNT(DISTINCT patient_nbr) AS unique_patients
        FROM encounters;
```

**Query Result Summary**:

|   total_encounters |   readmitted_within_30_days |   readmission_rate_pct |   unique_patients |
|-------------------:|----------------------------:|-----------------------:|------------------:|
|              69990 |                        6285 |                   8.98 |             69990 |

---

## Readmission Rates by Age Bracket
**Business / Clinical Question**: *How does 30-day readmission risk vary across age brackets?*

```sql
SELECT 
            age,
            COUNT(*) AS encounter_count,
            SUM(target) AS readmissions_30d,
            ROUND(AVG(target) * 100.0, 2) AS readmission_rate_pct,
            CASE WHEN COUNT(*) < 100 THEN 'Small Sample (<100)' ELSE 'Adequate Sample' END AS sample_size_flag
        FROM encounters
        GROUP BY age
        ORDER BY age;
```

**Query Result Summary**:

| age      |   encounter_count |   readmissions_30d |   readmission_rate_pct | sample_size_flag   |
|:---------|------------------:|-------------------:|-----------------------:|:-------------------|
| [0-10)   |               153 |                  3 |                   1.96 | Adequate Sample    |
| [10-20)  |               534 |                 26 |                   4.87 | Adequate Sample    |
| [20-30)  |              1121 |                 83 |                   7.4  | Adequate Sample    |
| [30-40)  |              2692 |                188 |                   6.98 | Adequate Sample    |
| [40-50)  |              6828 |                506 |                   7.41 | Adequate Sample    |
| [50-60)  |             12351 |                880 |                   7.12 | Adequate Sample    |
| [60-70)  |             15689 |               1415 |                   9.02 | Adequate Sample    |
| [70-80)  |             17751 |               1817 |                  10.24 | Adequate Sample    |
| [80-90)  |             11110 |               1199 |                  10.79 | Adequate Sample    |
| [90-100) |              1761 |                168 |                   9.54 | Adequate Sample    |

---

## Readmission Rates by Hospital Length of Stay
**Business / Clinical Question**: *Is prolonged acute hospitalization (>7 days) associated with higher readmission risk?*

```sql
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
```

**Query Result Summary**:

| los_category       |   total_patients |   readmitted_patients |   readmission_rate_pct |   avg_medications_prescribed |
|:-------------------|-----------------:|----------------------:|-----------------------:|-----------------------------:|
| 1-2 Days (Short)   |            22653 |                  1621 |                   7.16 |                         11.6 |
| 3-5 Days (Medium)  |            28615 |                  2516 |                   8.79 |                         15.4 |
| 6-8 Days (Long)    |            11796 |                  1346 |                  11.41 |                         19.5 |
| 9+ Days (Extended) |             6926 |                   802 |                  11.58 |                         23.6 |

---

## Readmission Risk by Prior Inpatient Admissions
**Business / Clinical Question**: *What is the relationship between prior inpatient visits in the preceding year and 30-day readmission?*

```sql
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
```

**Query Result Summary**:

| prior_inpatient_tier                |   total_encounters |   readmissions |   readmission_rate_pct |   avg_comorbidities |
|:------------------------------------|-------------------:|---------------:|-----------------------:|--------------------:|
| 0 Prior Admissions                  |              61795 |           5023 |                   8.13 |                 7.2 |
| 1 Prior Admission                   |               5798 |            747 |                  12.88 |                 7.6 |
| 2 Prior Admissions                  |               1501 |            278 |                  18.52 |                 7.7 |
| 3+ Prior Admissions (High Utilizer) |                896 |            237 |                  26.45 |                 7.8 |

---

## Readmission Rates by Discharge Disposition
**Business / Clinical Question**: *How do post-acute discharge pathways (Home vs Home Health / SNF vs Transfer) correlate with readmission?*

```sql
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
```

**Query Result Summary**:

| discharge_destination             |   encounter_count |   readmissions |   readmission_rate_pct | reliability_flag   |
|:----------------------------------|------------------:|---------------:|-----------------------:|:-------------------|
| Transferred to Acute/Rehab        |              4407 |            828 |                  18.79 | Reliable           |
| Home Health Care or SNF           |             17081 |           1971 |                  11.54 | Reliable           |
| Other Disposition                 |              3771 |            368 |                   9.76 | Reliable           |
| Left Against Medical Advice (AMA) |               409 |             39 |                   9.54 | Reliable           |
| Discharged to Home                |             44322 |           3079 |                   6.95 | Reliable           |

---

## Model Prediction Distribution across Risk Tiers
**Business / Clinical Question**: *How are test cohort patients distributed across Low, Moderate, and High risk tiers, and what is the empirical readmission rate in each?*

```sql
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
```

**Query Result Summary**:

| prediction_tier   |   patient_volume |   volume_share_pct |   observed_readmissions |   observed_readmission_rate_pct |   mean_predicted_risk_pct |   min_predicted_risk_pct |   max_predicted_risk_pct |
|:------------------|-----------------:|-------------------:|------------------------:|--------------------------------:|--------------------------:|-------------------------:|-------------------------:|
| Low               |             3292 |               31.4 |                     182 |                            5.53 |                      5.02 |                     1.7  |                     6.42 |
| Moderate          |             5040 |               48   |                     424 |                            8.41 |                      8.65 |                     6.42 |                    11.55 |
| High              |             2167 |               20.6 |                     337 |                           15.55 |                     15.57 |                    11.55 |                    38.8  |

---

## Confusion Matrix & Prediction Error Characteristics
**Business / Clinical Question**: *What are the clinical characteristics of False Positives vs False Negatives in the test set?*

```sql
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
```

**Query Result Summary**:

| error_type     |   patient_count |   share_of_test_cohort_pct |   mean_predicted_prob |   avg_length_of_stay |   avg_prior_inpatient |   avg_medications |   avg_diagnoses |
|:---------------|----------------:|---------------------------:|----------------------:|---------------------:|----------------------:|------------------:|----------------:|
| True Negative  |            5536 |                       52.7 |                 0.061 |                  3.2 |                  0.03 |              13.9 |             6.7 |
| False Positive |            4020 |                       38.3 |                 0.124 |                  5.6 |                  0.34 |              17.7 |             7.9 |
| True Positive  |             578 |                        5.5 |                 0.142 |                  5.7 |                  0.6  |              18.2 |             7.9 |
| False Negative |             365 |                        3.5 |                 0.064 |                  3.3 |                  0.04 |              14.6 |             7   |

---

## Model Performance Across Clinical Subgroups
**Business / Clinical Question**: *How does model sensitivity and precision perform across key vulnerable subgroups (Elderly 65+, Prior Inpatient History, Insulin Users)?*

```sql
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
```

**Query Result Summary**:

| subgroup                  |   total_patients |   actual_readmissions |   flagged_by_model |   true_positives |   sensitivity_pct |   precision_pct |
|:--------------------------|-----------------:|----------------------:|-------------------:|-----------------:|------------------:|----------------:|
| Elderly (Age 65+)         |             6966 |                   694 |               3784 |              479 |              69   |            12.7 |
| Prior Inpatient Admission |             1235 |                   187 |               1085 |              173 |              92.5 |            15.9 |
| Insulin Therapy Active    |             5318 |                   515 |               2623 |              339 |              65.8 |            12.9 |

---

