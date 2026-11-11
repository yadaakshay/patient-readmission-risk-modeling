# Exploratory Data Analysis & Clinical Cohort Profile

## 1. Dataset Dimensions & Completeness
- **Raw Encounter Count**: 101,766 hospital encounters
- **Unique Patients**: 71,518
- **Encounter Overlap**: 16,773 patients (23.5%) have multiple encounters (up to 40 encounters for a single patient).
- **Missing Value Audit**:
  - `weight`: 96.86% missing (dropped due to extreme sparsity).
  - `payer_code`: 39.56% missing (administrative feature, non-clinical).
  - `medical_specialty`: 49.08% missing.
  - `race`: 2.23% marked with '?' (imputed as 'Unknown' / 'Other').
  - `diag_1`, `diag_2`, `diag_3`: < 1.5% missing codes mapped to 'Missing'.

## 2. Target Definition & Class Imbalance
- Raw 3-Class Target:
  - `NO`: 54,864 (53.91%)
  - `>30`: 35,545 (34.93%)
  - `<30`: 11,357 (11.16%)
- **Binary Classification Target**:
  - Positive Class (`1`): Readmission strictly within 30 days (`<30`)
  - Negative Class (`0`): Not readmitted or readmitted after 30 days (`NO` or `>30`)
- **Clinical Cohort (CMS HRRP Criteria)**:
  - Excluded 2,423 records corresponding to in-hospital mortality or hospice discharge (discharge disposition IDs 11, 13, 14, 19, 20, 21).
  - Filtered to index encounter per patient to prevent multi-encounter patient leakage: 69,990 patients.
  - Positive class prevalence in final cohort: **8.98%** (6,285 cases).

## 3. Key Exploratory Findings
1. **Prior Utilization Gradient**: Patients with 4+ prior visits experience readmission rates >16%, roughly double the baseline prevalence (~8.8%). Prior inpatient admissions are particularly strong correlational markers.
2. **Length of Stay**: Patients readmitted within 30 days stayed on average longer (median 4 days, IQR 2-6) vs non-readmitted (median 3 days).
3. **Polypharmacy & Comorbidity**: Increased number of inpatient medications and diagnoses is strongly associated with elevated readmission risk.
4. **Glycemic Testing**: HbA1c testing was ordered in only ~18% of encounters; unmonitored glycemic status and elevated HbA1c (>8%) represent key clinical complexity indicators.

> [!NOTE]
> All observed relationships are statistical associations from historical observational EHR records. They do not represent causal effects.
