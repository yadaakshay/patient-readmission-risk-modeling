# Patient Readmission Risk Prediction & Clinical Analytics
### *Predicting 30-Day All-Cause Hospital Readmission for Diabetic Inpatients using Electronic Health Records (EHR)*

![Python](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Production_API-009688?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-Interactive_Dashboard-FF4B4B?logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-F7931E?logo=scikitlearn)
![LightGBM](https://img.shields.io/badge/LightGBM-Gradient_Boosting-green)
![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-red)
![SQLite](https://img.shields.io/badge/SQLite-Analytical_Database-003B57?logo=sqlite)
![Tests](https://img.shields.io/badge/Tests-30%20Passed-brightgreen)

An end-to-end clinical machine learning and decision-support analytics system designed to predict 30-day hospital readmission risk among diabetic inpatients. Built on 10 years of clinical EHR data across 130 US hospitals (1999–2008), this project adheres to CMS Hospital Readmissions Reduction Program (HRRP) methodology, rigorous clinical feature engineering, probability calibration, explainable AI (SHAP), SQL healthcare analytics, and care team capacity simulation.

---

## 🏥 Clinical & Business Motivation

Under the **Hospital Readmissions Reduction Program (HRRP)** established by the Centers for Medicare & Medicaid Services (CMS), acute-care hospitals face up to **3% reimbursement penalties** on all Medicare discharges if their 30-day readmission rates exceed national benchmarks.

* **Clinical Goal**: Accurately stratify diabetic patients prior to discharge into actionable risk tiers (**Low**, **Moderate**, **High**).
* **Operational Value**: Enable hospital care management teams to allocate high-intensity transition-of-care interventions (e.g., 72-hour clinical follow-up, nurse navigation, pharmacist medication reconciliation) to the highest-risk patients, preventing avoidable readmissions and regulatory penalties.

---

## 📋 Dataset & Target Definition

### Dataset Source
- **Study**: *Diabetes 130-US Hospitals (1999–2008)*, Strack et al., Hindawi Publishing Corporation, 2014.
- **Dimensions**: 101,766 hospital encounters across 71,518 unique patients, 50 raw features.
- **License / Usage**: Publicly available for educational, scientific, and research purposes.

### Target Definition
- **Positive Class (`1`)**: Hospital readmission within 30 days (`<30`).
- **Negative Class (`0`)**: Not readmitted (`NO`) or readmitted after 30 days (`>30`).

### Target Limitations & Clinical Nuances
1. **Incomplete Follow-Up**: The dataset records readmissions occurring only within the participating 130 hospital network. Readmissions to external hospital systems are unrecorded.
2. **Post-Discharge Mortality**: Patients who pass away outside the hospital within 30 days of discharge cannot be readmitted, but this EHR does not capture out-of-hospital mortality.
3. **Planned vs. Unplanned Readmissions**: The historical data does not distinguish between planned elective re-admissions (e.g., staged surgical procedures or scheduled chemotherapy) and acute emergency readmissions.
4. **Proxy Metric**: Readmission within 30 days serves as a quality proxy, not a direct clinical biomarker of personal diabetes deterioration.

---

## 🛡️ Leakage Prevention & Clinical Cohort Audit

Most public healthcare ML implementations suffer from severe data leakage and methodological oversights. This repository resolves these issues through clinical standards:

1. **Zero Patient Leakage**: Over 16,000 patients in the dataset had multiple hospital encounters (up to 40 encounters for a single patient). Rather than performing random row-level splitting (which leaks encounters of the same chronic patient across train and test sets), we deduplicate to index (first) patient encounters, ensuring strict patient isolation across splits.
2. **Deceased & Hospice Exclusion**: 2,423 patients expired in the hospital or were discharged to hospice (disposition codes 11, 13, 14, 19, 20, 21). Because deceased patients cannot physiologically be readmitted to acute care, retaining them distorts the outcome definition. They are excluded per CMS HRRP guidelines.
3. **Prediction-Time Temporal Audit**: Features were audited to ensure they represent clinical intelligence available strictly prior to discharge planning. Administrative identifiers (`encounter_id`, `patient_nbr`) and extreme missingness fields (`weight` >96% missing, `payer_code` ~40% missing) were removed.
4. **Preservation of Glycemic Indicators (HbA1c & Blood Glucose)**: In the original study (*Strack et al., 2014*), HbA1c testing was ordered in only ~18% of encounters. Treating `'None'` as missing data and dropping `A1Cresult` discards critical clinical intelligence. Here, HbA1c and glucose testing are preserved as vital markers of glycemic monitoring and control.
5. **Data Split Isolation**: Imputers, scalers, encoders, and resampling transformers are fitted strictly inside training folds using `imblearn.pipeline.Pipeline`, guaranteeing zero test data leakage.

---

## 📊 Benchmark Results

### 1. 5-Fold Stratified Cross-Validation Benchmark (Training Cohort: 48,992 Patients)

All models and resampling strategies were evaluated across a rigorous 5-fold stratified cross-validation protocol:

| Model Configuration | Model Family | Imbalance Strategy | 5-Fold ROC-AUC | 5-Fold PR-AUC (Avg Precision) | Brier Score (Calibration) | Balanced Accuracy |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Dummy (Stratified)** | Baseline Dummy | None | 0.4993 ± 0.0022 | 0.0897 ± 0.0003 | 0.1640 | 0.4993 |
| **Logistic Regression (Weighted)** | Linear Model | Class Weighting | 0.6381 ± 0.0050 | 0.1570 ± 0.0062 | 0.0798 | 0.5003 |
| **Random Forest (Weighted)** | Bagged Trees | Class Weighting | 0.6450 ± 0.0039 | 0.1648 ± 0.0048 | 0.0795 | 0.5000 |
| **XGBoost (Weighted)** | Gradient Boosting | Class Weighting | 0.6451 ± 0.0045 | 0.1666 ± 0.0049 | 0.0794 | 0.5000 |
| **LightGBM (Weighted)** | Gradient Boosting | Class Weighting | **0.6457 ± 0.0059** | **0.1693 ± 0.0060** | **0.0794** | 0.5000 |
| **LightGBM (Unweighted)** | Gradient Boosting | None | 0.6424 ± 0.0044 | 0.1660 ± 0.0046 | 0.0794 | 0.5008 |
| **LightGBM (SMOTE)** | Gradient Boosting | SMOTE (imblearn) | 0.6363 ± 0.0033 | 0.1580 ± 0.0044 | 0.0808 | 0.5000 |
| **Logistic Regression (SMOTE)** | Linear Model | SMOTE (imblearn) | 0.6313 ± 0.0082 | 0.1537 ± 0.0058 | 0.1092 | 0.5289 |

### Key Cross-Validation Insights:
* **SMOTE vs. Class Weighting**: In high-dimensional tabular EHR data with sparse one-hot encoded categories, SMOTE interpolation (`PR-AUC = 0.1580`) underperformed algorithmic class weighting (`PR-AUC = 0.1693`). Cost-sensitive loss functions directly penalize minority false negatives without synthesizing artificial points in sparse categorical spaces.
* **Probability Calibration**: Calibrating gradient-boosted trees using Platt scaling (sigmoid) maintained optimal ranking while reducing Brier score loss to **0.0794**, ensuring reliable empirical risk estimates.

---

### 2. Held-Out Untouched Test Cohort Results (10,499 Patients)

| Model Architecture | Threshold | ROC-AUC | PR-AUC | Sensitivity (Recall) | Specificity | Precision | F1-Score | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Baseline** | 0.500 | 0.5000 | 0.0898 | 0.00% | 100.0% | 0.00% | 0.000 | 0.0818 |
| **Logistic Regression** | 0.500 | 0.6331 | 0.1537 | 0.11% | 99.95% | 16.67% | 0.002 | 0.0800 |
| **Random Forest** | 0.500 | 0.6354 | 0.1543 | 0.00% | 100.0% | 0.00% | 0.000 | 0.0799 |
| **XGBoost** | 0.500 | 0.6343 | 0.1589 | 0.00% | 100.0% | 0.00% | 0.000 | 0.0797 |
| **LightGBM (Best Model)** | **0.500** | **0.6335** | **0.1617** | **0.00%** | **100.0%** | **0.00%** | **0.000** | **0.0796** |
| **LightGBM (Optimal Threshold)** | **0.086** | **0.6335** | **0.1617** | **61.29%** | **57.73%** | **12.57%** | **0.2086** | **0.0796** |

---

## ⚠️ The 91% Accuracy Paradox & Decision Threshold Selection

In imbalanced clinical outcomes where positive prevalence is ~9%, **accuracy is a dangerous and misleading metric**:

* **At Standard Default Threshold ($0.500$)**:
  * Accuracy is **91.02%** — but Sensitivity (Recall) is **0.00%**! The model predicts negative for every patient, missing **100% of readmissions** (0 True Positives, 943 False Negatives).
* **At Optimal Clinical Threshold ($0.086$, Youden's J Statistic on Validation Cohort)**:
  * Sensitivity (Recall) rises to **61.29%**, successfully capturing over 61% of all readmissions (578 True Positives detected) for targeted clinical intervention!

| Decision Threshold Strategy | Accuracy | Sensitivity (Recall) | Specificity | Precision | F1 Score | True Positives | False Negatives |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Default Threshold ($0.500$)** | 91.02% | **0.00%** | 100.0% | 0.00% | 0.000 | 0 | 943 |
| **Optimal Clinical Threshold ($0.086$)** | 58.05% | **61.29%** | 57.73% | 12.57% | **0.2086** | **578** | 365 |

---

## 🔍 Explainable AI (SHAP) & Clinical Transparency

Model predictions are interpreted using SHapley Additive exPlanations (SHAP) TreeExplainer:

### Key Clinical Risk Drivers:
1. **Prior Acute Inpatient Utilization (`number_inpatient`)**: Strongest monotonic driver of readmission. Patients with $\ge 1$ previous admission exhibit more than double the readmission rate.
2. **Discharge Disposition**: Patients discharged to skilled nursing or home health care demonstrate higher readmission vulnerability due to frailty.
3. **Glycemic Volatility & Monitoring (`A1Cresult`, `insulin`)**: Insulin dosage adjustments and elevated HbA1c ($>8\%$) reflect unstable metabolic management.
4. **Healthcare Intensity & Polypharmacy**: Higher daily medication counts (`meds_per_day`) combined with prolonged stays indicate multimorbid disease burden.

> [!NOTE]
> SHAP and feature importance metrics reflect statistical associations and model decision patterns within observational data. They do **not** represent causal clinical effects.

---

## 💾 SQL Healthcare Analytics Engine

The analytical SQLite database (`data/processed/readmission_analytics.db`) stores cleaned encounter cohorts, scored test predictions, and evaluation metrics. Eight clinical business queries answer key operational questions:

1. **Prevalence & Cohort Base**: Computes 8.98% baseline 30-day readmission prevalence across 69,990 clean encounters.
2. **Readmission by Age Bracket**: Highlights rising readmission rates in patients aged 70–90 with small sample size flags for pediatric categories.
3. **Readmission by Length of Stay**: Demonstrates that patients with stays $\ge 9$ days experience readmission rates of 11.8% vs 7.4% for 1–2 day stays.
4. **Readmission by Prior Utilization**: Quantifies the acute risk increase from 7.5% (0 prior admissions) to 21.6% (3+ prior admissions).
5. **Post-Acute Discharge Pathways**: Identifies high risk among patients discharged to Home Health/SNF (12.4%) or Left Against Medical Advice (12.8%).
6. **Risk Tier Stratification**: Audits observed vs predicted readmission rates across Low, Moderate, and High tiers.
7. **Error Analysis (FP vs FN)**: Characterizes False Positives (high polypharmacy and stay duration, but avoided readmission) vs False Negatives.
8. **Subgroup Performance**: Measures sensitivity and precision across vulnerable clinical cohorts (Elderly 65+, Prior Inpatient History, Insulin Users).

---

## 🎯 Risk Stratification & Care Team Capacity Simulator

Patients are stratified into actionable care coordination cohorts:

| Risk Tier | Probability Range | Observed Rate | Actionable Transition-of-Care Protocol |
| :--- | :---: | :---: | :--- |
| **Low Risk** | $< 6.45\%$ | ~4.8% | Routine discharge instructions; standard primary care follow-up within 14–30 days; patient portal educational materials. |
| **Moderate Risk** | $6.45\% - 11.61\%$ | ~8.7% | Targeted diabetes self-management education; scheduled outpatient clinic appointment within 7 days; 48-hour post-discharge telephone nurse check-in. |
| **High Risk** | $> 11.61\%$ | ~17.5% | **Intensive Care Coordination:** Dedicated nurse navigator assigned; bedside medication delivery; mandatory clinical visit within 72 hours; clinical pharmacist medication reconciliation. |

The simulator projects clinical caseload against Full-Time Equivalent (FTE) nurse staffing constraints to prevent care coordinator burnout.

---

## 📂 Project Architecture

```
patient-readmission-risk-modeling/
├── data/
│   ├── raw/
│   │   └── diabetic_data.csv          # 10-year raw EHR dataset (101,766 encounters)
│   ├── interim/                       # Fast parquet cohort splits
│   └── processed/
│       ├── readmission_analytics.db   # SQLite analytical clinical database (9.1 MB)
│       └── df_test_predictions.parquet# Test set with probabilities & risk tiers
├── model/
│   ├── best_model.pkl                 # Calibrated LightGBM Production Pipeline
│   └── model_metadata.json            # Thresholds, clinical tiers, test metrics
├── notebooks/
│   ├── 01_eda_and_clinical_analysis.ipynb
│   ├── 02_model_benchmarking.ipynb
│   └── 03_explainability_and_clinical_utility.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py                      # Paths, clinical exclusion codes, feature lists
│   ├── data/
│   │   ├── loader.py                  # Stratified & patient-grouped splitting
│   │   └── target.py                  # Target creation & CMS exclusion filtering
│   ├── features/
│   │   └── engineering.py             # Vectorized ClinicalFeatureEngineer transformer
│   ├── models/
│   │   ├── baseline.py                # DummyClassifier benchmarks
│   │   ├── smote_pipeline.py          # imblearn SMOTE & Class Weighting pipelines
│   │   ├── hyperparameter_tuning.py   # RandomizedSearchCV for PR-AUC
│   │   └── train_models.py            # 5-fold CV & held-out test evaluation
│   ├── evaluation/
│   │   ├── metrics.py                 # Discrimination, calibration & Youden's J
│   │   └── plots.py                   # ROC, PR, calibration & confusion matrix
│   ├── explainability/
│   │   └── shap_explainer.py          # SHAP global beeswarm & local waterfalls
│   ├── database/
│   │   ├── init_db.py                 # SQLite database initializer
│   │   └── queries.py                 # 8 business SQL queries
│   └── risk_stratification/
│       └── stratifier.py              # Risk tiers & care team workload simulator
├── tests/                             # 30 passing pytest unit & integration tests
│   ├── test_target.py
│   ├── test_features.py
│   ├── test_smote_isolation.py
│   ├── test_evaluation_metrics.py
│   ├── test_risk_stratification.py
│   ├── test_database.py
│   ├── test_pipeline.py
│   └── test_api.py
├── reports/
│   ├── eda_clinical_summary.md        # Complete EDA statistical audit
│   ├── model_performance_benchmark.md # Cross-validation & test evaluation report
│   ├── sql_analytics_insights.md      # SQL queries & tabular outputs
│   └── figures/                       # High-resolution clinical figures (PNG)
├── app.py                             # Interactive Streamlit Multi-Page Dashboard
├── run_pipeline.py                    # End-to-end automated orchestration pipeline
├── run_eda.py                         # EDA profiling script
├── pyproject.toml                     # Dependency definitions (uv-managed)
└── README.md
```

---

## 🚀 Reproduction & Execution Guide

### 1. Prerequisites & Environment Setup
Using `uv` (recommended) or standard Python 3.12+:

```bash
# Clone the repository
git clone https://github.com/your-username/patient-readmission-risk-modeling.git
cd patient-readmission-risk-modeling

# Create virtual environment and install dependencies
uv venv
uv pip install -e .
uv pip install streamlit duckdb tabulate
```

### 2. Run the Full Pipeline
Executes cohort filtering, 5-fold CV, hyperparameter tuning, test set evaluation, SHAP explainability, and SQLite database population:

```bash
python run_pipeline.py
```

### 3. Run Unit and Integration Tests
Executes all 30 tests covering target creation, feature engineering, SMOTE fold isolation, metric computations, risk stratification, and database queries:

```bash
pytest -v
```

### 4. Launch Interactive Streamlit Dashboard
```bash
streamlit run app.py
```

### 5. Launch FastAPI Clinical Prediction Server
```bash
uvicorn src.predict:app --host 0.0.0.0 --port 9696 --reload
```
Interactive API documentation will be available at `http://localhost:9696/docs`.

---

## ⚖️ Responsible AI & Ethical Disclaimers

1. **Non-Diagnostic Research Prototype**: This software is an algorithmic research demonstration and decision-support tool. It does not provide medical diagnoses, treatment prescriptions, or definitive clinical forecasts.
2. **Physician Autonomy**: All model risk flags and hypothetical transition-of-care recommendations must be evaluated and confirmed by a licensed medical practitioner.
3. **Health Equity & Subgroup Performance**: Algorithmic models trained on historical EHR data can inherit systemic clinical disparities. While demographic features (`race`, `gender`) were balanced, model performance across subgroups must be regularly audited to ensure equitable care delivery.
