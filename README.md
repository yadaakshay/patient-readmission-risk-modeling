# 🏥 Hospital Readmission Prediction using Machine Learning  

##  Problem Statement  

Hospital readmission occurs when a patient who has been discharged from a hospital is admitted again within a short period.  
Frequent readmissions not only reflect the **quality of healthcare services** but also increase **medical costs and strain hospital resources**.  

The goal of this project is to **identify key factors** that influence hospital readmissions and **predict which patients** are more likely to be readmitted, especially among **diabetic patients**.  

---

## 🎯 Objective  

1. Analyze patient data to determine the major factors contributing to readmissions.  
2. Build machine learning models to predict whether a patient will be readmitted or not.  
3. Compare multiple algorithms to find the best-performing one.  

---

##  Dataset Overview  

- The dataset contains **over 100,000 hospital records** of diabetic patients.  
- Each record includes patient demographics, hospital visit details, diagnoses, and medication information.  
- The **target variable** is `readmitted`, indicating whether the patient was readmitted (Yes/No).  

###  Key Features:
- `age`, `gender`, `race`  
- `time_in_hospital`, `num_lab_procedures`, `num_medications`  
- `number_outpatient`, `number_inpatient`, `number_emergency`  
- `number_diagnoses`, `A1Cresult`, `max_glu_serum`  
- `diag_1`, `diag_2`, `diag_3`  
- `insulin`, `metformin`, and other diabetes-related medications  

---

## ⚙️ Key Data Processing Steps  

1. **Data Copy & Cleaning**  
   - Created a clean copy of the dataset for preprocessing.  
   - Removed irrelevant or high-missing-value columns (`weight`, `medical_specialty`, `payer_code`).  

2. **Handling Missing Values**  
   - Dropped rows with `'?'` in key diagnosis and demographic columns.  
   - Removed invalid values in `gender` and `race`.  

3. **Feature Engineering**  
   - Created **dummy variables** for categorical features.  
   - Applied **log transformation** to highly skewed features like `number_outpatient`, `number_emergency`, and `number_inpatient`.  

4. **Class Imbalance Handling**  
   - Used **SMOTE (Synthetic Minority Oversampling Technique)** to balance the target variable (`readmitted`).  

5. **Feature Selection**  
   - Selected important demographic, diagnostic, and medication features for model training.  

6. **Train-Test Split**  
   - Split the data into **80% training** and **20% testing** sets.  

---

##  Machine Learning Models Used  

We implemented and compared the following models:

| S.No | Model | Technique | Cross-Validation Score | Test Accuracy | Precision | Recall | AUC |
|------|--------|------------|------------------------|---------------|------------|---------|------|
| 1 | Logistic Regression | L1 Regularization | 61.29% | 61.35% | 0.63 | 0.59 | 0.61 |
| 2 | Decision Tree (Entropy) | Information Gain | 88.99% | 89.44% | 0.92 | 0.87 | 0.89 |
| 3 | Decision Tree (Gini) | Gini Index | 89.10% | 89.46% | 0.92 | 0.87 | 0.89 |
| 4 | Random Forest (Entropy) | Ensemble of Trees | 92.23% | 92.47% | 0.98 | 0.87 | 0.93 |
| 5 | Random Forest (Gini) | Ensemble of Trees | 92.29% | 92.61% | 0.98 | 0.87 | 0.93 |
| **6** | **XGBoost (Tuned)** | Gradient Boosting | **94.00%** | **94.00%** | **0.98** | **0.90** | **0.94** |

✅ **Best Model:** XGBoost  
It achieved the **highest accuracy and AUC**, outperforming all other models.  

---

## ⚡ XGBoost Model Tuning  

**Key Parameters Tuned:**

| Parameter | Description | Tuned Value |
|------------|--------------|--------------|
| `max_depth` | Controls model complexity | 8 |
| `eta` | Learning rate | 0.02 |
| `colsample_bytree` | Fraction of features per tree | 0.9 |
| `num_rounds` | Boosting iterations | 20000 |
| `objective` | Binary classification | `binary:logistic` |

The model was trained using **early stopping** and **cross-validation**, ensuring high accuracy and generalization capability.  

---

## 🌟 Feature Importance  

Top predictors of hospital readmission identified by XGBoost and Random Forest:  
- `time_in_hospital`  
- `number_inpatient`  
- `number_diagnoses`  
- `num_medications`  
- `insulin` usage  

These features correlate strongly with **disease severity and treatment duration**, both crucial for predicting readmission.  

---

## 🧾 Conclusion  

- Machine learning models can effectively predict hospital readmissions among diabetic patients.  
- **XGBoost** emerged as the most accurate model with **94% accuracy and 0.94 AUC**.  
- The key influencing factors were **hospital stay duration**, **number of diagnoses**, and **inpatient frequency**.  
- These insights can help hospitals **proactively identify high-risk patients** and take preventive measures.  

