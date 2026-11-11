# Model Performance Benchmark & Evaluation Report

## 1. Executive Summary
- **Clinical Task**: 30-day all-cause hospital readmission prediction for diabetic inpatients.
- **Cohort Size**: 69,990 patients (10,499 in held-out test cohort).
- **Target Prevalence**: 8.98% positive readmissions.
- **Top Performing Architecture**: **LIGHTGBM** with probability calibration and class weighting.
- **Optimal Clinical Decision Threshold**: **0.086** (tuned via Youden's J statistic on validation cohort).

---

## 2. 5-Fold Stratified Cross-Validation Benchmark

| Model Configuration | Model Family | Imbalance Strategy | ROC-AUC (Mean ± Std) | PR-AUC (Mean ± Std) | Brier Score | Balanced Acc |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| Dummy (Stratified) | dummy | none | 0.4993 ± 0.0022 | 0.0897 ± 0.0003 | 0.1640 | 0.4993 |
| Logistic Regression (Weighted) | logistic_regression | class_weight | 0.6381 ± 0.0050 | 0.1570 ± 0.0062 | 0.0798 | 0.5003 |
| Random Forest (Weighted) | random_forest | class_weight | 0.6450 ± 0.0039 | 0.1648 ± 0.0048 | 0.0795 | 0.5000 |
| XGBoost (Weighted) | xgboost | class_weight | 0.6451 ± 0.0045 | 0.1666 ± 0.0049 | 0.0794 | 0.5000 |
| LightGBM (Weighted) | lightgbm | class_weight | 0.6457 ± 0.0059 | 0.1693 ± 0.0060 | 0.0794 | 0.5000 |
| LightGBM (Unweighted) | lightgbm | none | 0.6424 ± 0.0044 | 0.1660 ± 0.0046 | 0.0794 | 0.5008 |
| LightGBM (SMOTE) | lightgbm | smote | 0.6363 ± 0.0033 | 0.1580 ± 0.0044 | 0.0808 | 0.5000 |
| Logistic Regression (SMOTE) | logistic_regression | smote | 0.6313 ± 0.0082 | 0.1537 ± 0.0058 | 0.1092 | 0.5289 |

### Key Methodological Insights:
1. **Class Weighting vs SMOTE**: LightGBM with cost-sensitive class weighting (`scale_pos_weight=5.0`) delivered superior PR-AUC compared to SMOTE (0.1693 vs 0.1580). In complex tabular clinical data with many sparse one-hot encoded categories, synthetic interpolation in SMOTE can create unnatural feature combinations, whereas class weighting directly re-balances the loss function without distorting feature manifolds.
2. **Probability Calibration**: Calibrating tree ensemble probabilities via Platt scaling (sigmoid) preserves discriminative ROC-AUC while reducing Brier score loss to ~0.0796, ensuring that output probabilities reflect true empirical clinical event frequencies.

---

## 3. Held-Out Untouched Test Cohort Results

| Model Architecture | Threshold | ROC-AUC | PR-AUC | Sensitivity (Recall) | Specificity | Precision | F1-Score | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Dummy Prior | 0.50 | 0.5000 | 0.0898 | 0.0% | 100.0% | 0.0% | 0.0000 | 0.0818 |
| Logistic Regression | 0.50 | 0.6331 | 0.1537 | 0.1% | 99.9% | 16.7% | 0.0021 | 0.0800 |
| Random Forest | 0.50 | 0.6354 | 0.1543 | 0.0% | 100.0% | 0.0% | 0.0000 | 0.0799 |
| Xgboost | 0.50 | 0.6343 | 0.1589 | 0.0% | 100.0% | 0.0% | 0.0000 | 0.0797 |
| Lightgbm | 0.50 | 0.6335 | 0.1617 | 0.0% | 100.0% | 0.0% | 0.0000 | 0.0796 |

---

## 4. Decision Threshold Analysis: The 91% Accuracy Paradox

In an imbalanced healthcare setting where the positive outcome prevalence is ~9%, standard accuracy is misleading:

| Decision Threshold Strategy | Accuracy | Sensitivity (Recall) | Specificity | Precision | F1 Score | True Positives | False Negatives |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard Threshold ($0.500$)** | 91.02% | **0.0%** | 100.0% | 0.0% | 0.000 | 0 | 943 |
| **Optimal Clinical Threshold ($0.086$)** | 58.23% | **61.3%** | 57.9% | 12.6% | **0.209** | **578** | **365** |

> **Clinical Takeaway**: At standard 0.5 threshold, the model misses virtually all readmissions (0-1% sensitivity), making it clinically useless despite having >90% raw accuracy. Setting the operational threshold to 0.086 captures over 60% of all readmitted patients for proactive transitional care intervention.
