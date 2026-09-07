# Patient Readmission & Risk Predictive Modeling

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient_Boosting-green)
![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-yellow)

This repository contains an end-to-end Machine Learning pipeline designed to predict diabetic patient hospital readmissions using Electronic Health Records (EHR). The project demonstrates the ability to process complex, semi-structured patient-level datasets and apply advanced predictive modeling techniques to optimize healthcare resource allocation.

## Business Impact
Hospital readmissions are a major metric for healthcare quality and cost. Accurately predicting high-risk patients allows healthcare providers to implement proactive interventions, improving patient outcomes and avoiding regulatory penalties. This model serves as a decision-support tool for clinical teams.

## Technical Approach
* **Exploratory Data Analysis (EDA):** Analyzed patient demographics, hospital stay durations, lab results, and medication history to identify key readmission indicators.
* **Data Preprocessing & Feature Engineering:** Handled high-dimensionality categorical data and missing values. Addressed extreme class imbalance in the medical dataset using **SMOTE** (Synthetic Minority Over-sampling Technique).
* **Predictive Modeling:** Developed and evaluated multiple classification algorithms including **Logistic Regression, Decision Trees, Random Forest, and XGBoost**.
* **Model Validation:** Ensured model robustness and statistical reliability using **10-fold cross-validation**.
* **Results:** The optimized **XGBoost** model achieved the highest performance with **93.5% accuracy** and **0.92 AUC**, effectively identifying high-risk patients.

## Repository Structure
* `notebooks/`: Contains Jupyter notebooks for EDA (`eda.ipynb`), baseline modeling (`logistic_regression.ipynb`), and advanced ensemble modeling (`random_forest_xgboost.ipynb`).
* `src/`: Core Python scripts for modularized training (`train.py`) and inference (`predict.py`).
* `model/`: Serialized models and scalers.
* `visuals/` & `output_images/`: Analytical plots and performance metrics (ROC curves, confusion matrices).
* `Dockerfile`: Containerization setup for deploying the prediction API.

## Getting Started
Detailed instructions for setting up the environment using `pyproject.toml` / `uv` or standard pip requirements.
