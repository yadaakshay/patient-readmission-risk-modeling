#!/usr/bin/env python
# coding: utf-8

# To save our model, we use pickle, a system import
import pickle

import pandas as pd
import numpy as np
import sklearn

from sklearn.pipeline import make_pipeline # What does this do
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier

print(f'pandas=={pd.__version__}')
print(f'numpy=={np.__version__}')
print(f'sklearn=={sklearn.__version__}')

# Feature engineering function
def engineer_features(df):
    # Create additional predictive features while preserving the original columns.

    # -------- Visit history / intensity features --------
    df['total_previous_visits'] = (
        df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']
    )

    df['had_previous_inpatient'] = (df['number_inpatient'] > 0).astype(int)

    # Avoid division by zero by adding +1 to the denominator
    df['avg_medications_per_day'] = df['num_medications'] / (df['time_in_hospital'] + 1)
    df['procedure_to_lab_ratio'] = df['num_procedures'] / (df['num_lab_procedures'] + 1)

    # -------- Medication change summaries --------
    med_cols = [
        'metformin','repaglinide','nateglinide','chlorpropamide','glimepiride',
        'acetohexamide','glipizide','glyburide','tolbutamide','pioglitazone',
        'rosiglitazone','acarbose','miglitol','troglitazone','tolazamide',
        'examide','citoglipton','insulin'
    ]

    # Count how many meds are used at all (value != 'no')
    df['num_medications_used'] = df[med_cols].apply(lambda row: (row != 'no').sum(), axis=1)

    # Count meds with dose adjustment (value in {'up','down'})
    df['num_adjusted_medications'] = df[med_cols].apply(lambda row: row.isin(['up', 'down']).sum(), axis=1)

    # Binary helper features
    df['any_medication_change'] = (df['num_adjusted_medications'] > 0).astype(int)
    df['on_insulin'] = (df['insulin'] != 'no').astype(int)

    return df

# Load data and return dataframe
def load_data():

    data_url = 'data/diabetic_data.csv'

    df = pd.read_csv(data_url)

    df = df.replace('?', np.nan)

    # Drop columns with 30 to 90% null values
    df = df.drop(columns=[
        'weight',
        'max_glu_serum',
        'A1Cresult',
        'medical_specialty',
        'payer_code'
    ], axis=1)

    # Handle missing values and format categorical values
    cat_cols = df.select_dtypes(include=['object']).columns
    num_cols = df.select_dtypes(include=['number']).columns

    for c in cat_cols:
        df[c] = df[c].str.lower().str.replace(' ', '_')

    df[cat_cols] = df[cat_cols].fillna('NA')
    df[num_cols] = df[num_cols].fillna(0.0)

    # Map age
    age_map = {
        '[0-10)': 5,
        '[10-20)': 15,
        '[20-30)': 25,
        '[30-40)': 35,
        '[40-50)': 45,
        '[50-60)': 55,
        '[60-70)': 65,
        '[70-80)': 75,
        '[80-90)': 85,
        '[90-100)': 95
    }
    df['age'] = df['age'].map(age_map)

    df['readmitted'] = df['readmitted'].apply(lambda x: 1 if x=='<30' else 0 )

    df = df.drop(columns=['encounter_id', 'patient_nbr'], errors='ignore')

    df = engineer_features(df)
    
    return df

def train_model(df):

    y_train = df.readmitted.values

    numerical = [
        'time_in_hospital',
        'num_lab_procedures',
        'num_procedures',
        'num_medications',
        'number_outpatient',
        'number_emergency',
        'number_inpatient',
        'number_diagnoses',
        'total_previous_visits',
        'had_previous_inpatient',
        'avg_medications_per_day',
        'procedure_to_lab_ratio',
        'num_medications_used',
        'num_adjusted_medications',
        'any_medication_change',
        'on_insulin'
    ]

    categorical = [
        'race',
        'gender', 
        'age', 
        'diag_1', 
        'diag_2', 
        'diag_3', 
        'metformin',
        'repaglinide',
        'nateglinide',
        'chlorpropamide',
        'glimepiride',
        'acetohexamide',
        'glipizide',
        'glyburide', 
        'tolbutamide',
        'pioglitazone',
        'rosiglitazone',
        'acarbose',
        'miglitol',
        'troglitazone',
        'tolazamide',
        'examide',
        'citoglipton',
        'insulin',
        'glyburide-metformin',
        'glipizide-metformin',
        'glimepiride-pioglitazone',
        'metformin-rosiglitazone',
        'metformin-pioglitazone',
        'change',
        'diabetesMed'
    ]

    pipeline = make_pipeline(
        DictVectorizer(),
        RandomForestClassifier(
            max_depth=15,
            min_samples_leaf=3,
            class_weight='balanced',
            max_features='sqrt',
            n_jobs=-1,
            random_state=1)
    )

    train_dict = df[categorical + numerical].to_dict(orient='records') # DictVectorizer turns the dataframe into a dictionary, which is one-hot encoding
    pipeline.fit(train_dict, y_train)

    return pipeline

def save_model(filename, model):
    with open(filename, 'wb') as f_out:
        pickle.dump(model, f_out)

    print(f'model saved to {filename}')

df = load_data()
pipeline = train_model(df)
save_model('model/model.bin', pipeline)