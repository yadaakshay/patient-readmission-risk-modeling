import pickle

from typing import Dict, Any, Literal, Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

# request
class Patient(BaseModel):
    time_in_hospital: int
    num_lab_procedures: int
    num_procedures: int
    num_medications: int
    number_outpatient: int
    number_emergency: int
    number_inpatient: int
    number_diagnoses: int
    race: Optional[Literal['Caucasian', 'AfricanAmerican', 'Asian', 'Hispanic', 'Other']] = Field(None, description="Race of the patient")
    gender: Optional[Literal['Male', 'Female', 'Unknown/Invalid']] = Field(None, description="Gender of the patient")
    age: Optional[Literal['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)']] = Field(None, description="Age group of the patient")
    diag_1: Optional[str] = Field(None, description="Primary diagnosis")
    diag_2: Optional[str] = Field(None, description="Secondary diagnosis")
    diag_3: Optional[str] = Field(None, description="Additional diagnosis")
    metformin: Optional[Literal['No', 'Steady', 'Up', 'Down']] = Field(None, description="Metformin medication status")
    repaglinide: Optional[Literal['No', 'Steady', 'Up', 'Down']] = Field(None, description="Repaglinide medication status")
    nateglinide: Optional[Literal['No', 'Steady', 'Up', 'Down']] = Field(None, description="Nateglinide medication status")
    chlorpropamide: Optional[Literal['No', 'Steady', 'Up', 'Down']] = Field(None, description="Chlorpropamide medication status")
    glimepiride: Optional[Literal['No', 'Steady', 'Up', 'Down']] = Field(None, description="Glimepiride medication status")
    acetohexamide: Optional[Literal['No', 'Steady', 'Up', 'Down']] = Field(None, description="Acetohexamide medication status")
    glipizide: Optional[Literal['No', 'Steady', 'Up', 'Down']] = Field(None, description="Glipizide medication status")
    glyburide: Optional[Literal['No', 'Steady', 'Up', 'Down']] = Field(None, description="Glyburide medication status")
    tolbutamide: Optional[Literal['No', 'Steady', 'Up', 'Down']] = Field(None, description="Tolbutamide medication status")
    pioglitazone: Optional[Literal['No', 'Steady', 'Up', 'Down']] = Field(None, description="Pioglitazone medication status")
    rosiglitazone: Optional[Literal['No', 'Steady', 'Up', 'Down']] = Field(None, description="Rosiglitazone medication status")
    acarbose: Optional[Literal['No', 'Steady', 'Up', 'Down']] = Field(None, description="Acarbose medication status")
    miglitol: Optional[Literal['No', 'Steady', 'Up', 'Down']] = Field(None, description="Miglitol medication status")
    troglitazone: Optional[Literal['No', 'Steady', 'Up', 'Down']] = Field(None, description="Troglitazone medication status")
    tolazamide: Optional[Literal['No', 'Steady', 'Up', 'Down']] = Field(None, description="Tolazamide medication status")
    examide: Optional[Literal['No', 'Steady', 'Up', 'Down']] = Field(None, description="Examide medication status")
    citoglipton: Optional[Literal['No', 'Steady', 'Up', 'Down']] = Field(None, description="Citoglipton medication status")
    insulin: Optional[Literal['No', 'Steady', 'Up', 'Down']] = Field(None, description="Insulin medication status") 


# response
class PredictResponse(BaseModel):
    readmitted_probability: float
    readmitted: bool

app = FastAPI(title="Readmitted-Prediction", version="0.1")

with open('model/model.bin', 'rb') as f_in: # Loading or reading
    pipeline = pickle.load(f_in)
    
def predict_single(patient):
    result = pipeline.predict_proba(patient)[0, 1]  # probability of readmitted
    return float(result)

@app.post("/predict")

def predict(patient: Patient) -> PredictResponse:
    prob = predict_single(patient.dict())
    return PredictResponse(
        readmitted_probability=prob,
        readmitted=bool(prob >= 0.5),
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)








