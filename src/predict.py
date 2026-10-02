import os
import pickle
from typing import Any, Literal, Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Import the same transformer module used when the model was trained.
# The serialized pipeline contains this transformer.
from src.preprocessing import ReadmissionFeatureEngineer  # noqa: F401


class Patient(BaseModel):
    time_in_hospital: int
    num_lab_procedures: int
    num_procedures: int
    num_medications: int
    number_outpatient: int
    number_emergency: int
    number_inpatient: int
    number_diagnoses: int

    race: Optional[Literal[
        "Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other"
    ]] = Field(None, description="Race of the patient")
    gender: Optional[Literal[
        "Male", "Female", "Unknown/Invalid"
    ]] = Field(None, description="Gender of the patient")
    age: Optional[Literal[
        "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
        "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"
    ]] = Field(None, description="Age group of the patient")

    diag_1: Optional[str] = Field(None, description="Primary diagnosis")
    diag_2: Optional[str] = Field(None, description="Secondary diagnosis")
    diag_3: Optional[str] = Field(None, description="Additional diagnosis")

    metformin: Optional[Literal["No", "Steady", "Up", "Down"]] = None
    repaglinide: Optional[Literal["No", "Steady", "Up", "Down"]] = None
    nateglinide: Optional[Literal["No", "Steady", "Up", "Down"]] = None
    chlorpropamide: Optional[Literal["No", "Steady", "Up", "Down"]] = None
    glimepiride: Optional[Literal["No", "Steady", "Up", "Down"]] = None
    acetohexamide: Optional[Literal["No", "Steady", "Up", "Down"]] = None
    glipizide: Optional[Literal["No", "Steady", "Up", "Down"]] = None
    glyburide: Optional[Literal["No", "Steady", "Up", "Down"]] = None
    tolbutamide: Optional[Literal["No", "Steady", "Up", "Down"]] = None
    pioglitazone: Optional[Literal["No", "Steady", "Up", "Down"]] = None
    rosiglitazone: Optional[Literal["No", "Steady", "Up", "Down"]] = None
    acarbose: Optional[Literal["No", "Steady", "Up", "Down"]] = None
    miglitol: Optional[Literal["No", "Steady", "Up", "Down"]] = None
    troglitazone: Optional[Literal["No", "Steady", "Up", "Down"]] = None
    tolazamide: Optional[Literal["No", "Steady", "Up", "Down"]] = None
    examide: Optional[Literal["No", "Steady", "Up", "Down"]] = None
    citoglipton: Optional[Literal["No", "Steady", "Up", "Down"]] = None
    insulin: Optional[Literal["No", "Steady", "Up", "Down"]] = None


class PredictResponse(BaseModel):
    readmitted_probability: float
    readmitted: bool
    decision_threshold: float


app = FastAPI(title="Readmitted-Prediction", version="0.2")


MODEL_PATH = os.getenv("MODEL_PATH", "model/model.bin")

with open(MODEL_PATH, "rb") as f_in:
    artifact = pickle.load(f_in)

if not isinstance(artifact, dict) or "pipeline" not in artifact or "threshold" not in artifact:
    raise RuntimeError(
        "model/model.bin uses the legacy artifact format. "
        "Run 'uv run python -m src.train' to regenerate the corrected model."
    )

pipeline = artifact["pipeline"]
THRESHOLD = float(artifact["threshold"])


def predict_single(patient: dict[str, Any]) -> float:
    """Run the exact training-time preprocessing + model pipeline."""
    return float(
        pipeline.predict_proba([patient])[0, 1]
    )


@app.post("/predict", response_model=PredictResponse)
def predict(patient: Patient) -> PredictResponse:
    patient_data = patient.model_dump()
    probability = predict_single(patient_data)

    return PredictResponse(
        readmitted_probability=probability,
        readmitted=bool(probability >= THRESHOLD),
        decision_threshold=THRESHOLD,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
