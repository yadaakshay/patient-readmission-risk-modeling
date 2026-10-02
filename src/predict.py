import os
import pickle
from typing import Any, Literal, Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

# The serialized artifact contains this custom transformer.
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

    race: Optional[
        Literal[
            "Caucasian",
            "AfricanAmerican",
            "Asian",
            "Hispanic",
            "Other",
        ]
    ] = Field(None, description="Race of the patient")

    gender: Optional[
        Literal["Male", "Female", "Unknown/Invalid"]
    ] = Field(None, description="Gender of the patient")

    age: Optional[
        Literal[
            "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
            "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)",
        ]
    ] = Field(None, description="Age group of the patient")

    admission_type_id: Optional[int] = None
    discharge_disposition_id: Optional[int] = None
    admission_source_id: Optional[int] = None

    max_glu_serum: Optional[
        Literal[">200", ">300", "Norm", "None"]
    ] = None

    A1Cresult: Optional[
        Literal[">7", ">8", "Norm", "None"]
    ] = None

    diag_1: Optional[str] = Field(
        None,
        description="Primary ICD-9 diagnosis",
    )
    diag_2: Optional[str] = Field(
        None,
        description="Secondary ICD-9 diagnosis",
    )
    diag_3: Optional[str] = Field(
        None,
        description="Additional ICD-9 diagnosis",
    )

    metformin: Optional[
        Literal["No", "Steady", "Up", "Down"]
    ] = None
    repaglinide: Optional[
        Literal["No", "Steady", "Up", "Down"]
    ] = None
    nateglinide: Optional[
        Literal["No", "Steady", "Up", "Down"]
    ] = None
    chlorpropamide: Optional[
        Literal["No", "Steady", "Up", "Down"]
    ] = None
    glimepiride: Optional[
        Literal["No", "Steady", "Up", "Down"]
    ] = None
    acetohexamide: Optional[
        Literal["No", "Steady", "Up", "Down"]
    ] = None
    glipizide: Optional[
        Literal["No", "Steady", "Up", "Down"]
    ] = None
    glyburide: Optional[
        Literal["No", "Steady", "Up", "Down"]
    ] = None
    tolbutamide: Optional[
        Literal["No", "Steady", "Up", "Down"]
    ] = None
    pioglitazone: Optional[
        Literal["No", "Steady", "Up", "Down"]
    ] = None
    rosiglitazone: Optional[
        Literal["No", "Steady", "Up", "Down"]
    ] = None
    acarbose: Optional[
        Literal["No", "Steady", "Up", "Down"]
    ] = None
    miglitol: Optional[
        Literal["No", "Steady", "Up", "Down"]
    ] = None
    troglitazone: Optional[
        Literal["No", "Steady", "Up", "Down"]
    ] = None
    tolazamide: Optional[
        Literal["No", "Steady", "Up", "Down"]
    ] = None
    examide: Optional[
        Literal["No", "Steady", "Up", "Down"]
    ] = None
    citoglipton: Optional[
        Literal["No", "Steady", "Up", "Down"]
    ] = None
    insulin: Optional[
        Literal["No", "Steady", "Up", "Down"]
    ] = None
    glyburide_metformin: Optional[
        Literal["No", "Steady", "Up", "Down"]
    ] = Field(None, alias="glyburide-metformin")
    glipizide_metformin: Optional[
        Literal["No", "Steady", "Up", "Down"]
    ] = Field(None, alias="glipizide-metformin")
    glimepiride_pioglitazone: Optional[
        Literal["No", "Steady", "Up", "Down"]
    ] = Field(None, alias="glimepiride-pioglitazone")
    metformin_rosiglitazone: Optional[
        Literal["No", "Steady", "Up", "Down"]
    ] = Field(None, alias="metformin-rosiglitazone")
    metformin_pioglitazone: Optional[
        Literal["No", "Steady", "Up", "Down"]
    ] = Field(None, alias="metformin-pioglitazone")

    change: Optional[Literal["Ch", "No"]] = None
    diabetesMed: Optional[Literal["Yes", "No"]] = None


class PredictResponse(BaseModel):
    readmitted_probability: float
    readmitted: bool
    decision_threshold: float
    model_name: str


app = FastAPI(
    title="Patient Readmission Risk API",
    version="0.3",
)

MODEL_PATH = os.getenv("MODEL_PATH", "model/model.bin")

with open(MODEL_PATH, "rb") as f_in:
    artifact = pickle.load(f_in)

if (
    not isinstance(artifact, dict)
    or "pipeline" not in artifact
    or "threshold" not in artifact
):
    raise RuntimeError(
        "model/model.bin uses the legacy artifact format. "
        "Run 'uv run python -m src.train' to regenerate the "
        "corrected model."
    )

pipeline = artifact["pipeline"]
THRESHOLD = float(artifact["threshold"])
MODEL_NAME = str(artifact.get("model_name", "unknown"))


def predict_single(patient: dict[str, Any]) -> float:
    """Run the exact serialized preprocessing + model pipeline."""
    return float(
        pipeline.predict_proba([patient])[0, 1]
    )


@app.post("/predict", response_model=PredictResponse)
def predict(patient: Patient) -> PredictResponse:
    patient_data = patient.model_dump(
        by_alias=True
    )
    probability = predict_single(patient_data)

    return PredictResponse(
        readmitted_probability=probability,
        readmitted=bool(probability >= THRESHOLD),
        decision_threshold=THRESHOLD,
        model_name=MODEL_NAME,
    )


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=9696,
    )
