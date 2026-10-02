import os
import pickle
from typing import Any, Literal, Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

# The serialized model contains ReadmissionFeatureEngineer. Importing the class
# here ensures pickle can resolve it when the API starts.
from src.preprocessing import ReadmissionFeatureEngineer  # noqa: F401


Medication = Optional[
    Literal["No", "Steady", "Up", "Down"]
]


class Patient(BaseModel):
    time_in_hospital: int = Field(..., ge=0)
    num_lab_procedures: int = Field(..., ge=0)
    num_procedures: int = Field(..., ge=0)
    num_medications: int = Field(..., ge=0)
    number_outpatient: int = Field(..., ge=0)
    number_emergency: int = Field(..., ge=0)
    number_inpatient: int = Field(..., ge=0)
    number_diagnoses: int = Field(..., ge=0)

    race: Optional[
        Literal[
            "Caucasian",
            "AfricanAmerican",
            "Asian",
            "Hispanic",
            "Other",
        ]
    ] = None

    gender: Optional[
        Literal[
            "Male",
            "Female",
            "Unknown/Invalid",
        ]
    ] = None

    age: Optional[
        Literal[
            "[0-10)", "[10-20)", "[20-30)", "[30-40)",
            "[40-50)", "[50-60)", "[60-70)", "[70-80)",
            "[80-90)", "[90-100)",
        ]
    ] = None

    admission_type_id: Optional[int] = None
    discharge_disposition_id: Optional[int] = None
    admission_source_id: Optional[int] = None

    max_glu_serum: Optional[
        Literal[">200", ">300", "Norm", "None"]
    ] = None

    A1Cresult: Optional[
        Literal[">7", ">8", "Norm", "None"]
    ] = None

    diag_1: Optional[str] = None
    diag_2: Optional[str] = None
    diag_3: Optional[str] = None

    metformin: Medication = None
    repaglinide: Medication = None
    nateglinide: Medication = None
    chlorpropamide: Medication = None
    glimepiride: Medication = None
    glipizide: Medication = None
    glyburide: Medication = None
    tolbutamide: Medication = None
    pioglitazone: Medication = None
    rosiglitazone: Medication = None
    acarbose: Medication = None
    miglitol: Medication = None
    insulin: Medication = None
    glyburide_metformin: Medication = Field(
        None,
        alias="glyburide-metformin",
    )
    tolazamide: Medication = None
    metformin_pioglitazone: Medication = Field(
        None,
        alias="metformin-pioglitazone",
    )
    metformin_rosiglitazone: Medication = Field(
        None,
        alias="metformin-rosiglitazone",
    )
    glimepiride_pioglitazone: Medication = Field(
        None,
        alias="glimepiride-pioglitazone",
    )
    glipizide_metformin: Medication = Field(
        None,
        alias="glipizide-metformin",
    )
    troglitazone: Medication = None
    acetohexamide: Medication = None

    change: Optional[
        Literal["Ch", "No"]
    ] = None

    diabetesMed: Optional[
        Literal["Yes", "No"]
    ] = None

    model_config = {
        "populate_by_name": True,
    }


class PredictResponse(BaseModel):
    readmitted_probability: float
    readmitted: bool
    decision_threshold: float
    model_name: str


app = FastAPI(
    title="Hospital Readmission Risk API",
    version="0.3",
)


MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "model/model.bin",
)

with open(MODEL_PATH, "rb") as f_in:
    artifact = pickle.load(f_in)

if not isinstance(artifact, dict):
    raise RuntimeError(
        "Invalid model artifact. "
        "Run 'uv run python -m src.train'."
    )

required_keys = {
    "pipeline",
    "threshold",
    "model_name",
}

if not required_keys.issubset(artifact):
    raise RuntimeError(
        "model/model.bin uses an old artifact format. "
        "Run 'uv run python -m src.train' to regenerate "
        "the leakage-safe model."
    )

pipeline = artifact["pipeline"]
THRESHOLD = float(artifact["threshold"])
MODEL_NAME = str(artifact["model_name"])


def predict_single(patient: dict[str, Any]) -> float:
    """Run the exact training-time feature pipeline."""
    return float(
        pipeline.predict_proba([patient])[0, 1]
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
)
def predict(patient: Patient) -> PredictResponse:
    patient_data = patient.model_dump(
        by_alias=True
    )

    probability = predict_single(patient_data)

    return PredictResponse(
        readmitted_probability=probability,
        readmitted=bool(
            probability >= THRESHOLD
        ),
        decision_threshold=THRESHOLD,
        model_name=MODEL_NAME,
    )


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=9696,
    )
