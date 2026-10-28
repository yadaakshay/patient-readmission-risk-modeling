"""
Production FastAPI prediction service for Patient Readmission Risk.
Accepts raw EHR clinical encounter records and outputs calibrated readmission risk,
actionable clinical risk tiers (Low / Moderate / High), and patient-specific risk drivers.
"""
from typing import List, Optional, Literal, Dict, Any
from pathlib import Path
from contextlib import asynccontextmanager
import sys
import os

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import uvicorn
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from src.config import MODEL_PATH, METADATA_PATH
from src.pipeline import load_full_pipeline

# Global model and metadata references
pipeline = None
metadata = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model pipeline and metadata on server startup."""
    global pipeline, metadata
    try:
        pipeline, metadata = load_full_pipeline(MODEL_PATH, METADATA_PATH)
        print("Model pipeline and metadata successfully loaded.")
    except Exception as e:
        print(f"Warning: Model could not be loaded at startup: {e}")
    yield


app = FastAPI(
    title="Hospital Patient Readmission Risk Prediction API",
    description="Clinical Decision Support Service predicting 30-day all-cause hospital readmission risk for diabetic patients.",
    version="2.0.0",
    lifespan=lifespan
)


def get_risk_tier(prob: float, low_max: float, mod_max: float) -> str:
    """Classifies risk into actionable clinical cohorts."""
    if prob < low_max:
        return "Low"
    elif prob <= mod_max:
        return "Moderate"
    else:
        return "High"


def identify_patient_risk_factors(patient: Dict[str, Any]) -> List[str]:
    """Identifies notable clinical risk factors from the patient encounter."""
    factors = []
    
    inpatient_visits = patient.get('number_inpatient', 0)
    emergency_visits = patient.get('number_emergency', 0)
    total_prior = inpatient_visits + emergency_visits + patient.get('number_outpatient', 0)
    
    if inpatient_visits > 0:
        factors.append(f"Prior inpatient hospitalization history ({inpatient_visits} previous admissions)")
    elif emergency_visits >= 2:
        factors.append(f"Frequent emergency room utilization ({emergency_visits} prior visits)")
    elif total_prior >= 4:
        factors.append(f"High prior healthcare utilization ({total_prior} total outpatient/ED visits)")

    a1c = str(patient.get('A1Cresult', 'None'))
    if a1c == '>8':
        factors.append("Poor glycemic control: Elevated HbA1c > 8.0%")
    elif a1c == '>7':
        factors.append("Suboptimal glycemic control: HbA1c > 7.0%")
    elif a1c == 'None':
        factors.append("Glycemic unmonitored: HbA1c was not tested during encounter")

    time_in_hosp = patient.get('time_in_hospital', 1)
    if time_in_hosp >= 7:
        factors.append(f"Prolonged acute hospital stay ({time_in_hosp} days)")

    num_meds = patient.get('num_medications', 0)
    if num_meds >= 16:
        factors.append(f"Polypharmacy complexity ({num_meds} active inpatient medications)")

    insulin_status = str(patient.get('insulin', 'No'))
    if insulin_status in ['Up', 'Down']:
        factors.append("Active insulin dose titration / instability during encounter")
    elif insulin_status == 'Steady':
        factors.append("Maintenance insulin therapy")

    diag_1 = str(patient.get('diag_1', ''))
    try:
        val = float(diag_1)
        if (390 <= val <= 459) or val == 785:
            factors.append("Primary cardiovascular / circulatory acute diagnosis")
        elif (460 <= val <= 519) or val == 786:
            factors.append("Primary respiratory acute diagnosis")
        elif 250 <= val < 251:
            factors.append("Primary diabetic acute complication")
    except ValueError:
        pass

    disp = patient.get('discharge_disposition_id', 1)
    if disp in [3, 6]:
        factors.append("Post-discharge transitional care need: Discharged with Home Health Care or SNF")

    if not factors:
        factors.append("Standard clinical risk profile; no acute prior utilization spikes detected.")
        
    return factors


def get_clinical_action_plan(risk_tier: str) -> str:
    """Provides evidence-based clinical recommendations for care coordination."""
    if risk_tier == "High":
        return (
            "URGENT: High 30-day readmission risk. Enroll patient in intensive transition-of-care program. "
            "Schedule clinical follow-up within 72 hours of discharge, assign a nurse navigator, "
            "and conduct full clinical pharmacist medication reconciliation."
        )
    elif risk_tier == "Moderate":
        return (
            "ATTENTION: Moderate risk. Provide targeted disease education, verify post-discharge outpatient "
            "follow-up appointment within 7 days, and conduct telephone check-in 48 hours post-discharge."
        )
    else:
        return (
            "ROUTINE: Low risk profile. Standard post-discharge instructions, routine primary care "
            "follow-up within 14-30 days, and patient portal communication."
        )


class PatientInput(BaseModel):
    time_in_hospital: int = Field(default=3, description="Duration of hospital stay in days", ge=1, le=14)
    num_lab_procedures: int = Field(default=40, description="Total laboratory tests performed", ge=0)
    num_procedures: int = Field(default=1, description="Number of non-lab medical procedures performed", ge=0)
    num_medications: int = Field(default=14, description="Distinct medication count", ge=0)
    number_outpatient: int = Field(default=0, description="Outpatient encounters in past year", ge=0)
    number_emergency: int = Field(default=0, description="Emergency room visits in past year", ge=0)
    number_inpatient: int = Field(default=0, description="Inpatient admissions in past year", ge=0)
    number_diagnoses: int = Field(default=7, description="Number of diagnoses recorded", ge=1)
    race: Optional[Literal['Caucasian', 'AfricanAmerican', 'Asian', 'Hispanic', 'Other', 'Unknown']] = 'Caucasian'
    gender: Optional[Literal['Male', 'Female', 'Unknown']] = 'Female'
    age: Optional[str] = Field(default='[60-70)', description="Age bucket: [0-10) ... [90-100)")
    admission_type_id: Optional[int] = Field(default=1, description="1=Emergency, 2=Urgent, 3=Elective")
    discharge_disposition_id: Optional[int] = Field(default=1, description="1=Discharged Home, 3=SNF, 6=Home Health")
    admission_source_id: Optional[int] = Field(default=7, description="7=Emergency Room, 1=Physician Referral")
    diag_1: Optional[str] = Field(default="414", description="Primary ICD-9 diagnosis code")
    diag_2: Optional[str] = Field(default="250.0", description="Secondary ICD-9 diagnosis code")
    diag_3: Optional[str] = Field(default="401", description="Tertiary ICD-9 diagnosis code")
    max_glu_serum: Optional[Literal['None', 'Norm', '>200', '>300']] = 'None'
    A1Cresult: Optional[Literal['None', 'Norm', '>7', '>8']] = '>8'
    metformin: Optional[Literal['No', 'Steady', 'Up', 'Down']] = 'Steady'
    repaglinide: Optional[Literal['No', 'Steady', 'Up', 'Down']] = 'No'
    nateglinide: Optional[Literal['No', 'Steady', 'Up', 'Down']] = 'No'
    chlorpropamide: Optional[Literal['No', 'Steady', 'Up', 'Down']] = 'No'
    glimepiride: Optional[Literal['No', 'Steady', 'Up', 'Down']] = 'No'
    acetohexamide: Optional[Literal['No', 'Steady', 'Up', 'Down']] = 'No'
    glipizide: Optional[Literal['No', 'Steady', 'Up', 'Down']] = 'No'
    glyburide: Optional[Literal['No', 'Steady', 'Up', 'Down']] = 'No'
    tolbutamide: Optional[Literal['No', 'Steady', 'Up', 'Down']] = 'No'
    pioglitazone: Optional[Literal['No', 'Steady', 'Up', 'Down']] = 'No'
    rosiglitazone: Optional[Literal['No', 'Steady', 'Up', 'Down']] = 'No'
    acarbose: Optional[Literal['No', 'Steady', 'Up', 'Down']] = 'No'
    miglitol: Optional[Literal['No', 'Steady', 'Up', 'Down']] = 'No'
    troglitazone: Optional[Literal['No', 'Steady', 'Up', 'Down']] = 'No'
    tolazamide: Optional[Literal['No', 'Steady', 'Up', 'Down']] = 'No'
    insulin: Optional[Literal['No', 'Steady', 'Up', 'Down']] = 'Up'
    glyburide_metformin: Optional[Literal['No', 'Steady', 'Up', 'Down']] = Field(default='No', alias="glyburide-metformin")
    change: Optional[Literal['No', 'Ch']] = 'Ch'
    diabetesMed: Optional[Literal['No', 'Yes', 'YES']] = 'Yes'

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "time_in_hospital": 4,
                "num_lab_procedures": 48,
                "num_procedures": 1,
                "num_medications": 18,
                "number_outpatient": 0,
                "number_emergency": 1,
                "number_inpatient": 2,
                "number_diagnoses": 9,
                "race": "Caucasian",
                "gender": "Female",
                "age": "[70-80)",
                "admission_type_id": 1,
                "discharge_disposition_id": 1,
                "admission_source_id": 7,
                "diag_1": "428",
                "diag_2": "250.02",
                "diag_3": "585",
                "max_glu_serum": "None",
                "A1Cresult": ">8",
                "insulin": "Up",
                "change": "Ch",
                "diabetesMed": "Yes"
            }
        }
    }


class PredictionResponse(BaseModel):
    readmission_probability: float = Field(..., description="Calibrated risk probability between 0 and 1")
    readmitted_flag: bool = Field(..., description="Binary readmission risk flag based on clinical threshold")
    risk_tier: Literal['Low', 'Moderate', 'High'] = Field(..., description="Actionable clinical stratification tier")
    decision_threshold: float = Field(..., description="Threshold applied for binary classification")
    key_risk_drivers: List[str] = Field(..., description="Notable patient factors contributing to risk")
    clinical_action_plan: str = Field(..., description="Recommended care management intervention protocol")


class BatchPredictionRequest(BaseModel):
    patients: List[PatientInput]


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_processed: int


def init_model():
    """Manual model load helper if needed."""
    global pipeline, metadata
    try:
        pipeline, metadata = load_full_pipeline(MODEL_PATH, METADATA_PATH)
    except Exception as e:
        print(f"Warning: Model could not be loaded: {e}")


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Health status and model readiness endpoint."""
    return {
        "status": "healthy",
        "model_loaded": pipeline is not None,
        "model_name": metadata.get('model_name', 'Not trained yet'),
        "optimal_threshold": metadata.get('optimal_threshold', 0.15)
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_patient_readmission(patient: PatientInput):
    """Predicts 30-day hospital readmission risk for a single patient encounter."""
    global pipeline, metadata
    if pipeline is None:
        try:
            pipeline, metadata = load_full_pipeline(MODEL_PATH, METADATA_PATH)
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model pipeline not available: {e}")

    patient_dict = patient.model_dump(by_alias=True)

    try:
        prob = float(pipeline.predict_proba([patient_dict])[0, 1])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {e}")

    thr = metadata.get('optimal_threshold', 0.15)
    risk_tiers_cfg = metadata.get('clinical_risk_tiers', {
        'low_risk_max': round(thr * 0.75, 4),
        'moderate_risk_max': round(thr * 1.35, 4)
    })

    tier = get_risk_tier(prob, risk_tiers_cfg['low_risk_max'], risk_tiers_cfg['moderate_risk_max'])
    drivers = identify_patient_risk_factors(patient_dict)
    action_plan = get_clinical_action_plan(tier)

    return PredictionResponse(
        readmission_probability=round(prob, 4),
        readmitted_flag=bool(prob >= thr),
        risk_tier=tier,
        decision_threshold=round(thr, 4),
        key_risk_drivers=drivers,
        clinical_action_plan=action_plan
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(batch_req: BatchPredictionRequest):
    """Batch prediction endpoint for multi-patient risk stratification."""
    global pipeline, metadata
    if pipeline is None:
        try:
            pipeline, metadata = load_full_pipeline(MODEL_PATH, METADATA_PATH)
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model pipeline not available: {e}")

    records = [p.model_dump(by_alias=True) for p in batch_req.patients]
    if not records:
        return BatchPredictionResponse(predictions=[], total_processed=0)

    probs = pipeline.predict_proba(records)[:, 1]
    thr = metadata.get('optimal_threshold', 0.15)
    risk_tiers_cfg = metadata.get('clinical_risk_tiers', {
        'low_risk_max': round(thr * 0.75, 4),
        'moderate_risk_max': round(thr * 1.35, 4)
    })

    results = []
    for p_dict, prob in zip(records, probs):
        prob = float(prob)
        tier = get_risk_tier(prob, risk_tiers_cfg['low_risk_max'], risk_tiers_cfg['moderate_risk_max'])
        drivers = identify_patient_risk_factors(p_dict)
        action_plan = get_clinical_action_plan(tier)
        results.append(PredictionResponse(
            readmission_probability=round(prob, 4),
            readmitted_flag=bool(prob >= thr),
            risk_tier=tier,
            decision_threshold=round(thr, 4),
            key_risk_drivers=drivers,
            clinical_action_plan=action_plan
        ))

    return BatchPredictionResponse(predictions=results, total_processed=len(results))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
