"""
Clinical Risk Stratification and Operational Decision Support Module.
Provides:
1. Configurable risk tier assignment (Low, Moderate, High) with thresholds derived from validation data.
2. Observed vs predicted readmission rate audits.
3. Care management follow-up workload simulation based on clinical staff capacity.
4. Hypothetical decision-support action protocols with clinical non-diagnostic disclaimers.
"""
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd


def compute_risk_tiers(
    probs: np.ndarray,
    low_threshold: float,
    moderate_threshold: float
) -> np.ndarray:
    """
    Categorizes predicted probabilities into three clinical risk tiers:
    - Low Risk: prob < low_threshold
    - Moderate Risk: low_threshold <= prob < moderate_threshold
    - High Risk: prob >= moderate_threshold
    """
    probs = np.asarray(probs)
    tiers = np.empty(len(probs), dtype=object)
    tiers[probs < low_threshold] = "Low"
    tiers[(probs >= low_threshold) & (probs < moderate_threshold)] = "Moderate"
    tiers[probs >= moderate_threshold] = "High"
    return tiers


def summarize_risk_stratification(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    low_threshold: float = 0.065,
    moderate_threshold: float = 0.120
) -> pd.DataFrame:
    """
    Summarizes encounter volume, observed readmission rate, and predicted risk distribution
    across each risk tier.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    tiers = compute_risk_tiers(y_prob, low_threshold, moderate_threshold)

    df_summary = pd.DataFrame({
        'target': y_true,
        'prob': y_prob,
        'tier': tiers
    })

    tier_order = ['Low', 'Moderate', 'High']
    summary_rows = []
    total_encounters = len(y_true)

    for tier_name in tier_order:
        sub = df_summary[df_summary['tier'] == tier_name]
        cnt = len(sub)
        if cnt == 0:
            continue
        readmitted = sub['target'].sum()
        obs_rate = (readmitted / cnt) * 100.0
        vol_pct = (cnt / total_encounters) * 100.0

        summary_rows.append({
            'Risk Tier': tier_name,
            'Encounter Count': cnt,
            'Cohort Share (%)': round(vol_pct, 1),
            'Observed Readmissions': int(readmitted),
            'Observed Readmission Rate (%)': round(obs_rate, 2),
            'Mean Predicted Probability (%)': round(sub['prob'].mean() * 100.0, 2),
            'Min Predicted Probability (%)': round(sub['prob'].min() * 100.0, 2),
            'Max Predicted Probability (%)': round(sub['prob'].max() * 100.0, 2)
        })

    return pd.DataFrame(summary_rows)


def simulate_care_team_workload(
    df_scored: pd.DataFrame,
    nurse_navigator_fte: int = 5,
    patients_per_nurse_per_month: int = 40,
    operational_threshold: float = 0.085
) -> Dict[str, Any]:
    """
    Simulates clinical care management workload against staff capacity.
    
    Parameters:
    - nurse_navigator_fte: Number of Full-Time Equivalent nurse care coordinators.
    - patients_per_nurse_per_month: Number of high-intensity transitional care cases per coordinator.
    - operational_threshold: Cutoff above which patients receive proactive intervention.
    """
    total_monthly_capacity = nurse_navigator_fte * patients_per_nurse_per_month
    
    # Calculate flagged patients
    flagged = df_scored[df_scored['readmission_prob'] >= operational_threshold]
    high_tier = df_scored[df_scored['prediction_tier'] == 'High']
    moderate_tier = df_scored[df_scored['prediction_tier'] == 'Moderate']

    flagged_count = len(flagged)
    high_count = len(high_tier)
    mod_count = len(moderate_tier)

    capacity_utilization_flagged = (flagged_count / total_monthly_capacity) * 100.0 if total_monthly_capacity > 0 else 0
    capacity_utilization_high_only = (high_count / total_monthly_capacity) * 100.0 if total_monthly_capacity > 0 else 0

    return {
        'nurse_fte': nurse_navigator_fte,
        'monthly_caseload_per_nurse': patients_per_nurse_per_month,
        'total_care_team_capacity': total_monthly_capacity,
        'operational_threshold': operational_threshold,
        'total_cohort_evaluated': len(df_scored),
        'flagged_patient_count': flagged_count,
        'high_tier_count': high_count,
        'moderate_tier_count': mod_count,
        'capacity_utilization_flagged_pct': round(capacity_utilization_flagged, 1),
        'capacity_utilization_high_pct': round(capacity_utilization_high_only, 1),
        'can_cover_all_flagged': flagged_count <= total_monthly_capacity,
        'recommended_operational_action': (
            "Care team has sufficient capacity to enroll all flagged patients in proactive follow-up."
            if flagged_count <= total_monthly_capacity else
            "Flagged patient volume exceeds care team capacity! Recommend raising operational threshold "
            "or prioritizing High Risk tier only for intensive navigation."
        )
    }


def get_hypothetical_intervention_recommendation(risk_tier: str) -> Dict[str, Any]:
    """
    Returns evidence-based decision-support interventions.
    
    DISCLAIMER:
    These recommendations represent an algorithmic prototype designed for clinical workflow research.
    They do NOT constitute medical diagnoses, clinical guidelines, or independent orders for patient care.
    Final clinical decisions remain strictly within the clinician's professional judgment.
    """
    if risk_tier == "High":
        return {
            "tier": "High Risk",
            "intensity": "Intensive Transition of Care",
            "recommended_actions": [
                "Schedule primary care or endocrinology follow-up within 72 hours of hospital discharge.",
                "Assign designated Nurse Navigator for active outreach on days 2, 7, 14, and 28.",
                "Comprehensive clinical pharmacist medication reconciliation prior to discharge.",
                "Provide patient with home blood glucose testing kit and verify glucometer literacy."
            ],
            "disclaimer": "Research decision-support prototype only. Does not replace physician orders."
        }
    elif risk_tier == "Moderate":
        return {
            "tier": "Moderate Risk",
            "intensity": "Standard Transition of Care",
            "recommended_actions": [
                "Schedule routine outpatient clinical follow-up within 7-10 days.",
                "Automated post-discharge wellness check call on day 3 post-discharge.",
                "Provide standard diabetes self-management education materials and discharge instructions."
            ],
            "disclaimer": "Research decision-support prototype only. Does not replace physician orders."
        }
    else:
        return {
            "tier": "Low Risk",
            "intensity": "Routine Discharge Protocol",
            "recommended_actions": [
                "Provide standard written discharge summary.",
                "Recommend follow-up appointment within 30 days with primary care physician.",
                "Reinforce general diabetes nutritional and lifestyle maintenance guidance."
            ],
            "disclaimer": "Research decision-support prototype only. Does not replace physician orders."
        }
