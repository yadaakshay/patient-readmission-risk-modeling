"""
Unit tests for clinical risk stratification and care team workload simulation.
"""
import pytest
import numpy as np
import pandas as pd
from src.risk_stratification.stratifier import (
    compute_risk_tiers,
    summarize_risk_stratification,
    simulate_care_team_workload,
    get_hypothetical_intervention_recommendation
)


def test_compute_risk_tiers():
    probs = np.array([0.02, 0.05, 0.08, 0.12, 0.15, 0.25])
    tiers = compute_risk_tiers(probs, low_threshold=0.06, moderate_threshold=0.14)
    assert list(tiers) == ["Low", "Low", "Moderate", "Moderate", "High", "High"]


def test_summarize_risk_stratification():
    y_true = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])
    y_prob = np.array([0.02, 0.03, 0.04, 0.05, 0.08, 0.09, 0.15, 0.20, 0.25, 0.30])
    summary_df = summarize_risk_stratification(y_true, y_prob, low_threshold=0.06, moderate_threshold=0.14)

    assert isinstance(summary_df, pd.DataFrame)
    assert len(summary_df) == 3
    assert set(summary_df['Risk Tier'].tolist()) == {"Low", "Moderate", "High"}
    # Total encounters sum to 10
    assert summary_df['Encounter Count'].sum() == 10
    # Cohort shares sum to 100%
    assert np.isclose(summary_df['Cohort Share (%)'].sum(), 100.0)


def test_simulate_care_team_workload():
    df_scored = pd.DataFrame({
        'readmission_prob': [0.03, 0.05, 0.10, 0.15, 0.22],
        'prediction_tier': ['Low', 'Low', 'Moderate', 'High', 'High']
    })
    sim = simulate_care_team_workload(
        df_scored,
        nurse_navigator_fte=2,
        patients_per_nurse_per_month=10,
        operational_threshold=0.08
    )

    assert sim['total_care_team_capacity'] == 20
    assert sim['flagged_patient_count'] == 3  # >= 0.08
    assert sim['high_tier_count'] == 2
    assert sim['can_cover_all_flagged'] is True


def test_hypothetical_intervention_recommendation():
    high_rec = get_hypothetical_intervention_recommendation("High")
    assert high_rec['tier'] == "High Risk"
    assert "disclaimer" in high_rec
    assert len(high_rec['recommended_actions']) > 0

    low_rec = get_hypothetical_intervention_recommendation("Low")
    assert low_rec['tier'] == "Low Risk"
