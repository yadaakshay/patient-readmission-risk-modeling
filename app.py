"""
Patient Readmission Risk Prediction & Clinical Decision Support Dashboard.
Streamlit Web Application featuring:
1. Clinical Overview & EDA
2. Model Comparison & Threshold Tuning
3. Risk Stratification & Care Team Capacity Simulation
4. Explainable AI (SHAP Global & Local Interpretations)
5. SQL Clinical Healthcare Intelligence Engine
"""
import sqlite3
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Clinical Readmission Risk Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
REPORTS_DIR = BASE_DIR / "reports"
FIG_DIR = REPORTS_DIR / "figures"
DB_PATH = DATA_DIR / "processed" / "readmission_analytics.db"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
TEST_PRED_PATH = DATA_DIR / "processed" / "df_test_predictions.parquet"

# Custom CSS for clinical dashboard aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background: #f8fafc;
        border-left: 4px solid #2563eb;
        padding: 1rem 1.2rem;
        border-radius: 6px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .disclaimer-box {
        background: #fffbeb;
        border: 1px solid #fef3c7;
        border-left: 5px solid #f59e0b;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        font-size: 0.88rem;
        color: #92400e;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_metadata():
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_test_predictions():
    if TEST_PRED_PATH.exists():
        return pd.read_parquet(TEST_PRED_PATH)
    return pd.DataFrame()


@st.cache_data
def run_db_query(sql: str):
    if not DB_PATH.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(sql, conn)
    finally:
        conn.close()
    return df


def show_disclaimer():
    st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ CLINICAL RESEARCH & DECISION SUPPORT DISCLAIMER:</strong><br>
        This machine learning system is an analytical decision-support prototype developed for healthcare quality 
        improvement under CMS HRRP paradigms. It is <strong>NOT</strong> a medical diagnostic device, does not establish 
        a standard of clinical care, and does not replace the professional diagnostic judgment of a licensed physician.
    </div>
    """, unsafe_allow_html=True)


# Sidebar navigation
st.sidebar.image("https://img.icons8.com/color/96/hospital-3.png", width=64)
st.sidebar.title("Clinical Navigation")
page = st.sidebar.radio(
    "Select Clinical Module:",
    [
        "📊 Clinical Overview & EDA",
        "📈 Model Comparison & Benchmark",
        "🎯 Risk Stratification & Workload",
        "🔍 Explainable AI (SHAP)",
        "💾 SQL Healthcare Analytics"
    ]
)

metadata = load_metadata()
df_test = load_test_predictions()

# -------------------------------------------------------------
# PAGE 1: CLINICAL OVERVIEW & EDA
# -------------------------------------------------------------
if page == "📊 Clinical Overview & EDA":
    st.markdown('<div class="main-header">Hospital Inpatient Readmission & Clinical Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">EHR Cohort Profiling, CMS Inclusions/Exclusions, and Baseline Readmission Epidemiology</div>', unsafe_allow_html=True)
    show_disclaimer()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Raw EHR Encounters", "101,766", "130 US Hospitals")
    with col2:
        st.metric("Clean Patient Cohort", "69,990", "First Encounter Only")
    with col3:
        st.metric("30-Day Readmission Prevalence", "8.98%", "6,285 Readmitted (<30d)")
    with col4:
        st.metric("CMS Mortality/Hospice Excluded", "2,423", "2.38% Deceased/Hospice")

    st.markdown("---")
    st.subheader("Key Clinical Exploratory Insights")
    
    tab1, tab2, tab3 = st.tabs(["Prior Utilization Impact", "Outcome Distribution", "Length of Stay & Medications"])

    with tab1:
        st.markdown("**Prior Healthcare Utilization Gradient**")
        st.write("Patients with frequent prior admissions in the preceding year have dramatically elevated readmission rates (over 20% in high utilizers vs 7.5% in patients with zero prior visits).")
        prior_util_fig = FIG_DIR / "eda_prior_utilization_impact.png"
        if prior_util_fig.exists():
            st.image(str(prior_util_fig), use_container_width=True)
        else:
            st.info("Prior utilization plot will render once pipeline completes.")

    with tab2:
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.markdown("**3-Class vs Binary Target Distribution**")
            target_fig = FIG_DIR / "eda_readmission_distribution.png"
            if target_fig.exists():
                st.image(str(target_fig), use_container_width=True)
        with col_b:
            st.markdown("**Special Missingness Audit**")
            st.write("EHR missing indicators like `?` were audited across all columns. Features like `weight` (96.8% missing) and administrative billing codes (`payer_code`, 39.6% missing) were removed, while glycemic markers (`A1Cresult`, `max_glu_serum`) were preserved as vital clinical indicators.")
            missing_fig = FIG_DIR / "eda_missingness_audit.png"
            if missing_fig.exists():
                st.image(str(missing_fig), use_container_width=True)

    with tab3:
        col_c, col_d = st.columns([1, 1])
        with col_c:
            st.markdown("**Hospital Length of Stay**")
            los_fig = FIG_DIR / "eda_length_of_stay_distribution.png"
            if los_fig.exists():
                st.image(str(los_fig), use_container_width=True)
        with col_d:
            st.markdown("**Age Distribution & Medication Complexity**")
            age_fig = FIG_DIR / "eda_age_and_diagnoses.png"
            if age_fig.exists():
                st.image(str(age_fig), use_container_width=True)


# -------------------------------------------------------------
# PAGE 2: MODEL COMPARISON & BENCHMARK
# -------------------------------------------------------------
elif page == "📈 Model Comparison & Benchmark":
    st.markdown('<div class="main-header">Model Performance Benchmark & Threshold Optimization</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluation Across 5-Fold Stratified Cross-Validation and Untouched Held-Out Test Set</div>', unsafe_allow_html=True)
    show_disclaimer()

    st.subheader("1. Discrimination & Calibration Evaluation Curves")
    c1, c2 = st.columns(2)
    with c1:
        roc_pr_fig = FIG_DIR / "roc_and_pr_curves_comparison.png"
        if not roc_pr_fig.exists():
            roc_pr_fig = BASE_DIR / "output_images" / "roc_and_pr_curves_comparison.png"
        if roc_pr_fig.exists():
            st.image(str(roc_pr_fig), caption="Receiver Operating Characteristic (ROC) & Precision-Recall (PR) Curves", use_container_width=True)
    with c2:
        calib_fig = FIG_DIR / "calibration_curves_comparison.png"
        if not calib_fig.exists():
            calib_fig = BASE_DIR / "output_images" / "calibration_curves_comparison.png"
        if calib_fig.exists():
            st.image(str(calib_fig), caption="Reliability Diagram (Probability Calibration via Platt Scaling)", use_container_width=True)

    st.markdown("---")
    st.subheader("2. The 91% Accuracy Paradox & Decision Threshold Selection")
    
    opt_thr = metadata.get('optimal_threshold', 0.085)
    std_metrics = metadata.get('test_metrics_standard_threshold', {})
    opt_metrics = metadata.get('test_metrics_optimal_threshold', {})

    st.write(f"""
    In clinical settings with low positive outcome prevalence (~9%), **default 0.50 decision thresholds fail completely**.
    At threshold 0.50, the model achieves 91% accuracy by predicting negative for nearly all encounters, resulting in **0% sensitivity (missing 100% of readmitted patients)**.
    By optimizing the decision threshold to **{opt_thr:.3f}** (via Youden's J statistic on the validation set), the model detects over **60% of all readmissions**.
    """)

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown(f"#### Standard Threshold (`0.500`)")
        st.write(f"- **Accuracy**: {std_metrics.get('accuracy', 0.9102)*100:.2f}%")
        st.write(f"- **Sensitivity (Recall)**: {std_metrics.get('sensitivity_recall', 0.0)*100:.2f}% (Misses nearly all readmissions)")
        st.write(f"- **Specificity**: {std_metrics.get('specificity', 1.0)*100:.2f}%")
        st.write(f"- **Precision**: {std_metrics.get('precision', 0.0)*100:.2f}%")
        st.write(f"- **F1 Score**: {std_metrics.get('f1_score', 0.0):.4f}")

    with col_m2:
        st.markdown(f"#### Optimal Clinical Threshold (`{opt_thr:.3f}`)")
        st.write(f"- **Accuracy**: {opt_metrics.get('accuracy', 0.5741)*100:.2f}%")
        st.write(f"- **Sensitivity (Recall)**: **{opt_metrics.get('sensitivity_recall', 0.6235)*100:.2f}%** (Captures >60% of readmissions)")
        st.write(f"- **Specificity**: {opt_metrics.get('specificity', 0.5692)*100:.2f}%")
        st.write(f"- **Precision**: {opt_metrics.get('precision', 0.1250)*100:.2f}%")
        st.write(f"- **F1 Score**: **{opt_metrics.get('f1_score', 0.208):.4f}**")

    cm_fig = FIG_DIR / "confusion_matrix_optimal_threshold.png"
    if not cm_fig.exists():
        cm_fig = BASE_DIR / "output_images" / "confusion_matrix_optimal_threshold.png"
    if cm_fig.exists():
        st.image(str(cm_fig), caption=f"Confusion Matrix at Optimal Decision Threshold ({opt_thr:.3f})", width=600)


# -------------------------------------------------------------
# PAGE 3: RISK STRATIFICATION & WORKLOAD
# -------------------------------------------------------------
elif page == "🎯 Risk Stratification & Workload":
    st.markdown('<div class="main-header">Clinical Risk Stratification & Workload Simulation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Actionable Risk Tiers, Observed Prevalence, and Care Management Resource Allocation</div>', unsafe_allow_html=True)
    show_disclaimer()

    st.sidebar.subheader("Interactive Threshold Controls")
    default_opt = metadata.get('optimal_threshold', 0.085)
    low_cutoff = st.sidebar.slider("Low / Moderate Risk Cutoff", 0.02, 0.15, float(round(default_opt * 0.75, 3)), 0.005)
    high_cutoff = st.sidebar.slider("Moderate / High Risk Cutoff", 0.08, 0.30, float(round(default_opt * 1.35, 3)), 0.005)

    if not df_test.empty and 'readmission_prob' in df_test.columns:
        probs = df_test['readmission_prob'].values
        targets = df_test['target'].values
        
        # Calculate dynamic risk tiers
        dynamic_tiers = np.where(probs < low_cutoff, "Low", np.where(probs < high_cutoff, "Moderate", "High"))
        df_strat = pd.DataFrame({'tier': dynamic_tiers, 'target': targets, 'prob': probs})

        st.subheader("1. Cohort Distribution Across Actionable Risk Tiers")
        
        tier_stats = []
        for t in ["Low", "Moderate", "High"]:
            sub = df_strat[df_strat['tier'] == t]
            cnt = len(sub)
            obs_readm = sub['target'].sum() if cnt > 0 else 0
            obs_rate = (obs_readm / cnt * 100.0) if cnt > 0 else 0.0
            tier_stats.append({
                "Risk Tier": t,
                "Encounter Count": cnt,
                "Cohort Share (%)": round(cnt / len(df_strat) * 100.0, 1),
                "Observed Readmissions": int(obs_readm),
                "Observed Readmission Rate (%)": round(obs_rate, 2),
                "Avg Predicted Risk (%)": round(sub['prob'].mean() * 100.0, 2) if cnt > 0 else 0.0
            })
        st.dataframe(pd.DataFrame(tier_stats), use_container_width=True)

        st.markdown("---")
        st.subheader("2. Care Team Follow-Up Workload Simulator")
        st.write("Simulate whether hospital nurse care coordinators can accommodate high-intensity post-discharge interventions under current clinical capacity.")

        w_col1, w_col2, w_col3 = st.columns(3)
        with w_col1:
            nurse_fte = st.number_input("Care Management FTEs (Nurses)", min_value=1, max_value=50, value=5)
        with w_col2:
            caseload = st.number_input("Max Monthly Caseload per Nurse", min_value=10, max_value=150, value=40)
        with w_col3:
            total_capacity = nurse_fte * caseload
            st.metric("Total Monthly Care Capacity", f"{total_capacity:,} Patients", f"{nurse_fte} FTEs × {caseload} cases")

        high_volume = (dynamic_tiers == "High").sum()
        flagged_volume = (probs >= default_opt).sum()

        c_stat1, c_stat2 = st.columns(2)
        with c_stat1:
            st.markdown(f"**High Risk Tier Only** ({high_volume:,} patients)")
            util_high = (high_volume / total_capacity) * 100.0 if total_capacity > 0 else 0
            st.progress(min(util_high / 100.0, 1.0))
            st.write(f"Capacity Utilization: **{util_high:.1f}%**")
            if high_volume <= total_capacity:
                st.success("✅ Capacity sufficient to enroll 100% of High Risk patients in intensive care navigation!")
            else:
                st.warning("⚠️ High Risk volume exceeds nurse capacity. Consider tightening cutoff or requesting additional FTEs.")

        with c_stat2:
            st.markdown(f"**All Flagged Patients (Threshold ≥ {default_opt:.3f})** ({flagged_volume:,} patients)")
            util_flagged = (flagged_volume / total_capacity) * 100.0 if total_capacity > 0 else 0
            st.progress(min(util_flagged / 100.0, 1.0))
            st.write(f"Capacity Utilization: **{util_flagged:.1f}%**")
            if flagged_volume <= total_capacity:
                st.success("✅ Capacity sufficient to cover all flagged patients.")
            else:
                st.info("ℹ️ Total flagged volume exceeds capacity; prioritize High Risk tier first.")


# -------------------------------------------------------------
# PAGE 4: EXPLAINABLE AI (SHAP)
# -------------------------------------------------------------
elif page == "🔍 Explainable AI (SHAP)":
    st.markdown('<div class="main-header">Explainable AI (XAI) & Clinical Model Transparency</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Global Feature Attribution, Directional Impact, and Patient-Specific SHAP Waterfall Explanations</div>', unsafe_allow_html=True)
    show_disclaimer()

    st.subheader("1. Global Risk Drivers (SHAP Beeswarm & Importance)")
    st.write("SHAP values quantify how each clinical feature pushes the model's prediction toward or away from readmission relative to base expectation. (Note: SHAP describes statistical association within model logic, not clinical causation).")

    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        beeswarm_fig = FIG_DIR / "shap_summary_beeswarm.png"
        if not beeswarm_fig.exists():
            beeswarm_fig = BASE_DIR / "output_images" / "shap_feature_importance.png"
        if beeswarm_fig.exists():
            st.image(str(beeswarm_fig), caption="SHAP Summary Beeswarm (Red = High Feature Value, Blue = Low)", use_container_width=True)
    with exp_col2:
        bar_fig = FIG_DIR / "shap_feature_importance_bar.png"
        if bar_fig.exists():
            st.image(str(bar_fig), caption="Top 15 Predictive Risk Drivers (Mean Absolute SHAP Value)", use_container_width=True)

    st.markdown("---")
    st.subheader("2. Patient-Specific Local Explanations (Waterfall)")
    st.write("Compare how clinical factors drive predictions for high-risk versus low-risk patient encounters.")

    w1, w2 = st.columns(2)
    with w1:
        wf_high = FIG_DIR / "shap_waterfall_high_risk.png"
        if wf_high.exists():
            st.image(str(wf_high), caption="High-Risk Patient Case: Prior Inpatient Admissions + Prolonged Stay Drive Risk Up", use_container_width=True)
    with w2:
        wf_low = FIG_DIR / "shap_waterfall_low_risk.png"
        if wf_low.exists():
            st.image(str(wf_low), caption="Low-Risk Patient Case: Zero Prior Admissions + Normal Routine Markers Anchor Low Risk", use_container_width=True)


# -------------------------------------------------------------
# PAGE 5: SQL HEALTHCARE ANALYTICS
# -------------------------------------------------------------
elif page == "💾 SQL Healthcare Analytics":
    st.markdown('<div class="main-header">SQL Clinical Analytics & Healthcare Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Interactive Query Engine Powered by SQLite Clinical Database</div>', unsafe_allow_html=True)
    show_disclaimer()

    query_options = {
        "1. Baseline Readmission Prevalence": """
        SELECT 
            COUNT(*) AS total_encounters,
            SUM(target) AS readmitted_within_30_days,
            ROUND(AVG(target) * 100.0, 2) AS readmission_rate_pct,
            COUNT(DISTINCT patient_nbr) AS unique_patients
        FROM encounters;
        """,
        "2. Readmission Rates by Age Bracket": """
        SELECT 
            age,
            COUNT(*) AS encounter_count,
            SUM(target) AS readmissions_30d,
            ROUND(AVG(target) * 100.0, 2) AS readmission_rate_pct,
            CASE WHEN COUNT(*) < 100 THEN 'Small Sample (<100)' ELSE 'Adequate Sample' END AS sample_size_flag
        FROM encounters
        GROUP BY age
        ORDER BY age;
        """,
        "3. Readmission by Length of Hospital Stay": """
        SELECT 
            CASE 
                WHEN time_in_hospital BETWEEN 1 AND 2 THEN '1-2 Days (Short)'
                WHEN time_in_hospital BETWEEN 3 AND 5 THEN '3-5 Days (Medium)'
                WHEN time_in_hospital BETWEEN 6 AND 8 THEN '6-8 Days (Long)'
                ELSE '9+ Days (Extended)'
            END AS los_category,
            COUNT(*) AS total_patients,
            SUM(target) AS readmitted_patients,
            ROUND(AVG(target) * 100.0, 2) AS readmission_rate_pct,
            ROUND(AVG(num_medications), 1) AS avg_medications_prescribed
        FROM encounters
        GROUP BY los_category
        ORDER BY MIN(time_in_hospital);
        """,
        "4. Readmission by Prior Inpatient History": """
        SELECT 
            CASE 
                WHEN number_inpatient = 0 THEN '0 Prior Admissions'
                WHEN number_inpatient = 1 THEN '1 Prior Admission'
                WHEN number_inpatient = 2 THEN '2 Prior Admissions'
                ELSE '3+ Prior Admissions (High Utilizer)'
            END AS prior_inpatient_tier,
            COUNT(*) AS total_encounters,
            SUM(target) AS readmissions,
            ROUND(AVG(target) * 100.0, 2) AS readmission_rate_pct,
            ROUND(AVG(number_diagnoses), 1) AS avg_comorbidities
        FROM encounters
        GROUP BY prior_inpatient_tier
        ORDER BY MIN(number_inpatient);
        """,
        "5. Readmission by Discharge Disposition": """
        SELECT 
            CASE 
                WHEN discharge_disposition_id = 1 THEN 'Discharged to Home'
                WHEN discharge_disposition_id IN (3, 6) THEN 'Home Health Care or SNF'
                WHEN discharge_disposition_id IN (2, 4, 5, 22) THEN 'Transferred to Acute/Rehab'
                WHEN discharge_disposition_id = 7 THEN 'Left Against Medical Advice (AMA)'
                ELSE 'Other Disposition'
            END AS discharge_destination,
            COUNT(*) AS encounter_count,
            SUM(target) AS readmissions,
            ROUND(AVG(target) * 100.0, 2) AS readmission_rate_pct,
            CASE WHEN COUNT(*) < 100 THEN 'Small Subgroup' ELSE 'Reliable' END AS reliability_flag
        FROM encounters
        GROUP BY discharge_destination
        ORDER BY readmission_rate_pct DESC;
        """,
        "6. Test Cohort Prediction Distribution across Tiers": """
        SELECT 
            prediction_tier,
            COUNT(*) AS patient_volume,
            ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM predictions), 1) AS volume_share_pct,
            SUM(target) AS observed_readmissions,
            ROUND(AVG(target) * 100.0, 2) AS observed_readmission_rate_pct,
            ROUND(AVG(readmission_prob) * 100.0, 2) AS mean_predicted_risk_pct
        FROM predictions
        GROUP BY prediction_tier
        ORDER BY mean_predicted_risk_pct ASC;
        """,
        "7. False Positive & False Negative Clinical Characteristics": """
        SELECT 
            error_type,
            COUNT(*) AS patient_count,
            ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM predictions), 1) AS share_of_test_cohort_pct,
            ROUND(AVG(readmission_prob), 3) AS mean_predicted_prob,
            ROUND(AVG(time_in_hospital), 1) AS avg_length_of_stay,
            ROUND(AVG(number_inpatient), 2) AS avg_prior_inpatient,
            ROUND(AVG(num_medications), 1) AS avg_medications,
            ROUND(AVG(number_diagnoses), 1) AS avg_diagnoses
        FROM predictions
        GROUP BY error_type
        ORDER BY patient_count DESC;
        """,
        "8. Model Performance across Clinical Subgroups": """
        SELECT 
            'Elderly (Age 65+)' AS subgroup,
            COUNT(*) AS total_patients,
            SUM(target) AS actual_readmissions,
            SUM(predicted_flag) AS flagged_by_model,
            ROUND(SUM(CASE WHEN target = 1 AND predicted_flag = 1 THEN 1.0 ELSE 0.0 END) / NULLIF(SUM(target), 0) * 100.0, 1) AS sensitivity_pct,
            ROUND(SUM(CASE WHEN target = 1 AND predicted_flag = 1 THEN 1.0 ELSE 0.0 END) / NULLIF(SUM(predicted_flag), 0) * 100.0, 1) AS precision_pct
        FROM predictions
        WHERE age IN ('[60-70)', '[70-80)', '[80-90)', '[90-100)')
        UNION ALL
        SELECT 
            'Prior Inpatient History',
            COUNT(*),
            SUM(target),
            SUM(predicted_flag),
            ROUND(SUM(CASE WHEN target = 1 AND predicted_flag = 1 THEN 1.0 ELSE 0.0 END) / NULLIF(SUM(target), 0) * 100.0, 1),
            ROUND(SUM(CASE WHEN target = 1 AND predicted_flag = 1 THEN 1.0 ELSE 0.0 END) / NULLIF(SUM(predicted_flag), 0) * 100.0, 1)
        FROM predictions
        WHERE number_inpatient > 0
        UNION ALL
        SELECT 
            'Active Insulin Therapy',
            COUNT(*),
            SUM(target),
            SUM(predicted_flag),
            ROUND(SUM(CASE WHEN target = 1 AND predicted_flag = 1 THEN 1.0 ELSE 0.0 END) / NULLIF(SUM(target), 0) * 100.0, 1),
            ROUND(SUM(CASE WHEN target = 1 AND predicted_flag = 1 THEN 1.0 ELSE 0.0 END) / NULLIF(SUM(predicted_flag), 0) * 100.0, 1)
        FROM predictions
        WHERE insulin IN ('Steady', 'Up', 'Down');
        """
    }

    selected_query_label = st.selectbox("Select Business / Clinical Question:", list(query_options.keys()))
    sql_text = query_options[selected_query_label]

    st.markdown("**Executed SQL Query:**")
    st.code(sql_text.strip(), language="sql")

    df_res = run_db_query(sql_text)
    if not df_res.empty:
        st.markdown("**Query Results:**")
        st.dataframe(df_res, use_container_width=True)

        csv_bytes = df_res.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv_bytes,
            file_name=f"{selected_query_label.replace(' ', '_')}.csv",
            mime="text/csv"
        )
    else:
        st.info("Database results will appear once the pipeline executes and populates SQLite.")
