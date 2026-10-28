"""
Computes SHAP values and generates clinical risk stratification visualizations.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_PATH, METADATA_PATH, OUTPUT_DIR
from src.pipeline import load_full_pipeline


def generate_explainability_artifacts():
    print("Loading pipeline and test cohort for SHAP analysis...")
    pipeline, metadata = load_full_pipeline(MODEL_PATH, METADATA_PATH)
    
    df_test = pd.read_parquet(PROJECT_ROOT / "data" / "interim" / "df_test.parquet")
    
    # 1. Extract feature engineer & preprocessor
    feature_engineer = pipeline.named_steps['features']
    preprocessor = pipeline.named_steps['preprocessor']
    classifier = pipeline.named_steps['classifier']
    
    # In CalibratedClassifierCV, get the underlying base estimators
    if hasattr(classifier, 'calibrated_classifiers_'):
        base_estimator = classifier.calibrated_classifiers_[0].estimator
    else:
        base_estimator = classifier

    # Preprocess test set sample for SHAP
    df_feat = feature_engineer.transform(df_test.iloc[:1500])
    X_sample = preprocessor.transform(df_feat)
    
    # Get feature names from ColumnTransformer
    cat_encoder = preprocessor.named_transformers_['cat']
    cat_names = cat_encoder.get_feature_names_out(feature_engineer.categorical_cols)
    feature_names = list(feature_engineer.numerical_cols) + list(cat_names)
    
    print(f"Calculating TreeExplainer on {len(feature_names)} features...")
    explainer = shap.TreeExplainer(base_estimator)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_vals = shap_values[1]  # positive class (readmission)
    else:
        shap_vals = shap_values

    # Plot SHAP summary
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_vals,
        X_sample,
        feature_names=feature_names,
        max_display=15,
        show=False
    )
    plt.title("SHAP Feature Importance (Clinical Risk Drivers)", fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    shap_path = OUTPUT_DIR / "shap_feature_importance.png"
    plt.savefig(shap_path, dpi=300)
    plt.close()
    print(f"Saved SHAP plot to {shap_path}")

    # 2. Risk Stratification Plot
    y_test_probs = pipeline.predict_proba(df_test)[:, 1]
    y_test_true = df_test['target'].values
    
    thr = metadata.get('optimal_threshold', 0.085)
    tiers_cfg = metadata.get('clinical_risk_tiers', {
        'low_risk_max': round(thr * 0.75, 4),
        'moderate_risk_max': round(thr * 1.35, 4)
    })
    
    def assign_tier(p):
        if p < tiers_cfg['low_risk_max']: return 'Low Risk'
        elif p <= tiers_cfg['moderate_risk_max']: return 'Moderate Risk'
        else: return 'High Risk'
        
    tiers = [assign_tier(p) for p in y_test_probs]
    df_eval = pd.DataFrame({'tier': tiers, 'readmitted': y_test_true})
    tier_order = ['Low Risk', 'Moderate Risk', 'High Risk']
    
    summary = df_eval.groupby('tier', observed=True)['readmitted'].agg(['count', 'mean']).reindex(tier_order)
    summary['mean'] = summary['mean'] * 100
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color = '#1f77b4'
    bars = ax1.bar(summary.index, summary['count'], color=['#2ca02c', '#ff7f0e', '#d62728'], alpha=0.85, width=0.5)
    ax1.set_ylabel('Number of Patients in Cohort', fontsize=11, fontweight='bold')
    ax1.set_title('Clinical Risk Stratification & Empirical Readmission Rates', fontsize=12, fontweight='bold')
    
    for bar in bars:
        h = bar.get_height()
        ax1.annotate(f"{int(h):,} pts", xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=10)

    ax2 = ax1.twinx()
    ax2.plot(summary.index, summary['mean'], color='black', marker='o', linewidth=2.5, markersize=8)
    ax2.set_ylabel('Observed 30-Day Readmission Rate (%)', fontsize=11, fontweight='bold', color='black')
    ax2.set_ylim(0, max(summary['mean']) * 1.3)
    
    for i, txt in enumerate(summary['mean']):
        ax2.annotate(f"{txt:.1f}%", xy=(i, txt), xytext=(0, 7),
                     textcoords="offset points", ha='center', fontweight='bold', fontsize=10)
        
    plt.tight_layout()
    strat_path = OUTPUT_DIR / "clinical_risk_stratification.png"
    plt.savefig(strat_path, dpi=300)
    plt.close()
    print(f"Saved stratification plot to {strat_path}")


if __name__ == "__main__":
    generate_explainability_artifacts()
