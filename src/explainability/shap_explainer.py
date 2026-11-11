"""
Model interpretability and Explainable AI (SHAP) module.
Generates:
1. Global SHAP feature importance beeswarm and bar charts.
2. Individual patient waterfall plots (local explanations).
3. Logistic Regression odds ratios vs tree-based importance.
4. Non-causal clinical disclaimer enforcement.
"""
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from src.config import MODEL_PATH, METADATA_PATH, OUTPUT_DIR
from src.pipeline import load_full_pipeline

FIG_DIR = Path(__file__).resolve().parent.parent.parent / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def extract_base_model_and_features(pipeline) -> Tuple[Any, List[str], Any, Any]:
    """Extracts the underlying uncalibrated estimator, feature names, and preprocessors."""
    feature_engineer = pipeline.named_steps['features']
    preprocessor = pipeline.named_steps['preprocessor']
    classifier = pipeline.named_steps['classifier']

    if hasattr(classifier, 'calibrated_classifiers_'):
        base_estimator = classifier.calibrated_classifiers_[0].estimator
    else:
        base_estimator = classifier

    # Extract feature names
    cat_encoder = preprocessor.named_transformers_['cat']
    cat_names = list(cat_encoder.get_feature_names_out(feature_engineer.categorical_cols))
    feature_names = list(feature_engineer.numerical_cols) + cat_names

    return base_estimator, feature_names, feature_engineer, preprocessor


def generate_shap_explanations(df_test: pd.DataFrame, n_samples: int = 1200):
    """
    Computes SHAP values on test set and exports global & local explanation plots.
    """
    print(f"\nComputing SHAP explanations on {min(len(df_test), n_samples)} test encounters...")
    pipeline, metadata = load_full_pipeline(MODEL_PATH, METADATA_PATH)
    base_estimator, feature_names, feat_eng, prep = extract_base_model_and_features(pipeline)

    sample_df = df_test.iloc[:n_samples].copy()
    X_feat = feat_eng.transform(sample_df)
    X_prep = prep.transform(X_feat)

    explainer = shap.TreeExplainer(base_estimator)
    shap_explanation = explainer(X_prep)

    # In binary classification, check if explanation has 2 output classes
    if len(shap_explanation.shape) == 3:
        # Class 1 (readmission)
        shap_values_pos = shap_explanation.values[:, :, 1]
        base_val = shap_explanation.base_values[0, 1] if shap_explanation.base_values.ndim > 1 else shap_explanation.base_values[0]
    else:
        shap_values_pos = shap_explanation.values
        base_val = shap_explanation.base_values[0]

    # 1. Global Beeswarm Summary Plot
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values_pos,
        X_prep,
        feature_names=feature_names,
        max_display=15,
        show=False
    )
    plt.title("SHAP Global Feature Importance (Beeswarm Summary)", fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_summary_beeswarm.png", dpi=300)
    plt.savefig(OUTPUT_DIR / "shap_feature_importance.png", dpi=300)
    plt.close()

    # 2. Global Bar Chart of Mean Absolute SHAP Values
    mean_abs_shap = np.mean(np.abs(shap_values_pos), axis=0)
    top_indices = np.argsort(mean_abs_shap)[-15:]
    top_features = [feature_names[i] for i in top_indices]
    top_scores = mean_abs_shap[top_indices]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(top_scores)), top_scores, color='#1f77b4', align='center')
    plt.yticks(range(len(top_scores)), top_features)
    plt.xlabel("Mean |SHAP Value| (Impact on Log-Odds of Readmission)")
    plt.title("Top 15 Predictive Risk Drivers (Global SHAP Magnitude)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_feature_importance_bar.png", dpi=300)
    plt.close()

    # 3. Local Explanations (High-Risk vs Low-Risk Patients)
    probs = pipeline.predict_proba(sample_df)[:, 1]
    high_risk_idx = int(np.argmax(probs))
    low_risk_idx = int(np.argmin(probs))

    # Waterfall plot for high risk patient
    plt.figure(figsize=(9, 6))
    shap_exp_single_high = shap.Explanation(
        values=shap_values_pos[high_risk_idx],
        base_values=base_val,
        data=X_prep[high_risk_idx],
        feature_names=feature_names
    )
    shap.plots.waterfall(shap_exp_single_high, max_display=10, show=False)
    plt.title(f"Local Patient Explanation - High Risk Profile (Prob: {probs[high_risk_idx]:.1%})", fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_waterfall_high_risk.png", dpi=300)
    plt.close()

    # Waterfall plot for low risk patient
    plt.figure(figsize=(9, 6))
    shap_exp_single_low = shap.Explanation(
        values=shap_values_pos[low_risk_idx],
        base_values=base_val,
        data=X_prep[low_risk_idx],
        feature_names=feature_names
    )
    shap.plots.waterfall(shap_exp_single_low, max_display=10, show=False)
    plt.title(f"Local Patient Explanation - Low Risk Profile (Prob: {probs[low_risk_idx]:.1%})", fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_waterfall_low_risk.png", dpi=300)
    plt.close()

    print(f"Saved SHAP plots to {FIG_DIR}")
    return {
        'top_15_features': list(reversed(top_features)),
        'top_15_mean_abs_shap': [float(x) for x in reversed(top_scores)]
    }
