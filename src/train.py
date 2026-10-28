"""
End-to-end model training, cross-validation, evaluation, and artifact generation script.
"""
import argparse
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.config import (
    MODEL_PATH,
    METADATA_PATH,
    OUTPUT_DIR,
    RANDOM_SEED
)
from src.data_loader import (
    load_raw_data,
    prepare_clinical_cohort,
    split_patient_cohort,
    save_splits
)
from src.pipeline import (
    build_full_pipeline,
    save_full_pipeline
)
from src.evaluate import (
    evaluate_binary_predictions,
    find_optimal_threshold,
    plot_roc_and_pr_curves,
    plot_calibration_curves,
    plot_confusion_matrix_annotated
)


def run_cross_validation(
    df_train: pd.DataFrame,
    model_types=['logistic_regression', 'random_forest', 'xgboost', 'lightgbm'],
    cv: int = 5,
    calibrate: bool = True
):
    """Executes Stratified K-Fold cross-validation across specified model architectures."""
    print(f"\n========================================================")
    print(f"Running {cv}-Fold Stratified Cross-Validation Benchmark")
    print(f"========================================================")
    
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
    y_train = df_train['target'].values
    cv_summary = {}

    for model_name in model_types:
        print(f"\nEvaluating: {model_name.upper()}...")
        roc_aucs, pr_aucs, briers = [], [], []

        for fold, (train_idx, val_idx) in enumerate(skf.split(df_train, y_train), 1):
            fold_train = df_train.iloc[train_idx]
            fold_val = df_train.iloc[val_idx]

            pipeline = build_full_pipeline(
                model_type=model_name,
                calibrate=calibrate,
                random_state=RANDOM_SEED + fold
            )
            pipeline.fit(fold_train, fold_train['target'].values)
            y_val_prob = pipeline.predict_proba(fold_val)[:, 1]

            metrics = evaluate_binary_predictions(fold_val['target'].values, y_val_prob)
            roc_aucs.append(metrics['roc_auc'])
            pr_aucs.append(metrics['pr_auc'])
            briers.append(metrics['brier_score'])

        cv_summary[model_name] = {
            'mean_roc_auc': float(np.mean(roc_aucs)),
            'std_roc_auc': float(np.std(roc_aucs)),
            'mean_pr_auc': float(np.mean(pr_aucs)),
            'std_pr_auc': float(np.std(pr_aucs)),
            'mean_brier': float(np.mean(briers))
        }
        print(f"  -> ROC-AUC: {np.mean(roc_aucs):.4f} +/- {np.std(roc_aucs):.4f}")
        print(f"  -> PR-AUC:  {np.mean(pr_aucs):.4f} +/- {np.std(pr_aucs):.4f}")
        print(f"  -> Brier:   {np.mean(briers):.4f}")

    return cv_summary


def train_and_evaluate_final_model(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    best_model_name: str = 'lightgbm',
    calibrate: bool = True,
    save_artifacts: bool = True
):
    """
    Trains all candidate models, tunes decision threshold on Validation set,
    and reports unbiased performance on held-out Test set.
    """
    print(f"\n========================================================")
    print(f"Training and Evaluating Models on Test Set")
    print(f"========================================================")

    model_names = ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm']
    test_results_dict = {}
    fitted_pipelines = {}
    val_probs_dict = {}

    y_train = df_train['target'].values
    y_val = df_val['target'].values
    y_test = df_test['target'].values

    for name in model_names:
        print(f"Fitting {name}...")
        pipe = build_full_pipeline(model_type=name, calibrate=calibrate, random_state=RANDOM_SEED)
        pipe.fit(df_train, y_train)
        
        y_val_prob = pipe.predict_proba(df_val)[:, 1]
        y_test_prob = pipe.predict_proba(df_test)[:, 1]

        val_probs_dict[name] = y_val_prob
        test_results_dict[name] = (y_test, y_test_prob)
        fitted_pipelines[name] = pipe

    # Plot comparative evaluation figures
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_roc_and_pr_curves(
        test_results_dict,
        save_path=OUTPUT_DIR / "roc_and_pr_curves_comparison.png"
    )
    plot_calibration_curves(
        test_results_dict,
        save_path=OUTPUT_DIR / "calibration_curves_comparison.png"
    )
    print(f"Saved evaluation curves to {OUTPUT_DIR}")

    # Optimize threshold on Validation set for the primary best model
    best_pipe = fitted_pipelines[best_model_name]
    best_val_prob = val_probs_dict[best_model_name]
    best_test_prob = test_results_dict[best_model_name][1]

    # Threshold tuned using Youden's J for balanced sensitivity/specificity
    opt_thr, opt_val_metrics = find_optimal_threshold(y_val, best_val_prob, method='youden')
    
    # Test evaluation at standard 0.5 threshold vs optimal threshold
    standard_test_metrics = evaluate_binary_predictions(y_test, best_test_prob, threshold=0.5)
    optimal_test_metrics = evaluate_binary_predictions(y_test, best_test_prob, threshold=opt_thr)

    print(f"\n--- {best_model_name.upper()} FINAL PERFORMANCE ON UNSEEN TEST SET ---")
    print(f"ROC-AUC:               {optimal_test_metrics['roc_auc']:.4f}")
    print(f"PR-AUC (Avg Precision): {optimal_test_metrics['pr_auc']:.4f}")
    print(f"Brier Score:           {optimal_test_metrics['brier_score']:.4f}")
    print(f"\n[Standard Threshold = 0.500]")
    print(f"  Accuracy:            {standard_test_metrics['accuracy']:.4f}")
    print(f"  Sensitivity (Recall): {standard_test_metrics['sensitivity_recall']:.4f}")
    print(f"  Specificity:         {standard_test_metrics['specificity']:.4f}")
    print(f"  Precision:           {standard_test_metrics['precision']:.4f}")
    print(f"  F1 Score:            {standard_test_metrics['f1_score']:.4f}")
    print(f"\n[Optimal Clinical Threshold = {opt_thr:.3f}]")
    print(f"  Accuracy:            {optimal_test_metrics['accuracy']:.4f}")
    print(f"  Sensitivity (Recall): {optimal_test_metrics['sensitivity_recall']:.4f}")
    print(f"  Specificity:         {optimal_test_metrics['specificity']:.4f}")
    print(f"  Precision:           {optimal_test_metrics['precision']:.4f}")
    print(f"  F1 Score:            {optimal_test_metrics['f1_score']:.4f}")

    # Plot Confusion Matrix at optimal threshold
    y_test_pred_opt = (best_test_prob >= opt_thr).astype(int)
    plot_confusion_matrix_annotated(
        y_test,
        y_test_pred_opt,
        threshold=opt_thr,
        model_name=best_model_name.upper(),
        save_path=OUTPUT_DIR / "confusion_matrix_optimal_threshold.png"
    )

    metadata = {
        'model_name': best_model_name,
        'calibrated': calibrate,
        'optimal_threshold': float(opt_thr),
        'clinical_risk_tiers': {
            'low_risk_max': float(round(opt_thr * 0.75, 4)),
            'moderate_risk_max': float(round(opt_thr * 1.35, 4))
        },
        'test_metrics_optimal_threshold': optimal_test_metrics,
        'test_metrics_standard_threshold': standard_test_metrics,
        'dataset_summary': {
            'train_size': len(df_train),
            'val_size': len(df_val),
            'test_size': len(df_test),
            'positive_prevalence_pct': float(np.mean(y_train) * 100)
        }
    }

    if save_artifacts:
        save_full_pipeline(best_pipe, metadata, MODEL_PATH, METADATA_PATH)

    return best_pipe, metadata


def main():
    parser = argparse.ArgumentParser(description="Train Patient Readmission Risk Prediction Pipeline")
    parser.add_argument("--model", type=str, default="lightgbm", choices=["lightgbm", "xgboost", "random_forest", "logistic_regression", "all"], help="Model architecture")
    parser.add_argument("--no-calibrate", dest="calibrate", action="store_false", help="Disable probability calibration")
    parser.add_argument("--cv", type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument("--no-save", dest="save", action="store_false", help="Do not persist model artifacts")
    parser.set_defaults(calibrate=True, save=True)
    args = parser.parse_args()

    print("Step 1: Loading raw dataset and preparing clinical cohort...")
    raw_df = load_raw_data()
    cohort_df = prepare_clinical_cohort(raw_df, keep_first_encounter=True)
    print(f"Cohort size: {len(cohort_df)} patients, Target positive prevalence: {cohort_df['target'].mean()*100:.2f}%")

    print("\nStep 2: Splitting into Train (70%), Validation (15%), Test (15%) with zero patient leakage...")
    df_train, df_val, df_test = split_patient_cohort(cohort_df, test_size=0.15, val_size=0.15)
    save_splits(df_train, df_val, df_test)

    print("\nStep 3: Running Cross-Validation benchmark across models...")
    cv_models = ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm'] if args.model == 'all' else [args.model]
    cv_summary = run_cross_validation(df_train, model_types=cv_models, cv=args.cv, calibrate=args.calibrate)

    best_model_name = args.model if args.model != 'all' else 'lightgbm'
    print(f"\nStep 4: Training final production model ({best_model_name.upper()}) and evaluating on held-out test cohort...")
    pipe, metadata = train_and_evaluate_final_model(
        df_train,
        df_val,
        df_test,
        best_model_name=best_model_name,
        calibrate=args.calibrate,
        save_artifacts=args.save
    )
    print("\nTraining and validation complete!")


if __name__ == "__main__":
    main()
