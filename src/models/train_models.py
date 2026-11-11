"""
Comprehensive model training, benchmarking, and evaluation orchestrator.
Compares:
1. DummyClassifier (Baseline)
2. Logistic Regression (Unweighted, Class-Weighted, SMOTE)
3. Random Forest (Unweighted, Class-Weighted, SMOTE)
4. XGBoost (Unweighted, Class-Weighted, SMOTE)
5. LightGBM (Unweighted, Class-Weighted, SMOTE)
"""
import sys
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    MODEL_PATH,
    METADATA_PATH,
    OUTPUT_DIR,
    RANDOM_SEED
)
from src.data.loader import load_raw_data, split_stratified_cohort, save_dataset_splits
from src.data.target import prepare_cohort
from src.models.baseline import build_dummy_baseline_pipeline
from src.models.smote_pipeline import build_pipeline_with_strategy
from src.models.hyperparameter_tuning import run_hyperparameter_search
from src.evaluation.metrics import calculate_clinical_metrics, find_optimal_threshold
from src.evaluation.plots import (
    plot_roc_and_pr_curves,
    plot_calibration_curves,
    plot_confusion_matrix_annotated
)
from src.pipeline import save_full_pipeline


def run_benchmark_cross_validation(
    df_train: pd.DataFrame,
    cv_splits: int = 5,
    sample_size: int = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Runs cross-validation across all model architectures and imbalance strategies.
    Reports ROC-AUC, PR-AUC, Brier score, and balanced accuracy.
    """
    print(f"\n========================================================")
    print(f"Running {cv_splits}-Fold Stratified Cross-Validation Benchmark")
    print(f"========================================================")

    if sample_size and len(df_train) > sample_size:
        eval_df = df_train.sample(n=sample_size, random_state=RANDOM_SEED).reset_index(drop=True)
    else:
        eval_df = df_train

    y_train = eval_df['target'].values
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_SEED)

    experiments = [
        # Baseline
        ('Dummy (Stratified)', 'dummy', 'none'),
        # Core Models with Class Weighting
        ('Logistic Regression (Weighted)', 'logistic_regression', 'class_weight'),
        ('Random Forest (Weighted)', 'random_forest', 'class_weight'),
        ('XGBoost (Weighted)', 'xgboost', 'class_weight'),
        ('LightGBM (Weighted)', 'lightgbm', 'class_weight'),
        # Imbalance Comparison Experiments
        ('LightGBM (Unweighted)', 'lightgbm', 'none'),
        ('LightGBM (SMOTE)', 'lightgbm', 'smote'),
        ('Logistic Regression (SMOTE)', 'logistic_regression', 'smote'),
    ]

    benchmark_records = []
    cv_detailed = {}

    for label, m_type, strat in experiments:
        print(f"Evaluating: {label}...")
        roc_list, pr_list, brier_list, bal_acc_list = [], [], [], []

        for fold, (train_idx, val_idx) in enumerate(skf.split(eval_df, y_train), 1):
            f_train = eval_df.iloc[train_idx]
            f_val = eval_df.iloc[val_idx]
            f_y_train = y_train[train_idx]
            f_y_val = y_train[val_idx]

            if m_type == 'dummy':
                pipe = build_dummy_baseline_pipeline(strategy='stratified', random_state=RANDOM_SEED + fold)
            else:
                pipe = build_pipeline_with_strategy(
                    model_type=m_type,
                    imbalance_strategy=strat,
                    calibrate=(strat == 'class_weight'),
                    random_state=RANDOM_SEED + fold
                )

            pipe.fit(f_train, f_y_train)
            val_probs = pipe.predict_proba(f_val)[:, 1]

            metrics = calculate_clinical_metrics(f_y_val, val_probs)
            roc_list.append(metrics['roc_auc'])
            pr_list.append(metrics['pr_auc'])
            brier_list.append(metrics['brier_score'])
            bal_acc_list.append(metrics['balanced_accuracy'])

        record = {
            'Model Configuration': label,
            'Model Family': m_type,
            'Imbalance Strategy': strat,
            'ROC-AUC (Mean)': float(np.mean(roc_list)),
            'ROC-AUC (Std)': float(np.std(roc_list)),
            'PR-AUC (Mean)': float(np.mean(pr_list)),
            'PR-AUC (Std)': float(np.std(pr_list)),
            'Brier Score (Mean)': float(np.mean(brier_list)),
            'Balanced Acc (Mean)': float(np.mean(bal_acc_list))
        }
        benchmark_records.append(record)
        cv_detailed[label] = record
        print(f"  -> ROC-AUC: {np.mean(roc_list):.4f} +/- {np.std(roc_list):.4f} | PR-AUC: {np.mean(pr_list):.4f} +/- {np.std(pr_list):.4f}")

    benchmark_df = pd.DataFrame(benchmark_records)
    return benchmark_df, cv_detailed


def train_and_evaluate_all_on_test(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    best_model_name: str = 'lightgbm',
    best_strategy: str = 'class_weight'
) -> Tuple[Any, Dict[str, Any], pd.DataFrame]:
    """
    Fits final pipelines on df_train, tunes decision threshold on df_val,
    and evaluates unbiased performance on untouched df_test.
    """
    print(f"\n========================================================")
    print(f"Training Final Models & Evaluating on Untouched Test Cohort")
    print(f"========================================================")

    y_train = df_train['target'].values
    y_val = df_val['target'].values
    y_test = df_test['target'].values

    models_to_test = {
        'dummy_prior': build_dummy_baseline_pipeline(strategy='prior'),
        'logistic_regression': build_pipeline_with_strategy('logistic_regression', 'class_weight', calibrate=True),
        'random_forest': build_pipeline_with_strategy('random_forest', 'class_weight', calibrate=True),
        'xgboost': build_pipeline_with_strategy('xgboost', 'class_weight', calibrate=True),
        'lightgbm': build_pipeline_with_strategy('lightgbm', 'class_weight', calibrate=True),
    }

    test_predictions = {}
    val_predictions = {}
    test_metrics_summary = []

    for name, pipe in models_to_test.items():
        print(f"Fitting final {name}...")
        pipe.fit(df_train, y_train)

        val_probs = pipe.predict_proba(df_val)[:, 1]
        test_probs = pipe.predict_proba(df_test)[:, 1]

        val_predictions[name] = val_probs
        test_predictions[name] = (y_test, test_probs)

        # Standard 0.5 threshold metrics
        m_std = calculate_clinical_metrics(y_test, test_probs, threshold=0.5)
        test_metrics_summary.append({
            'Model': name,
            'Threshold': 0.5,
            'ROC-AUC': m_std['roc_auc'],
            'PR-AUC': m_std['pr_auc'],
            'Sensitivity (Recall)': m_std['sensitivity_recall'],
            'Specificity': m_std['specificity'],
            'Precision': m_std['precision'],
            'F1-Score': m_std['f1_score'],
            'Brier Score': m_std['brier_score']
        })

    # Save comparative evaluation plots
    plot_roc_and_pr_curves(test_predictions, OUTPUT_DIR / "roc_and_pr_curves_comparison.png")
    plot_calibration_curves(test_predictions, OUTPUT_DIR / "calibration_curves_comparison.png")

    # Optimal threshold for chosen best model (LightGBM) using validation set
    best_pipe = models_to_test[best_model_name]
    best_val_probs = val_predictions[best_model_name]
    best_test_probs = test_predictions[best_model_name][1]

    opt_thr, opt_val_metrics = find_optimal_threshold(y_val, best_val_probs, method='youden')
    opt_test_metrics = calculate_clinical_metrics(y_test, best_test_probs, threshold=opt_thr)
    std_test_metrics = calculate_clinical_metrics(y_test, best_test_probs, threshold=0.5)

    print(f"\nOptimal Clinical Decision Threshold (Youden's J on Val): {opt_thr:.3f}")
    print(f"Test Recall at {opt_thr:.3f}: {opt_test_metrics['sensitivity_recall']*100:.2f}% (vs {std_test_metrics['sensitivity_recall']*100:.2f}% at 0.50)")
    print(f"Test Precision at {opt_thr:.3f}: {opt_test_metrics['precision']*100:.2f}%")
    print(f"Test F1 Score: {opt_test_metrics['f1_score']:.4f}")
    print(f"Test ROC-AUC: {opt_test_metrics['roc_auc']:.4f} | PR-AUC: {opt_test_metrics['pr_auc']:.4f}")

    # Plot optimal confusion matrix
    opt_test_preds = (best_test_probs >= opt_thr).astype(int)
    plot_confusion_matrix_annotated(
        y_test, opt_test_preds,
        threshold=opt_thr,
        model_name="LightGBM (Calibrated)",
        save_path=OUTPUT_DIR / "confusion_matrix_optimal_threshold.png"
    )

    metadata = {
        'model_name': best_model_name,
        'strategy': best_strategy,
        'calibrated': True,
        'optimal_threshold': float(opt_thr),
        'clinical_risk_tiers': {
            'low_risk_max': float(round(opt_thr * 0.75, 4)),
            'moderate_risk_max': float(round(opt_thr * 1.35, 4))
        },
        'test_metrics_optimal_threshold': opt_test_metrics,
        'test_metrics_standard_threshold': std_test_metrics,
        'dataset_summary': {
            'train_size': len(df_train),
            'val_size': len(df_val),
            'test_size': len(df_test),
            'positive_prevalence_pct': float(np.mean(y_train) * 100)
        }
    }

    save_full_pipeline(best_pipe, metadata, MODEL_PATH, METADATA_PATH)

    # Attach predictions to df_test for SQL analytics
    df_test_scored = df_test.copy()
    df_test_scored['readmission_prob'] = best_test_probs
    df_test_scored['predicted_flag'] = opt_test_preds
    df_test_scored['prediction_tier'] = pd.cut(
        df_test_scored['readmission_prob'],
        bins=[-0.01, metadata['clinical_risk_tiers']['low_risk_max'], metadata['clinical_risk_tiers']['moderate_risk_max'], 1.01],
        labels=['Low', 'Moderate', 'High']
    ).astype(str)

    df_test_scored.to_parquet(PROJECT_ROOT / "data" / "processed" / "df_test_predictions.parquet", index=False)

    return best_pipe, metadata, pd.DataFrame(test_metrics_summary)
