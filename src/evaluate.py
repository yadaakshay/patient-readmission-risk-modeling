"""
Model evaluation, calibration assessment, and clinical threshold optimization utilities.
"""
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)
from sklearn.calibration import calibration_curve

from src.config import OUTPUT_DIR


def evaluate_binary_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Computes a comprehensive suite of discrimination, calibration, and classification metrics.
    """
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true)
    balanced_acc = (sensitivity + specificity) / 2.0

    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    loss = log_loss(y_true, y_prob)

    return {
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'brier_score': float(brier),
        'log_loss': float(loss),
        'threshold': float(threshold),
        'accuracy': float(accuracy),
        'balanced_accuracy': float(balanced_acc),
        'sensitivity_recall': float(sensitivity),
        'specificity': float(specificity),
        'precision': float(precision),
        'f1_score': float(f1),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn)
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    method: str = 'youden',
    target_recall: float = 0.50
) -> Tuple[float, Dict[str, Any]]:
    """
    Finds the optimal decision threshold for clinical risk classification:
    - 'youden': Maximizes Sensitivity + Specificity - 1 (Youden's J statistic).
    - 'f1': Maximizes the F1 harmonic mean.
    - 'target_recall': Finds the threshold that captures at least `target_recall` of positive cases.
    """
    thresholds = np.linspace(0.01, 0.99, 200)
    best_thr = 0.5
    best_score = -1.0

    if method == 'youden':
        fpr, tpr, roc_thrs = roc_curve(y_true, y_prob)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_thr = float(roc_thrs[best_idx])
    elif method == 'f1':
        for thr in thresholds:
            y_pred = (y_prob >= thr).astype(int)
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            if f1 > best_score:
                best_score = f1
                best_thr = float(thr)
    elif method == 'target_recall':
        for thr in thresholds[::-1]:
            y_pred = (y_prob >= thr).astype(int)
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if rec >= target_recall:
                best_thr = float(thr)
                break

    metrics = evaluate_binary_predictions(y_true, y_prob, threshold=best_thr)
    return best_thr, metrics


def plot_roc_and_pr_curves(
    results_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path=None
):
    """
    Plots comparative ROC and Precision-Recall curves across models.
    results_dict format: { 'Model Name': (y_true, y_prob) }
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. ROC Curve
    for name, (y_true, y_prob) in results_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = roc_auc_score(y_true, y_prob)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", lw=2)

    axes[0].plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.7)
    axes[0].set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[0].set_ylabel("True Positive Rate (Sensitivity / Recall)")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    # 2. Precision-Recall Curve
    for name, (y_true, y_prob) in results_dict.items():
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        axes[1].plot(rec, prec, label=f"{name} (PR-AUC = {pr_auc:.3f})", lw=2)

    base_rate = np.mean(list(results_dict.values())[0][0])
    axes[1].plot([0, 1], [base_rate, base_rate], 'k--', lw=1, label=f"No Skill (Prevalence = {base_rate:.2f})")
    axes[1].set_title("Precision-Recall (PR) Curve", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Recall (Sensitivity)")
    axes[1].set_ylabel("Precision (Positive Predictive Value)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_calibration_curves(
    results_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path=None
):
    """Plots probability calibration curves (reliability diagrams)."""
    plt.figure(figsize=(8, 6))
    for name, (y_true, y_prob) in results_dict.items():
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')
        brier = brier_score_loss(y_true, y_prob)
        plt.plot(prob_pred, prob_true, marker='o', label=f"{name} (Brier = {brier:.4f})", lw=2)

    plt.plot([0, 1], [0, 1], 'k--', label="Perfectly Calibrated", lw=1)
    plt.title("Probability Calibration Curves (Reliability Diagram)", fontsize=12, fontweight='bold')
    plt.xlabel("Mean Predicted Risk Probability")
    plt.ylabel("Observed Readmission Proportion")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_confusion_matrix_annotated(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
    model_name: str,
    save_path=None
):
    """Plots annotated confusion matrix at the optimal clinical threshold."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Not Readmitted (<30)', 'Readmitted (<30)'],
        yticklabels=['Not Readmitted (<30)', 'Readmitted (<30)']
    )
    plt.title(f"{model_name} Confusion Matrix\n(Decision Threshold = {threshold:.3f})", fontsize=11, fontweight='bold')
    plt.ylabel("Actual Outcome")
    plt.xlabel("Predicted Outcome")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()
