"""
Visualization functions for model evaluation, discrimination curves, and calibration.
"""
from pathlib import Path
from typing import Dict, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from sklearn.calibration import calibration_curve

sns.set_theme(style="whitegrid", font="DejaVu Sans")


def plot_roc_and_pr_curves(
    results_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path: Path
):
    """Generates dual-panel ROC and Precision-Recall comparison plots."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = {
        'dummy_prior': '#7f7f7f',
        'logistic_regression': '#1f77b4',
        'random_forest': '#2ca02c',
        'xgboost': '#ff7f0e',
        'lightgbm': '#d62728'
    }

    # Left: ROC Curve
    for name, (y_true, y_prob) in results_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        color = colors.get(name.lower(), None)
        axes[0].plot(fpr, tpr, label=name.replace('_', ' ').title(), lw=2.2, color=color)

    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random Chance (AUC=0.50)')
    axes[0].set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[0].set_ylabel("True Positive Rate (Sensitivity / Recall)")
    axes[0].legend(loc="lower right")

    # Right: PR Curve
    for name, (y_true, y_prob) in results_dict.items():
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        color = colors.get(name.lower(), None)
        axes[1].plot(rec, prec, label=name.replace('_', ' ').title(), lw=2.2, color=color)

    base_prev = np.mean(list(results_dict.values())[0][0])
    axes[1].axhline(y=base_prev, color='k', linestyle='--', alpha=0.6, label=f'Baseline Prevalence ({base_prev*100:.1f}%)')
    axes[1].set_title("Precision-Recall (PR) Curve", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Recall (Sensitivity)")
    axes[1].set_ylabel("Precision (Positive Predictive Value)")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_calibration_curves(
    results_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path: Path
):
    """Plots probability calibration reliability diagram."""
    plt.figure(figsize=(8, 7))

    for name, (y_true, y_prob) in results_dict.items():
        if name.startswith('dummy'):
            continue
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')
        plt.plot(prob_pred, prob_true, marker='o', lw=2, label=name.replace('_', ' ').title())

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfect Calibration')
    plt.title("Reliability Diagram (Probability Calibration)", fontsize=13, fontweight='bold')
    plt.xlabel("Mean Predicted Readmission Probability")
    plt.ylabel("Observed Fraction of Positives (<30 Days)")
    plt.legend(loc="lower right")
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_confusion_matrix_annotated(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
    model_name: str,
    save_path: Path
):
    """Plots confusion matrix with detailed clinical annotations."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    total = len(y_true)

    labels = [
        [f"True Negative\n{tn:,}\n({tn/total*100:.1f}%)", f"False Positive\n{fp:,}\n({fp/total*100:.1f}%)"],
        [f"False Negative\n{fn:,}\n({fn/total*100:.1f}%)", f"True Positive\n{tp:,}\n({tp/total*100:.1f}%)"]
    ]

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", cbar=False,
                xticklabels=['Predicted No Readmission', 'Predicted Readmission (<30)'],
                yticklabels=['Actual No Readmission', 'Actual Readmitted (<30)'])
    plt.title(f"{model_name} Confusion Matrix (Threshold = {threshold:.3f})", fontsize=12, fontweight='bold')
    plt.ylabel("Actual Clinical Outcome")
    plt.xlabel("Model Decision")
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
