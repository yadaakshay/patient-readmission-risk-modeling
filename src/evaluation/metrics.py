"""
Comprehensive clinical model evaluation metrics and threshold optimization module.
"""
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score
)
from sklearn.calibration import calibration_curve


def calculate_clinical_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Computes a comprehensive dictionary of discrimination, calibration, and classification metrics
    at a specific decision threshold.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true)
    balanced_acc = (sensitivity + specificity) / 2.0

    roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    pr_auc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float(np.mean(y_true))
    brier = float(brier_score_loss(y_true, y_prob))

    # Clamp probs to avoid log_loss overflow
    clipped_probs = np.clip(y_prob, 1e-7, 1 - 1e-7)
    loss = float(log_loss(y_true, clipped_probs))

    return {
        'threshold': float(threshold),
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'brier_score': brier,
        'log_loss': loss,
        'accuracy': float(accuracy),
        'balanced_accuracy': float(balanced_acc),
        'sensitivity_recall': float(sensitivity),
        'specificity': float(specificity),
        'precision': float(precision),
        'f1_score': float(f1),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'total_eval_samples': len(y_true),
        'positive_prevalence': float(np.mean(y_true))
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    method: str = 'youden',
    beta: float = 2.0
) -> Tuple[float, Dict[str, Any]]:
    """
    Finds the optimal decision threshold on validation data.
    
    Methods:
    - 'youden': Maximizes Youden's J statistic = Sensitivity + Specificity - 1
    - 'f_beta': Maximizes F-beta score (beta > 1 prioritizes recall/sensitivity over precision)
    - 'pr_balance': Balances precision and recall (closest to intersection)
    """
    thresholds = np.linspace(0.01, 0.95, 200)
    best_thr = 0.5
    best_score = -float('inf')

    for thr in thresholds:
        metrics = calculate_clinical_metrics(y_true, y_prob, threshold=thr)
        
        if method == 'youden':
            # Youden's J statistic
            score = metrics['sensitivity_recall'] + metrics['specificity'] - 1.0
        elif method == 'f_beta':
            # F-beta favoring recall (default beta=2.0)
            p = metrics['precision']
            r = metrics['sensitivity_recall']
            score = (1 + beta**2) * (p * r) / ((beta**2 * p) + r) if ((beta**2 * p) + r) > 0 else 0.0
        elif method == 'pr_balance':
            # Minimum absolute gap between precision and recall
            score = -abs(metrics['precision'] - metrics['sensitivity_recall'])
        else:
            raise ValueError(f"Unknown threshold method: {method}")

        if score > best_score:
            best_score = score
            best_thr = float(thr)

    best_metrics = calculate_clinical_metrics(y_true, y_prob, threshold=best_thr)
    return best_thr, best_metrics
