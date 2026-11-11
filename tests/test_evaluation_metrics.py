"""
Unit tests for evaluation metrics and clinical threshold optimization.
"""
import pytest
import numpy as np
from src.evaluation.metrics import calculate_clinical_metrics, find_optimal_threshold


def test_calculate_clinical_metrics_perfect():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = calculate_clinical_metrics(y_true, y_prob, threshold=0.5)

    assert metrics['accuracy'] == 1.0
    assert metrics['sensitivity_recall'] == 1.0
    assert metrics['specificity'] == 1.0
    assert metrics['precision'] == 1.0
    assert metrics['f1_score'] == 1.0
    assert metrics['true_positives'] == 2
    assert metrics['true_negatives'] == 2
    assert metrics['false_positives'] == 0
    assert metrics['false_negatives'] == 0
    assert metrics['roc_auc'] == 1.0


def test_calculate_clinical_metrics_thresholding():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.3, 0.4, 0.7])
    # At threshold 0.5, y_pred = [0, 0, 0, 1] -> 1 TP, 1 FN
    m_50 = calculate_clinical_metrics(y_true, y_prob, threshold=0.5)
    assert m_50['sensitivity_recall'] == 0.5
    assert m_50['specificity'] == 1.0

    # At threshold 0.35, y_pred = [0, 0, 1, 1] -> 2 TP, 0 FN
    m_35 = calculate_clinical_metrics(y_true, y_prob, threshold=0.35)
    assert m_35['sensitivity_recall'] == 1.0
    assert m_35['specificity'] == 1.0


def test_find_optimal_threshold_youden():
    y_true = np.array([0]*90 + [1]*10)
    # Predicted probabilities where positives have higher mean
    y_prob = np.array([0.05]*90 + [0.25]*10)
    
    best_thr, best_metrics = find_optimal_threshold(y_true, y_prob, method='youden')
    assert 0.05 < best_thr <= 0.25
    assert best_metrics['sensitivity_recall'] == 1.0
    assert best_metrics['specificity'] == 1.0
