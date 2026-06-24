"""Tests for evaluation module."""
import numpy as np
import pytest
from evaluation import compute_metrics, measure_iot_metrics


class TestComputeMetrics:
    def test_perfect_predictions(self):
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 1])
        m = compute_metrics(y_true, y_pred)
        assert m['accuracy'] == 1.0
        assert m['precision'] == 1.0
        assert m['recall'] == 1.0
        assert m['f1'] == 1.0

    def test_all_wrong(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        m = compute_metrics(y_true, y_pred)
        assert m['accuracy'] == 0.0

    def test_with_proba(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        y_proba = np.array([0.1, 0.9, 0.2, 0.8])
        m = compute_metrics(y_true, y_pred, y_proba)
        assert 0 < m['auc_roc'] <= 1.0


class TestIoTMetrics:
    def test_measure_returns_expected_keys(self):
        def dummy_predict(X):
            return np.ones(len(X))

        X = np.random.rand(100, 10).astype(np.float32)
        metrics = measure_iot_metrics(dummy_predict, X, n_runs=10)
        assert 'inference_time_ms' in metrics
        assert 'ram_peak_mb' in metrics
        assert metrics['inference_time_ms'] > 0
