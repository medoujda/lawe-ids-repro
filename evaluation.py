"""Evaluation metrics and IoT measurement utilities."""
import numpy as np
import time
import tracemalloc
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

from config import RESULTS_DIR


def compute_metrics(y_true, y_pred, y_proba=None):
    """Compute classification metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics['auc_roc'] = 0.0
    else:
        metrics['auc_roc'] = 0.0

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics.update({'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)})
        metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0
    return metrics


def measure_iot_metrics(predict_fn, X_sample, n_runs=1000):
    """Measure IoT deployment metrics."""
    predict_fn(X_sample[:1])  # warm up

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        predict_fn(X_sample[:1])
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    tracemalloc.start()
    predict_fn(X_sample)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        'inference_time_ms': np.mean(times),
        'inference_time_std_ms': np.std(times),
        'ram_peak_mb': peak / (1024 * 1024),
    }


def measure_model_size(model_path):
    """Measure saved model size in MB."""
    if os.path.exists(model_path):
        return os.path.getsize(model_path) / (1024 * 1024)
    return 0.0


def print_metrics(metrics, label=''):
    """Pretty-print metrics."""
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}Acc={metrics['accuracy']:.4f}  "
          f"Prec={metrics['precision']:.4f}  "
          f"Rec={metrics['recall']:.4f}  "
          f"F1={metrics['f1']:.4f}  "
          f"AUC={metrics['auc_roc']:.4f}")
