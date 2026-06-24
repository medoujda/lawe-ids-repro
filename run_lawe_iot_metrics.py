"""Measure IoT deployment metrics for LAWE-IDS.

Produces, for each dataset:
  - Per-sample inference latency (ms, mean/std over n_runs)
  - Batched throughput (samples/sec) over the full test set
  - Per-component on-disk model size (MB)
  - Total ensemble size (MB)
  - Meta-learner-only latency (ms) and size
  - Meta-learner FLOPs and parameter count

Output: results/lawe_iot_metrics.csv
"""
import argparse
import glob
import os
import warnings

import numpy as np
import pandas as pd
import time
import torch

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from config import DATASETS, RESULTS_DIR, SEED, DATASET_EXCLUDE_CATEGORIES
from preprocessing import load_dataset, create_splits
from lawe_ids import LAWEIDS
from evaluation import measure_iot_metrics, measure_model_size

SPLIT_CONFIG = {
    'unsw-nb15':  dict(split_type='random',   val_ratio=0.1,  test_ratio=0.2),
    'nsl-kdd':    dict(split_type='random',   val_ratio=0.1,  test_ratio=0.1),
    'cicids2017': dict(split_type='random',   val_ratio=0.16, test_ratio=0.2),
    'ton-iot':    dict(split_type='random',   val_ratio=0.1,  test_ratio=0.2),
    'cicids2023': dict(split_type='official', val_ratio=0.1,  test_ratio=0.2),
}


def count_flops_meta_learner(n_features, n_models=4):
    """Estimate FLOPs for the meta-learner + feature gating forward pass."""
    fg_flops = n_features * n_features + n_features * 3
    fc1_flops = (n_features + n_models) * 128 + 128
    fc2_flops = 128 * 64 + 64
    fc3_flops = 64 * n_models + n_models + n_models * 4
    combine_flops = n_models * 2
    return fg_flops + fc1_flops + fc2_flops + fc3_flops + combine_flops


def _dir_size_mb(path):
    """Return total size (MB) of all files under ``path``."""
    total = 0
    if os.path.isdir(path):
        for f in glob.glob(os.path.join(path, '**'), recursive=True):
            if os.path.isfile(f):
                total += os.path.getsize(f)
    elif os.path.isfile(path):
        total = os.path.getsize(path)
    return total / (1024 * 1024)


def _measure_component_sizes(model_dir):
    """Return a dict of per-component on-disk sizes (MB)."""
    sizes = {}
    for name in ('xgboost', 'lightgbm', 'catboost'):
        sizes[name] = measure_model_size(os.path.join(model_dir, f'{name}.joblib'))
    # Keras saves cnn_bilstm_attn.keras (could be a zip-dir on some versions)
    keras_path = os.path.join(model_dir, 'cnn_bilstm_attn.keras')
    sizes['cnn-bilstm-attn'] = _dir_size_mb(keras_path)
    sizes['meta_learner'] = measure_model_size(os.path.join(model_dir, 'meta_learner.pt'))
    sizes['pipeline_meta'] = measure_model_size(os.path.join(model_dir, 'pipeline_meta.joblib'))
    return sizes


def _measure_batched_throughput(predict_fn, X, n_runs=5):
    """Measure throughput (samples/sec) of ``predict_fn`` on the full X."""
    # Warm up
    predict_fn(X[: min(128, len(X))])
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        predict_fn(X)
        times.append(time.perf_counter() - t0)
    mean_t = float(np.mean(times))
    return len(X) / mean_t, mean_t


def main():
    parser = argparse.ArgumentParser(description='LAWE-IDS IoT Metrics')
    parser.add_argument('--dataset', choices=DATASETS + ['all'], default='all')
    parser.add_argument('--no-optuna', action='store_true',
                        help='Skip loading Optuna-tuned hyperparameters')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--n_latency_runs', type=int, default=100,
                        help='Per-sample latency repetitions (default: 100)')
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == 'all' else [args.dataset]
    use_optuna = not args.no_optuna
    all_metrics = []

    for ds_name in datasets:
        print(f"\n{'='*70}")
        print(f"  IoT METRICS: {ds_name}")
        print(f"{'='*70}")

        load_kwargs = {}
        excl = DATASET_EXCLUDE_CATEGORIES.get(ds_name)
        if excl and ds_name == 'unsw-nb15':
            load_kwargs['exclude_attack_categories'] = excl
        data = load_dataset(ds_name, **load_kwargs)
        splits = create_splits(data, random_state=args.seed, **SPLIT_CONFIG[ds_name])

        # Train LAWE-IDS
        lawe = LAWEIDS(use_optuna=use_optuna, dataset_name=ds_name,
                       random_state=args.seed)
        lawe.fit(splits['X_train'], splits['y_train'],
                 splits['X_val'], splits['y_val'],
                 n_features=splits['n_features'])

        X_test = splits['X_test']

        # Full pipeline per-sample latency
        print("\n  [Full Pipeline - per sample]")
        full_metrics = measure_iot_metrics(
            lawe.predict_proba, X_test, n_runs=args.n_latency_runs
        )
        print(f"    Inference: {full_metrics['inference_time_ms']:.3f} "
              f"± {full_metrics['inference_time_std_ms']:.3f} ms/sample")
        print(f"    RAM peak: {full_metrics['ram_peak_mb']:.2f} MB")

        # Full pipeline batched throughput
        print("\n  [Full Pipeline - batched throughput]")
        throughput, batch_time = _measure_batched_throughput(
            lawe.predict_proba, X_test, n_runs=3
        )
        print(f"    Throughput: {throughput:,.0f} samples/sec "
              f"(batch of {len(X_test):,} in {batch_time:.2f}s)")

        # Meta-learner-only latency
        print("\n  [Meta-Learner Only]")
        base_probas = lawe._get_base_probas(X_test)

        def meta_only_predict(X):
            feat_t = torch.tensor(X, dtype=torch.float32)
            prob_t = torch.tensor(base_probas[:len(X)], dtype=torch.float32)
            lawe.meta_learner_.eval()
            with torch.no_grad():
                p, _, _ = lawe.meta_learner_(feat_t, prob_t)
            return p.numpy()

        meta_metrics = measure_iot_metrics(
            meta_only_predict, X_test, n_runs=max(args.n_latency_runs * 10, 1000)
        )
        print(f"    Inference: {meta_metrics['inference_time_ms']:.3f} "
              f"± {meta_metrics['inference_time_std_ms']:.3f} ms/sample")

        # Save pipeline and measure per-component sizes
        model_dir = os.path.join(RESULTS_DIR, f'models_{ds_name}')
        lawe.save(model_dir)
        sizes = _measure_component_sizes(model_dir)
        total_size = sum(v for k, v in sizes.items() if k != 'pipeline_meta')
        print(f"\n  [Model Sizes]")
        for name, sz in sizes.items():
            print(f"    {name:<18}: {sz:>8.3f} MB")
        print(f"    {'TOTAL (ensemble)':<18}: {total_size:>8.3f} MB")

        # FLOPs / params (meta-learner)
        flops = count_flops_meta_learner(splits['n_features'])
        n_params_meta = sum(p.numel() for p in lawe.meta_learner_.parameters())
        print(f"\n  [Meta-Learner Complexity]")
        print(f"    FLOPs  : {flops}")
        print(f"    Params : {n_params_meta}")

        row = {
            'dataset':                 ds_name,
            'seed':                    args.seed,
            'n_features':              splits['n_features'],
            'n_test':                  len(X_test),
            'per_sample_ms':           round(full_metrics['inference_time_ms'], 4),
            'per_sample_std_ms':       round(full_metrics['inference_time_std_ms'], 4),
            'meta_per_sample_ms':      round(meta_metrics['inference_time_ms'], 4),
            'meta_per_sample_std_ms':  round(meta_metrics['inference_time_std_ms'], 4),
            'throughput_samples_sec':  round(throughput, 1),
            'full_ram_peak_mb':        round(full_metrics['ram_peak_mb'], 2),
            'size_xgboost_mb':         round(sizes['xgboost'], 4),
            'size_lightgbm_mb':        round(sizes['lightgbm'], 4),
            'size_catboost_mb':        round(sizes['catboost'], 4),
            'size_cnn_bilstm_mb':      round(sizes['cnn-bilstm-attn'], 4),
            'size_meta_mb':            round(sizes['meta_learner'], 4),
            'size_total_mb':           round(total_size, 4),
            'meta_flops':              flops,
            'meta_params':             n_params_meta,
            'use_optuna':              use_optuna,
        }
        all_metrics.append(row)

    # --- Save CSV (append, dedup on (dataset, seed)) ---
    df_new = pd.DataFrame(all_metrics)
    path = os.path.join(RESULTS_DIR, 'lawe_iot_metrics.csv')
    if os.path.exists(path):
        df_old = pd.read_csv(path)
        if 'seed' not in df_old.columns:
            df_old['seed'] = SEED
        new_keys = set(zip(df_new['dataset'], df_new['seed']))
        mask = df_old.apply(
            lambda r: (r['dataset'], r['seed']) not in new_keys, axis=1
        )
        df_old = df_old[mask]
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(path, index=False)
    print(f"\nIoT metrics saved to {path}")


if __name__ == '__main__':
    main()
