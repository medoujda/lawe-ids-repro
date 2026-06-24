"""run_lawe.py — Main execution script for LAWE-IDS.

Usage
-----
    python run_lawe.py --dataset all
    python run_lawe.py --dataset unsw-nb15
    python run_lawe.py --dataset nsl-kdd --no-optuna
    python run_lawe.py --dataset cicids2017
    python run_lawe.py --dataset all --no-optuna --n_seeds 3
"""

import argparse
import os
import sys
import time
import logging
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from config import DATASETS, RESULTS_DIR, SEED, DATASET_EXCLUDE_CATEGORIES
from preprocessing import load_dataset, create_splits
from lawe_ids import LAWEIDS
from evaluation import compute_metrics, print_metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# Split configuration per dataset
SPLIT_CONFIG = {
    'unsw-nb15':  dict(split_type='random',   val_ratio=0.1,  test_ratio=0.2),
    'nsl-kdd':    dict(split_type='random',   val_ratio=0.1,  test_ratio=0.1),
    'cicids2017': dict(split_type='random',   val_ratio=0.16, test_ratio=0.2),
    'ton-iot':    dict(split_type='random',   val_ratio=0.1,  test_ratio=0.2),
    'bot-iot':    dict(split_type='random',   val_ratio=0.1,  test_ratio=0.2),
    'cicids2023': dict(split_type='official', val_ratio=0.1,  test_ratio=0.2),
}


def run_dataset(dataset_name: str, use_optuna: bool, seed: int = SEED) -> dict:
    """Train and evaluate LAWE-IDS on a single dataset.

    Parameters
    ----------
    dataset_name : str
        One of 'unsw-nb15', 'nsl-kdd', 'cicids2017', 'ton-iot', 'cicids2023'.
    use_optuna : bool
        Whether to load Optuna-tuned hyperparameters from disk.
    seed : int
        Random seed controlling splits and model initialization.

    Returns
    -------
    dict with metrics and metadata.
    """
    logger.info("=" * 60)
    logger.info("Dataset: %s  (seed=%d)", dataset_name.upper(), seed)
    logger.info("=" * 60)

    # --- Load data ---
    logger.info("Loading %s …", dataset_name)
    t0 = time.perf_counter()
    load_kwargs = {}
    excl = DATASET_EXCLUDE_CATEGORIES.get(dataset_name)
    if excl and dataset_name == 'unsw-nb15':
        load_kwargs['exclude_attack_categories'] = excl
        logger.info("  Excluding attack categories: %s", excl)
    data = load_dataset(dataset_name, **load_kwargs)
    logger.info(
        "  Loaded %d samples, %d features  (%.1fs)",
        data['X'].shape[0], data['n_features'],
        time.perf_counter() - t0,
    )

    # --- Create splits ---
    logger.info("Creating splits …")
    cfg = SPLIT_CONFIG[dataset_name]
    splits = create_splits(data, random_state=seed, **cfg)
    X_train, y_train = splits['X_train'], splits['y_train']
    X_val,   y_val   = splits['X_val'],   splits['y_val']
    X_test,  y_test  = splits['X_test'],  splits['y_test']

    logger.info(
        "  train=%d  val=%d  test=%d",
        len(y_train), len(y_val), len(y_test),
    )

    # --- Train LAWE-IDS ---
    logger.info("Training LAWE-IDS (use_optuna=%s, seed=%d) …", use_optuna, seed)
    t_fit = time.perf_counter()
    lawe = LAWEIDS(dataset_name=dataset_name, use_optuna=use_optuna, random_state=seed)
    lawe.fit(X_train, y_train, X_val, y_val, n_features=data['n_features'])
    fit_time = time.perf_counter() - t_fit
    logger.info("  Training done in %.1fs", fit_time)
    logger.info("  Decision threshold: %.3f", lawe.threshold_)

    # --- Evaluate on test set ---
    logger.info("Evaluating on test set …")
    t_eval = time.perf_counter()
    y_proba = lawe.predict_proba(X_test)
    y_pred  = (y_proba >= lawe.threshold_).astype(int)
    eval_time = time.perf_counter() - t_eval

    metrics = compute_metrics(y_test, y_pred, y_proba)
    print_metrics(metrics, label=dataset_name)

    # --- Per-sample weight statistics ---
    logger.info("Computing per-sample weight statistics …")
    weights = lawe.get_sample_weights(X_test)   # (N_test, 4)
    model_keys = ['xgboost', 'lightgbm', 'catboost', 'cnn-bilstm-attn']
    print(f"\n  Per-sample weight statistics on test set ({len(y_test)} samples):")
    print(f"  {'Model':<20} {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
    print(f"  {'-'*56}")
    for i, mname in enumerate(model_keys):
        w_col = weights[:, i]
        print(
            f"  {mname:<20} {w_col.mean():>8.4f}  {w_col.std():>8.4f}"
            f"  {w_col.min():>8.4f}  {w_col.max():>8.4f}"
        )
    print()

    result = {
        'dataset':          dataset_name,
        'seed':             seed,
        'n_train':          len(y_train),
        'n_val':            len(y_val),
        'n_test':           len(y_test),
        'n_features':       data['n_features'],
        'threshold':        round(float(lawe.threshold_), 4),
        'accuracy':         round(metrics['accuracy'],  4),
        'precision':        round(metrics['precision'], 4),
        'recall':           round(metrics['recall'],    4),
        'f1':               round(metrics['f1'],        4),
        'auc_roc':          round(metrics['auc_roc'],   4),
        'fpr':              round(metrics.get('fpr', 0.0), 4),
        'fnr':              round(metrics.get('fnr', 0.0), 4),
        'tp':               metrics.get('tp', ''),
        'tn':               metrics.get('tn', ''),
        'fp':               metrics.get('fp', ''),
        'fn':               metrics.get('fn', ''),
        'fit_time_s':       round(fit_time,  1),
        'eval_time_s':      round(eval_time, 4),
        'use_optuna':       use_optuna,
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Run LAWE-IDS on one or all datasets.'
    )
    parser.add_argument(
        '--dataset',
        choices=['all'] + DATASETS,
        default='all',
        help='Dataset to run (default: all)',
    )
    parser.add_argument(
        '--no-optuna',
        action='store_true',
        help='Skip loading Optuna-tuned hyperparameters',
    )
    parser.add_argument(
        '--n_seeds',
        type=int,
        default=1,
        help='Number of random seeds to run (default: 1). '
             'Seeds are [base_seed, base_seed+1, …, base_seed+n_seeds-1].',
    )
    parser.add_argument(
        '--base_seed',
        type=int,
        default=SEED,
        help=f'Base random seed (default: {SEED}).',
    )
    args = parser.parse_args()

    use_optuna = not args.no_optuna
    datasets = DATASETS if args.dataset == 'all' else [args.dataset]
    seeds = [args.base_seed + i for i in range(args.n_seeds)]

    all_results = []
    for seed in seeds:
        for ds in datasets:
            try:
                result = run_dataset(ds, use_optuna=use_optuna, seed=seed)
                all_results.append(result)
            except Exception as exc:
                logger.error("Failed on %s (seed=%d): %s", ds, seed, exc,
                             exc_info=True)

    if not all_results:
        logger.error("No results to save.")
        sys.exit(1)

    # --- Save summary CSV (append, de-duplicate on (dataset, seed)) ---
    out_path = os.path.join(RESULTS_DIR, 'lawe_ids_results.csv')
    df_new = pd.DataFrame(all_results)
    if os.path.exists(out_path):
        df_old = pd.read_csv(out_path)
        # Ensure seed column exists in legacy CSVs
        if 'seed' not in df_old.columns:
            df_old['seed'] = SEED
        # Remove old rows for (dataset, seed) pairs we just ran
        new_keys = set(zip(df_new['dataset'], df_new['seed']))
        mask = df_old.apply(
            lambda r: (r['dataset'], r['seed']) not in new_keys, axis=1
        )
        df_old = df_old[mask]
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(out_path, index=False)
    logger.info("Results saved to %s", out_path)

    # --- Print summary table (mean across seeds if n_seeds>1) ---
    print("\n" + "=" * 70)
    print("LAWE-IDS SUMMARY")
    print("=" * 70)
    if args.n_seeds > 1:
        agg = df_new.groupby('dataset').agg(
            acc_mean=('accuracy', 'mean'),
            acc_std=('accuracy', 'std'),
            f1_mean=('f1', 'mean'),
            f1_std=('f1', 'std'),
            auc_mean=('auc_roc', 'mean'),
        ).round(4)
        print(agg.to_string())
    else:
        print(df_new[['dataset', 'seed', 'accuracy', 'precision', 'recall', 'f1',
                  'auc_roc', 'threshold']].to_string(index=False))
    print("=" * 70)


if __name__ == '__main__':
    main()
