"""Analyze per-sample attention weights - interpretability figures for LAWE-IDS.

For each dataset:
  1. Train a LAWE-IDS pipeline (or reuse a saved one if available).
  2. Compute per-sample attention weights on the test set: (N, 4) softmax rows.
  3. Compute per-sample feature gates: (N, n_features) values in [0, 1].
  4. Split weight statistics by predicted class (benign vs attack).
  5. Produce:
       - Per-dataset boxplot of weights by base learner
       - Per-dataset boxplot of weights split by predicted class
       - Mean feature gate heatmap (top-k features per dataset)
       - CSV of summary statistics

Outputs
-------
  results/interpretability/weight_stats.csv       Summary stats
  results/interpretability/weights_box_{ds}.pdf   Weight boxplot per dataset
  results/interpretability/weights_by_class_{ds}.pdf
  results/interpretability/gates_topk_{ds}.pdf    Top-k mean feature gates
  results/interpretability/weights_grid.pdf       All datasets in one grid

Note: Per-attack-type analysis requires preserving ``attack_cat``/labels
through the loader, which is currently dropped in ``preprocessing.py``.
That is left as a follow-up; here we provide by-predicted-class splits.

Usage
-----
    python analyze_attention_weights.py --dataset all
    python analyze_attention_weights.py --dataset cicids2023 --no-optuna
"""
import argparse
import os
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from config import DATASETS, RESULTS_DIR, SEED, DATASET_EXCLUDE_CATEGORIES
from preprocessing import load_dataset, create_splits
from lawe_ids import LAWEIDS, ALL_MODEL_KEYS

SPLIT_CONFIG = {
    'unsw-nb15':  dict(split_type='random',   val_ratio=0.1,  test_ratio=0.2),
    'nsl-kdd':    dict(split_type='random',   val_ratio=0.1,  test_ratio=0.1),
    'cicids2017': dict(split_type='random',   val_ratio=0.16, test_ratio=0.2),
    'ton-iot':    dict(split_type='random',   val_ratio=0.1,  test_ratio=0.2),
    'cicids2023': dict(split_type='official', val_ratio=0.1,  test_ratio=0.2),
}

MODEL_LABELS = ['XGBoost', 'LightGBM', 'CatBoost', 'CNN-BiLSTM-Attn']

OUT_DIR = os.path.join(RESULTS_DIR, 'interpretability')


def _boxplot_weights(weights, dataset_name, out_path):
    """Boxplot of weight distributions, one box per base learner."""
    fig, ax = plt.subplots(figsize=(6, 3.8))
    bp = ax.boxplot(
        [weights[:, i] for i in range(weights.shape[1])],
        labels=MODEL_LABELS, showfliers=False, patch_artist=True,
    )
    for patch, color in zip(bp['boxes'], ['#4C72B0', '#55A868', '#C44E52', '#8172B3']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Per-sample attention weight')
    ax.set_title(f'{dataset_name.upper()} - weights per base learner')
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def _boxplot_by_class(weights, y_pred, dataset_name, out_path):
    """Side-by-side boxplot: weights for predicted-benign vs predicted-attack."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    positions_benign = np.arange(len(MODEL_LABELS)) * 3 - 0.5
    positions_attack = np.arange(len(MODEL_LABELS)) * 3 + 0.5

    benign_mask = (y_pred == 0)
    attack_mask = (y_pred == 1)

    if benign_mask.sum() > 0:
        bp_b = ax.boxplot(
            [weights[benign_mask, i] for i in range(weights.shape[1])],
            positions=positions_benign, widths=0.8, showfliers=False,
            patch_artist=True,
        )
        for p in bp_b['boxes']:
            p.set_facecolor('#4C72B0'); p.set_alpha(0.7)
    if attack_mask.sum() > 0:
        bp_a = ax.boxplot(
            [weights[attack_mask, i] for i in range(weights.shape[1])],
            positions=positions_attack, widths=0.8, showfliers=False,
            patch_artist=True,
        )
        for p in bp_a['boxes']:
            p.set_facecolor('#C44E52'); p.set_alpha(0.7)

    ax.set_xticks(np.arange(len(MODEL_LABELS)) * 3)
    ax.set_xticklabels(MODEL_LABELS)
    ax.set_ylabel('Per-sample attention weight')
    ax.set_title(
        f'{dataset_name.upper()} - weights by predicted class '
        f'(blue=benign n={int(benign_mask.sum())}, '
        f'red=attack n={int(attack_mask.sum())})'
    )
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def _topk_gates(gates, feature_names, dataset_name, out_path, k=15):
    """Bar chart of top-k mean feature gate values."""
    mean_gates = gates.mean(axis=0)  # (n_features,)
    k = min(k, len(mean_gates))
    top_idx = np.argsort(mean_gates)[::-1][:k]

    fig, ax = plt.subplots(figsize=(7, max(3.5, 0.25 * k + 1.5)))
    y_pos = np.arange(k)
    ax.barh(y_pos, mean_gates[top_idx][::-1], color='#4C72B0', alpha=0.8)
    labels = (
        [feature_names[i] for i in top_idx[::-1]]
        if feature_names is not None
        else [f'f{i}' for i in top_idx[::-1]]
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Mean feature gate value')
    ax.set_title(f'{dataset_name.upper()} - top-{k} features by mean gate')
    ax.set_xlim(0.0, 1.0)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def _grid_plot(all_weights, dataset_names, out_path):
    """Single figure with one boxplot subplot per dataset."""
    n = len(dataset_names)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.5 * rows),
                             squeeze=False)
    for idx, (ds, w) in enumerate(zip(dataset_names, all_weights)):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        bp = ax.boxplot(
            [w[:, i] for i in range(w.shape[1])],
            labels=MODEL_LABELS, showfliers=False, patch_artist=True,
        )
        for patch, color in zip(bp['boxes'],
                                 ['#4C72B0', '#55A868', '#C44E52', '#8172B3']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(ds.upper(), fontsize=10)
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', labelsize=7, rotation=20)
    # hide unused axes
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].axis('off')
    fig.suptitle('LAWE-IDS per-sample attention weights (test set)',
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def _summary_stats(weights, gates, dataset_name, seed, n_test):
    """Return a dict of summary statistics for the CSV."""
    row = {'dataset': dataset_name, 'seed': seed, 'n_test': n_test}
    for i, lbl in enumerate(ALL_MODEL_KEYS):
        w = weights[:, i]
        row[f'w_{lbl}_mean'] = round(float(w.mean()), 4)
        row[f'w_{lbl}_std']  = round(float(w.std()),  4)
        row[f'w_{lbl}_min']  = round(float(w.min()),  4)
        row[f'w_{lbl}_max']  = round(float(w.max()),  4)
    # gate sparsity: fraction of features whose mean gate < 0.5
    mean_gates = gates.mean(axis=0)
    row['gate_mean']            = round(float(mean_gates.mean()), 4)
    row['gate_frac_below_0_5']  = round(float((mean_gates < 0.5).mean()), 4)
    row['gate_frac_above_0_8']  = round(float((mean_gates > 0.8).mean()), 4)
    return row


def analyze_dataset(dataset_name, use_optuna, seed):
    """Train LAWE-IDS on one dataset and return (weights, gates, y_pred, feat_names)."""
    print(f"\n{'='*70}\n  INTERPRETABILITY: {dataset_name}\n{'='*70}")

    load_kwargs = {}
    excl = DATASET_EXCLUDE_CATEGORIES.get(dataset_name)
    if excl and dataset_name == 'unsw-nb15':
        load_kwargs['exclude_attack_categories'] = excl
    data = load_dataset(dataset_name, **load_kwargs)
    splits = create_splits(data, random_state=seed, **SPLIT_CONFIG[dataset_name])

    lawe = LAWEIDS(dataset_name=dataset_name, use_optuna=use_optuna,
                   random_state=seed)
    lawe.fit(splits['X_train'], splits['y_train'],
             splits['X_val'], splits['y_val'],
             n_features=splits['n_features'])

    X_test = splits['X_test']
    weights = lawe.get_sample_weights(X_test)     # (N, 4)
    gates   = lawe.get_feature_gates(X_test)      # (N, F)
    y_proba = lawe.predict_proba(X_test)
    y_pred  = (y_proba >= lawe.threshold_).astype(int)

    feat_names = data.get('feature_names')
    return weights, gates, y_pred, feat_names, len(X_test)


def main():
    parser = argparse.ArgumentParser(description='LAWE-IDS Interpretability')
    parser.add_argument('--dataset', choices=DATASETS + ['all'], default='all')
    parser.add_argument('--no-optuna', action='store_true')
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    datasets = DATASETS if args.dataset == 'all' else [args.dataset]
    use_optuna = not args.no_optuna

    stats_rows = []
    all_weights = []
    processed_names = []

    for ds in datasets:
        try:
            weights, gates, y_pred, feat_names, n_test = analyze_dataset(
                ds, use_optuna, args.seed
            )
        except Exception as exc:
            print(f"  [FAILED] {ds}: {exc}")
            continue

        _boxplot_weights(weights, ds,
                         os.path.join(OUT_DIR, f'weights_box_{ds}.pdf'))
        _boxplot_by_class(weights, y_pred, ds,
                          os.path.join(OUT_DIR, f'weights_by_class_{ds}.pdf'))
        _topk_gates(gates, feat_names, ds,
                    os.path.join(OUT_DIR, f'gates_topk_{ds}.pdf'))
        stats_rows.append(_summary_stats(weights, gates, ds, args.seed, n_test))
        all_weights.append(weights)
        processed_names.append(ds)

    if not stats_rows:
        print('\nNo interpretability results to save.')
        return

    # Grid figure
    _grid_plot(all_weights, processed_names,
               os.path.join(OUT_DIR, 'weights_grid.pdf'))

    # Stats CSV (append, dedup on (dataset, seed))
    df_new = pd.DataFrame(stats_rows)
    path = os.path.join(OUT_DIR, 'weight_stats.csv')
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
    print(f'\nInterpretability stats saved to {path}')
    print(f'Figures saved to {OUT_DIR}/')


if __name__ == '__main__':
    main()
