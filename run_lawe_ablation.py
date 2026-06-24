"""Ablation study for LAWE-IDS — 5 conditions per dataset.

Optimised: base learners are trained ONCE per dataset and reused across
conditions that share the same model set. Only the meta-learner / weighting
strategy changes between conditions.

Training groups per dataset:
  - Standard 4 models (XGB, LGBM, CatBoost, CNN-BiLSTM-Attn):
      → reused by Full, No Feature Gating, Fixed Weights
  - Vanilla CNN-BiLSTM (no self-attention) + 3 GBTs:
      → used by No Self-Attention
  - 3 models only (XGB, LGBM, CNN-BiLSTM-Attn, no CatBoost):
      → used by No CatBoost
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from config import DATASETS, RESULTS_DIR, SEED, THRESHOLD_MIN, THRESHOLD_MAX, DATASET_EXCLUDE_CATEGORIES
from preprocessing import load_dataset, create_splits, apply_smote
from models import create_base_learner, CNNBiLSTMAttention
from lawe_ids import LAWEIDS, BASE_LEARNER_NAMES
from attention_meta_learner import train_meta_learner, find_best_threshold
from evaluation import compute_metrics, print_metrics

SPLIT_CONFIG = {
    'unsw-nb15':  dict(split_type='random',   val_ratio=0.1,  test_ratio=0.2),
    'nsl-kdd':    dict(split_type='random',   val_ratio=0.1,  test_ratio=0.1),
    'cicids2017': dict(split_type='random',   val_ratio=0.16, test_ratio=0.2),
    'ton-iot':    dict(split_type='random',   val_ratio=0.1,  test_ratio=0.2),
    'cicids2023': dict(split_type='official', val_ratio=0.1,  test_ratio=0.2),
}


# ------------------------------------------------------------------
# Shared training helpers
# ------------------------------------------------------------------

def _train_standard_models(X_train_sm, y_train_sm, n_features):
    """Train the standard 4 base learners (3 GBTs + CNN-BiLSTM-Attn)."""
    models = {}
    for name in BASE_LEARNER_NAMES:
        print(f"      Training {name} …")
        model = create_base_learner(name)
        model.fit(X_train_sm, y_train_sm)
        models[name] = model

    print("      Training CNN-BiLSTM-Attn …")
    dl_model = CNNBiLSTMAttention(n_features=n_features)
    dl_model.fit(X_train_sm, y_train_sm)
    models['cnn-bilstm-attn'] = dl_model
    return models


def _get_probas(models, X, model_names):
    """Get class-1 probabilities from a dict of models."""
    return np.column_stack([
        models[n].predict_proba(X)[:, 1] for n in model_names
    ]).astype(np.float32)


# ------------------------------------------------------------------
# Ablation conditions (now receive pre-trained models + probas)
# ------------------------------------------------------------------

def run_full(splits, val_probas, test_probas, models, n_features, **kw):
    """Full LAWE-IDS (train meta-learner + threshold)."""
    print("\n  [Ablation] Full LAWE-IDS")
    ml, _ = train_meta_learner(
        splits['X_val'], val_probas, splits['y_val'],
        n_features=n_features, n_models=val_probas.shape[1],
    )
    tau, _ = find_best_threshold(ml, splits['X_val'], val_probas, splits['y_val'])

    feat_t = torch.tensor(splits['X_test'], dtype=torch.float32)
    prob_t = torch.tensor(test_probas, dtype=torch.float32)
    ml.eval()
    with torch.no_grad():
        p_final, _, _ = ml(feat_t, prob_t)
    p_final = p_final.numpy()

    y_pred = (p_final > tau).astype(int)
    return compute_metrics(splits['y_test'], y_pred, p_final)


def run_no_feature_gating(splits, val_probas, test_probas, models, n_features, **kw):
    """No Feature Gating — freeze gates to ~1.0."""
    print("\n  [Ablation] No Feature Gating")
    ml, _ = train_meta_learner(
        splits['X_val'], val_probas, splits['y_val'],
        n_features=n_features, n_models=val_probas.shape[1],
    )
    with torch.no_grad():
        ml.feature_gating.gate_layer.weight.fill_(0)
        ml.feature_gating.gate_layer.bias.fill_(10)  # sigmoid(10) ≈ 1.0

    tau, _ = find_best_threshold(ml, splits['X_val'], val_probas, splits['y_val'])

    feat_t = torch.tensor(splits['X_test'], dtype=torch.float32)
    prob_t = torch.tensor(test_probas, dtype=torch.float32)
    ml.eval()
    with torch.no_grad():
        p_final, _, _ = ml(feat_t, prob_t)
    p_final = p_final.numpy()

    y_pred = (p_final > tau).astype(int)
    return compute_metrics(splits['y_test'], y_pred, p_final)


def run_fixed_weights(splits, val_probas, test_probas, models, n_features, **kw):
    """Fixed weights via grid search — no attention meta-learner."""
    print("\n  [Ablation] Fixed Weights (no attention)")
    n_models = val_probas.shape[1]

    best_acc, best_w, best_tau = 0, None, 0.5
    for w0 in np.arange(0, 1.05, 0.1):
        for w1 in np.arange(0, 1.05 - w0, 0.1):
            for w2 in np.arange(0, 1.05 - w0 - w1, 0.1):
                w3 = round(1.0 - w0 - w1 - w2, 2)
                if w3 < -0.01:
                    continue
                w3 = max(0, w3)
                p = w0*val_probas[:,0] + w1*val_probas[:,1] + w2*val_probas[:,2] + w3*val_probas[:,3]
                for tau in np.arange(0.3, 0.71, 0.02):
                    acc = ((p > tau).astype(int) == splits['y_val']).mean()
                    if acc > best_acc:
                        best_acc, best_w, best_tau = acc, [w0,w1,w2,w3], tau

    print(f"    Best weights: {[f'{w:.2f}' for w in best_w]}, tau={best_tau:.2f}, val_acc={best_acc:.4f}")
    p_test = sum(best_w[i] * test_probas[:, i] for i in range(4))
    y_pred = (p_test > best_tau).astype(int)
    return compute_metrics(splits['y_test'], y_pred, p_test)


def run_no_self_attention(splits, val_probas_vanilla, test_probas_vanilla,
                          models, n_features, **kw):
    """Vanilla CNN-BiLSTM (no self-attention) — uses pre-trained vanilla models."""
    print("\n  [Ablation] No Self-Attention (vanilla CNN-BiLSTM)")
    ml, _ = train_meta_learner(
        splits['X_val'], val_probas_vanilla, splits['y_val'],
        n_features=n_features, n_models=val_probas_vanilla.shape[1],
    )
    tau, _ = find_best_threshold(ml, splits['X_val'], val_probas_vanilla, splits['y_val'])

    feat_t = torch.tensor(splits['X_test'], dtype=torch.float32)
    prob_t = torch.tensor(test_probas_vanilla, dtype=torch.float32)
    ml.eval()
    with torch.no_grad():
        p_final, _, _ = ml(feat_t, prob_t)
    p_final = p_final.numpy()

    y_pred = (p_final > tau).astype(int)
    return compute_metrics(splits['y_test'], y_pred, p_final)


def run_no_catboost(splits, val_probas_3, test_probas_3, models, n_features, **kw):
    """3 base learners only (no CatBoost) — uses pre-trained 3-model probas."""
    print("\n  [Ablation] No CatBoost (3 models)")
    ml, _ = train_meta_learner(
        splits['X_val'], val_probas_3, splits['y_val'],
        n_features=n_features, n_models=3,
    )
    tau, _ = find_best_threshold(ml, splits['X_val'], val_probas_3, splits['y_val'])

    feat_t = torch.tensor(splits['X_test'], dtype=torch.float32)
    prob_t = torch.tensor(test_probas_3, dtype=torch.float32)
    ml.eval()
    with torch.no_grad():
        p_final, _, _ = ml(feat_t, prob_t)
    p_final = p_final.numpy()

    y_pred = (p_final > tau).astype(int)
    return compute_metrics(splits['y_test'], y_pred, p_final)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    ablation_datasets = list(SPLIT_CONFIG.keys())

    parser = argparse.ArgumentParser(description='LAWE-IDS Ablation Study')
    parser.add_argument('--dataset', choices=ablation_datasets + ['all'], default='all')
    parser.add_argument('--append', action='store_true',
                        help='Append to existing CSV instead of overwriting')
    parser.add_argument('--no-optuna', action='store_true')
    parser.add_argument('--seed', type=int, default=SEED,
                        help=f'Random seed (default: {SEED})')
    args = parser.parse_args()

    datasets = ablation_datasets if args.dataset == 'all' else [args.dataset]

    all_results = []
    for ds_name in datasets:
        print(f"\n{'='*70}")
        print(f"  ABLATION STUDY: {ds_name}  (seed={args.seed})")
        print(f"{'='*70}")

        load_kwargs = {}
        excl = DATASET_EXCLUDE_CATEGORIES.get(ds_name)
        if excl and ds_name == 'unsw-nb15':
            load_kwargs['exclude_attack_categories'] = excl
        data = load_dataset(ds_name, **load_kwargs)
        splits = create_splits(data, random_state=args.seed, **SPLIT_CONFIG[ds_name])

        X_train_sm, y_train_sm = apply_smote(splits['X_train'], splits['y_train'])
        n_features = splits['n_features']

        # ==============================================================
        # Group 1: Standard 4 models (reused by Full, NoFG, FixedWeights)
        # ==============================================================
        print("\n  [Training Group 1] Standard 4 base learners")
        std_models = _train_standard_models(X_train_sm, y_train_sm, n_features)
        all_names_4 = BASE_LEARNER_NAMES + ['cnn-bilstm-attn']
        val_probas_4  = _get_probas(std_models, splits['X_val'],  all_names_4)
        test_probas_4 = _get_probas(std_models, splits['X_test'], all_names_4)

        # Conditions using Group 1
        for cond_name, func in [
            ('Full LAWE-IDS',     run_full),
            ('No Feature Gating', run_no_feature_gating),
            ('Fixed Weights',     run_fixed_weights),
        ]:
            t0 = time.perf_counter()
            try:
                metrics = func(splits, val_probas_4, test_probas_4,
                               std_models, n_features)
            except Exception as exc:
                print(f"    [FAILED] {cond_name}: {exc}")
                import traceback; traceback.print_exc()
                continue
            elapsed = time.perf_counter() - t0
            metrics['condition'] = cond_name
            metrics['dataset'] = ds_name
            metrics['seed'] = args.seed
            metrics['time_s'] = round(elapsed, 1)
            print_metrics(metrics, label=f"{ds_name}/{cond_name}")
            all_results.append(metrics)

        # ==============================================================
        # Group 2: Vanilla CNN (no self-attention) + 3 GBTs
        # ==============================================================
        print("\n  [Training Group 2] Vanilla CNN-BiLSTM (no self-attention)")
        from models import CNNBiLSTMVanilla

        print("      Training vanilla CNN-BiLSTM …")
        vanilla_dl = CNNBiLSTMVanilla(n_features=n_features)
        vanilla_dl.fit(X_train_sm, y_train_sm)

        # Reuse GBTs from Group 1, swap CNN only
        vanilla_models = {k: std_models[k] for k in BASE_LEARNER_NAMES}
        vanilla_models['cnn-bilstm-attn'] = vanilla_dl
        val_probas_vanilla  = _get_probas(vanilla_models, splits['X_val'],  all_names_4)
        test_probas_vanilla = _get_probas(vanilla_models, splits['X_test'], all_names_4)

        t0 = time.perf_counter()
        try:
            metrics = run_no_self_attention(
                splits, val_probas_vanilla, test_probas_vanilla,
                vanilla_models, n_features)
        except Exception as exc:
            print(f"    [FAILED] No Self-Attention: {exc}")
            import traceback; traceback.print_exc()
        else:
            elapsed = time.perf_counter() - t0
            metrics['condition'] = 'No Self-Attention'
            metrics['dataset'] = ds_name
            metrics['seed'] = args.seed
            metrics['time_s'] = round(elapsed, 1)
            print_metrics(metrics, label=f"{ds_name}/No Self-Attention")
            all_results.append(metrics)

        # ==============================================================
        # Group 3: No CatBoost (reuse XGB + LGBM from Group 1 + CNN)
        # ==============================================================
        print("\n  [Training Group 3] No CatBoost (3 models)")
        names_3 = ['xgboost', 'lightgbm', 'cnn-bilstm-attn']
        val_probas_3  = _get_probas(std_models, splits['X_val'],  names_3)
        test_probas_3 = _get_probas(std_models, splits['X_test'], names_3)

        t0 = time.perf_counter()
        try:
            metrics = run_no_catboost(
                splits, val_probas_3, test_probas_3,
                std_models, n_features)
        except Exception as exc:
            print(f"    [FAILED] No CatBoost: {exc}")
            import traceback; traceback.print_exc()
        else:
            elapsed = time.perf_counter() - t0
            metrics['condition'] = 'No CatBoost'
            metrics['dataset'] = ds_name
            metrics['seed'] = args.seed
            metrics['time_s'] = round(elapsed, 1)
            print_metrics(metrics, label=f"{ds_name}/No CatBoost")
            all_results.append(metrics)

    if not all_results:
        print("\nNo ablation results to save.")
        return

    df_new = pd.DataFrame(all_results)
    path = os.path.join(RESULTS_DIR, 'lawe_ablation_results.csv')
    if args.append and os.path.exists(path):
        df_old = pd.read_csv(path)
        if 'seed' not in df_old.columns:
            df_old['seed'] = SEED
        new_keys = set(zip(df_new['dataset'], df_new['condition'], df_new['seed']))
        mask = df_old.apply(
            lambda r: (r['dataset'], r['condition'], r.get('seed', SEED)) not in new_keys,
            axis=1
        )
        df_old = df_old[mask]
        df = pd.concat([df_old, df_new], ignore_index=True)
        df.to_csv(path, index=False)
        print(f"\nAblation results appended (dedup on dataset,condition,seed) to {path}")
    else:
        df_new.to_csv(path, index=False)
        print(f"\nAblation results saved to {path}")


if __name__ == '__main__':
    main()
