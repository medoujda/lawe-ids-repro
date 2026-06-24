# LAWE-IDS: A Lightweight Attention-Weighted Ensemble for Intrusion Detection in IoT Networks

Reproducibility package for the paper submitted to [journal name].

> **Anonymous review**: this repository is anonymized via
> [anonymous.4open.science](https://anonymous.4open.science).
> Author information will be disclosed upon acceptance.

---

## Overview

LAWE-IDS is a modular ensemble IDS that fuses four heterogeneous classifiers
(XGBoost, LightGBM, CatBoost, CNN-BiLSTM with self-attention) through a
lightweight attention-based meta-learner. The meta-learner produces per-sample
fusion weights used diagnostically to identify redundant components, motivating
**LAWE-IDS Lite** (−79% model size, <0.03% accuracy loss).

## Results

Pre-computed results matching all paper tables are in `results/`:

| File | Content |
|------|---------|
| `results/lawe_ids_results.csv` | Binary classification (3 seeds, 5 datasets) |
| `results/lawe_ablation_results.csv` | Ablation study (5 conditions, seed 42) |
| `results/lawe_iot_metrics.csv` | Inference latency and model sizes |

## Repository structure

```
├── config.py                   # Paths, hyperparameters
├── lawe_ids.py                 # Main pipeline orchestrator
├── attention_meta_learner.py   # Attention-based meta-learner (PyTorch)
├── feature_gating.py           # Learnable per-sample feature gates
├── models.py                   # Base learners + CNNBiLSTMVanilla (ablation)
├── preprocessing.py            # Dataset loaders, SMOTE, train/val/test splits
├── evaluation.py               # Metrics (accuracy, F1, AUC-ROC, FPR, FNR)
├── optimize_base_learners.py   # Optuna hyperparameter search
├── run_lawe.py                 # Reproduce main binary classification results
├── run_lawe_ablation.py        # Reproduce ablation study
├── run_lawe_iot_metrics.py     # Reproduce inference latency / model size table
├── analyze_ablation_stats.py   # Friedman + Wilcoxon statistical tests
├── analyze_attention_weights.py# Attention weight analysis (paper figures)
├── tests/                      # Unit tests
├── data/
│   ├── README.md               # Dataset download instructions
│   └── subsample_cicids2017.py # Reproducible CIC-IDS2017 subsample script
└── results/                    # Pre-computed CSVs
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested on Python 3.10–3.12. GPU is optional; all experiments ran on CPU.

## Reproducing the paper results

### 1. Prepare datasets

Follow `data/README.md` to download the five datasets. For CIC-IDS2017,
generate the 500 k-row subsample first:

```bash
python data/subsample_cicids2017.py \
    --raw_dir /path/to/raw/cicids2017 \
    --out_dir data/cicids2017
```

### 2. Run binary classification (Table 2 in paper)

Single dataset, one seed:
```bash
python run_lawe.py --dataset unsw-nb15 --seed 42
```

All datasets, all seeds (reproduces full Table 2; ~8–12 h per seed):
```bash
for seed in 42 43 44; do
    python run_lawe.py --dataset all --seed $seed --append
done
```

Results are saved to `results/lawe_ids_results.csv`.

### 3. Run ablation study (Table 3 in paper)

```bash
python run_lawe_ablation.py --dataset all --seed 42
```

Then run the statistical tests (Friedman + Wilcoxon, reported in §4.2):
```bash
python analyze_ablation_stats.py
```

### 4. Inference latency and model sizes (Table 4)

```bash
python run_lawe_iot_metrics.py --dataset all --seed 42
```

### 5. Attention weight analysis (reported in §4.4)

```bash
python analyze_attention_weights.py
```

## LAWE-IDS Lite

LAWE-IDS Lite removes CatBoost from the ensemble. The "No CatBoost" row in
the ablation results is equivalent to Lite. To train Lite directly:

```bash
python run_lawe.py --dataset unsw-nb15 --no-catboost
```

## Hyperparameter tuning (optional)

The default hyperparameters (in `config.py`) were found via Optuna on each
dataset. To re-run the search (adds ~2 h per dataset):

```bash
python run_lawe.py --dataset unsw-nb15 --optuna
```

## Unit tests

```bash
python -m pytest tests/ -v
```

## Hardware

All experiments were run on a machine with:
- CPU: Apple M-series (or equivalent x86-64)
- RAM: 16 GB
- No GPU required

Approximate wall-clock times per dataset (seed, no Optuna):
- NSL-KDD: ~15 min
- UNSW-NB15: ~20 min
- CIC-IDS2017, TON-IoT, CICIoT2023: ~60–90 min each
