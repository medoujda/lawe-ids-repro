"""Configuration for LAWE-IDS.

Dataset paths are resolved from DATA_DIR, which defaults to ./data/ relative
to this file. Override by setting the DATA_DIR environment variable:

    DATA_DIR=/path/to/datasets python run_lawe.py
"""
import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get('DATA_DIR', os.path.join(PROJECT_DIR, 'data'))
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Dataset paths
UNSW_TRAIN = os.path.join(DATA_DIR, 'unsw-nb15', 'UNSW_NB15_training-set.csv')
UNSW_TEST  = os.path.join(DATA_DIR, 'unsw-nb15', 'UNSW_NB15_testing-set.csv')
NSL_KDD_DIR   = os.path.join(DATA_DIR, 'nsl-kdd')
CICIDS_DIR    = os.path.join(DATA_DIR, 'cicids2017')
TON_IOT_DIR   = os.path.join(DATA_DIR, 'ton-iot')
CICIDS2023_DIR = os.path.join(DATA_DIR, 'cicids2023')

# Active datasets (BoT-IoT excluded: ~99.99% attack traffic makes binary
# metrics uninterpretable on a stratified 500 k-row sample)
DATASETS = ['unsw-nb15', 'nsl-kdd', 'cicids2017', 'ton-iot', 'cicids2023']

# UNSW-NB15: Fuzzers are excluded because payload-dependent attacks are
# structurally indistinguishable from Normal traffic in the 42-feature
# flow-based representation (~39% of Fuzzers overlap exactly with Normal).
# Flow-based IDS cannot detect them; excluding them avoids label noise.
DATASET_EXCLUDE_CATEGORIES = {
    'unsw-nb15': ['Fuzzers'],
}

SEED = 42

# Optuna search spaces for gradient boosting base learners
GBT_SEARCH_SPACE = {
    'n_estimators':    (500, 3000),
    'max_depth':       (6, 20),
    'learning_rate':   (0.01, 0.1),
    'subsample':       (0.6, 0.9),
    'colsample_bytree':(0.6, 0.9),
    'reg_alpha':       (1e-8, 10.0),
    'reg_lambda':      (1e-8, 10.0),
}
OPTUNA_TRIALS    = 50
OPTUNA_CV_FOLDS  = 3

# CNN-BiLSTM-Attention
DL_EPOCHS     = 5
DL_BATCH_SIZE = 64
DL_PATIENCE   = 3

# Attention meta-learner
META_EPOCHS   = 100
META_PATIENCE = 15
META_LR       = 0.001
META_HIDDEN_1 = 128
META_HIDDEN_2 = 64
META_DROPOUT  = 0.2

# Decision threshold search range (validation set)
THRESHOLD_MIN  = 0.30
THRESHOLD_MAX  = 0.70
THRESHOLD_STEP = 0.01
