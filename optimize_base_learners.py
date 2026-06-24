"""
Optuna Hyperparameter Optimization for LAWE-IDS Base Learners.

Runs Bayesian optimization (Optuna TPE sampler) to find the best
hyperparameters for XGBoost, LightGBM, and CatBoost on each dataset.

Usage
-----
    python optimize_base_learners.py --dataset all --model all
    python optimize_base_learners.py --dataset unsw-nb15 --model xgboost
    python optimize_base_learners.py --dataset nsl-kdd --model lightgbm catboost

Saved params location
---------------------
    results/optuna_params/{dataset}_{model}_best_params.json
"""

import argparse
import json
import logging
import os
import sys
import warnings

import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from config import (
    SEED, GBT_SEARCH_SPACE, OPTUNA_TRIALS, OPTUNA_CV_FOLDS,
    RESULTS_DIR, DATASETS,
)
from preprocessing import load_dataset, create_splits, apply_smote

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PARAMS_DIR = os.path.join(RESULTS_DIR, 'optuna_params')
os.makedirs(PARAMS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# Silence Optuna's own verbose output (keep WARNING and above)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def _build_model(model_name, params):
    """Instantiate a model from a hyperparameter dict.

    CatBoost uses 'iterations' and 'depth' instead of
    'n_estimators' and 'max_depth', and does not accept 'colsample_bytree'.
    """
    if model_name == 'xgboost':
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            reg_alpha=params['reg_alpha'],
            reg_lambda=params['reg_lambda'],
            use_label_encoder=False,
            eval_metric='logloss',
            tree_method='hist',
            random_state=SEED,
            n_jobs=-1,
            verbosity=0,
        )

    elif model_name == 'lightgbm':
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            reg_alpha=params['reg_alpha'],
            reg_lambda=params['reg_lambda'],
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
        )

    elif model_name == 'catboost':
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=params['n_estimators'],
            depth=min(params['max_depth'], 8),   # CatBoost safe cap
            learning_rate=params['learning_rate'],
            l2_leaf_reg=max(params['reg_lambda'], 1.0),  # prevent degenerate solutions
            random_seed=SEED,
            thread_count=-1,
            verbose=0,
        )

    else:
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            "Choose from 'xgboost', 'lightgbm', 'catboost'."
        )


# ---------------------------------------------------------------------------
# Objective factory
# ---------------------------------------------------------------------------

def _create_objective(model_name, X_train, y_train):
    """Return an Optuna objective function for the given model and training data.

    The objective performs stratified k-fold CV (OPTUNA_CV_FOLDS folds) and
    returns the mean macro F1-score across folds (higher is better).

    Parameters
    ----------
    model_name : str
        One of 'xgboost', 'lightgbm', 'catboost'.
    X_train : np.ndarray, shape (n_samples, n_features)
    y_train : np.ndarray, shape (n_samples,)

    Returns
    -------
    objective : callable
        A function compatible with optuna.study.optimize().
    """
    ss = GBT_SEARCH_SPACE
    # CatBoost crashes with depth > 8 on some versions
    max_depth_hi = 8 if model_name == 'catboost' else ss['max_depth'][1]

    def objective(trial):
        # ---- Suggest hyperparameters ----------------------------------------
        params = {
            'n_estimators':    trial.suggest_int(
                'n_estimators', ss['n_estimators'][0], ss['n_estimators'][1]),
            'max_depth':       trial.suggest_int(
                'max_depth', ss['max_depth'][0], max_depth_hi),
            'learning_rate':   trial.suggest_float(
                'learning_rate', ss['learning_rate'][0], ss['learning_rate'][1],
                log=True),
            'subsample':       trial.suggest_float(
                'subsample', ss['subsample'][0], ss['subsample'][1]),
            'colsample_bytree': trial.suggest_float(
                'colsample_bytree',
                ss['colsample_bytree'][0], ss['colsample_bytree'][1]),
            'reg_alpha':       trial.suggest_float(
                'reg_alpha', ss['reg_alpha'][0], ss['reg_alpha'][1], log=True),
            'reg_lambda':      trial.suggest_float(
                'reg_lambda', ss['reg_lambda'][0], ss['reg_lambda'][1],
                log=True),
        }

        # ---- Cross-validation -----------------------------------------------
        skf = StratifiedKFold(
            n_splits=OPTUNA_CV_FOLDS, shuffle=True, random_state=SEED
        )
        fold_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_fold_train = X_train[train_idx]
            y_fold_train = y_train[train_idx]
            X_fold_val   = X_train[val_idx]
            y_fold_val   = y_train[val_idx]

            model = _build_model(model_name, params)

            if model_name == 'xgboost':
                model.fit(
                    X_fold_train, y_fold_train,
                    eval_set=[(X_fold_val, y_fold_val)],
                    verbose=False,
                )
            elif model_name == 'lightgbm':
                import lightgbm as lgb
                model.fit(
                    X_fold_train, y_fold_train,
                    eval_set=[(X_fold_val, y_fold_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False),
                               lgb.log_evaluation(period=-1)],
                )
            else:
                model.fit(X_fold_train, y_fold_train)

            preds = model.predict(X_fold_val)
            score = f1_score(y_fold_val, preds, average='macro', zero_division=0)
            fold_scores.append(score)

            # Pruning: report intermediate value after each fold
            trial.report(np.mean(fold_scores), step=fold_idx)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return float(np.mean(fold_scores))

    return objective


# ---------------------------------------------------------------------------
# Public API: optimize_model
# ---------------------------------------------------------------------------

def optimize_model(model_name, dataset_name, X_train, y_train,
                   n_trials=None, save=True):
    """Run an Optuna study to find optimal hyperparameters.

    Parameters
    ----------
    model_name : str
        One of 'xgboost', 'lightgbm', 'catboost'.
    dataset_name : str
        Name used for saving (e.g. 'unsw-nb15').
    X_train : np.ndarray
    y_train : np.ndarray
    n_trials : int or None
        Number of Optuna trials. Defaults to OPTUNA_TRIALS from config.
    save : bool
        Whether to persist the best params as JSON.

    Returns
    -------
    best_params : dict
    """
    if n_trials is None:
        n_trials = OPTUNA_TRIALS

    logger.info(
        "Starting Optuna study | model=%s | dataset=%s | trials=%d | cv=%d",
        model_name, dataset_name, n_trials, OPTUNA_CV_FOLDS,
    )

    sampler = optuna.samplers.TPESampler(seed=SEED)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)

    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        study_name=f"{dataset_name}_{model_name}",
    )

    objective = _create_objective(model_name, X_train, y_train)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False,
                   catch=(Exception,))

    best_trial  = study.best_trial
    best_params = best_trial.params
    best_value  = best_trial.value

    logger.info(
        "Best trial #%d | macro-F1=%.4f | params=%s",
        best_trial.number, best_value, best_params,
    )

    if save:
        _save_params(model_name, dataset_name, best_params, best_value)

    return best_params


# ---------------------------------------------------------------------------
# Public API: load_best_params / save
# ---------------------------------------------------------------------------

def _params_path(model_name, dataset_name):
    """Return the canonical JSON path for a (model, dataset) pair."""
    fname = f"{dataset_name}_{model_name}_best_params.json"
    return os.path.join(PARAMS_DIR, fname)


def _save_params(model_name, dataset_name, params, best_value=None):
    """Persist best hyperparameters to JSON."""
    payload = {'params': params}
    if best_value is not None:
        payload['best_macro_f1'] = best_value

    path = _params_path(model_name, dataset_name)
    with open(path, 'w') as fh:
        json.dump(payload, fh, indent=2)
    logger.info("Saved best params → %s", path)


def load_best_params(model_name, dataset_name):
    """Load previously optimised hyperparameters from disk.

    Parameters
    ----------
    model_name : str
    dataset_name : str

    Returns
    -------
    dict or None
        The params dict (suitable for passing to the model constructor),
        or None if no saved file exists.
    """
    path = _params_path(model_name, dataset_name)
    if not os.path.exists(path):
        logger.debug(
            "No saved params found for %s / %s at %s",
            model_name, dataset_name, path,
        )
        return None

    with open(path, 'r') as fh:
        payload = json.load(fh)

    params = payload.get('params', payload)  # backwards-compat fallback
    logger.info(
        "Loaded best params for %s / %s from %s",
        model_name, dataset_name, path,
    )
    return params


# ---------------------------------------------------------------------------
# Dataset loading helpers
# ---------------------------------------------------------------------------

def _load_train_data(dataset_name):
    """Load a dataset and return (X_train, y_train) ready for optimisation.

    Uses the same split strategy as the main pipeline:
    - UNSW-NB15  : official split
    - NSL-KDD    : random, val=0.1, test=0.1
    - CIC-IDS2017: random, val=0.16, test=0.2
    """
    name = dataset_name.lower()
    logger.info("Loading dataset: %s", name)
    data = load_dataset(name)

    if name == 'unsw-nb15':
        splits = create_splits(data, split_type='random',
                               val_ratio=0.1, test_ratio=0.2)
    elif name == 'nsl-kdd':
        splits = create_splits(data, split_type='random',
                               val_ratio=0.1, test_ratio=0.1)
    elif name == 'cicids2017':
        splits = create_splits(data, split_type='random',
                               val_ratio=0.16, test_ratio=0.2)
    else:
        raise ValueError(f"Unknown dataset: '{dataset_name}'")

    X_train = splits['X_train']
    y_train = splits['y_train']
    logger.info(
        "Dataset %s | train shape: %s | class balance: %.2f%%",
        name, X_train.shape, 100 * y_train.mean(),
    )
    return X_train, y_train


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

SUPPORTED_MODELS   = ['xgboost', 'lightgbm', 'catboost']
SUPPORTED_DATASETS = DATASETS  # from config: ['unsw-nb15', 'nsl-kdd', 'cicids2017']


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Optuna hyperparameter search for LAWE-IDS base learners.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--dataset', nargs='+', default=['all'],
        help=(
            "Dataset(s) to optimise. Use 'all' or any combination of: "
            + ', '.join(SUPPORTED_DATASETS)
        ),
    )
    parser.add_argument(
        '--model', nargs='+', default=['all'],
        help=(
            "Model(s) to optimise. Use 'all' or any combination of: "
            + ', '.join(SUPPORTED_MODELS)
        ),
    )
    parser.add_argument(
        '--trials', type=int, default=None,
        help='Number of Optuna trials (overrides config.OPTUNA_TRIALS).',
    )
    parser.add_argument(
        '--no-smote', action='store_true',
        help='Skip SMOTE resampling before optimisation.',
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    # Resolve dataset list
    if 'all' in [d.lower() for d in args.dataset]:
        datasets = SUPPORTED_DATASETS
    else:
        datasets = [d.lower() for d in args.dataset]
        for d in datasets:
            if d not in SUPPORTED_DATASETS:
                logger.error("Unknown dataset '%s'. Choose from %s", d, SUPPORTED_DATASETS)
                sys.exit(1)

    # Resolve model list
    if 'all' in [m.lower() for m in args.model]:
        models = SUPPORTED_MODELS
    else:
        models = [m.lower() for m in args.model]
        for m in models:
            if m not in SUPPORTED_MODELS:
                logger.error("Unknown model '%s'. Choose from %s", m, SUPPORTED_MODELS)
                sys.exit(1)

    n_trials = args.trials  # may be None → uses config default inside optimize_model

    logger.info(
        "Optuna search | datasets=%s | models=%s | trials=%s",
        datasets, models, n_trials or OPTUNA_TRIALS,
    )

    for dataset_name in datasets:
        # Load training data once per dataset
        try:
            X_train, y_train = _load_train_data(dataset_name)
        except FileNotFoundError as exc:
            logger.warning("Skipping dataset '%s': %s", dataset_name, exc)
            continue

        # Apply SMOTE if requested (default: apply)
        if not args.no_smote:
            logger.info("Applying SMOTE to %s training data ...", dataset_name)
            X_train, y_train = apply_smote(X_train, y_train)
            logger.info("After SMOTE: shape=%s", X_train.shape)

        for model_name in models:
            try:
                optimize_model(
                    model_name, dataset_name,
                    X_train, y_train,
                    n_trials=n_trials,
                )
            except Exception as exc:
                logger.error(
                    "Error optimising %s / %s: %s",
                    model_name, dataset_name, exc, exc_info=True,
                )

    logger.info("All optimisations complete.")


if __name__ == '__main__':
    main()
