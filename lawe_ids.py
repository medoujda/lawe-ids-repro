"""LAWE-IDS Pipeline Orchestrator.

Ties together:
  - SMOTE resampling
  - 4 base learners (XGBoost, LightGBM, CatBoost, CNN-BiLSTM-Attention)
  - Attention meta-learner (per-sample adaptive weighting)
  - Threshold optimisation on the validation set

Typical usage
-------------
    lawe = LAWEIDS()
    lawe.fit(X_train, y_train, X_val, y_val, n_features=X_train.shape[1])
    preds  = lawe.predict(X_test)
    probas = lawe.predict_proba(X_test)
"""

import os
import logging
import warnings
import numpy as np

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import joblib
import torch

from models import create_base_learner, CNNBiLSTMAttention
from attention_meta_learner import AttentionMetaLearner, train_meta_learner, find_best_threshold
from preprocessing import apply_smote
from optimize_base_learners import load_best_params
from config import (
    SEED, DL_EPOCHS, DL_BATCH_SIZE, META_EPOCHS, META_PATIENCE,
    META_LR, THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEP, RESULTS_DIR,
)

logger = logging.getLogger(__name__)

# Names of the three gradient-boosting base learners (order determines column
# index in the (N, 4) probability array).
BASE_LEARNER_NAMES = ['xgboost', 'lightgbm', 'catboost']
CNN_KEY = 'cnn-bilstm-attn'
ALL_MODEL_KEYS = BASE_LEARNER_NAMES + [CNN_KEY]   # length 4, column order fixed


class LAWEIDS:
    """LAWE-IDS end-to-end pipeline.

    Parameters
    ----------
    dataset_name : str or None
        Used to load Optuna-tuned hyperparameters when ``use_optuna=True``.
    use_optuna : bool
        If True, attempt to load pre-optimised GBT hyperparameters from disk.
    dl_epochs : int
        Maximum training epochs for the CNN-BiLSTM-Attention model.
    dl_batch_size : int
        Mini-batch size for the CNN-BiLSTM-Attention model.
    meta_epochs : int
        Maximum training epochs for the attention meta-learner.
    meta_patience : int
        Early-stopping patience for the meta-learner.
    meta_lr : float
        Adam learning rate for the meta-learner.
    threshold_min / threshold_max / threshold_step : float
        Grid-search range for the decision threshold.
    random_state : int
        Global random seed.
    """

    def __init__(
        self,
        dataset_name: str = None,
        use_optuna: bool = True,
        dl_epochs: int = DL_EPOCHS,
        dl_batch_size: int = DL_BATCH_SIZE,
        meta_epochs: int = META_EPOCHS,
        meta_patience: int = META_PATIENCE,
        meta_lr: float = META_LR,
        threshold_min: float = THRESHOLD_MIN,
        threshold_max: float = THRESHOLD_MAX,
        threshold_step: float = THRESHOLD_STEP,
        random_state: int = SEED,
    ):
        self.dataset_name = dataset_name
        self.use_optuna = use_optuna
        self.dl_epochs = dl_epochs
        self.dl_batch_size = dl_batch_size
        self.meta_epochs = meta_epochs
        self.meta_patience = meta_patience
        self.meta_lr = meta_lr
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self.threshold_step = threshold_step
        self.random_state = random_state

        # Set after fit()
        self.base_learners_ = {}          # dict: name -> fitted model
        self.meta_learner_: AttentionMetaLearner = None
        self.threshold_: float = 0.5
        self.n_features_: int = None
        self.is_fitted_: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_features: int = None,
    ) -> 'LAWEIDS':
        """Train the full LAWE-IDS pipeline.

        Steps
        -----
        1. SMOTE on training data.
        2. Train 4 base learners on the (resampled) training set.
        3. Collect base-learner probabilities on the validation set.
        4. Train the attention meta-learner on validation set probabilities.
        5. Find the best decision threshold on the validation set.

        Parameters
        ----------
        X_train, y_train : training data (before SMOTE).
        X_val, y_val     : validation data (not resampled).
        n_features       : number of input features; inferred if None.

        Returns
        -------
        self
        """
        if n_features is None:
            n_features = X_train.shape[1]
        self.n_features_ = n_features

        # --- 1. SMOTE ---------------------------------------------------
        logger.info("Applying SMOTE …")
        X_tr_sm, y_tr_sm = apply_smote(X_train, y_train, random_state=self.random_state)

        # --- 2. Train base learners ------------------------------------
        logger.info("Training base learners …")
        self.base_learners_ = {}

        for name in BASE_LEARNER_NAMES:
            logger.info("  Fitting %s …", name)
            params = {}
            if self.use_optuna and self.dataset_name:
                loaded = load_best_params(name, self.dataset_name)
                if loaded:
                    params = loaded
                    logger.info("    Using Optuna params for %s / %s", name, self.dataset_name)

            model = create_base_learner(name, random_state=self.random_state, **params)
            model.fit(X_tr_sm, y_tr_sm)
            self.base_learners_[name] = model

        # CNN-BiLSTM-Attention
        logger.info("  Fitting CNN-BiLSTM-Attention …")
        cnn = CNNBiLSTMAttention(
            n_features=n_features,
            epochs=self.dl_epochs,
            batch_size=self.dl_batch_size,
            random_state=self.random_state,
        )
        cnn.fit(X_tr_sm, y_tr_sm)
        self.base_learners_[CNN_KEY] = cnn

        # --- 3. Base-learner probabilities on val set ------------------
        logger.info("Collecting validation set probabilities …")
        val_probas = self._get_base_probas(X_val)   # (N_val, 4)

        # --- 4. Train meta-learner ------------------------------------
        logger.info("Training attention meta-learner …")
        self.meta_learner_, _ = train_meta_learner(
            features=X_val.astype(np.float32),
            probas=val_probas.astype(np.float32),
            y_true=y_val,
            n_features=n_features,
            n_models=len(ALL_MODEL_KEYS),
            epochs=self.meta_epochs,
            patience=self.meta_patience,
            lr=self.meta_lr,
            seed=self.random_state,
        )

        # --- 5. Find best threshold ------------------------------------
        logger.info("Finding best decision threshold …")
        self.threshold_, best_acc = find_best_threshold(
            model=self.meta_learner_,
            features=X_val.astype(np.float32),
            probas=val_probas.astype(np.float32),
            y_true=y_val,
            low=self.threshold_min,
            high=self.threshold_max,
            step=self.threshold_step,
        )
        logger.info("  Best threshold: %.3f  (val acc: %.4f)", self.threshold_, best_acc)

        self.is_fitted_ = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return the meta-learner's weighted ensemble probability for class 1.

        Parameters
        ----------
        X : (N, n_features)

        Returns
        -------
        proba : (N,) probabilities in [0, 1]
        """
        self._check_fitted()
        base_probas = self._get_base_probas(X)  # (N, 4)

        self.meta_learner_.eval()
        X_t = torch.tensor(X.astype(np.float32), dtype=torch.float32)
        P_t = torch.tensor(base_probas.astype(np.float32), dtype=torch.float32)

        with torch.no_grad():
            p_final, _, _ = self.meta_learner_(X_t, P_t)

        return p_final.numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels using the optimised threshold.

        Parameters
        ----------
        X : (N, n_features)

        Returns
        -------
        labels : (N,) integer array with values in {0, 1}
        """
        proba = self.predict_proba(X)
        return (proba >= self.threshold_).astype(int)

    def get_sample_weights(self, X: np.ndarray) -> np.ndarray:
        """Return per-sample base-learner weights from the meta-learner.

        Parameters
        ----------
        X : (N, n_features)

        Returns
        -------
        weights : (N, 4)  softmax weights, rows sum to 1.
        """
        self._check_fitted()
        base_probas = self._get_base_probas(X)

        self.meta_learner_.eval()
        X_t = torch.tensor(X.astype(np.float32), dtype=torch.float32)
        P_t = torch.tensor(base_probas.astype(np.float32), dtype=torch.float32)

        with torch.no_grad():
            _, weights, _ = self.meta_learner_(X_t, P_t)

        return weights.numpy()

    def get_feature_gates(self, X: np.ndarray) -> np.ndarray:
        """Return feature gate values from the meta-learner's FeatureGating layer.

        Parameters
        ----------
        X : (N, n_features)

        Returns
        -------
        gates : (N, n_features) gate values in [0, 1]
        """
        self._check_fitted()
        base_probas = self._get_base_probas(X)

        self.meta_learner_.eval()
        X_t = torch.tensor(X.astype(np.float32), dtype=torch.float32)
        P_t = torch.tensor(base_probas.astype(np.float32), dtype=torch.float32)

        with torch.no_grad():
            _, _, gates = self.meta_learner_(X_t, P_t)

        return gates.numpy()

    def save(self, path: str) -> None:
        """Persist all fitted models to ``path``.

        File layout
        -----------
        ``path/``
          ├── xgboost.joblib
          ├── lightgbm.joblib
          ├── catboost.joblib
          ├── cnn_bilstm_attn.keras
          ├── meta_learner.pt
          └── pipeline_meta.joblib   (threshold, n_features, …)
        """
        self._check_fitted()
        os.makedirs(path, exist_ok=True)

        # GBT models
        for name in BASE_LEARNER_NAMES:
            joblib.dump(self.base_learners_[name],
                        os.path.join(path, f'{name}.joblib'))

        # CNN-BiLSTM-Attention (Keras)
        cnn = self.base_learners_[CNN_KEY]
        if cnn.model is not None:
            cnn.model.save(os.path.join(path, 'cnn_bilstm_attn.keras'))

        # Meta-learner (PyTorch)
        torch.save(self.meta_learner_.state_dict(),
                   os.path.join(path, 'meta_learner.pt'))

        # Pipeline metadata
        meta = {
            'threshold': self.threshold_,
            'n_features': self.n_features_,
            'dataset_name': self.dataset_name,
        }
        joblib.dump(meta, os.path.join(path, 'pipeline_meta.joblib'))
        logger.info("Saved LAWE-IDS pipeline to %s", path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_base_probas(self, X: np.ndarray) -> np.ndarray:
        """Collect class-1 probabilities from all 4 base learners.

        Parameters
        ----------
        X : (N, n_features)

        Returns
        -------
        probas : (N, 4)  column order matches ``ALL_MODEL_KEYS``
        """
        cols = []
        for name in ALL_MODEL_KEYS:
            model = self.base_learners_[name]
            p = model.predict_proba(X)   # (N, 2)
            cols.append(np.clip(p[:, 1], 0.0, 1.0))  # class-1 probability
        return np.column_stack(cols).astype(np.float32)  # (N, 4)

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError(
                "LAWEIDS is not fitted yet. Call fit() before predict / transform methods."
            )
