"""
Base learners for LAWE-IDS.
4 models: XGBoost, LightGBM, CatBoost, CNN-BiLSTM-Attention.
"""
import numpy as np
import warnings
import os
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Bidirectional, LSTM,
    Dropout, Dense, BatchNormalization, GlobalAveragePooling1D,
    Lambda
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from config import SEED, DL_EPOCHS, DL_BATCH_SIZE, DL_PATIENCE


def create_base_learner(name, random_state=SEED, **kwargs):
    """Factory for gradient boosting base learners.

    Args:
        name: 'xgboost', 'lightgbm', or 'catboost'
        random_state: random seed
        **kwargs: override default hyperparameters (from Optuna)

    Returns:
        sklearn-compatible classifier
    """
    defaults = {
        'xgboost': {
            'n_estimators': 2000, 'max_depth': 15, 'learning_rate': 0.02,
            'subsample': 0.8, 'colsample_bytree': 0.8,
            'random_state': random_state, 'n_jobs': -1,
            'eval_metric': 'logloss', 'verbosity': 0,
        },
        'lightgbm': {
            'n_estimators': 2000, 'max_depth': 18, 'learning_rate': 0.02,
            'subsample': 0.8, 'colsample_bytree': 0.8,
            'random_state': random_state, 'n_jobs': -1, 'verbose': -1,
        },
        'catboost': {
            'iterations': 2000, 'depth': 12, 'learning_rate': 0.02,
            'random_seed': random_state, 'verbose': 0,
            'task_type': 'CPU',
        },
    }

    name_lower = name.lower()
    if name_lower not in defaults:
        raise ValueError(f"Unknown model: {name}. Choose from {list(defaults.keys())}")

    params = {**defaults[name_lower], **kwargs}

    # Remap Optuna generic param names to CatBoost-specific names
    if name_lower == 'catboost':
        if 'n_estimators' in params:
            params['iterations'] = params.pop('n_estimators')
        if 'max_depth' in params:
            params['depth'] = min(params.pop('max_depth'), 10)
        if 'subsample' in params:
            params['subsample'] = params.pop('subsample')
            params['bootstrap_type'] = 'Bernoulli'
        params.pop('colsample_bytree', None)
        if 'reg_alpha' in params:
            params.pop('reg_alpha')
        if 'reg_lambda' in params:
            params['l2_leaf_reg'] = max(params.pop('reg_lambda'), 1.0)

    constructors = {
        'xgboost': XGBClassifier,
        'lightgbm': LGBMClassifier,
        'catboost': CatBoostClassifier,
    }
    return constructors[name_lower](**params)


class CNNBiLSTMAttention:
    """CNN-BiLSTM with Self-Attention. Sklearn-compatible wrapper."""

    def __init__(self, n_features, epochs=DL_EPOCHS, batch_size=DL_BATCH_SIZE,
                 random_state=SEED):
        self.n_features = n_features
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.model = None
        self.classes_ = np.array([0, 1])

    def _build_model(self):
        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)

        inp = Input(shape=(self.n_features, 1))

        # CNN block
        x = Conv1D(64, 3, activation='relu', padding='same')(inp)
        x = BatchNormalization()(x)
        x = MaxPooling1D(2)(x)
        x = Conv1D(128, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(2)(x)

        # BiLSTM
        x = Bidirectional(LSTM(64, return_sequences=True))(x)

        # Self-Attention
        d_model = 128  # 64*2 from BiLSTM
        query = Dense(d_model, use_bias=False)(x)
        key = Dense(d_model, use_bias=False)(x)
        value = Dense(d_model, use_bias=False)(x)

        scores = Lambda(lambda qk: tf.matmul(qk[0], qk[1], transpose_b=True) / tf.math.sqrt(tf.cast(d_model, tf.float32)))([query, key])
        attention_weights = Lambda(lambda s: tf.nn.softmax(s, axis=-1))(scores)
        attention_output = Lambda(lambda av: tf.matmul(av[0], av[1]))([attention_weights, value])

        # Pool
        x = GlobalAveragePooling1D()(attention_output)

        # Dense head
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        out = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=inp, outputs=out)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def fit(self, X, y):
        self.model = self._build_model()
        X_3d = X.reshape(-1, self.n_features, 1)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=DL_PATIENCE, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
        ]

        self.model.fit(
            X_3d, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            callbacks=callbacks,
            verbose=2,
        )
        return self

    def predict(self, X):
        X_3d = X.reshape(-1, self.n_features, 1)
        proba = self.model.predict(X_3d, verbose=0).flatten()
        return (proba > 0.5).astype(int)

    def predict_proba(self, X):
        X_3d = X.reshape(-1, self.n_features, 1)
        proba_pos = self.model.predict(X_3d, verbose=0).flatten()
        return np.column_stack([1 - proba_pos, proba_pos])


# ============================================================
# CNN-BiLSTM without self-attention (used by "No Self-Attention" ablation)
# ============================================================

class CNNBiLSTMVanilla:
    """Vanilla CNN-BiLSTM without self-attention (ablation baseline)."""

    def __init__(self, n_features, epochs=30, batch_size=64, random_state=42):
        self.n_features = n_features
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.model = None
        self.classes_ = np.array([0, 1])

    def _build_model(self):
        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)
        model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same',
                                   input_shape=(self.n_features, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Conv1D(32, 3, activation='relu', padding='same'),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid'),
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                      loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def fit(self, X, y):
        self.model = self._build_model()
        X_3d = X.reshape(-1, self.n_features, 1)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
        ]
        self.model.fit(X_3d, y, epochs=self.epochs, batch_size=self.batch_size,
                       validation_split=0.1, callbacks=callbacks, verbose=0)
        return self

    def predict(self, X):
        X_3d = X.reshape(-1, self.n_features, 1)
        return (self.model.predict(X_3d, verbose=0).flatten() > 0.5).astype(int)

    def predict_proba(self, X):
        X_3d = X.reshape(-1, self.n_features, 1)
        p = self.model.predict(X_3d, verbose=0).flatten()
        return np.column_stack([1 - p, p])
