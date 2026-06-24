"""Tests for LAWE-IDS pipeline."""
import numpy as np
import pytest
from lawe_ids import LAWEIDS

class TestLAWEIDS:
    @pytest.fixture
    def small_data(self):
        np.random.seed(42)
        n_train, n_val, n_test, n_feat = 300, 100, 100, 10
        return {
            'X_train': np.random.rand(n_train, n_feat).astype(np.float32),
            'y_train': np.random.randint(0, 2, n_train),
            'X_val': np.random.rand(n_val, n_feat).astype(np.float32),
            'y_val': np.random.randint(0, 2, n_val),
            'X_test': np.random.rand(n_test, n_feat).astype(np.float32),
            'y_test': np.random.randint(0, 2, n_test),
            'n_features': n_feat,
            'feature_names': [f'f{i}' for i in range(n_feat)],
            'dataset_name': 'test',
        }

    def test_fit_predict(self, small_data):
        lawe = LAWEIDS(
            use_optuna=False,
            dl_epochs=2,
            dl_batch_size=32,
            meta_epochs=5,
            meta_patience=50,
        )
        lawe.fit(
            small_data['X_train'], small_data['y_train'],
            small_data['X_val'], small_data['y_val'],
            n_features=small_data['n_features'],
        )
        preds = lawe.predict(small_data['X_test'])
        assert preds.shape == (100,)
        assert set(preds).issubset({0, 1})

    def test_predict_proba(self, small_data):
        lawe = LAWEIDS(
            use_optuna=False,
            dl_epochs=2,
            dl_batch_size=32,
            meta_epochs=5,
            meta_patience=50,
        )
        lawe.fit(
            small_data['X_train'], small_data['y_train'],
            small_data['X_val'], small_data['y_val'],
            n_features=small_data['n_features'],
        )
        proba = lawe.predict_proba(small_data['X_test'])
        assert proba.shape == (100,)
        assert proba.min() >= -0.01
        assert proba.max() <= 1.01

    def test_get_weights(self, small_data):
        lawe = LAWEIDS(
            use_optuna=False,
            dl_epochs=2,
            dl_batch_size=32,
            meta_epochs=5,
            meta_patience=50,
        )
        lawe.fit(
            small_data['X_train'], small_data['y_train'],
            small_data['X_val'], small_data['y_val'],
            n_features=small_data['n_features'],
        )
        weights = lawe.get_sample_weights(small_data['X_test'])
        assert weights.shape == (100, 4)
        assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-4)
