"""Tests for base learner models."""
import numpy as np
import pytest
from models import create_base_learner, CNNBiLSTMAttention

class TestGradientBoostingModels:
    @pytest.fixture
    def dummy_data(self):
        np.random.seed(42)
        X_train = np.random.rand(200, 10).astype(np.float32)
        y_train = np.random.randint(0, 2, 200)
        X_test = np.random.rand(50, 10).astype(np.float32)
        return X_train, y_train, X_test

    def test_xgboost_fit_predict(self, dummy_data):
        X_train, y_train, X_test = dummy_data
        model = create_base_learner('xgboost')
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        assert proba.shape == (50,)
        assert all(0 <= p <= 1 for p in proba)

    def test_lightgbm_fit_predict(self, dummy_data):
        X_train, y_train, X_test = dummy_data
        model = create_base_learner('lightgbm')
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        assert proba.shape == (50,)

    def test_catboost_fit_predict(self, dummy_data):
        X_train, y_train, X_test = dummy_data
        model = create_base_learner('catboost')
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        assert proba.shape == (50,)

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError):
            create_base_learner('unknown_model')

class TestCNNBiLSTMAttention:
    def test_fit_predict(self):
        np.random.seed(42)
        X_train = np.random.rand(100, 10).astype(np.float32)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.rand(20, 10).astype(np.float32)

        model = CNNBiLSTMAttention(n_features=10, epochs=2, batch_size=32)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)
        assert proba.shape == (20, 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=0.01)

    def test_predict_returns_binary(self):
        np.random.seed(42)
        X_train = np.random.rand(100, 10).astype(np.float32)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.rand(20, 10).astype(np.float32)

        model = CNNBiLSTMAttention(n_features=10, epochs=2, batch_size=32)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert set(preds).issubset({0, 1})
