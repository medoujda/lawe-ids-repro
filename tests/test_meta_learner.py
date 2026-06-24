"""Tests for Attention Meta-Learner."""
import torch
import numpy as np
import pytest
from attention_meta_learner import AttentionMetaLearner, train_meta_learner

class TestAttentionMetaLearner:
    def test_output_shape(self):
        ml = AttentionMetaLearner(n_features=42, n_models=4)
        features = torch.randn(16, 42)
        probas = torch.rand(16, 4)
        p_final, weights, gates = ml(features, probas)
        assert p_final.shape == (16,)
        assert weights.shape == (16, 4)
        assert gates.shape == (16, 42)

    def test_weights_sum_to_one(self):
        ml = AttentionMetaLearner(n_features=42, n_models=4)
        features = torch.randn(32, 42)
        probas = torch.rand(32, 4)
        _, weights, _ = ml(features, probas)
        sums = weights.sum(dim=1)
        assert torch.allclose(sums, torch.ones(32), atol=1e-5)

    def test_p_final_in_01_range(self):
        ml = AttentionMetaLearner(n_features=42, n_models=4)
        features = torch.randn(32, 42)
        probas = torch.rand(32, 4)
        p_final, _, _ = ml(features, probas)
        assert p_final.min() >= -0.01
        assert p_final.max() <= 1.01

    def test_gradient_flows_to_feature_gating(self):
        ml = AttentionMetaLearner(n_features=10, n_models=4)
        features = torch.randn(8, 10)
        probas = torch.rand(8, 4)
        p_final, _, _ = ml(features, probas)
        loss = ((p_final - torch.ones(8)) ** 2).mean()
        loss.backward()
        for p in ml.feature_gating.parameters():
            assert p.grad is not None

    def test_parameter_count(self):
        ml = AttentionMetaLearner(n_features=42, n_models=4)
        n_params = sum(p.numel() for p in ml.parameters())
        # FeatureGating: 42*42+42 = 1806
        # FC1: (42+4)*128+128 = 6016
        # FC2: 128*64+64 = 8256
        # FC3: 64*4+4 = 260
        # Total: 16338
        assert n_params == 16338

class TestTrainMetaLearner:
    def test_training_reduces_loss(self):
        np.random.seed(42)
        n_samples = 200
        n_features = 10

        features = np.random.rand(n_samples, n_features).astype(np.float32)
        y_true = np.random.randint(0, 2, n_samples)
        probas = np.random.rand(n_samples, 4).astype(np.float32) * 0.3
        probas[:, 0] = y_true * 0.9 + (1 - y_true) * 0.1

        ml, history = train_meta_learner(
            features, probas, y_true,
            n_features=n_features, n_models=4,
            epochs=20, patience=50, lr=0.01,
        )
        assert history['val_loss'][-1] < history['val_loss'][0]
