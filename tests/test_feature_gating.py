"""Tests for Feature Gating module."""
import torch
import numpy as np
import pytest
from feature_gating import FeatureGating

class TestFeatureGating:
    def test_output_shape(self):
        fg = FeatureGating(n_features=42)
        x = torch.randn(16, 42)
        gated, gates = fg(x)
        assert gated.shape == (16, 42)
        assert gates.shape == (16, 42)

    def test_gates_in_01_range(self):
        fg = FeatureGating(n_features=42)
        x = torch.randn(32, 42)
        _, gates = fg(x)
        assert gates.min() >= 0.0
        assert gates.max() <= 1.0

    def test_gated_output_is_elementwise_product(self):
        fg = FeatureGating(n_features=10)
        x = torch.randn(8, 10)
        gated, gates = fg(x)
        expected = x * gates
        assert torch.allclose(gated, expected, atol=1e-6)

    def test_parameter_count(self):
        fg = FeatureGating(n_features=42)
        n_params = sum(p.numel() for p in fg.parameters())
        # 42*42 (weight) + 42 (bias) = 1806
        assert n_params == 42 * 42 + 42

    def test_gradient_flows(self):
        fg = FeatureGating(n_features=10)
        x = torch.randn(4, 10, requires_grad=True)
        gated, _ = fg(x)
        loss = gated.sum()
        loss.backward()
        assert x.grad is not None
        for p in fg.parameters():
            assert p.grad is not None
