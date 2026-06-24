"""Feature Gating module - learns a soft mask over input features."""
import torch
import torch.nn as nn


class FeatureGating(nn.Module):
    """Learnable feature gating via sigmoid attention.

    For each sample, produces gates in [0,1] per feature.
    Output = input * gates (element-wise).
    """

    def __init__(self, n_features):
        super().__init__()
        self.gate_layer = nn.Linear(n_features, n_features)

    def forward(self, x):
        """
        Args:
            x: (batch, n_features) tensor

        Returns:
            gated: (batch, n_features) - input * gates
            gates: (batch, n_features) - gate values in [0, 1]
        """
        gates = torch.sigmoid(self.gate_layer(x))
        gated = x * gates
        return gated, gates
