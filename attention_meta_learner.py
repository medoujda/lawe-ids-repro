"""Attention Meta-Learner — per-sample adaptive weighting of base learners."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from feature_gating import FeatureGating
from config import (
    META_EPOCHS,
    META_PATIENCE,
    META_LR,
    META_HIDDEN_1,
    META_HIDDEN_2,
    META_DROPOUT,
)


class AttentionMetaLearner(nn.Module):
    """Per-sample attention-based meta-learner.

    Takes raw features and base-learner probabilities, produces softmax
    weights over models, and outputs a weighted combination probability.

    Architecture:
        FeatureGating(n_features)
        → concat(gated_features, model_probas)   # (n_features + n_models)
        → fc1 → ReLU → Dropout
        → fc2 → ReLU
        → fc3 → softmax weights
        → p_final = (weights * model_probas).sum(dim=1)
    """

    def __init__(self, n_features: int, n_models: int = 4):
        super().__init__()
        self.feature_gating = FeatureGating(n_features)
        self.fc1 = nn.Linear(n_features + n_models, META_HIDDEN_1)
        self.dropout = nn.Dropout(META_DROPOUT)
        self.fc2 = nn.Linear(META_HIDDEN_1, META_HIDDEN_2)
        self.fc3 = nn.Linear(META_HIDDEN_2, n_models)

    def forward(self, features: torch.Tensor, model_probas: torch.Tensor):
        """Forward pass.

        Args:
            features:     (batch, n_features) raw input features
            model_probas: (batch, n_models)   probabilities from base learners

        Returns:
            p_final: (batch,)        weighted ensemble probability
            weights: (batch, n_models) per-sample softmax model weights
            gates:   (batch, n_features) feature gate values in [0, 1]
        """
        gated, gates = self.feature_gating(features)          # (B, F), (B, F)
        x = torch.cat([gated, model_probas], dim=1)           # (B, F+M)
        x = torch.relu(self.fc1(x))                           # (B, H1)
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))                           # (B, H2)
        weights = torch.softmax(self.fc3(x), dim=1)           # (B, M)
        p_final = (weights * model_probas).sum(dim=1)         # (B,)
        return p_final, weights, gates


def train_meta_learner(
    features: np.ndarray,
    probas: np.ndarray,
    y_true: np.ndarray,
    n_features: int,
    n_models: int = 4,
    epochs: int = META_EPOCHS,
    patience: int = META_PATIENCE,
    lr: float = META_LR,
    val_split: float = 0.2,
    batch_size: int = 64,
    seed: int = 42,
):
    """Train the AttentionMetaLearner with early stopping.

    Args:
        features:   (N, n_features) numpy array
        probas:     (N, n_models)   numpy array of base-learner probabilities
        y_true:     (N,)            binary labels (0/1)
        n_features: number of input features
        n_models:   number of base learners
        epochs:     maximum training epochs
        patience:   early-stopping patience (epochs without val improvement)
        lr:         Adam learning rate
        val_split:  fraction of data for validation
        batch_size: mini-batch size
        seed:       random seed for reproducibility

    Returns:
        model:   trained AttentionMetaLearner
        history: dict with keys 'train_loss' and 'val_loss' (lists)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- split ---
    n = len(y_true)
    n_val = max(1, int(n * val_split))
    idx = np.random.permutation(n)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    def _to_tensors(indices):
        X = torch.tensor(features[indices], dtype=torch.float32)
        P = torch.tensor(probas[indices], dtype=torch.float32)
        y = torch.tensor(y_true[indices], dtype=torch.float32)
        return X, P, y

    X_tr, P_tr, y_tr = _to_tensors(train_idx)
    X_val, P_val, y_val = _to_tensors(val_idx)

    train_ds = TensorDataset(X_tr, P_tr, y_tr)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = AttentionMetaLearner(n_features=n_features, n_models=n_models)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        # --- training ---
        model.train()
        epoch_loss = 0.0
        for X_b, P_b, y_b in train_loader:
            optimizer.zero_grad()
            p_final, _, _ = model(X_b, P_b)
            p_final = p_final.clamp(1e-7, 1 - 1e-7)
            loss = criterion(p_final, y_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(y_b)
        train_loss = epoch_loss / len(y_tr)

        # --- validation ---
        model.eval()
        with torch.no_grad():
            p_val, _, _ = model(X_val, P_val)
            p_val = p_val.clamp(1e-7, 1 - 1e-7)
            val_loss = criterion(p_val, y_val).item()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # --- early stopping ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def find_best_threshold(
    model: AttentionMetaLearner,
    features: np.ndarray,
    probas: np.ndarray,
    y_true: np.ndarray,
    low: float = 0.30,
    high: float = 0.70,
    step: float = 0.01,
):
    """Grid-search the decision threshold that maximises accuracy.

    Args:
        model:    trained AttentionMetaLearner
        features: (N, n_features) numpy array
        probas:   (N, n_models)   numpy array
        y_true:   (N,)            binary labels
        low:      lower bound of threshold search range
        high:     upper bound (inclusive)
        step:     grid step size

    Returns:
        best_threshold: float
        best_accuracy:  float in [0, 1]
    """
    model.eval()
    X = torch.tensor(features, dtype=torch.float32)
    P = torch.tensor(probas, dtype=torch.float32)

    with torch.no_grad():
        p_final, _, _ = model(X, P)

    p_np = p_final.numpy()
    thresholds = np.arange(low, high + step / 2, step)

    best_threshold = low
    best_accuracy = 0.0

    for tau in thresholds:
        preds = (p_np >= tau).astype(int)
        acc = (preds == y_true).mean()
        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = tau

    return best_threshold, best_accuracy
