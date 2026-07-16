"""
Shared data loading for baseline models (XGBoost, MLP, LSTM).
Does NOT modify any existing code.
"""
import numpy as np
from sklearn.model_selection import train_test_split

from train.data_utils import load_raw_data, engineer_features, normalize_features


def load_baseline_data(feature_path, label_path, test_size=0.2, random_state=42,
                       use_last_timestep=True, return_train=False):
    """
    Load data for baseline models.

    Returns:
        X_val:      (N_val, n_features)         flat (last-timestep) for XGBoost/MLP
        X_seq_val:  (N_val, seq_len, n_features) sequential for LSTM
        y_val:      (N_val,)                     log10 target labels
        If return_train=True, also returns X_train, X_seq_train, y_train.
    """
    features, labels = load_raw_data(feature_path, label_path)
    features = engineer_features(features)

    n = len(features)
    indices = np.arange(n)
    train_idx, val_idx = train_test_split(indices, test_size=test_size, random_state=random_state)

    # Normalize
    features = normalize_features(features, train_idx)

    # Clip labels to log10 space (same as TCN training)
    eps = 1e-10
    labels = np.clip(np.log10(np.maximum(labels, eps)), -9.0, -0.3)

    # Sequential form (for LSTM)
    X_seq_train = features[train_idx]
    X_seq_val = features[val_idx]

    # Flat form: use last timestep (closest to TCA)  (for XGBoost, MLP)
    if use_last_timestep:
        X_train = features[train_idx, -1, :]   # (N_train, n_features)
        X_val = features[val_idx, -1, :]       # (N_val, n_features)
    else:
        # Flatten the full sequence: (N, seq_len * n_features)
        X_train = features[train_idx].reshape(len(train_idx), -1)
        X_val = features[val_idx].reshape(len(val_idx), -1)

    y_train = labels[train_idx]
    y_val = labels[val_idx]

    if return_train:
        return (X_train, X_val, X_seq_train, X_seq_val, y_train, y_val)
    return (X_val, X_seq_val, y_val)
