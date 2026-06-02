import numpy as np
from sklearn.model_selection import train_test_split


def load_raw_data(feature_path, label_path):
    features = np.load(feature_path)
    labels   = np.load(label_path).astype(np.float64)
    return features, labels


def split_indices(n, test_size=0.2, random_state=42):
    indices = np.arange(n)
    return train_test_split(indices, test_size=test_size, random_state=random_state)


def normalize_features(features, train_indices):
    mean = np.mean(features[train_indices], axis=(0, 1))
    std  = np.std(features[train_indices],  axis=(0, 1))
    std[std == 0] = 1.0
    return (features - mean) / std


def engineer_features(features):
    raw_geo  = features[:, :, :-18]   # 14 geo features
    raw_unc  = features[:, :, -18:]   # 18 uncertainty features (static, no diff)
    diff_geo = np.diff(raw_geo, axis=1, prepend=raw_geo[:, :1, :])
    return np.concatenate([raw_geo, diff_geo, raw_unc], axis=2)


def load_data(feature_path, label_path, test_size=0.2, random_state=42):
    features, labels             = load_raw_data(feature_path, label_path)
    train_indices, val_indices   = split_indices(len(features), test_size, random_state)
    features                     = engineer_features(features)   # diff computed on raw scale
    features                     = normalize_features(features, train_indices)  # normalize all 46 features together
    train_features, val_features = features[train_indices], features[val_indices]
    train_labels,   val_labels   = labels[train_indices],   labels[val_indices]
    return train_features, train_labels, val_features, val_labels
