import torch
import numpy as np
from torch.utils.data import Dataset


class SatelliteCollisionDataset(Dataset):
    def __init__(self, features, labels, seq_length=240, sample_weights=None):
        self.features = features
        self.labels = labels
        self.seq_length = seq_length
        n = len(labels)
        self.sample_weights = (
            torch.FloatTensor(sample_weights) if sample_weights is not None
            else torch.ones(n)
        )
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        seq_data = self.features[idx]
        if len(seq_data) > self.seq_length:
            seq_data = seq_data[-self.seq_length:]
        elif len(seq_data) < self.seq_length:
            pad_length = self.seq_length - len(seq_data)
            seq_data = np.pad(seq_data, ((pad_length, 0), (0, 0)), mode='edge')

        features_tensor = torch.FloatTensor(seq_data)
        label_tensor = torch.FloatTensor([self.labels[idx]])
        return features_tensor, label_tensor, self.sample_weights[idx]


class HybridDataset(Dataset):
    def __init__(self, geo_seq, leaf_indices, xgb_pred, labels, seq_length=85,
                 sample_weights=None):
        self.geo_seq = geo_seq
        self.leaf_indices = leaf_indices
        self.xgb_pred = xgb_pred
        self.labels = labels
        self.seq_length = seq_length
        n = len(labels)
        self.sample_weights = (
            torch.FloatTensor(sample_weights) if sample_weights is not None
            else torch.ones(n)
        )

    def __len__(self):
        return len(self.geo_seq)

    def __getitem__(self, idx):
        seq_data = self.geo_seq[idx]
        if len(seq_data) > self.seq_length:
            seq_data = seq_data[-self.seq_length:]
        elif len(seq_data) < self.seq_length:
            pad_length = self.seq_length - len(seq_data)
            seq_data = np.pad(seq_data, ((pad_length, 0), (0, 0)), mode='edge')

        return (torch.FloatTensor(seq_data),
                torch.LongTensor(self.leaf_indices[idx]),
                torch.FloatTensor([self.xgb_pred[idx]]),
                torch.FloatTensor([self.labels[idx]]),
                self.sample_weights[idx])
