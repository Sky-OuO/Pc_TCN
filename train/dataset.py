import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.ndimage import convolve1d
from train.models import get_lds_kernel_window


class SatelliteCollisionDataset(Dataset):
    def __init__(self, features, labels, seq_length=601, sample_weights=None):
        self.features   = features
        self.labels     = labels
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
        # padding or truncating to fixed length
        if len(seq_data) > self.seq_length:
            seq_data = seq_data[-self.seq_length:]
        elif len(seq_data) < self.seq_length:
            pad_length = self.seq_length - len(seq_data)
            seq_data = np.pad(seq_data, ((pad_length, 0), (0, 0)), mode='edge')

        features_tensor = torch.FloatTensor(seq_data)
        label_tensor    = torch.FloatTensor([self.labels[idx]])
        return features_tensor, label_tensor, self.sample_weights[idx]



def compute_lds_weights(labels, num_bins=100, lds_kernel='gaussian', lds_ks=5, lds_sigma=2,
                        log_min=-10.0, log_max=0.0):
    
    log_labels = np.log10(np.maximum(labels, 1e-10))
    bin_edges           = np.linspace(log_min, log_max, num_bins + 1)
    bin_index_per_label = np.clip(np.digitize(log_labels, bin_edges) - 1, 0, num_bins - 1)

    emp_label_dist = np.zeros(num_bins, dtype=np.float32)
    for idx in bin_index_per_label:
        emp_label_dist[idx] += 1

    lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
    eff_label_dist    = convolve1d(emp_label_dist, weights=lds_kernel_window, mode='constant')

    eff_num_per_label = np.array([eff_label_dist[b] for b in bin_index_per_label], dtype=np.float32)
    weights = np.where(eff_num_per_label > 0, 1.0 / eff_num_per_label, 0.0).astype(np.float32)
    if weights.mean() > 0:
        weights /= weights.mean()
    return weights
