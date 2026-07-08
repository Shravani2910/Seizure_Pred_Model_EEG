import numpy as np
import torch
from torch.utils.data import Dataset


class EEGDataset(Dataset):
    """Wraps windowed EEG (n_windows, n_channels, n_samples) + labels for PyTorch."""

    def __init__(self, X, y, normalize=True):
        self.y = torch.tensor(y, dtype=torch.long)
        X = X.astype(np.float32)
        if normalize:
            # per-channel z-score using train-set stats should be computed outside
            # and passed in for a real setup; here we do simple per-window normalization
            mean = X.mean(axis=2, keepdims=True)
            std = X.std(axis=2, keepdims=True) + 1e-6
            X = (X - mean) / std
        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
