"""
Data utilities for loading and preprocessing data for anomaly detection
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification


class AnomalyDataset(Dataset):
    """PyTorch Dataset for anomaly detection"""
    
    def __init__(self, data, labels=None):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels) if labels is not None else None
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.data[idx], self.labels[idx]
        return self.data[idx]


def generate_synthetic_data(n_samples=1000, n_features=10, contamination=0.1, random_state=42):
    """
    Generate synthetic data for anomaly detection
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        contamination: Proportion of anomalies
        random_state: Random seed
    
    Returns:
        X: Features
        y: Labels (0=normal, 1=anomaly)
    """
    np.random.seed(random_state)
    
    # Generate normal data
    n_normal = int(n_samples * (1 - contamination))
    n_anomalies = n_samples - n_normal
    
    # Normal data: samples from a standard distribution
    X_normal = np.random.randn(n_normal, n_features)
    y_normal = np.zeros(n_normal)
    
    # Anomalies: samples from a different distribution (shifted and scaled)
    X_anomalies = np.random.randn(n_anomalies, n_features) * 3 + 5
    y_anomalies = np.ones(n_anomalies)
    
    # Combine and shuffle
    X = np.vstack([X_normal, X_anomalies])
    y = np.concatenate([y_normal, y_anomalies])
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]
    
    return X, y


def prepare_data(X, y, train_split=0.7, val_split=0.15, normalize=True):
    """
    Prepare data for training
    
    Args:
        X: Features
        y: Labels
        train_split: Proportion of training data
        val_split: Proportion of validation data
        normalize: Whether to normalize features
    
    Returns:
        Dictionary with train, val, test splits
    """
    n_samples = len(X)
    n_train = int(n_samples * train_split)
    n_val = int(n_samples * val_split)
    
    # Split data
    X_train = X[:n_train]
    y_train = y[:n_train]
    
    X_val = X[n_train:n_train + n_val]
    y_val = y[n_train:n_train + n_val]
    
    X_test = X[n_train + n_val:]
    y_test = y[n_train + n_val:]
    
    # Normalize using training data statistics
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
    else:
        scaler = None
    
    return {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test),
        'scaler': scaler
    }


def create_dataloaders(data_dict, batch_size=32):
    """
    Create PyTorch DataLoaders
    
    Args:
        data_dict: Dictionary with train, val, test data
        batch_size: Batch size
    
    Returns:
        Dictionary with DataLoaders
    """
    train_dataset = AnomalyDataset(*data_dict['train'])
    val_dataset = AnomalyDataset(*data_dict['val'])
    test_dataset = AnomalyDataset(*data_dict['test'])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
