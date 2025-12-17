#!/usr/bin/env python3
"""
LSTM Encoder-Decoder for Meteorological Data Anomaly Detection

This model learns normal wind patterns from "good" quality data and detects
anomalies (bad/suspect) based on reconstruction error.

Architecture:
    - Encoder: Bidirectional LSTM compresses time series to latent space
    - Decoder: LSTM reconstructs time series from latent representation
    - Anomaly Score: Mean Squared Error between input and reconstruction

Training Strategy:
    - Train ONLY on "good" class data
    - Validate on held-out "good" data
    - Test on ALL classes to evaluate anomaly detection

Author: LANL Meteorological Data Analysis
Date: December 2025
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from multiprocessing import Pool, cpu_count
from functools import partial

# Deep Learning
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, precision_recall_curve, f1_score,
    roc_curve, average_precision_score
)
from tqdm import tqdm

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import clear_output
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving
plt.ion()  # Enable interactive mode for live updates

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration parameters"""
    
    # Data paths
    DATA_FILE = Path("/home/jose/ONRL_ENCODER_DECODER_CAPARS/aligned_5min_met_data.parquet")
    OUTPUT_DIR = Path("/home/jose/ONRL_ENCODER_DECODER_CAPARS/model_outputs_lstm")
    
    # Features
    FEATURES = ['wind direction', 'wind speed']
    TARGET_COLUMN = 'class'
    
    # Sequence parameters
    SEQUENCE_LENGTH = 24      # 24 steps × 5min = 2 hours of data
    STRIDE = 12               # 50% overlap between sequences
    MAX_NAN_RATIO = 0.3       # Max 30% NaN allowed per sequence
    
    # Model architecture
    INPUT_DIM = len(FEATURES)
    HIDDEN_DIM = 64           # LSTM hidden units
    LATENT_DIM = 32           # Bottleneck dimension
    NUM_LAYERS = 2            # Number of LSTM layers
    DROPOUT = 0.2             # Dropout rate
    BIDIRECTIONAL = True      # Use bidirectional encoder
    
    # Training parameters
    BATCH_SIZE = 64
    EPOCHS = 100
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5       # L2 regularization
    EARLY_STOPPING_PATIENCE = 15
    LR_SCHEDULER_PATIENCE = 5
    LR_SCHEDULER_FACTOR = 0.5
    
    # Data split (from GOOD data only)
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    RANDOM_SEED = 42
    
    # K-Fold Cross Validation
    USE_KFOLD = True           # Set to True to enable K-Fold CV
    N_FOLDS = 5               # Number of folds for cross-validation
    
    # Live plotting
    LIVE_PLOT = True          # Enable real-time training plots
    PLOT_INTERVAL = 1         # Update plot every N epochs
    
    # Anomaly detection
    THRESHOLD_PERCENTILE = 95  # Default threshold at 95th percentile of good data errors
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # GPU optimizations
    PIN_MEMORY = torch.cuda.is_available()  # Speed up CPU to GPU transfer
    NUM_WORKERS = 4 if torch.cuda.is_available() else 0  # Parallel data loading
    CUDNN_BENCHMARK = True  # Enable cuDNN auto-tuner for faster convolutions
    
    @classmethod
    def to_dict(cls) -> dict:
        """Convert config to dictionary for saving"""
        return {
            k: str(v) if isinstance(v, (Path, torch.device)) else v
            for k, v in vars(cls).items()
            if not k.startswith('_') and not callable(getattr(cls, k))
        }


# ============================================================================
# DATA PREPARATION
# ============================================================================

class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series sequences"""
    
    def __init__(self, sequences: np.ndarray, labels: Optional[np.ndarray] = None):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = labels
        
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.sequences[idx]


def load_data(config: Config) -> pd.DataFrame:
    """Load aligned parquet data"""
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    
    df = pd.read_parquet(config.DATA_FILE)
    
    print(f"Data file: {config.DATA_FILE}")
    print(f"Total rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nClass distribution:")
    print(df['class'].value_counts().to_string())
    
    return df


def process_station_sequences(args: Tuple) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process sequences for a single station using vectorized operations.
    Used for parallel processing.
    """
    station, station_df, features, seq_length, stride, max_nan_ratio, class_map = args
    
    # Sort by datetime
    if 'datetime' in station_df.columns:
        station_df = station_df.sort_values('datetime')
    else:
        station_df = station_df.sort_values('timestamp string')
    
    if len(station_df) < seq_length:
        return np.array([]), np.array([]), np.array([])
    
    # Extract features and labels as numpy arrays
    feature_data = station_df[features].values.astype(np.float32)
    label_data = station_df['class'].map(class_map).values
    
    n_samples = len(feature_data)
    n_features = len(features)
    
    # Calculate number of sequences using vectorized indexing
    n_sequences = (n_samples - seq_length) // stride + 1
    
    if n_sequences <= 0:
        return np.array([]), np.array([]), np.array([])
    
    # Create index array for all sequences at once (vectorized)
    starts = np.arange(0, n_sequences * stride, stride)
    
    # Use stride_tricks for memory-efficient view of sequences
    # This creates a view without copying data
    shape = (n_sequences, seq_length, n_features)
    strides_bytes = (stride * feature_data.strides[0], feature_data.strides[0], feature_data.strides[1])
    
    try:
        sequences_view = np.lib.stride_tricks.as_strided(
            feature_data, shape=shape, strides=strides_bytes
        )
        # Make a copy since we'll modify it
        sequences = sequences_view.copy()
    except:
        # Fallback to regular indexing if stride_tricks fails
        sequences = np.array([feature_data[i:i+seq_length] for i in starts])
    
    # Similarly for labels
    label_shape = (n_sequences, seq_length)
    label_strides = (stride * label_data.strides[0], label_data.strides[0])
    
    try:
        labels_view = np.lib.stride_tricks.as_strided(
            label_data, shape=label_shape, strides=label_strides
        )
        seq_labels_all = labels_view.copy()
    except:
        seq_labels_all = np.array([label_data[i:i+seq_length] for i in starts])
    
    # Vectorized NaN ratio check
    nan_counts = np.isnan(sequences).sum(axis=(1, 2))
    total_elements = seq_length * n_features
    nan_ratios = nan_counts / total_elements
    valid_mask = nan_ratios <= max_nan_ratio
    
    # Filter sequences
    sequences = sequences[valid_mask]
    seq_labels_all = seq_labels_all[valid_mask]
    
    if len(sequences) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Vectorized NaN filling using pandas (per sequence)
    # Fill NaN: forward fill, backward fill, then mean
    filled_sequences = []
    final_labels = []
    
    for i in range(len(sequences)):
        seq = sequences[i]
        seq_df = pd.DataFrame(seq, columns=features)
        seq_df = seq_df.ffill().bfill()
        
        # Fill remaining NaN with column mean
        for col in seq_df.columns:
            if seq_df[col].isna().any():
                col_mean = seq_df[col].mean()
                if np.isnan(col_mean):
                    col_mean = 0.0
                seq_df[col] = seq_df[col].fillna(col_mean)
        
        seq_filled = seq_df.values
        
        # Skip if still has NaN
        if np.isnan(seq_filled).any():
            continue
        
        filled_sequences.append(seq_filled)
        
        # Determine sequence label (vectorized)
        seq_label_arr = seq_labels_all[i]
        if 1 in seq_label_arr:
            final_labels.append(1)  # bad
        elif 2 in seq_label_arr:
            final_labels.append(2)  # suspect
        else:
            final_labels.append(0)  # good
    
    if len(filled_sequences) == 0:
        return np.array([]), np.array([]), np.array([])
    
    sequences_out = np.array(filled_sequences, dtype=np.float32)
    labels_out = np.array(final_labels)
    stations_out = np.array([station] * len(sequences_out))
    
    return sequences_out, labels_out, stations_out


def create_sequences(
    df: pd.DataFrame,
    config: Config
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create time series sequences for each station using parallel processing.
    
    Returns:
        sequences: (n_sequences, seq_length, n_features)
        labels: (n_sequences,) - 0=good, 1=bad, 2=suspect
        metadata: (n_sequences,) - station IDs
    """
    print("\n" + "=" * 70)
    print("CREATING SEQUENCES (Parallel + Vectorized)")
    print("=" * 70)
    
    class_map = {'good': 0, 'bad': 1, 'suspect': 2}
    
    stations = df['station'].unique()
    print(f"Processing {len(stations)} stations using {cpu_count()} CPU cores...")
    
    # Prepare arguments for parallel processing
    args_list = [
        (station, df[df['station'] == station].copy(), 
         config.FEATURES, config.SEQUENCE_LENGTH, config.STRIDE, 
         config.MAX_NAN_RATIO, class_map)
        for station in stations
    ]
    
    # Process in parallel
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(
            pool.imap(process_station_sequences, args_list),
            total=len(stations),
            desc="Creating sequences"
        ))
    
    # Combine results
    all_sequences = []
    all_labels = []
    all_stations = []
    
    for seq, lab, sta in results:
        if len(seq) > 0:
            all_sequences.append(seq)
            all_labels.append(lab)
            all_stations.append(sta)
    
    if not all_sequences:
        raise ValueError("No valid sequences created!")
    
    sequences = np.concatenate(all_sequences, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    stations = np.concatenate(all_stations, axis=0)
    
    print(f"\nSequences created: {len(sequences):,}")
    print(f"Sequence shape: {sequences.shape}")
    print(f"\nLabel distribution:")
    for label, name in [(0, 'Good'), (1, 'Bad'), (2, 'Suspect')]:
        count = (labels == label).sum()
        pct = count / len(labels) * 100 if len(labels) > 0 else 0
        print(f"  {name}: {count:,} ({pct:.1f}%)")
    
    return sequences, labels, stations


def prepare_data_splits(
    sequences: np.ndarray,
    labels: np.ndarray,
    config: Config
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Split data for training approach:
    - Train/Val: ONLY good data
    - Test: ALL data (for anomaly detection evaluation)
    """
    print("\n" + "=" * 70)
    print("PREPARING DATA SPLITS")
    print("=" * 70)
    
    # Separate good from anomaly data
    good_mask = labels == 0
    good_sequences = sequences[good_mask]
    good_labels = labels[good_mask]
    
    anomaly_mask = labels > 0
    anomaly_sequences = sequences[anomaly_mask]
    anomaly_labels = labels[anomaly_mask]
    
    print(f"Good sequences: {len(good_sequences):,}")
    print(f"Anomaly sequences: {len(anomaly_sequences):,}")
    
    # Split good data into train/val/test
    # First split: train vs (val + test)
    train_seq, temp_seq, train_labels, temp_labels = train_test_split(
        good_sequences, good_labels,
        test_size=(config.VAL_RATIO + config.TEST_RATIO),
        random_state=config.RANDOM_SEED
    )
    
    # Second split: val vs test
    val_ratio_adjusted = config.VAL_RATIO / (config.VAL_RATIO + config.TEST_RATIO)
    val_seq, test_good_seq, val_labels, test_good_labels = train_test_split(
        temp_seq, temp_labels,
        test_size=(1 - val_ratio_adjusted),
        random_state=config.RANDOM_SEED
    )
    
    # Combine test set: good test + ALL anomaly data
    test_sequences = np.concatenate([test_good_seq, anomaly_sequences])
    test_labels = np.concatenate([test_good_labels, anomaly_labels])
    
    # Shuffle test set
    test_indices = np.random.RandomState(config.RANDOM_SEED).permutation(len(test_sequences))
    test_sequences = test_sequences[test_indices]
    test_labels = test_labels[test_indices]
    
    print(f"\nData splits:")
    print(f"  Training:   {len(train_seq):,} sequences (100% good)")
    print(f"  Validation: {len(val_seq):,} sequences (100% good)")
    print(f"  Testing:    {len(test_sequences):,} sequences")
    print(f"    - Good:    {(test_labels == 0).sum():,}")
    print(f"    - Bad:     {(test_labels == 1).sum():,}")
    print(f"    - Suspect: {(test_labels == 2).sum():,}")
    
    return {
        'train': (train_seq, train_labels),
        'val': (val_seq, val_labels),
        'test': (test_sequences, test_labels)
    }


def normalize_data(
    train_seq: np.ndarray,
    val_seq: np.ndarray,
    test_seq: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Normalize data using StandardScaler fit on training data only"""
    
    print("\nNormalizing data...")
    
    n_train, seq_len, n_features = train_seq.shape
    
    # Fit scaler on training data
    scaler = StandardScaler()
    train_flat = train_seq.reshape(-1, n_features)
    scaler.fit(train_flat)
    
    # Transform all splits
    train_normalized = scaler.transform(train_flat).reshape(n_train, seq_len, n_features)
    
    n_val = val_seq.shape[0]
    val_flat = val_seq.reshape(-1, n_features)
    val_normalized = scaler.transform(val_flat).reshape(n_val, seq_len, n_features)
    
    n_test = test_seq.shape[0]
    test_flat = test_seq.reshape(-1, n_features)
    test_normalized = scaler.transform(test_flat).reshape(n_test, seq_len, n_features)
    
    print(f"Scaler mean: {scaler.mean_}")
    print(f"Scaler std:  {scaler.scale_}")
    
    return train_normalized, val_normalized, test_normalized, scaler


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class Encoder(nn.Module):
    """
    LSTM Encoder: Compresses time series to latent representation
    
    Input:  (batch, seq_len, input_dim)
    Output: (batch, latent_dim)
    """
    
    def __init__(self, config: Config):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=config.INPUT_DIM,
            hidden_size=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            batch_first=True,
            dropout=config.DROPOUT if config.NUM_LAYERS > 1 else 0,
            bidirectional=config.BIDIRECTIONAL
        )
        
        # Account for bidirectional
        lstm_output_dim = config.HIDDEN_DIM * (2 if config.BIDIRECTIONAL else 1)
        
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim, config.LATENT_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        _, (hidden, _) = self.lstm(x)
        
        # hidden: (num_layers * num_directions, batch, hidden_dim)
        if self.lstm.bidirectional:
            # Concatenate forward and backward final hidden states
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        
        # Project to latent space
        latent = self.fc(hidden)
        return latent


class Decoder(nn.Module):
    """
    LSTM Decoder: Reconstructs time series from latent representation
    
    Input:  (batch, latent_dim)
    Output: (batch, seq_len, input_dim)
    """
    
    def __init__(self, config: Config):
        super().__init__()
        
        self.seq_length = config.SEQUENCE_LENGTH
        self.hidden_dim = config.HIDDEN_DIM
        
        # Expand latent to hidden
        self.fc = nn.Sequential(
            nn.Linear(config.LATENT_DIM, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT)
        )
        
        self.lstm = nn.LSTM(
            input_size=config.HIDDEN_DIM,
            hidden_size=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            batch_first=True,
            dropout=config.DROPOUT if config.NUM_LAYERS > 1 else 0
        )
        
        self.output = nn.Linear(config.HIDDEN_DIM, config.INPUT_DIM)
        
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        # latent: (batch, latent_dim)
        
        # Expand to hidden dimension
        hidden = self.fc(latent)
        
        # Repeat for each time step
        hidden = hidden.unsqueeze(1).repeat(1, self.seq_length, 1)
        
        # Decode sequence
        output, _ = self.lstm(hidden)
        
        # Project to output dimension
        output = self.output(output)
        
        return output


class LSTMAutoencoder(nn.Module):
    """
    Complete LSTM Autoencoder for Anomaly Detection
    
    Architecture:
        Input -> Encoder (BiLSTM) -> Latent Space -> Decoder (LSTM) -> Reconstruction
    
    Anomaly Detection:
        - Train on normal data only
        - Anomalies have higher reconstruction error
    """
    
    def __init__(self, config: Config):
        super().__init__()
        
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.config = config
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
    
    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation for analysis"""
        return self.encoder(x)
    
    def compute_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Compute MSE reconstruction error per sample"""
        reconstructed = self.forward(x)
        mse = ((x - reconstructed) ** 2).mean(dim=(1, 2))
        return mse


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        batch = batch.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        reconstructed = model(batch)
        loss = criterion(reconstructed, batch)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device, non_blocking=True)
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


class LivePlotter:
    """Real-time training loss plotter"""
    
    def __init__(self, output_dir: Path, title: str = "Training Progress"):
        self.output_dir = output_dir
        self.title = title
        self.fig = None
        self.axes = None
        
    def initialize(self):
        """Initialize the plot"""
        self.fig, self.axes = plt.subplots(1, 3, figsize=(18, 5))
        self.fig.suptitle(self.title, fontsize=14, fontweight='bold')
        
    def update(self, history: Dict, epoch: int, patience: int, max_patience: int, best_val_loss: float):
        """Update the plot with current training state"""
        if self.fig is None:
            self.initialize()
        
        for ax in self.axes:
            ax.clear()
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss curves
        ax = self.axes[0]
        ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax.axhline(best_val_loss, color='green', linestyle='--', alpha=0.7, label=f'Best Val: {best_val_loss:.6f}')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Loss (MSE)', fontsize=11)
        ax.set_title(f'Loss Curves (Epoch {epoch})', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, max(len(epochs), 10))
        
        # Learning rate
        ax = self.axes[1]
        ax.plot(epochs, history['lr'], 'g-', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Learning Rate', fontsize=11)
        ax.set_title('Learning Rate Schedule', fontsize=12)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, max(len(epochs), 10))
        
        # Early stopping progress bar
        ax = self.axes[2]
        colors = ['green' if patience < max_patience * 0.5 else 'orange' if patience < max_patience * 0.8 else 'red']
        ax.barh(['Patience'], [patience], color=colors, height=0.5)
        ax.barh(['Patience'], [max_patience], color='lightgray', height=0.5, alpha=0.3)
        ax.set_xlim(0, max_patience)
        ax.set_xlabel('Epochs without improvement', fontsize=11)
        ax.set_title(f'Early Stopping: {patience}/{max_patience}', fontsize=12)
        ax.text(max_patience/2, 0, f'{patience}/{max_patience}', ha='center', va='center', fontsize=14, fontweight='bold')
        
        # Add current stats as text
        stats_text = f"Epoch: {epoch}\nTrain Loss: {history['train_loss'][-1]:.6f}\nVal Loss: {history['val_loss'][-1]:.6f}\nLR: {history['lr'][-1]:.2e}"
        self.fig.text(0.98, 0.02, stats_text, fontsize=10, ha='right', va='bottom', 
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save current state
        self.fig.savefig(self.output_dir / 'training_live.png', dpi=100, bbox_inches='tight')
        
    def close(self):
        """Close the plot"""
        if self.fig is not None:
            plt.close(self.fig)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Config,
    fold: int = None
) -> Dict[str, List[float]]:
    """
    Full training loop with early stopping, learning rate scheduling, and live plotting
    """
    fold_str = f" (Fold {fold})" if fold is not None else ""
    print("\n" + "=" * 70)
    print(f"TRAINING MODEL{fold_str}")
    print("=" * 70)
    
    device = config.DEVICE
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.LR_SCHEDULER_FACTOR,
        patience=config.LR_SCHEDULER_PATIENCE
    )
    
    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Initialize live plotter
    live_plotter = None
    if config.LIVE_PLOT:
        title = f"Training Progress{fold_str}"
        live_plotter = LivePlotter(config.OUTPUT_DIR, title)
    
    print(f"\nDevice: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training samples: {len(train_loader.dataset):,}")
    print(f"Validation samples: {len(val_loader.dataset):,}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Early stopping patience: {config.EARLY_STOPPING_PATIENCE}")
    print(f"LR scheduler patience: {config.LR_SCHEDULER_PATIENCE}")
    print(f"Live plotting: {'Enabled' if config.LIVE_PLOT else 'Disabled'}")
    print()
    
    # Progress bar for epochs
    pbar = tqdm(range(config.EPOCHS), desc="Training", unit="epoch")
    
    for epoch in pbar:
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, device)
        
        # Record history
        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)
        
        # Check for LR reduction
        old_lr = current_lr
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < old_lr:
            print(f"\n📉 LR reduced: {old_lr:.2e} → {new_lr:.2e}")
        
        # Early stopping
        if val_loss < best_val_loss:
            improvement = best_val_loss - val_loss
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            best_epoch = epoch + 1
        else:
            patience_counter += 1
        
        # Update progress bar
        pbar.set_postfix({
            'train': f'{train_loss:.5f}',
            'val': f'{val_loss:.5f}',
            'best': f'{best_val_loss:.5f}',
            'lr': f'{new_lr:.1e}',
            'pat': f'{patience_counter}/{config.EARLY_STOPPING_PATIENCE}'
        })
        
        # Update live plot
        if live_plotter and (epoch + 1) % config.PLOT_INTERVAL == 0:
            live_plotter.update(history, epoch + 1, patience_counter, 
                              config.EARLY_STOPPING_PATIENCE, best_val_loss)
        
        # Early stopping check
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"\n\n⚠️  Early stopping triggered at epoch {epoch+1}")
            print(f"   Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
            break
    
    # Close live plotter
    if live_plotter:
        live_plotter.update(history, epoch + 1, patience_counter,
                           config.EARLY_STOPPING_PATIENCE, best_val_loss)
        live_plotter.close()
    
    # Restore best model
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"\n✓ Restored best model from epoch {best_epoch} (val_loss: {best_val_loss:.6f})")
    
    return history


# ============================================================================
# ANOMALY DETECTION
# ============================================================================

def compute_reconstruction_errors(
    model: nn.Module,
    sequences: np.ndarray,
    device: torch.device,
    batch_size: int = 256
) -> np.ndarray:
    """Compute reconstruction error for each sequence"""
    model.eval()
    errors = []
    
    dataset = TimeSeriesDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                           pin_memory=torch.cuda.is_available())
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device, non_blocking=True)
            mse = model.compute_reconstruction_error(batch)
            errors.extend(mse.cpu().numpy())
    
    return np.array(errors)


def find_optimal_threshold(
    val_errors: np.ndarray,
    test_errors: np.ndarray,
    test_labels: np.ndarray,
    config: Config
) -> Tuple[float, Dict]:
    """
    Find optimal anomaly detection threshold
    
    Methods:
    1. Percentile of validation (good) data errors
    2. Maximize F1 score on test data
    """
    print("\n" + "=" * 70)
    print("FINDING OPTIMAL THRESHOLD")
    print("=" * 70)
    
    # Binary labels for anomaly detection
    binary_labels = (test_labels > 0).astype(int)
    
    results = {}
    
    # Method 1: Percentile-based threshold
    threshold_pct = np.percentile(val_errors, config.THRESHOLD_PERCENTILE)
    preds_pct = (test_errors > threshold_pct).astype(int)
    f1_pct = f1_score(binary_labels, preds_pct)
    
    results['percentile'] = {
        'threshold': threshold_pct,
        'percentile': config.THRESHOLD_PERCENTILE,
        'f1': f1_pct
    }
    print(f"\n{config.THRESHOLD_PERCENTILE}th percentile threshold: {threshold_pct:.6f} (F1={f1_pct:.4f})")
    
    # Method 2: Maximize F1 score
    best_f1 = 0
    best_threshold = threshold_pct
    
    # Try different percentiles
    for pct in range(80, 100):
        thresh = np.percentile(val_errors, pct)
        preds = (test_errors > thresh).astype(int)
        f1 = f1_score(binary_labels, preds)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
            best_pct = pct
    
    results['optimal_f1'] = {
        'threshold': best_threshold,
        'percentile': best_pct,
        'f1': best_f1
    }
    print(f"Optimal F1 threshold: {best_threshold:.6f} at {best_pct}th percentile (F1={best_f1:.4f})")
    
    # Use optimal F1 threshold
    return best_threshold, results


def evaluate_anomaly_detection(
    test_errors: np.ndarray,
    test_labels: np.ndarray,
    threshold: float
) -> Dict:
    """
    Comprehensive evaluation of anomaly detection performance
    """
    print("\n" + "=" * 70)
    print("ANOMALY DETECTION EVALUATION")
    print("=" * 70)
    
    # Binary classification: Good (0) vs Anomaly (1, 2)
    binary_labels = (test_labels > 0).astype(int)
    predictions = (test_errors > threshold).astype(int)
    
    # Classification metrics
    print(f"\nThreshold: {threshold:.6f}")
    print(f"\nClassification Report:")
    print(classification_report(
        binary_labels, predictions,
        target_names=['Good (Normal)', 'Anomaly (Bad/Suspect)'],
        digits=4
    ))
    
    # Confusion Matrix
    cm = confusion_matrix(binary_labels, predictions)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"Confusion Matrix:")
    print(f"                    Predicted")
    print(f"                Normal   Anomaly")
    print(f"Actual Normal   {tn:6d}   {fp:6d}")
    print(f"       Anomaly  {fn:6d}   {tp:6d}")
    
    # Additional metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nDetailed Metrics:")
    print(f"  Accuracy:    {accuracy:.4f}")
    print(f"  Precision:   {precision:.4f}")
    print(f"  Recall:      {recall:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  F1 Score:    {f1:.4f}")
    
    # ROC-AUC
    if len(np.unique(binary_labels)) > 1:
        auc_roc = roc_auc_score(binary_labels, test_errors)
        auc_pr = average_precision_score(binary_labels, test_errors)
        print(f"  ROC-AUC:     {auc_roc:.4f}")
        print(f"  PR-AUC:      {auc_pr:.4f}")
    else:
        auc_roc = None
        auc_pr = None
    
    # Per-class error statistics
    print(f"\n--- Reconstruction Error by Class ---")
    for label, name in [(0, 'Good'), (1, 'Bad'), (2, 'Suspect')]:
        mask = test_labels == label
        if mask.sum() > 0:
            class_errors = test_errors[mask]
            detected = (class_errors > threshold).sum()
            print(f"  {name:8s}: n={mask.sum():6d} | "
                  f"mean={class_errors.mean():.6f} | "
                  f"std={class_errors.std():.6f} | "
                  f"detected={detected:5d} ({detected/mask.sum()*100:.1f}%)")
    
    return {
        'threshold': threshold,
        'confusion_matrix': cm,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'roc_auc': auc_roc,
        'pr_auc': auc_pr,
        'predictions': predictions,
        'binary_labels': binary_labels
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_history(history: Dict, output_dir: Path):
    """Plot training curves"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    ax = axes[0]
    ax.plot(history['train_loss'], label='Training Loss', linewidth=2)
    ax.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Learning rate
    ax = axes[1]
    ax.plot(history['lr'], linewidth=2, color='green')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'training_history.png'}")


def plot_kfold_comparison(all_histories: List[Dict], fold_results: List[Dict], output_dir: Path):
    """Plot comparison of all K-Fold training runs"""
    n_folds = len(all_histories)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Colors for each fold
    colors = plt.cm.tab10(np.linspace(0, 1, n_folds))
    
    # Plot 1: Training loss for all folds
    ax = axes[0, 0]
    for i, history in enumerate(all_histories):
        ax.plot(history['train_loss'], color=colors[i], label=f'Fold {i+1}', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Training Loss (MSE)', fontsize=11)
    ax.set_title('Training Loss Across Folds', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Validation loss for all folds
    ax = axes[0, 1]
    for i, history in enumerate(all_histories):
        ax.plot(history['val_loss'], color=colors[i], label=f'Fold {i+1}', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Validation Loss (MSE)', fontsize=11)
    ax.set_title('Validation Loss Across Folds', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Final validation loss bar chart
    ax = axes[1, 0]
    val_losses = [r['final_val_loss'] for r in fold_results]
    fold_nums = [r['fold'] for r in fold_results]
    bars = ax.bar(fold_nums, val_losses, color=colors[:n_folds], edgecolor='black', linewidth=1)
    ax.axhline(np.mean(val_losses), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(val_losses):.6f}')
    ax.set_xlabel('Fold', fontsize=11)
    ax.set_ylabel('Final Validation Loss', fontsize=11)
    ax.set_title('Final Validation Loss by Fold', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, val_losses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{val:.5f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    # Create summary text
    summary_text = f"""
    K-Fold Cross-Validation Summary
    {'='*40}
    
    Number of Folds: {n_folds}
    
    Validation Loss Statistics:
      • Mean:    {np.mean(val_losses):.6f}
      • Std:     {np.std(val_losses):.6f}
      • Min:     {np.min(val_losses):.6f} (Fold {np.argmin(val_losses)+1})
      • Max:     {np.max(val_losses):.6f} (Fold {np.argmax(val_losses)+1})
    
    Epochs Trained:
      • Mean:    {np.mean([r['epochs_trained'] for r in fold_results]):.1f}
      • Range:   {min([r['epochs_trained'] for r in fold_results])} - {max([r['epochs_trained'] for r in fold_results])}
    """
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('K-Fold Cross-Validation Results', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'kfold_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'kfold_comparison.png'}")


def plot_error_distributions(
    val_errors: np.ndarray,
    test_errors: np.ndarray,
    test_labels: np.ndarray,
    threshold: float,
    output_dir: Path
):
    """Plot reconstruction error distributions"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram by class
    ax = axes[0]
    colors = {'Good': 'green', 'Bad': 'red', 'Suspect': 'orange'}
    
    for label, name in [(0, 'Good'), (1, 'Bad'), (2, 'Suspect')]:
        mask = test_labels == label
        if mask.sum() > 0:
            ax.hist(test_errors[mask], bins=50, alpha=0.5, 
                   label=f'{name} (n={mask.sum()})', color=colors[name])
    
    ax.axvline(threshold, color='black', linestyle='--', linewidth=2,
               label=f'Threshold={threshold:.4f}')
    ax.set_xlabel('Reconstruction Error (MSE)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Reconstruction Error Distribution by Class', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Box plot
    ax = axes[1]
    data = []
    labels_list = []
    box_colors = []
    
    for label, name, color in [(0, 'Good', 'lightgreen'), (1, 'Bad', 'lightcoral'), (2, 'Suspect', 'lightyellow')]:
        mask = test_labels == label
        if mask.sum() > 0:
            data.append(test_errors[mask])
            labels_list.append(name)
            box_colors.append(color)
    
    bp = ax.boxplot(data, labels=labels_list, patch_artist=True)
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
    
    ax.axhline(threshold, color='black', linestyle='--', linewidth=2, label='Threshold')
    ax.set_ylabel('Reconstruction Error (MSE)', fontsize=12)
    ax.set_title('Reconstruction Error Box Plot by Class', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'error_distributions.png'}")


def plot_roc_pr_curves(
    test_errors: np.ndarray,
    test_labels: np.ndarray,
    output_dir: Path
):
    """Plot ROC and Precision-Recall curves"""
    binary_labels = (test_labels > 0).astype(int)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ROC Curve
    ax = axes[0]
    fpr, tpr, _ = roc_curve(binary_labels, test_errors)
    auc_roc = roc_auc_score(binary_labels, test_errors)
    
    ax.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC={auc_roc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Precision-Recall Curve
    ax = axes[1]
    precision, recall, _ = precision_recall_curve(binary_labels, test_errors)
    auc_pr = average_precision_score(binary_labels, test_errors)
    
    ax.plot(recall, precision, linewidth=2, label=f'PR (AUC={auc_pr:.4f})')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_pr_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'roc_pr_curves.png'}")


def plot_confusion_matrix_heatmap(cm: np.ndarray, output_dir: Path):
    """Plot confusion matrix as heatmap"""
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Normal', 'Anomaly'],
        yticklabels=['Normal', 'Anomaly'],
        annot_kws={'size': 14}
    )
    
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'confusion_matrix.png'}")


def plot_reconstruction_examples(
    model: nn.Module,
    test_sequences: np.ndarray,
    test_labels: np.ndarray,
    scaler: StandardScaler,
    config: Config,
    output_dir: Path,
    n_examples: int = 4
):
    """Plot reconstruction examples for each class"""
    model.eval()
    device = config.DEVICE
    
    fig, axes = plt.subplots(3, n_examples, figsize=(16, 10))
    
    for class_idx, (label, name) in enumerate([(0, 'Good'), (1, 'Bad'), (2, 'Suspect')]):
        mask = test_labels == label
        if mask.sum() == 0:
            continue
        
        indices = np.where(mask)[0][:n_examples]
        
        for i, idx in enumerate(indices):
            if i >= n_examples:
                break
                
            seq = test_sequences[idx:idx+1]
            seq_tensor = torch.FloatTensor(seq).to(device)
            
            with torch.no_grad():
                reconstructed = model(seq_tensor).cpu().numpy()[0]
            
            original = seq[0]
            
            # Inverse transform
            original_inv = scaler.inverse_transform(original)
            reconstructed_inv = scaler.inverse_transform(reconstructed)
            
            # Plot wind speed (index 1)
            ax = axes[class_idx, i]
            time_steps = range(len(original_inv))
            
            ax.plot(time_steps, original_inv[:, 1], 'b-', 
                   label='Original', linewidth=2, alpha=0.8)
            ax.plot(time_steps, reconstructed_inv[:, 1], 'r--', 
                   label='Reconstructed', linewidth=2, alpha=0.8)
            
            mse = ((original - reconstructed) ** 2).mean()
            ax.set_title(f'{name} - MSE: {mse:.4f}', fontsize=11)
            
            if i == 0:
                ax.set_ylabel('Wind Speed', fontsize=10)
            if class_idx == 2:
                ax.set_xlabel('Time Step', fontsize=10)
            
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3)
    
    plt.suptitle('Reconstruction Examples by Class', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'reconstruction_examples.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'reconstruction_examples.png'}")


# ============================================================================
# SAVE / LOAD
# ============================================================================

def save_model_and_artifacts(
    model: nn.Module,
    scaler: StandardScaler,
    config: Config,
    history: Dict,
    threshold: float,
    eval_results: Dict,
    output_dir: Path
):
    """Save model, scaler, config, and results"""
    print("\n" + "=" * 70)
    print("SAVING MODEL AND ARTIFACTS")
    print("=" * 70)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_dir / 'lstm_autoencoder.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.to_dict()
    }, model_path)
    print(f"  Model saved: {model_path}")
    
    # Save scaler
    scaler_path = output_dir / 'scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  Scaler saved: {scaler_path}")
    
    # Save training history
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  History saved: {history_path}")
    
    # Save results - convert numpy types to Python types for JSON serialization
    results = {
        'threshold': float(threshold),
        'accuracy': float(eval_results['accuracy']),
        'precision': float(eval_results['precision']),
        'recall': float(eval_results['recall']),
        'specificity': float(eval_results['specificity']),
        'f1': float(eval_results['f1']),
        'roc_auc': float(eval_results['roc_auc']) if eval_results['roc_auc'] is not None else None,
        'pr_auc': float(eval_results['pr_auc']) if eval_results['pr_auc'] is not None else None,
        'confusion_matrix': eval_results['confusion_matrix'].tolist()
    }
    results_path = output_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved: {results_path}")
    
    # Save config
    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"  Config saved: {config_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main training and evaluation pipeline"""
    
    print("\n" + "=" * 70)
    print("LSTM ENCODER-DECODER ANOMALY DETECTION")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    config = Config()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Set random seeds
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    
    # GPU Setup
    print("\n" + "=" * 70)
    print("DEVICE CONFIGURATION")
    print("=" * 70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        torch.backends.cudnn.benchmark = config.CUDNN_BENCHMARK
        print(f"cuDNN benchmark: {config.CUDNN_BENCHMARK}")
        if config.RANDOM_SEED:
            torch.cuda.manual_seed(config.RANDOM_SEED)
    else:
        print("⚠️  No GPU detected - using CPU (training will be slower)")
    print(f"Using device: {config.DEVICE}")
    
    # 1. Load data
    df = load_data(config)
    
    # 2. Create sequences
    sequences, labels, stations = create_sequences(df, config)
    
    # 3. Prepare data splits
    data_splits = prepare_data_splits(sequences, labels, config)
    
    train_seq, train_labels = data_splits['train']
    val_seq, val_labels = data_splits['val']
    test_seq, test_labels = data_splits['test']
    
    # Check if using K-Fold Cross Validation
    if config.USE_KFOLD:
        print("\n" + "=" * 70)
        print(f"K-FOLD CROSS VALIDATION ({config.N_FOLDS} folds)")
        print("=" * 70)
        
        # Combine train and val for K-Fold
        combined_seq = np.concatenate([train_seq, val_seq], axis=0)
        combined_labels = np.concatenate([train_labels, val_labels], axis=0)
        
        kfold = KFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
        
        fold_results = []
        all_histories = []
        best_fold_model = None
        best_fold_val_loss = float('inf')
        best_fold_idx = 0
        
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(combined_seq)):
            print(f"\n{'='*70}")
            print(f"FOLD {fold_idx + 1}/{config.N_FOLDS}")
            print(f"{'='*70}")
            
            # Split data for this fold
            fold_train_seq = combined_seq[train_idx]
            fold_val_seq = combined_seq[val_idx]
            
            # Normalize data for this fold
            fold_train_norm, fold_val_norm, fold_test_norm, fold_scaler = normalize_data(
                fold_train_seq, fold_val_seq, test_seq
            )
            
            # Create data loaders
            fold_train_dataset = TimeSeriesDataset(fold_train_norm)
            fold_val_dataset = TimeSeriesDataset(fold_val_norm)
            
            fold_train_loader = DataLoader(
                fold_train_dataset,
                batch_size=config.BATCH_SIZE,
                shuffle=True,
                num_workers=config.NUM_WORKERS,
                pin_memory=config.PIN_MEMORY
            )
            fold_val_loader = DataLoader(
                fold_val_dataset,
                batch_size=config.BATCH_SIZE,
                shuffle=False,
                num_workers=config.NUM_WORKERS,
                pin_memory=config.PIN_MEMORY
            )
            
            # Create and train model for this fold
            fold_model = LSTMAutoencoder(config)
            fold_history = train_model(fold_model, fold_train_loader, fold_val_loader, config, fold=fold_idx+1)
            all_histories.append(fold_history)
            
            # Compute validation loss for this fold
            final_val_loss = min(fold_history['val_loss'])
            fold_results.append({
                'fold': fold_idx + 1,
                'final_val_loss': final_val_loss,
                'epochs_trained': len(fold_history['train_loss']),
                'min_train_loss': min(fold_history['train_loss'])
            })
            
            # Track best fold
            if final_val_loss < best_fold_val_loss:
                best_fold_val_loss = final_val_loss
                best_fold_model = fold_model
                best_fold_idx = fold_idx + 1
                best_scaler = fold_scaler
                best_test_norm = fold_test_norm
        
        # Print K-Fold summary
        print("\n" + "=" * 70)
        print("K-FOLD CROSS VALIDATION SUMMARY")
        print("=" * 70)
        
        val_losses = [r['final_val_loss'] for r in fold_results]
        print(f"\nFold Results:")
        for r in fold_results:
            marker = " *** BEST ***" if r['fold'] == best_fold_idx else ""
            print(f"  Fold {r['fold']}: Val Loss = {r['final_val_loss']:.6f}, "
                  f"Epochs = {r['epochs_trained']}{marker}")
        
        print(f"\nCross-Validation Statistics:")
        print(f"  Mean Val Loss: {np.mean(val_losses):.6f}")
        print(f"  Std Val Loss:  {np.std(val_losses):.6f}")
        print(f"  Best Fold:     {best_fold_idx} (Val Loss: {best_fold_val_loss:.6f})")
        
        # Use best fold model for evaluation
        model = best_fold_model
        scaler = best_scaler
        test_norm = best_test_norm
        history = all_histories[best_fold_idx - 1]
        
        # Plot all folds comparison
        plot_kfold_comparison(all_histories, fold_results, config.OUTPUT_DIR)
        
    else:
        # Standard train/val/test split
        print("\n" + "=" * 70)
        print("STANDARD TRAIN/VAL/TEST SPLIT")
        print("=" * 70)
        
        # 4. Normalize data
        train_norm, val_norm, test_norm, scaler = normalize_data(train_seq, val_seq, test_seq)
        
        # 5. Create data loaders
        train_dataset = TimeSeriesDataset(train_norm, train_labels)
        val_dataset = TimeSeriesDataset(val_norm, val_labels)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY
        )
        
        # 6. Create model
        model = LSTMAutoencoder(config)
        print(f"\nModel Architecture:")
        print(model)
        
        # 7. Train model
        history = train_model(model, train_loader, val_loader, config)
        
        val_norm_for_errors = val_norm
    
    # 8. Compute reconstruction errors
    print("\n" + "=" * 70)
    print("COMPUTING RECONSTRUCTION ERRORS")
    print("=" * 70)
    
    if config.USE_KFOLD:
        # For K-Fold, use the validation set from the best fold
        # Re-normalize all good data with best scaler for error computation
        good_mask = labels == 0
        good_seq = sequences[good_mask]
        n_good, seq_len, n_features = good_seq.shape
        good_flat = good_seq.reshape(-1, n_features)
        good_norm = scaler.transform(good_flat).reshape(n_good, seq_len, n_features)
        val_errors = compute_reconstruction_errors(model, good_norm, config.DEVICE)
    else:
        val_errors = compute_reconstruction_errors(model, val_norm, config.DEVICE)
    
    test_errors = compute_reconstruction_errors(model, test_norm, config.DEVICE)
    
    print(f"Validation errors: mean={val_errors.mean():.6f}, std={val_errors.std():.6f}")
    print(f"Test errors: mean={test_errors.mean():.6f}, std={test_errors.std():.6f}")
    
    # 9. Find optimal threshold
    threshold, threshold_results = find_optimal_threshold(
        val_errors, test_errors, test_labels, config
    )
    
    # 10. Evaluate anomaly detection
    eval_results = evaluate_anomaly_detection(test_errors, test_labels, threshold)
    
    # 11. Generate visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    plot_training_history(history, config.OUTPUT_DIR)
    plot_error_distributions(val_errors, test_errors, test_labels, threshold, config.OUTPUT_DIR)
    plot_roc_pr_curves(test_errors, test_labels, config.OUTPUT_DIR)
    plot_confusion_matrix_heatmap(eval_results['confusion_matrix'], config.OUTPUT_DIR)
    plot_reconstruction_examples(model, test_norm, test_labels, scaler, config, config.OUTPUT_DIR)
    
    # 12. Save model and artifacts
    save_model_and_artifacts(
        model, scaler, config, history, threshold, eval_results, config.OUTPUT_DIR
    )
    
    # 13. Print final summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nFinal Results:")
    print(f"  Threshold:   {threshold:.6f}")
    print(f"  Accuracy:    {eval_results['accuracy']:.4f}")
    print(f"  Precision:   {eval_results['precision']:.4f}")
    print(f"  Recall:      {eval_results['recall']:.4f}")
    print(f"  F1 Score:    {eval_results['f1']:.4f}")
    if eval_results['roc_auc']:
        print(f"  ROC-AUC:     {eval_results['roc_auc']:.4f}")
    if config.USE_KFOLD:
        print(f"\n  K-Fold CV:   {config.N_FOLDS} folds")
        print(f"  Best Fold:   {best_fold_idx}")
    print(f"\nOutputs saved to: {config.OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
