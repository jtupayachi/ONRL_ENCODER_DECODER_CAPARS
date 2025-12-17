#!/usr/bin/env python3
"""
TCN Masked Autoencoder for Meteorological Data Anomaly Detection

Improved architecture with:
    - Temporal Convolutional Network (TCN) - better than LSTM for time series
    - Masked Loss Function - properly handles missing values
    - Attention Mechanism - focuses on important time steps
    - Skip Connections - better gradient flow
    - Proper NaN handling with masking instead of imputation

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

# Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler
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
import matplotlib
matplotlib.use('Agg')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration parameters"""
    
    # Data paths
    DATA_FILE = Path("/home/jose/ONRL_ENCODER_DECODER_CAPARS/aligned_5min_met_data.parquet")
    OUTPUT_DIR = Path("/home/jose/ONRL_ENCODER_DECODER_CAPARS/model_outputs_tcn")
    
    # Features
    FEATURES = ['wind direction', 'wind speed']
    TARGET_COLUMN = 'class'
    
    # Sequence parameters
    SEQUENCE_LENGTH = 48      # 48 steps × 5min = 4 hours of data (longer context)
    STRIDE = 24               # 50% overlap between sequences
    
    # Model architecture - TCN
    INPUT_DIM = len(FEATURES)
    TCN_CHANNELS = [32, 64, 128, 64, 32]  # Channel sizes for each TCN layer
    KERNEL_SIZE = 3           # Convolution kernel size
    LATENT_DIM = 64           # Bottleneck dimension
    DROPOUT = 0.2             # Dropout rate
    USE_ATTENTION = True      # Use attention mechanism
    
    # Training parameters
    BATCH_SIZE = 128          # Larger batch for GPU
    EPOCHS = 150
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4       # L2 regularization
    EARLY_STOPPING_PATIENCE = 20
    LR_SCHEDULER_PATIENCE = 7
    LR_SCHEDULER_FACTOR = 0.5
    
    # Data split
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    RANDOM_SEED = 42
    
    # K-Fold Cross Validation
    USE_KFOLD = True
    N_FOLDS = 5
    
    # Live plotting
    LIVE_PLOT = True          # Enable real-time training plots
    PLOT_INTERVAL = 1         # Update plot every N epochs
    
    # Anomaly detection
    THRESHOLD_PERCENTILE = 95
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    PIN_MEMORY = torch.cuda.is_available()
    NUM_WORKERS = 4 if torch.cuda.is_available() else 0
    
    @classmethod
    def to_dict(cls) -> dict:
        return {
            k: str(v) if isinstance(v, (Path, torch.device)) else v
            for k, v in vars(cls).items()
            if not k.startswith('_') and not callable(getattr(cls, k))
        }


# ============================================================================
# DATA PREPARATION WITH MASKING
# ============================================================================

class MaskedTimeSeriesDataset(Dataset):
    """
    PyTorch Dataset that preserves NaN information as masks.
    Instead of imputing NaN values, we keep track of where they are.
    """
    
    def __init__(self, sequences: np.ndarray, labels: Optional[np.ndarray] = None):
        # Replace NaN with 0 but keep mask
        self.mask = ~np.isnan(sequences)  # True where valid, False where NaN
        self.sequences = np.nan_to_num(sequences, nan=0.0).astype(np.float32)
        self.labels = labels
        
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.BoolTensor(self.mask[idx])
        )


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
    
    # Check NaN statistics
    print(f"\nNaN statistics:")
    for col in config.FEATURES:
        nan_count = df[col].isna().sum()
        nan_pct = nan_count / len(df) * 100
        print(f"  {col}: {nan_count:,} NaN ({nan_pct:.2f}%)")
    
    return df


def create_sequences_with_nan(
    df: pd.DataFrame,
    config: Config
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create time series sequences PRESERVING NaN values.
    NaN values will be handled by masking during training.
    """
    print("\n" + "=" * 70)
    print("CREATING SEQUENCES (Preserving NaN for Masking)")
    print("=" * 70)
    
    class_map = {'good': 0, 'bad': 1, 'suspect': 2}
    
    all_sequences = []
    all_labels = []
    all_stations = []
    
    stations = df['station'].unique()
    print(f"Processing {len(stations)} stations...")
    
    for station in tqdm(stations, desc="Creating sequences"):
        station_df = df[df['station'] == station].copy()
        
        if 'datetime' in station_df.columns:
            station_df = station_df.sort_values('datetime')
        else:
            station_df = station_df.sort_values('timestamp string')
        
        if len(station_df) < config.SEQUENCE_LENGTH:
            continue
        
        # Extract features (KEEP NaN values)
        features = station_df[config.FEATURES].values.astype(np.float32)
        labels = station_df['class'].map(class_map).values
        
        # Create sequences with sliding window
        for i in range(0, len(features) - config.SEQUENCE_LENGTH + 1, config.STRIDE):
            seq = features[i:i + config.SEQUENCE_LENGTH]
            seq_labels = labels[i:i + config.SEQUENCE_LENGTH]
            
            # Check valid data ratio (at least 50% valid)
            valid_ratio = (~np.isnan(seq)).sum() / seq.size
            if valid_ratio < 0.5:
                continue
            
            # Determine sequence label
            if 1 in seq_labels:
                seq_label = 1  # bad
            elif 2 in seq_labels:
                seq_label = 2  # suspect
            else:
                seq_label = 0  # good
            
            all_sequences.append(seq)
            all_labels.append(seq_label)
            all_stations.append(station)
    
    sequences = np.array(all_sequences, dtype=np.float32)
    labels = np.array(all_labels)
    stations = np.array(all_stations)
    
    # Statistics
    total_values = sequences.size
    nan_values = np.isnan(sequences).sum()
    
    print(f"\nSequences created: {len(sequences):,}")
    print(f"Sequence shape: {sequences.shape}")
    print(f"NaN values preserved: {nan_values:,} ({nan_values/total_values*100:.2f}%)")
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
    """Split data for training"""
    print("\n" + "=" * 70)
    print("PREPARING DATA SPLITS")
    print("=" * 70)
    
    good_mask = labels == 0
    good_sequences = sequences[good_mask]
    good_labels = labels[good_mask]
    
    anomaly_mask = labels > 0
    anomaly_sequences = sequences[anomaly_mask]
    anomaly_labels = labels[anomaly_mask]
    
    print(f"Good sequences: {len(good_sequences):,}")
    print(f"Anomaly sequences: {len(anomaly_sequences):,}")
    
    # Split good data
    train_seq, temp_seq, train_labels, temp_labels = train_test_split(
        good_sequences, good_labels,
        test_size=(config.VAL_RATIO + config.TEST_RATIO),
        random_state=config.RANDOM_SEED
    )
    
    val_ratio_adjusted = config.VAL_RATIO / (config.VAL_RATIO + config.TEST_RATIO)
    val_seq, test_good_seq, val_labels, test_good_labels = train_test_split(
        temp_seq, temp_labels,
        test_size=(1 - val_ratio_adjusted),
        random_state=config.RANDOM_SEED
    )
    
    # Combine test set
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


def normalize_data_robust(
    train_seq: np.ndarray,
    val_seq: np.ndarray,
    test_seq: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, RobustScaler]:
    """
    Normalize data using RobustScaler (handles outliers better).
    NaN values are preserved.
    """
    print("\nNormalizing data with RobustScaler...")
    
    n_train, seq_len, n_features = train_seq.shape
    
    # Fit scaler on training data (ignoring NaN)
    scaler = RobustScaler()
    train_flat = train_seq.reshape(-1, n_features)
    
    # Fit only on non-NaN values
    valid_mask = ~np.isnan(train_flat).any(axis=1)
    scaler.fit(train_flat[valid_mask])
    
    # Transform all splits (NaN propagates)
    def transform_with_nan(data, scaler):
        n, s, f = data.shape
        flat = data.reshape(-1, f)
        result = np.full_like(flat, np.nan)
        valid = ~np.isnan(flat).any(axis=1)
        if valid.sum() > 0:
            result[valid] = scaler.transform(flat[valid])
        return result.reshape(n, s, f)
    
    train_normalized = transform_with_nan(train_seq, scaler)
    val_normalized = transform_with_nan(val_seq, scaler)
    test_normalized = transform_with_nan(test_seq, scaler)
    
    print(f"Scaler center: {scaler.center_}")
    print(f"Scaler scale:  {scaler.scale_}")
    
    return train_normalized, val_normalized, test_normalized, scaler


# ============================================================================
# LEARNABLE MASK LAYERS
# ============================================================================

class LearnableMaskLayer(nn.Module):
    """
    Learnable Mask Layer that learns to weight/mask features and time steps.
    
    This layer learns:
    1. Feature importance weights (which features are more important)
    2. Temporal attention (which time steps matter more)
    3. Combines with input NaN mask for robust handling
    """
    
    def __init__(self, input_dim: int, seq_length: int, hidden_dim: int = 32):
        super().__init__()
        
        self.input_dim = input_dim
        self.seq_length = seq_length
        
        # Feature-wise learnable mask (learns importance of each feature)
        self.feature_mask = nn.Parameter(torch.ones(input_dim))
        
        # Temporal mask network (learns importance of each time step)
        self.temporal_mask_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Confidence estimation (how confident are we about each value)
        self.confidence_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, nan_mask: torch.Tensor = None):
        """
        Args:
            x: (batch, seq_len, input_dim) - input data (NaN replaced with 0)
            nan_mask: (batch, seq_len, input_dim) - True where valid, False where was NaN
        
        Returns:
            masked_x: weighted input
            combined_mask: learned + NaN mask combined
            mask_weights: the learned mask weights for visualization
        """
        batch_size = x.shape[0]
        
        # 1. Apply feature importance mask (softmax to ensure valid weights)
        feature_weights = F.softmax(self.feature_mask, dim=0)  # (input_dim,)
        x_weighted = x * feature_weights.unsqueeze(0).unsqueeze(0)  # (batch, seq, input_dim)
        
        # 2. Compute temporal importance mask
        temporal_weights = self.temporal_mask_net(x)  # (batch, seq_len, 1)
        x_weighted = x_weighted * temporal_weights  # Apply temporal weighting
        
        # 3. Compute confidence mask (how reliable is each value)
        confidence = self.confidence_net(x)  # (batch, seq_len, input_dim)
        
        # 4. Combine learned mask with NaN mask
        if nan_mask is not None:
            nan_mask_float = nan_mask.float()
            # NaN positions get zero confidence
            combined_mask = confidence * nan_mask_float
        else:
            combined_mask = confidence
        
        # 5. Apply combined mask
        masked_x = x_weighted * combined_mask
        
        # Return mask weights for visualization/analysis
        mask_weights = {
            'feature_weights': feature_weights.detach(),
            'temporal_weights': temporal_weights.detach(),
            'confidence': confidence.detach(),
            'combined_mask': combined_mask.detach()
        }
        
        return masked_x, combined_mask, mask_weights


class InputMaskModule(nn.Module):
    """
    Input Masking Module that handles missing values more intelligently.
    
    Instead of just setting NaN to 0, this module:
    1. Learns embeddings for missing vs present values
    2. Creates a "missingness indicator" that the model can learn from
    3. Applies learnable interpolation for missing values
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        
        self.input_dim = input_dim
        
        # Missing value embedding (what to use when value is NaN)
        self.missing_embedding = nn.Parameter(torch.zeros(input_dim))
        
        # Missingness indicator projection (concatenated to features)
        self.missingness_proj = nn.Linear(input_dim, hidden_dim)
        
        # Interpolation network (learns to interpolate missing values from context)
        self.interpolation_net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),  # current + context
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x: torch.Tensor, nan_mask: torch.Tensor):
        """
        Args:
            x: (batch, seq_len, input_dim) - input with NaN replaced by 0
            nan_mask: (batch, seq_len, input_dim) - True where valid
        
        Returns:
            x_filled: data with learned interpolation for missing values
            missingness_features: features indicating where data was missing
        """
        batch_size, seq_len, _ = x.shape
        nan_mask_float = nan_mask.float()
        
        # 1. Create missingness indicator features
        missingness_indicator = 1.0 - nan_mask_float  # 1 where missing, 0 where valid
        missingness_features = self.missingness_proj(missingness_indicator)
        
        # 2. For missing values, use learned embedding as initial fill
        x_with_embedding = x * nan_mask_float + self.missing_embedding * missingness_indicator
        
        # 3. Compute local context (mean of valid neighbors)
        # Use a simple rolling mean for context
        x_padded = F.pad(x_with_embedding.transpose(1, 2), (1, 1), mode='replicate').transpose(1, 2)
        context = (x_padded[:, :-2, :] + x_padded[:, 2:, :]) / 2  # average of prev and next
        
        # 4. Learn interpolation
        interp_input = torch.cat([x_with_embedding, context], dim=-1)
        interpolated = self.interpolation_net(interp_input)
        
        # 5. Use original where valid, interpolated where missing
        x_filled = x * nan_mask_float + interpolated * missingness_indicator
        
        return x_filled, missingness_features


# ============================================================================
# TCN ARCHITECTURE WITH MASKING
# ============================================================================

class CausalConv1d(nn.Module):
    """Causal convolution - only looks at past, not future"""
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )
        
    def forward(self, x):
        # x: (batch, channels, seq_len)
        out = self.conv(x)
        # Remove future padding
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TCNBlock(nn.Module):
    """
    Temporal Convolutional Block with:
    - Dilated causal convolution
    - Residual connection
    - Weight normalization
    - Dropout
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        
        # Create causal conv layers
        self.causal_conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.causal_conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        
        # Apply weight normalization to the inner Conv1d layers
        nn.utils.parametrizations.weight_norm(self.causal_conv1.conv, name='weight')
        nn.utils.parametrizations.weight_norm(self.causal_conv2.conv, name='weight')
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
    def forward(self, x):
        # x: (batch, channels, seq_len)
        residual = x
        
        out = self.causal_conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.causal_conv2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Residual
        if self.downsample is not None:
            residual = self.downsample(residual)
        
        return self.relu(out + residual)


class TCNEncoder(nn.Module):
    """
    TCN Encoder: Compresses time series to latent representation
    Uses dilated causal convolutions for long-range dependencies
    """
    
    def __init__(self, config: Config):
        super().__init__()
        
        channels = [config.INPUT_DIM] + config.TCN_CHANNELS
        
        self.tcn_blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            dilation = 2 ** i  # Exponentially increasing dilation
            self.tcn_blocks.append(
                TCNBlock(
                    channels[i], channels[i+1],
                    config.KERNEL_SIZE, dilation, config.DROPOUT
                )
            )
        
        # Attention for weighted pooling
        if config.USE_ATTENTION:
            self.attention = nn.Sequential(
                nn.Linear(channels[-1], channels[-1] // 2),
                nn.Tanh(),
                nn.Linear(channels[-1] // 2, 1)
            )
        else:
            self.attention = None
        
        # Project to latent space
        self.fc = nn.Sequential(
            nn.Linear(channels[-1], config.LATENT_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT)
        )
        
    def forward(self, x, mask=None):
        # x: (batch, seq_len, input_dim)
        # Convert to (batch, channels, seq_len)
        x = x.transpose(1, 2)
        
        # Apply TCN blocks
        for block in self.tcn_blocks:
            x = block(x)
        
        # x: (batch, channels, seq_len)
        x = x.transpose(1, 2)  # (batch, seq_len, channels)
        
        if self.attention is not None:
            # Attention-weighted pooling
            attn_weights = self.attention(x)  # (batch, seq_len, 1)
            
            # Apply mask to attention if provided
            if mask is not None:
                # mask: (batch, seq_len, features) -> (batch, seq_len, 1)
                mask_pooled = mask.any(dim=-1, keepdim=True).float()
                attn_weights = attn_weights * mask_pooled
                attn_weights = attn_weights - 1e9 * (1 - mask_pooled)
            
            attn_weights = F.softmax(attn_weights, dim=1)
            x = (x * attn_weights).sum(dim=1)  # (batch, channels)
        else:
            # Global average pooling
            x = x.mean(dim=1)
        
        # Project to latent
        latent = self.fc(x)
        return latent


class TCNDecoder(nn.Module):
    """
    TCN Decoder: Reconstructs time series from latent representation
    Uses transposed convolutions for upsampling
    """
    
    def __init__(self, config: Config):
        super().__init__()
        
        self.seq_length = config.SEQUENCE_LENGTH
        channels = config.TCN_CHANNELS[::-1]  # Reverse channel order
        
        # Expand latent to sequence
        self.fc = nn.Sequential(
            nn.Linear(config.LATENT_DIM, channels[0] * config.SEQUENCE_LENGTH),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT)
        )
        
        self.channels_0 = channels[0]
        
        # Decoder TCN blocks (non-causal for reconstruction)
        self.tcn_blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.tcn_blocks.append(
                nn.Sequential(
                    nn.Conv1d(channels[i], channels[i+1], 3, padding=1),
                    nn.ReLU(),
                    nn.Dropout(config.DROPOUT)
                )
            )
        
        # Final output layer
        self.output = nn.Conv1d(channels[-1], config.INPUT_DIM, 1)
        
    def forward(self, latent):
        # latent: (batch, latent_dim)
        
        # Expand to sequence
        x = self.fc(latent)
        x = x.view(-1, self.channels_0, self.seq_length)  # (batch, channels, seq_len)
        
        # Decoder blocks
        for block in self.tcn_blocks:
            x = block(x)
        
        # Output
        x = self.output(x)  # (batch, input_dim, seq_len)
        x = x.transpose(1, 2)  # (batch, seq_len, input_dim)
        
        return x


class TCNAutoencoder(nn.Module):
    """
    Complete TCN Autoencoder with Learnable Masking for Anomaly Detection
    
    Key features:
    - TCN encoder with dilated causal convolutions
    - Attention-weighted encoding
    - LEARNABLE MASK LAYER - learns feature importance and confidence
    - INPUT MASK MODULE - handles missing values with learned interpolation
    - Masked loss for handling missing values
    - Skip connections throughout
    """
    
    def __init__(self, config: Config):
        super().__init__()
        
        # Learnable mask layers
        self.input_mask_module = InputMaskModule(
            input_dim=config.INPUT_DIM,
            hidden_dim=32
        )
        self.learnable_mask = LearnableMaskLayer(
            input_dim=config.INPUT_DIM,
            seq_length=config.SEQUENCE_LENGTH,
            hidden_dim=32
        )
        
        self.encoder = TCNEncoder(config)
        self.decoder = TCNDecoder(config)
        self.config = config
        
        # Output mask layer (learns to weight reconstruction confidence)
        self.output_mask = nn.Sequential(
            nn.Linear(config.INPUT_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, config.INPUT_DIM),
            nn.Sigmoid()
        )
        
    def forward(self, x, mask=None, return_masks=False):
        """
        Forward pass with learnable masking.
        
        Args:
            x: (batch, seq_len, input_dim) - input data
            mask: (batch, seq_len, input_dim) - NaN mask (True=valid)
            return_masks: whether to return mask information for visualization
        """
        # 1. Apply input mask module (handle missing values intelligently)
        if mask is not None:
            x_filled, missingness_features = self.input_mask_module(x, mask)
        else:
            x_filled = x
            missingness_features = None
        
        # 2. Apply learnable mask (learn feature/temporal importance)
        x_masked, combined_mask, mask_weights = self.learnable_mask(x_filled, mask)
        
        # 3. Encode
        latent = self.encoder(x_masked, combined_mask if mask is not None else None)
        
        # 4. Decode
        reconstructed = self.decoder(latent)
        
        # 5. Apply output confidence mask
        output_confidence = self.output_mask(reconstructed)
        reconstructed = reconstructed * output_confidence
        
        if return_masks:
            return reconstructed, {
                'mask_weights': mask_weights,
                'output_confidence': output_confidence.detach(),
                'combined_mask': combined_mask.detach()
            }
        
        return reconstructed
    
    def get_latent(self, x, mask=None):
        if mask is not None:
            x_filled, _ = self.input_mask_module(x, mask)
            x_masked, combined_mask, _ = self.learnable_mask(x_filled, mask)
            return self.encoder(x_masked, combined_mask)
        return self.encoder(x, None)
    
    def compute_masked_reconstruction_error(self, x, mask):
        """
        Compute MSE only on valid (non-NaN) positions.
        Uses the learned masks to weight the error appropriately.
        """
        reconstructed, mask_info = self.forward(x, mask, return_masks=True)
        
        # Compute squared error
        squared_error = (x - reconstructed) ** 2
        
        # Apply original NaN mask - only consider valid positions
        mask_float = mask.float()
        masked_error = squared_error * mask_float
        
        # Also weight by learned confidence (optional - can be tuned)
        # Higher confidence positions contribute more to the error
        learned_weight = mask_info['combined_mask']
        weighted_error = masked_error * (0.5 + 0.5 * learned_weight)  # Base weight + learned weight
        
        # Mean per sample
        error_sum = weighted_error.sum(dim=(1, 2))
        valid_count = mask_float.sum(dim=(1, 2)).clamp(min=1)
        
        mse = error_sum / valid_count
        return mse
    
    def get_mask_weights(self):
        """Return the learned mask weights for analysis."""
        return {
            'feature_importance': F.softmax(self.learnable_mask.feature_mask, dim=0).detach().cpu().numpy(),
            'missing_embedding': self.input_mask_module.missing_embedding.detach().cpu().numpy()
        }


# ============================================================================
# MASKED LOSS FUNCTION
# ============================================================================

class MaskedMSELoss(nn.Module):
    """
    MSE Loss that only computes loss on valid (non-NaN) positions.
    This properly handles missing data without introducing bias.
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target, mask):
        """
        pred: (batch, seq_len, features) - reconstructed values
        target: (batch, seq_len, features) - original values (NaN replaced with 0)
        mask: (batch, seq_len, features) - True where valid, False where was NaN
        """
        # Compute squared error
        squared_error = (pred - target) ** 2
        
        # Apply mask
        mask_float = mask.float()
        masked_error = squared_error * mask_float
        
        # Mean over valid positions
        loss = masked_error.sum() / mask_float.sum().clamp(min=1)
        
        return loss


# ============================================================================
# TRAINING WITH MASKING
# ============================================================================

def train_epoch_masked(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: MaskedMSELoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    """Train for one epoch with masked loss"""
    model.train()
    total_loss = 0
    
    for batch_data, batch_mask in dataloader:
        batch_data = batch_data.to(device, non_blocking=True)
        batch_mask = batch_mask.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        reconstructed = model(batch_data, batch_mask)
        loss = criterion(reconstructed, batch_data, batch_mask)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate_epoch_masked(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: MaskedMSELoss,
    device: torch.device
) -> float:
    """Validate for one epoch with masked loss"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch_data, batch_mask in dataloader:
            batch_data = batch_data.to(device, non_blocking=True)
            batch_mask = batch_mask.to(device, non_blocking=True)
            reconstructed = model(batch_data, batch_mask)
            loss = criterion(reconstructed, batch_data, batch_mask)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


class LivePlotter:
    """Real-time training loss plotter for TCN"""
    
    def __init__(self, output_dir: Path, title: str = "TCN Training Progress"):
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
        ax.set_ylabel('Masked MSE Loss', fontsize=11)
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
    """Full training loop with masked loss and live plotting"""
    
    fold_str = f" (Fold {fold})" if fold is not None else ""
    print("\n" + "=" * 70)
    print(f"TRAINING TCN AUTOENCODER{fold_str}")
    print("=" * 70)
    
    device = config.DEVICE
    model = model.to(device)
    
    criterion = MaskedMSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min',
        factor=config.LR_SCHEDULER_FACTOR,
        patience=config.LR_SCHEDULER_PATIENCE
    )
    
    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    best_epoch = 0
    
    # Initialize live plotter
    live_plotter = None
    if config.LIVE_PLOT:
        title = f"TCN Training Progress{fold_str}"
        live_plotter = LivePlotter(config.OUTPUT_DIR, title)
    
    print(f"\nDevice: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training samples: {len(train_loader.dataset):,}")
    print(f"Validation samples: {len(val_loader.dataset):,}")
    print(f"Using Masked Loss: Yes")
    print(f"Live plotting: {'Enabled' if config.LIVE_PLOT else 'Disabled'}")
    print()
    
    pbar = tqdm(range(config.EPOCHS), desc="Training", unit="epoch")
    
    for epoch in pbar:
        train_loss = train_epoch_masked(model, train_loader, criterion, optimizer, device)
        val_loss = validate_epoch_masked(model, val_loader, criterion, device)
        
        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)
        
        old_lr = current_lr
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < old_lr:
            print(f"\n📉 LR reduced: {old_lr:.2e} → {new_lr:.2e}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            best_epoch = epoch + 1
        else:
            patience_counter += 1
        
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
        
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"\n\n⚠️  Early stopping at epoch {epoch+1}")
            break
    
    # Close live plotter
    if live_plotter:
        live_plotter.update(history, epoch + 1, patience_counter,
                           config.EARLY_STOPPING_PATIENCE, best_val_loss)
        live_plotter.close()
    
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"\n✓ Restored best model from epoch {best_epoch} (val_loss: {best_val_loss:.6f})")
    
    return history


# ============================================================================
# ANOMALY DETECTION
# ============================================================================

def compute_reconstruction_errors_masked(
    model: nn.Module,
    sequences: np.ndarray,
    device: torch.device,
    batch_size: int = 256
) -> np.ndarray:
    """Compute masked reconstruction error for each sequence"""
    model.eval()
    errors = []
    
    dataset = MaskedTimeSeriesDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                           pin_memory=torch.cuda.is_available())
    
    with torch.no_grad():
        for batch_data, batch_mask in dataloader:
            batch_data = batch_data.to(device, non_blocking=True)
            batch_mask = batch_mask.to(device, non_blocking=True)
            mse = model.compute_masked_reconstruction_error(batch_data, batch_mask)
            errors.extend(mse.cpu().numpy())
    
    return np.array(errors)


def find_optimal_threshold(
    val_errors: np.ndarray,
    test_errors: np.ndarray,
    test_labels: np.ndarray,
    config: Config
) -> Tuple[float, Dict]:
    """Find optimal anomaly detection threshold"""
    print("\n" + "=" * 70)
    print("FINDING OPTIMAL THRESHOLD")
    print("=" * 70)
    
    binary_labels = (test_labels > 0).astype(int)
    results = {}
    
    # Percentile-based threshold
    threshold_pct = np.percentile(val_errors, config.THRESHOLD_PERCENTILE)
    preds_pct = (test_errors > threshold_pct).astype(int)
    f1_pct = f1_score(binary_labels, preds_pct)
    
    results['percentile'] = {
        'threshold': float(threshold_pct),
        'percentile': config.THRESHOLD_PERCENTILE,
        'f1': float(f1_pct)
    }
    print(f"\n{config.THRESHOLD_PERCENTILE}th percentile threshold: {threshold_pct:.6f} (F1={f1_pct:.4f})")
    
    # Optimize F1
    best_f1 = 0
    best_threshold = threshold_pct
    best_pct = config.THRESHOLD_PERCENTILE
    
    for pct in range(80, 100):
        thresh = np.percentile(val_errors, pct)
        preds = (test_errors > thresh).astype(int)
        f1 = f1_score(binary_labels, preds)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
            best_pct = pct
    
    results['optimal_f1'] = {
        'threshold': float(best_threshold),
        'percentile': best_pct,
        'f1': float(best_f1)
    }
    print(f"Optimal F1 threshold: {best_threshold:.6f} at {best_pct}th percentile (F1={best_f1:.4f})")
    
    return best_threshold, results


def evaluate_anomaly_detection(
    test_errors: np.ndarray,
    test_labels: np.ndarray,
    threshold: float
) -> Dict:
    """Comprehensive evaluation"""
    print("\n" + "=" * 70)
    print("ANOMALY DETECTION EVALUATION")
    print("=" * 70)
    
    binary_labels = (test_labels > 0).astype(int)
    predictions = (test_errors > threshold).astype(int)
    
    print(f"\nThreshold: {threshold:.6f}")
    print(f"\nClassification Report:")
    print(classification_report(
        binary_labels, predictions,
        target_names=['Good (Normal)', 'Anomaly (Bad/Suspect)'],
        digits=4
    ))
    
    cm = confusion_matrix(binary_labels, predictions)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"Confusion Matrix:")
    print(f"                    Predicted")
    print(f"                Normal   Anomaly")
    print(f"Actual Normal   {tn:6d}   {fp:6d}")
    print(f"       Anomaly  {fn:6d}   {tp:6d}")
    
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
    
    auc_roc = roc_auc_score(binary_labels, test_errors) if len(np.unique(binary_labels)) > 1 else None
    auc_pr = average_precision_score(binary_labels, test_errors) if len(np.unique(binary_labels)) > 1 else None
    
    if auc_roc:
        print(f"  ROC-AUC:     {auc_roc:.4f}")
        print(f"  PR-AUC:      {auc_pr:.4f}")
    
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
    
    ax = axes[0]
    ax.plot(history['train_loss'], label='Training Loss', linewidth=2)
    ax.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Masked MSE Loss', fontsize=12)
    ax.set_title('Training and Validation Loss (TCN)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
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


def plot_error_distributions(
    val_errors: np.ndarray,
    test_errors: np.ndarray,
    test_labels: np.ndarray,
    threshold: float,
    output_dir: Path
):
    """Plot reconstruction error distributions"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax = axes[0]
    colors = {'Good': 'green', 'Bad': 'red', 'Suspect': 'orange'}
    
    for label, name in [(0, 'Good'), (1, 'Bad'), (2, 'Suspect')]:
        mask = test_labels == label
        if mask.sum() > 0:
            ax.hist(test_errors[mask], bins=50, alpha=0.5, 
                   label=f'{name} (n={mask.sum()})', color=colors[name])
    
    ax.axvline(threshold, color='black', linestyle='--', linewidth=2,
               label=f'Threshold={threshold:.4f}')
    ax.set_xlabel('Reconstruction Error (Masked MSE)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Error Distribution by Class (TCN)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
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
    ax.set_ylabel('Reconstruction Error', fontsize=12)
    ax.set_title('Error Box Plot by Class', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'error_distributions.png'}")


def plot_roc_pr_curves(test_errors: np.ndarray, test_labels: np.ndarray, output_dir: Path):
    """Plot ROC and PR curves"""
    binary_labels = (test_labels > 0).astype(int)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax = axes[0]
    fpr, tpr, _ = roc_curve(binary_labels, test_errors)
    auc_roc = roc_auc_score(binary_labels, test_errors)
    ax.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC={auc_roc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve (TCN Autoencoder)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
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
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'],
                annot_kws={'size': 14})
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title('Confusion Matrix (TCN)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'confusion_matrix.png'}")


def plot_learned_masks(model: nn.Module, config: Config, output_dir: Path):
    """Visualize the learned mask weights."""
    print("\nPlotting learned mask weights...")
    
    mask_weights = model.get_mask_weights()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Feature importance
    ax = axes[0]
    features = config.FEATURES
    importance = mask_weights['feature_importance']
    bars = ax.bar(features, importance, color=['steelblue', 'coral'])
    ax.set_ylabel('Learned Importance', fontsize=11)
    ax.set_title('Feature Importance (Learned)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, importance):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Missing value embedding
    ax = axes[1]
    embedding = mask_weights['missing_embedding']
    bars = ax.bar(features, embedding, color=['steelblue', 'coral'])
    ax.set_ylabel('Learned Missing Value', fontsize=11)
    ax.set_title('Missing Value Embedding (Learned)', fontsize=12, fontweight='bold')
    for bar, val in zip(bars, embedding):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'learned_masks.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'learned_masks.png'}")


def save_model_and_artifacts(
    model: nn.Module,
    scaler,
    config: Config,
    history: Dict,
    threshold: float,
    eval_results: Dict,
    output_dir: Path
):
    """Save model and artifacts"""
    print("\n" + "=" * 70)
    print("SAVING MODEL AND ARTIFACTS")
    print("=" * 70)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_dir / 'tcn_autoencoder.pt'
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
    
    # Save history
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  History saved: {history_path}")
    
    # Save results
    results = {
        'threshold': float(threshold),
        'accuracy': float(eval_results['accuracy']),
        'precision': float(eval_results['precision']),
        'recall': float(eval_results['recall']),
        'specificity': float(eval_results['specificity']),
        'f1': float(eval_results['f1']),
        'roc_auc': float(eval_results['roc_auc']) if eval_results['roc_auc'] else None,
        'pr_auc': float(eval_results['pr_auc']) if eval_results['pr_auc'] else None,
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
    
    # Save learned mask weights
    mask_weights = model.get_mask_weights()
    mask_path = output_dir / 'learned_mask_weights.json'
    with open(mask_path, 'w') as f:
        json.dump({k: v.tolist() for k, v in mask_weights.items()}, f, indent=2)
    print(f"  Mask weights saved: {mask_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main training and evaluation pipeline"""
    
    print("\n" + "=" * 70)
    print("TCN MASKED AUTOENCODER - ANOMALY DETECTION")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    config = Config()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Random seeds
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    
    # GPU Setup
    print("\n" + "=" * 70)
    print("DEVICE CONFIGURATION")
    print("=" * 70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
        torch.cuda.manual_seed(config.RANDOM_SEED)
    print(f"Using device: {config.DEVICE}")
    
    # Load data
    df = load_data(config)
    
    # Create sequences (preserving NaN)
    sequences, labels, stations = create_sequences_with_nan(df, config)
    
    # Prepare splits
    data_splits = prepare_data_splits(sequences, labels, config)
    
    train_seq, train_labels = data_splits['train']
    val_seq, val_labels = data_splits['val']
    test_seq, test_labels = data_splits['test']
    
    if config.USE_KFOLD:
        print("\n" + "=" * 70)
        print(f"K-FOLD CROSS VALIDATION ({config.N_FOLDS} folds)")
        print("=" * 70)
        
        combined_seq = np.concatenate([train_seq, val_seq], axis=0)
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
            
            fold_train_seq = combined_seq[train_idx]
            fold_val_seq = combined_seq[val_idx]
            
            # Normalize
            fold_train_norm, fold_val_norm, fold_test_norm, fold_scaler = normalize_data_robust(
                fold_train_seq, fold_val_seq, test_seq
            )
            
            # Create datasets with masking
            fold_train_dataset = MaskedTimeSeriesDataset(fold_train_norm)
            fold_val_dataset = MaskedTimeSeriesDataset(fold_val_norm)
            
            fold_train_loader = DataLoader(
                fold_train_dataset, batch_size=config.BATCH_SIZE,
                shuffle=True, num_workers=config.NUM_WORKERS,
                pin_memory=config.PIN_MEMORY
            )
            fold_val_loader = DataLoader(
                fold_val_dataset, batch_size=config.BATCH_SIZE,
                shuffle=False, num_workers=config.NUM_WORKERS,
                pin_memory=config.PIN_MEMORY
            )
            
            # Create and train model
            fold_model = TCNAutoencoder(config)
            fold_history = train_model(fold_model, fold_train_loader, fold_val_loader, config, fold=fold_idx+1)
            all_histories.append(fold_history)
            
            final_val_loss = min(fold_history['val_loss'])
            fold_results.append({
                'fold': fold_idx + 1,
                'final_val_loss': final_val_loss,
                'epochs_trained': len(fold_history['train_loss'])
            })
            
            if final_val_loss < best_fold_val_loss:
                best_fold_val_loss = final_val_loss
                best_fold_model = fold_model
                best_fold_idx = fold_idx + 1
                best_scaler = fold_scaler
                best_test_norm = fold_test_norm
        
        # Summary
        print("\n" + "=" * 70)
        print("K-FOLD SUMMARY")
        print("=" * 70)
        val_losses = [r['final_val_loss'] for r in fold_results]
        for r in fold_results:
            marker = " *** BEST ***" if r['fold'] == best_fold_idx else ""
            print(f"  Fold {r['fold']}: Val Loss = {r['final_val_loss']:.6f}{marker}")
        print(f"\nMean Val Loss: {np.mean(val_losses):.6f} ± {np.std(val_losses):.6f}")
        
        model = best_fold_model
        scaler = best_scaler
        test_norm = best_test_norm
        history = all_histories[best_fold_idx - 1]
        
    else:
        # Standard training
        train_norm, val_norm, test_norm, scaler = normalize_data_robust(train_seq, val_seq, test_seq)
        
        train_dataset = MaskedTimeSeriesDataset(train_norm)
        val_dataset = MaskedTimeSeriesDataset(val_norm)
        
        train_loader = DataLoader(
            train_dataset, batch_size=config.BATCH_SIZE,
            shuffle=True, num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.BATCH_SIZE,
            shuffle=False, num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY
        )
        
        model = TCNAutoencoder(config)
        print(f"\nModel Architecture:")
        print(model)
        
        history = train_model(model, train_loader, val_loader, config)
    
    # Compute errors
    print("\n" + "=" * 70)
    print("COMPUTING RECONSTRUCTION ERRORS")
    print("=" * 70)
    
    if config.USE_KFOLD:
        good_mask = labels == 0
        good_seq = sequences[good_mask]
        _, _, good_norm, _ = normalize_data_robust(good_seq, good_seq, good_seq)
        val_errors = compute_reconstruction_errors_masked(model, good_norm, config.DEVICE)
    else:
        val_errors = compute_reconstruction_errors_masked(model, val_norm, config.DEVICE)
    
    test_errors = compute_reconstruction_errors_masked(model, test_norm, config.DEVICE)
    
    print(f"Validation errors: mean={val_errors.mean():.6f}, std={val_errors.std():.6f}")
    print(f"Test errors: mean={test_errors.mean():.6f}, std={test_errors.std():.6f}")
    
    # Find threshold
    threshold, _ = find_optimal_threshold(val_errors, test_errors, test_labels, config)
    
    # Evaluate
    eval_results = evaluate_anomaly_detection(test_errors, test_labels, threshold)
    
    # Visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    plot_training_history(history, config.OUTPUT_DIR)
    plot_error_distributions(val_errors, test_errors, test_labels, threshold, config.OUTPUT_DIR)
    plot_roc_pr_curves(test_errors, test_labels, config.OUTPUT_DIR)
    plot_confusion_matrix_heatmap(eval_results['confusion_matrix'], config.OUTPUT_DIR)
    plot_learned_masks(model, config, config.OUTPUT_DIR)  # Plot learned mask weights
    
    # Save
    save_model_and_artifacts(model, scaler, config, history, threshold, eval_results, config.OUTPUT_DIR)
    
    # Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nFinal Results (TCN Masked Autoencoder):")
    print(f"  Threshold:   {threshold:.6f}")
    print(f"  Accuracy:    {eval_results['accuracy']:.4f}")
    print(f"  Precision:   {eval_results['precision']:.4f}")
    print(f"  Recall:      {eval_results['recall']:.4f}")
    print(f"  F1 Score:    {eval_results['f1']:.4f}")
    if eval_results['roc_auc']:
        print(f"  ROC-AUC:     {eval_results['roc_auc']:.4f}")
    print(f"\nOutputs saved to: {config.OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
