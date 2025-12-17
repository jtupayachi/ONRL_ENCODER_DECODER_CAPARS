# ONRL_ENCODER_DECODER_CAPARS

## Meteorological Data Anomaly Detection using Deep Learning

This project implements encoder-decoder architectures for detecting anomalies in LANL meteorological wind data. The system learns normal wind patterns from high-quality data and identifies anomalous readings based on reconstruction error.

---

## 📊 Dataset Overview

| Metric | Value |
|--------|-------|
| **Total Records** | 9,175,591 |
| **Features** | Wind Direction (°), Wind Speed (m/s) |
| **Stations** | 130 unique meteorological stations |
| **Time Range** | January 2023 - December 2023 |
| **Storage Intervals** | 5, 10, 15 minutes |

### Quality Class Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| Bad | 4,680,789 | 51.01% |
| Good | 4,095,146 | 44.63% |
| Suspect | 399,656 | 4.36% |

### Storage Interval Distribution

| Interval | Count | Percentage |
|----------|-------|------------|
| 5 min | 7,306,767 | 79.63% |
| 10 min | 950,089 | 10.35% |
| 15 min | 918,735 | 10.01% |

---

## 🔍 EDA Key Findings

### Missing Data Analysis

| Feature | Missing Count | Missing % |
|---------|---------------|-----------|
| Wind Direction | 3,690,340 | 40.22% |
| Wind Speed | 1,656,169 | 18.05% |

**Missing by Quality Class:**
- **Bad class**: 60.34% missing wind direction, 34.98% missing wind speed
- **Good class**: 19.90% missing wind direction, 0.44% missing wind speed
- **Suspect class**: 12.72% missing wind direction, 0.19% missing wind speed

### Wind Speed Statistics

| Statistic | Value |
|-----------|-------|
| Mean | 1.66 m/s |
| Std | 2.09 m/s |
| Median | 0.90 m/s |
| Max | 40.20 m/s |
| 95th Percentile | 5.80 m/s |

**Wind Speed by Class:**
- Good: mean=1.88, std=2.15
- Bad: mean=1.20, std=1.67
- Suspect: mean=2.98, std=3.15

### Temporal Patterns

- **Diurnal Pattern**: Wind speed peaks at 20:00-21:00 (2.60-2.63 m/s) and is lowest at 11:00-13:00 (1.05-1.07 m/s)
- **Seasonal Pattern**: Higher winds in March-April (2.13-2.17 m/s), lower in September-November (1.30-1.47 m/s)
- **Correlation**: Wind speed shows weak positive correlation with hour (0.163) and negative with month (-0.107)

---

## 🛠️ Data Processing Pipeline

### 1. Data Merging (`merge_met_data.py`)

```
├── Reads CSV files from storage interval directories (5, 10, 15 min)
├── Extracts metadata: station ID, quality class, storage interval
├── Timestamp alignment to common 5-minute intervals
├── Parallel processing using multiprocessing (all CPU cores)
└── Output: aligned_5min_met_data.parquet
```

**Alignment Methods:**
- **Downsampling** (10/15 min → 5 min): Linear interpolation for wind speed, nearest for direction
- **Upsampling**: Aggregation using mean
- **NaN Handling**: Preserved for masking in training

### 2. Sequence Generation

- **Sequence Length**: 48 timesteps (4 hours of data)
- **Stride**: 24 timesteps (50% overlap)
- **Valid Ratio**: Sequences with <50% NaN are discarded
- **Parallel Processing**: Vectorized NumPy operations with multiprocessing

### 3. Normalization

- **Method**: RobustScaler (handles outliers better than StandardScaler)
- **Fit**: Only on training data (good class)
- **NaN Preservation**: NaN values propagate through normalization

---

## 🧠 Model Architectures

### Architecture 1: LSTM Autoencoder (`lstm_masked_autoencoder.py`)

```
Input (batch, 48, 2)
    │
    ▼
┌─────────────────────────────┐
│  Encoder (Bidirectional)    │
│  BiLSTM: 2 → 64 × 2 = 128   │
│  FC: 128 → 32 (latent)      │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  Decoder                    │
│  FC: 32 → 64                │
│  LSTM: 64 → 64              │
│  Output: 64 → 2             │
└─────────────────────────────┘
    │
    ▼
Output (batch, 48, 2)
```

**Hyperparameters:**
- Hidden Dim: 64
- Latent Dim: 32
- Layers: 2
- Dropout: 0.2

### Architecture 2: TCN Masked Autoencoder (`tcn_masked_autoencoder.py`)

```
Input (batch, 48, 2)
    │
    ▼
┌─────────────────────────────┐
│  Input Mask Module          │
│  - Missing value embedding  │
│  - Learned interpolation    │
│  - Missingness indicators   │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  Learnable Mask Layer       │
│  - Feature importance       │
│  - Temporal attention       │
│  - Confidence estimation    │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  TCN Encoder                │
│  Channels: [32,64,128,64,32]│
│  Dilations: [1,2,4,8,16]    │
│  + Attention pooling        │
│  → Latent (64)              │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  TCN Decoder                │
│  FC expansion → Conv1d      │
│  → Output (batch, 48, 2)    │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  Output Confidence Mask     │
│  Learned reconstruction     │
│  confidence weighting       │
└─────────────────────────────┘
```

**Key Features:**
- **Dilated Causal Convolutions**: Exponentially increasing receptive field
- **Learnable Mask Layer**: Learns feature importance and temporal attention
- **Input Mask Module**: Intelligent missing value handling with learned interpolation
- **Masked MSE Loss**: Only computes loss on valid (non-NaN) positions

---

## 📈 Training Strategy

### Anomaly Detection Approach

1. **Train ONLY on "good" class data** - model learns normal patterns
2. **Validate on held-out "good" data** - tune hyperparameters
3. **Test on ALL classes** - evaluate anomaly detection

### Data Splits (from Good data only)

| Split | Ratio | Purpose |
|-------|-------|---------|
| Train | 70% | Learn normal patterns |
| Validation | 15% | Early stopping, LR scheduling |
| Test | 15% good + ALL anomalies | Evaluate detection |

### K-Fold Cross Validation

- **Folds**: 5
- **Strategy**: Combine train+val, split into K folds
- **Selection**: Best fold model used for final evaluation

### Training Configuration

| Parameter | LSTM | TCN |
|-----------|------|-----|
| Batch Size | 64 | 128 |
| Epochs | 100 | 150 |
| Learning Rate | 1e-3 | 1e-3 |
| Weight Decay | 1e-5 | 1e-4 |
| Early Stopping | 15 epochs | 20 epochs |
| LR Scheduler | ReduceLROnPlateau | ReduceLROnPlateau |
| Optimizer | Adam | AdamW |

### GPU Optimizations

- Pin Memory for faster CPU→GPU transfer
- cuDNN Benchmark mode enabled
- Non-blocking data transfers
- Multi-worker data loading (4 workers)

---

## 🎯 Anomaly Detection

### Threshold Selection

1. **Percentile-based**: 95th percentile of validation (good) reconstruction errors
2. **F1-optimized**: Search 80th-99th percentile for best F1 score

### Evaluation Metrics

- Accuracy, Precision, Recall, F1 Score
- ROC-AUC, PR-AUC
- Confusion Matrix
- Per-class detection rates

### Anomaly Score

```
Anomaly Score = Masked MSE(Input, Reconstruction)
              = Σ(valid positions) (x - x̂)² / count(valid)
```

---

## 📁 Project Structure

```
ONRL_ENCODER_DECODER_CAPARS/
├── merge_met_data.py           # Data merging & alignment
├── lstm_masked_autoencoder.py  # LSTM architecture
├── tcn_masked_autoencoder.py   # TCN architecture with learnable masks
├── aligned_5min_met_data.parquet  # Processed data
├── model_outputs/              # LSTM outputs
│   ├── lstm_autoencoder.pt
│   ├── scaler.pkl
│   ├── training_history.json
│   ├── evaluation_results.json
│   └── *.png (visualizations)
├── model_outputs_tcn/          # TCN outputs
│   ├── tcn_autoencoder.pt
│   ├── learned_mask_weights.json
│   └── *.png (visualizations)
└── eda_outputs/                # EDA results
    └── eda_summary_report.txt
```

---

## 🚀 Usage

### 1. Merge Data
```bash
python merge_met_data.py
```

### 2. Train LSTM Model
```bash
python lstm_masked_autoencoder.py
```

### 3. Train TCN Model (Recommended)
```bash
python tcn_masked_autoencoder.py
```

---

## 📦 Requirements

```
torch>=2.0
numpy
pandas
scikit-learn
matplotlib
seaborn
tqdm
pyarrow
```

---

## 🔑 Key Innovations

1. **Masked Loss Function**: Properly handles missing values without introducing bias
2. **Learnable Mask Layers**: Model learns feature importance and optimal missing value representations
3. **TCN Architecture**: Better long-range temporal dependencies than LSTM
4. **Robust Normalization**: RobustScaler handles outliers in meteorological data
5. **K-Fold CV**: Reliable model selection and variance estimation

---
