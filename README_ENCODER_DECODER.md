# LSTM Encoder-Decoder for Meteorological Data Anomaly Detection

## Overview

This project implements an **LSTM Autoencoder** (Encoder-Decoder architecture) for detecting anomalies in meteorological wind data from the LANL dataset. The approach is based on **unsupervised anomaly detection** where the model learns to reconstruct normal patterns and flags deviations as anomalies.

## Method

### Training Strategy

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TRAINING APPROACH                                │
├─────────────────────────────────────────────────────────────────────────┤
│  1. Train ONLY on "good" (normal) class data                            │
│  2. Model learns to reconstruct normal wind patterns                     │
│  3. High reconstruction error = anomaly (bad/suspect data)              │
└─────────────────────────────────────────────────────────────────────────┘

Data Split:
├── Training:   70% of GOOD data only (learn normal patterns)
├── Validation: 15% of GOOD data only (hyperparameter tuning)
└── Testing:    15% GOOD + ALL BAD + ALL SUSPECT (evaluate detection)
```

### Why This Approach?

1. **No need for labeled anomalies during training**: The model learns what "normal" looks like
2. **Generalizes to unseen anomaly types**: Can detect anomalies it has never seen
3. **Interpretable**: Reconstruction error provides a meaningful anomaly score
4. **Handles temporal patterns**: LSTM captures time-series dependencies

## Architecture

```
                    LSTM Autoencoder Architecture
    
    Input Sequence                              Output Sequence
    (24 timesteps × 2 features)                 (24 timesteps × 2 features)
           │                                           ▲
           ▼                                           │
    ┌──────────────┐                          ┌──────────────┐
    │   Encoder    │                          │   Decoder    │
    │  (BiLSTM)    │                          │   (LSTM)     │
    │              │                          │              │
    │  64 hidden   │                          │  64 hidden   │
    │  2 layers    │                          │  2 layers    │
    └──────────────┘                          └──────────────┘
           │                                           ▲
           ▼                                           │
    ┌──────────────┐                          ┌──────────────┐
    │    Dense     │                          │    Dense     │
    │   + ReLU     │                          │   + ReLU     │
    └──────────────┘                          └──────────────┘
           │                                           ▲
           └────────────► Latent Space ────────────────┘
                          (32 dims)
```

### Model Components

| Component | Description |
|-----------|-------------|
| **Encoder** | Bidirectional LSTM that compresses the input sequence into a fixed-size latent representation |
| **Latent Space** | Compressed representation (32 dimensions) capturing essential patterns |
| **Decoder** | LSTM that reconstructs the original sequence from the latent representation |

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `SEQUENCE_LENGTH` | 24 | Number of time steps (24 × 5min = 2 hours) |
| `STRIDE` | 12 | Step size between sequences (50% overlap) |
| `HIDDEN_DIM` | 64 | LSTM hidden layer size |
| `LATENT_DIM` | 32 | Bottleneck dimension |
| `NUM_LAYERS` | 2 | Number of LSTM layers |
| `DROPOUT` | 0.2 | Dropout rate for regularization |
| `BATCH_SIZE` | 64 | Training batch size |
| `LEARNING_RATE` | 1e-3 | Initial learning rate |
| `EPOCHS` | 100 | Maximum training epochs |

## Features Used

| Feature | Description |
|---------|-------------|
| `wind direction` | Wind direction in degrees (0-360°) |
| `wind speed` | Wind speed measurement |

## Anomaly Detection

### How It Works

1. **Training Phase**: 
   - Feed sequences of "good" data through the autoencoder
   - Minimize reconstruction error (MSE loss)
   - Model learns to accurately reconstruct normal patterns

2. **Detection Phase**:
   - Feed new sequences through the trained model
   - Calculate reconstruction error (MSE)
   - If error > threshold → **Anomaly detected**

### Threshold Selection

Two methods for selecting the anomaly threshold:

1. **Percentile-based**: Use the 95th percentile of validation set errors
2. **F1-optimized**: Search for threshold that maximizes F1 score on test set

```
                     Error Distribution
                     
     Good Data                    Anomalies
         │                            │
         ▼                            ▼
    ┌─────────┐                ┌───────────┐
    │ █████   │                │     ████  │
    │ ██████  │                │    █████  │
    │ ███████ │                │   ██████  │
    │ ████████│      Threshold │  ███████  │
    │█████████│ ◄──────────────┼─►████████ │
    └─────────┘                └───────────┘
       Low Error                  High Error
```

## Data Pipeline

```
Raw Data (aligned_5min_met_data.parquet)
         │
         ▼
┌─────────────────────────────────────┐
│     Per-Station Sequence Creation    │
│  • Sliding window (24 steps)        │
│  • Handle missing values            │
│  • Label assignment (majority vote) │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│         Data Splitting              │
│  • Good → Train/Val/Test            │
│  • Bad/Suspect → Test only          │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│       Normalization                 │
│  • StandardScaler on training data  │
│  • Transform all splits             │
└─────────────────────────────────────┘
```

## Files

| File | Description |
|------|-------------|
| `encoder_decoder_anomaly.py` | Main training and evaluation script |
| `aligned_5min_met_data.parquet` | Input data (aligned timestamps) |
| `model_outputs/` | Output directory for results |

### Output Files

| File | Description |
|------|-------------|
| `lstm_autoencoder.pt` | Trained PyTorch model |
| `scaler.pkl` | Fitted StandardScaler |
| `config.json` | Training configuration |
| `training_history.json` | Loss history per epoch |
| `evaluation_results.json` | Final metrics |
| `training_history.png` | Training/validation loss curves |
| `error_distributions.png` | Reconstruction error by class |
| `roc_pr_curves.png` | ROC and Precision-Recall curves |
| `confusion_matrix.png` | Confusion matrix heatmap |
| `reconstruction_examples.png` | Visual examples of reconstructions |

## Usage

### Training

```bash
# Activate conda environment
conda activate met_data

# Install dependencies
pip install torch scikit-learn matplotlib seaborn tqdm

# Run training
python encoder_decoder_anomaly.py
```

### Configuration

Edit the `Config` class in `encoder_decoder_anomaly.py` to modify:

```python
class Config:
    DATA_FILE = Path("aligned_5min_met_data.parquet")
    SEQUENCE_LENGTH = 24      # Adjust for longer/shorter patterns
    HIDDEN_DIM = 64           # Increase for more capacity
    LATENT_DIM = 32           # Adjust bottleneck size
    EPOCHS = 100              # Maximum training epochs
    THRESHOLD_PERCENTILE = 95 # Anomaly threshold percentile
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correct predictions |
| **Precision** | Of predicted anomalies, how many are true anomalies |
| **Recall** | Of actual anomalies, how many were detected |
| **Specificity** | Of actual normal data, how many were correctly identified |
| **F1 Score** | Harmonic mean of precision and recall |
| **ROC-AUC** | Area under ROC curve |
| **PR-AUC** | Area under Precision-Recall curve |

## Expected Results

Based on the data characteristics:

- **Good detection of BAD class**: Higher reconstruction errors expected
- **Moderate detection of SUSPECT class**: May overlap with good data
- **Trade-off**: Adjusting threshold affects precision vs recall

## References

1. Malhotra, P., et al. (2016). "LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection"
2. Park, D., et al. (2018). "A Multimodal Anomaly Detector for Robot-Assisted Feeding Using an LSTM-based Variational Autoencoder"
3. Su, Y., et al. (2019). "Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network"

## License

This project is part of the LANL Meteorological Data Analysis pipeline.

---

**Author**: Generated for ONRL Encoder-Decoder CAPARS Project  
**Date**: December 2024
