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




# NOW VISION TRANSFORMERS!


# Question what to do with the nan values categories? EXCLUDE!IF NOT MANY TO THE BAD CATEGORY  A MONTH OF BATH DATA EXCLUDE IT ---> GOOD VS BAD .... 

# is it fine if we train all for a model? or different models? WE SHOULD ... ALL DATA INTERVAL/ BREAKIT UP DIFFERENT INTERVALS ...
# ...  IF WE JSUT LOOK AT THE INDIVIDUAL .... 5 ,10 ,15 .... (ISSUE WITH THE NUMBER OF DOTS  .... CARE MORE ABOUT RANDOMNESS DIRECTION OF THE GRAPH AND VALUES OF THE SPEED.). SPEED AND WIND SEPARATE TOO ... 
---------- GET BEST MODEL ----------

# 5 ten minutes ... 10 minutes ...

--------------------------------------------------
ENSEMBLE .... METHOD .... LAST OUTPUT

# Script
```
nohup python3 multi_model_training.py > training_log.log 2>&1 &
```


## Now we have:
```
# During Training (per fold):
Real-time progress bars
Train/Val loss and accuracy
Confusion matrix saved after each fold
Classification report JSON
Predictions CSV

# After Each Model:
Final confusion matrix (aggregated across all folds)
K-fold summary with statistics
All predictions combined

# Final Comparison:
Bar chart comparing all 3 models
Summary table with mean ± std for each model
Best model identification

```


## Image Preprocessing:



Input:
├── /lanl_met_data/images/speed/*.png
└── /lanl_met_data/images/dir/*.png
    (Wind speed and direction time-series plots)

Process (OpenCV):
1. Read image as BGR
2. Convert to grayscale
3. Apply threshold (250) to detect white borders *******NEED TO TUNE THIS BETTER!*******
4. Find largest contour (plot content)
5. Get bounding box: x, y, w, h
6. Crop image to content only
7. Save trimmed image

Output:
└── /trimmed_images/
    ├── StationName_speed.png  (172 images)
    └── StationName_dir.png    (171 images)
    Total: 343 images (~565×426 avg size)









Input: Trimmed image (variable size, RGB)

Process Chain:

┌─────────────────────────────────────┐
│ 1. Load Image                       │
│    PIL.Image.open(path)             │
│    → RGB (H × W × 3)                │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│ 2. Convert to Grayscale ✅          │
│    image.convert('L')               │
│    → Single channel (H × W × 1)     │
│                                     │
│ Why? Focuses on temporal patterns   │
│      not color artifacts            │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│ 3. Expand Back to 3 Channels        │
│    Image.merge('RGB', [L, L, L])    │
│    → (H × W × 3) grayscale RGB      │
│                                     │
│ Why? Models expect 3-channel input  │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│ 4. NO Augmentation                  │
│    (Previously had flip/rotation)   │
│    NOW: Preserve temporal structure │
│                                     │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│ 5. Model-Specific Processing        │
│    processor(images=img)            │
│                                     │
│    a) Resize to model input:        │
│       - ViT/ConvNeXt/Swin: 384×384  │
│       - DINOv2: 518×518             │
│                                     │
│    b) Normalize (ImageNet stats):   │
│       mean = [0.485, 0.456, 0.406]  │
│       std = [0.229, 0.224, 0.225]   │
│                                     │
│    c) Convert to PyTorch tensor:    │
│       (3, H, W) float32             │
└─────────────────────────────────────┘
              ↓
Output: Preprocessed tensor ready for model








Input: 
├── List of (image_path, label, station_name) tuples
├── Image processor
└── Preprocessing flags

Process:
1. Load image from path
2. Apply RGB conversion
3. Process with model processor
4. Map label to ID: {good:0, bad:1, suspect:2}
5. Return batch dict:
   {
     'pixel_values': tensor(3, 384, 384),
     'labels': tensor(int),
     'station_name': str,
     'image_path': str
   }

Output: PyTorch Dataset ready for DataLoader





# To run:



for v3 (Ensemble swin transformer for the suspect class)

```
 nohup python3 multi_model_trainingv3.py > training_logv3.txt 2>&1 &
```


for v1
```
sleep 1800 && nohup python3 multi_model_training.py > training_log.txt 2>&1 &
```

for v2

```
sleep 3600 && nohup python3 multi_model_trainingv2.py > training_logv2.txt 2>&1 &
```


for v4

```
sleep 3600 && nohup python3 multi_model_trainingv4.py > training_logv4.txt 2>&1 &
```




Input: 343 images with labels

Process:
┌───────────────────────────────────────┐
│ Fold 1:                               │
│   Train: ~229 images (2/3)            │
│   Val:   ~114 images (1/3)            │
│   Stratified by class (good/bad/susp) │
└───────────────────────────────────────┘
┌───────────────────────────────────────┐
│ Fold 2:                               │
│   Train: ~229 images (different 2/3)  │
│   Val:   ~114 images (different 1/3)  │
└───────────────────────────────────────┘
┌───────────────────────────────────────┐
│ Fold 3:                               │
│   Train: ~229 images (remaining 2/3)  │
│   Val:   ~114 images (remaining 1/3)  │
└───────────────────────────────────────┘

Output: 3 independent train/val splits


