# ONRL Encoder-Decoder for Anomaly Detection

A PyTorch implementation of an encoder-decoder (autoencoder) architecture for anomaly detection. This project provides a complete pipeline for training, evaluating, and deploying anomaly detection models using deep learning.

## Overview

This implementation uses an autoencoder neural network to detect anomalies in data. The key principle is that the model learns to reconstruct normal data with low error, while anomalies produce higher reconstruction errors.

### Key Features

- **Flexible Architecture**: Configurable encoder-decoder with customizable hidden layers
- **Complete Pipeline**: Data generation, preprocessing, training, and inference
- **Visualization Tools**: Plot training history, reconstruction errors, and latent space
- **Anomaly Detection**: Automatic threshold setting and anomaly prediction
- **Easy to Use**: Simple configuration file and example scripts

## Architecture

The autoencoder consists of:

1. **Encoder**: Compresses input data to a lower-dimensional latent representation
2. **Latent Space**: Bottleneck layer capturing the most important features
3. **Decoder**: Reconstructs the original input from the latent representation

```
Input (n features) → Encoder → Latent Space (2D) → Decoder → Reconstruction (n features)
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/jtupayachi/ONRL_ENCODER_DECODER_CAPARS.git
cd ONRL_ENCODER_DECODER_CAPARS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Train the Model

```bash
python train.py
```

This will:
- Generate synthetic data with normal samples and anomalies
- Train the autoencoder model
- Save the trained model to `models/autoencoder.pth`
- Generate training history plots in `results/`

### 2. Run Inference

```bash
python inference.py
```

This will:
- Load the trained model
- Detect anomalies in test data
- Generate visualization plots (reconstructions, error distributions, latent space)
- Print performance metrics

### 3. See Examples

```bash
python examples/usage_examples.py
```

This demonstrates various use cases including custom architectures, latent space analysis, and data pipelines.

## Configuration

Edit `config.py` to customize the model and training parameters:

```python
# Model Architecture
INPUT_DIM = 10              # Input feature dimensions
HIDDEN_DIMS = [8, 4, 2]     # Encoder hidden layers (decoder mirrors this)
LATENT_DIM = 2              # Bottleneck dimension

# Training Parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 100

# Anomaly Detection
THRESHOLD_PERCENTILE = 95   # Percentile for anomaly threshold
```

## Usage Examples

### Basic Usage

```python
from src.models.autoencoder import Autoencoder, AnomalyDetector
import torch

# Create model
model = Autoencoder(input_dim=10, hidden_dims=[8, 4], latent_dim=2)

# Train model (simplified)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

# ... training loop ...

# Detect anomalies
detector = AnomalyDetector(model)
detector.set_threshold(normal_data, percentile=95)
predictions, errors = detector.predict(test_data)
```

### Custom Architecture

```python
# Create a deeper autoencoder
model = Autoencoder(
    input_dim=20,
    hidden_dims=[16, 12, 8, 4],
    latent_dim=2
)
```

### Working with Latent Space

```python
# Encode data to latent space
latent_representation = model.encode(data)

# Decode from latent space
reconstruction = model.decode(latent_representation)
```

## Project Structure

```
ONRL_ENCODER_DECODER_CAPARS/
├── config.py                    # Configuration parameters
├── train.py                     # Training script
├── inference.py                 # Inference and evaluation script
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── src/
│   ├── models/
│   │   └── autoencoder.py      # Model architecture
│   └── utils/
│       └── data_utils.py       # Data utilities
├── examples/
│   └── usage_examples.py       # Example scripts
├── models/                      # Saved models (created during training)
└── results/                     # Plots and results (created during training)
```

## How It Works

### Training Phase

1. **Data Preparation**: Normal data is used to train the autoencoder
2. **Learning**: The model learns to reconstruct normal patterns
3. **Threshold Setting**: Reconstruction errors on normal data determine the anomaly threshold

### Detection Phase

1. **Reconstruction**: New data is passed through the trained autoencoder
2. **Error Calculation**: Reconstruction error (MSE) is computed
3. **Classification**: Samples with errors above the threshold are flagged as anomalies

## Metrics

The system evaluates performance using:

- **Accuracy**: Overall classification accuracy
- **Precision**: Ratio of true anomalies among detected anomalies
- **Recall**: Ratio of detected anomalies among all true anomalies
- **F1-Score**: Harmonic mean of precision and recall

## Visualizations

The system generates several visualizations:

1. **Training History**: Loss curves for training and validation
2. **Reconstruction Comparison**: Original vs reconstructed samples
3. **Error Distribution**: Histogram and box plots of reconstruction errors
4. **Latent Space**: 2D visualization of latent representations (for 2D latent spaces)

## Customization

### Using Your Own Data

Replace the data generation in `train.py` with your own data loader:

```python
# Instead of generate_synthetic_data()
X, y = load_your_data()

# Ensure X is normalized
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
```

### Adjusting Sensitivity

Modify the `THRESHOLD_PERCENTILE` in `config.py`:
- Lower values (e.g., 90) → More sensitive (more anomalies detected)
- Higher values (e.g., 99) → Less sensitive (fewer anomalies detected)

## Requirements

- Python 3.7+
- PyTorch 1.9+
- NumPy
- scikit-learn
- Matplotlib
- Pandas
- Seaborn
- tqdm

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Citation

If you use this code in your research, please cite:

```
@software{onrl_encoder_decoder,
  author = {ONRL},
  title = {Encoder-Decoder for Anomaly Detection},
  year = {2025},
  url = {https://github.com/jtupayachi/ONRL_ENCODER_DECODER_CAPARS}
}
```

## References

- Autoencoder-based anomaly detection methodology
- Deep learning for unsupervised anomaly detection
- Reconstruction error as anomaly score