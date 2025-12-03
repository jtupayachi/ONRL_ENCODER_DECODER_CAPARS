"""
Configuration file for Encoder-Decoder Anomaly Detection
"""

# Model Architecture
INPUT_DIM = 10  # Dimensionality of input features
HIDDEN_DIMS = [8, 4, 2]  # Encoder hidden layer dimensions (decoder mirrors this)
LATENT_DIM = 2  # Bottleneck dimension

# Training Parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
WEIGHT_DECAY = 1e-5

# Anomaly Detection
CONTAMINATION = 0.1  # Expected proportion of anomalies in the dataset
THRESHOLD_PERCENTILE = 95  # Percentile for anomaly threshold

# Data
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# Paths
MODEL_SAVE_PATH = "models/autoencoder.pth"
RESULTS_PATH = "results/"
