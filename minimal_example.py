#!/usr/bin/env python3
"""
Minimal example of using the Encoder-Decoder for Anomaly Detection
"""

import torch
import sys
import os

# Add project root to path to allow importing src modules
# This is a simple solution for standalone scripts without requiring package installation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.autoencoder import Autoencoder, AnomalyDetector
from src.utils.data_utils import generate_synthetic_data

def main():
    print("=" * 60)
    print("Minimal Encoder-Decoder Anomaly Detection Example")
    print("=" * 60)
    print()
    
    # 1. Generate synthetic data
    print("1. Generating synthetic data...")
    X, y = generate_synthetic_data(n_samples=200, n_features=10, contamination=0.1)
    print(f"   Created {len(X)} samples with {X.shape[1]} features")
    print(f"   Normal samples: {(y == 0).sum()}, Anomalies: {(y == 1).sum()}")
    print()
    
    # 2. Create and display model
    print("2. Creating autoencoder model...")
    model = Autoencoder(input_dim=10, hidden_dims=[8, 4], latent_dim=2)
    print(f"   Architecture: 10 -> 8 -> 4 -> 2 -> 4 -> 8 -> 10")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters())}")
    print()
    
    # 3. Quick training (just a few epochs for demo)
    print("3. Training model (30 epochs)...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    
    X_tensor = torch.FloatTensor(X)
    model.train()
    
    for epoch in range(30):
        reconstruction = model(X_tensor)
        loss = criterion(reconstruction, X_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1}/30, Loss: {loss.item():.6f}")
    print()
    
    # 4. Set up anomaly detector
    print("4. Setting up anomaly detector...")
    detector = AnomalyDetector(model)
    
    # Use only normal samples for threshold
    X_normal = X[y == 0]
    X_normal_tensor = torch.FloatTensor(X_normal)
    threshold = detector.set_threshold(X_normal_tensor, percentile=95)
    print(f"   Anomaly threshold: {threshold:.6f}")
    print()
    
    # 5. Detect anomalies
    print("5. Detecting anomalies...")
    predictions, errors = detector.predict(X_tensor)
    predictions = predictions.cpu().tolist()
    
    # Calculate accuracy
    correct = sum(1 for p, t in zip(predictions, y) if p == t)
    accuracy = correct / len(y)
    
    print(f"   Detected {sum(predictions)} anomalies")
    print(f"   True anomalies: {(y == 1).sum()}")
    print(f"   Accuracy: {accuracy:.2%}")
    print()
    
    # 6. Show some examples
    print("6. Example reconstruction errors:")
    print("   Normal samples (first 5):")
    y_tensor = torch.tensor(y.tolist())
    normal_errors = [e for e, label in zip(errors.cpu().tolist(), y) if label == 0][:5]
    for i, err in enumerate(normal_errors, 1):
        print(f"     Sample {i}: {err:.6f}")
    
    print("   Anomaly samples (first 5):")
    anomaly_errors = [e for e, label in zip(errors.cpu().tolist(), y) if label == 1][:5]
    for i, err in enumerate(anomaly_errors, 1):
        print(f"     Sample {i}: {err:.6f}")
    print()
    
    print("=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
    print()
    print("For more features, run:")
    print("  - python train.py       # Full training with visualization")
    print("  - python inference.py   # Inference with plots")
    print("  - python examples/usage_examples.py  # More examples")


if __name__ == '__main__':
    main()
