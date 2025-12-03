"""
Example usage of the Encoder-Decoder Anomaly Detection system
"""

import numpy as np
import torch
import sys
import os

# Add project root to path to allow importing src modules  
# This is a simple solution for standalone examples without requiring package installation
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from src.models.autoencoder import Autoencoder, AnomalyDetector
from src.utils.data_utils import generate_synthetic_data, prepare_data, create_dataloaders
import config


def example_basic_usage():
    """Example: Basic usage of the autoencoder for anomaly detection"""
    print("="*60)
    print("Example 1: Basic Usage")
    print("="*60)
    
    # Generate data
    X, y = generate_synthetic_data(n_samples=100, n_features=5, contamination=0.1)
    print(f"Generated {len(X)} samples with {X.shape[1]} features")
    print(f"Normal samples: {(y == 0).sum()}, Anomalies: {(y == 1).sum()}")
    
    # Create and train a simple model
    model = Autoencoder(input_dim=5, hidden_dims=[4, 3], latent_dim=2)
    print(f"\nModel created with architecture: 5 -> 4 -> 3 -> 2 -> 3 -> 4 -> 5")
    
    # Quick training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    
    X_tensor = torch.FloatTensor(X)
    model.train()
    
    for epoch in range(50):
        reconstruction = model(X_tensor)
        loss = criterion(reconstruction, X_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/50, Loss: {loss.item():.6f}")
    
    # Detect anomalies
    detector = AnomalyDetector(model)
    X_normal = X[y == 0]
    detector.set_threshold(torch.FloatTensor(X_normal), percentile=95)
    
    predictions, errors = detector.predict(X_tensor)
    predictions = predictions.cpu().detach()
    
    print(f"\nDetected {predictions.sum().item():.0f} anomalies")
    print(f"True anomalies: {(y == 1).sum()}")


def example_custom_architecture():
    """Example: Using custom architecture"""
    print("\n" + "="*60)
    print("Example 2: Custom Architecture")
    print("="*60)
    
    # Create a deeper autoencoder
    model = Autoencoder(
        input_dim=20,
        hidden_dims=[16, 12, 8, 4],
        latent_dim=2
    )
    
    print("Created deeper autoencoder:")
    print(f"  Input: 20 features")
    print(f"  Encoder: 20 -> 16 -> 12 -> 8 -> 4 -> 2")
    print(f"  Decoder: 2 -> 4 -> 8 -> 12 -> 16 -> 20")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")


def example_latent_representation():
    """Example: Working with latent representations"""
    print("\n" + "="*60)
    print("Example 3: Latent Space Representation")
    print("="*60)
    
    # Generate data
    X, y = generate_synthetic_data(n_samples=50, n_features=10, contamination=0.1)
    
    # Create model
    model = Autoencoder(input_dim=10, hidden_dims=[8, 4], latent_dim=2)
    
    # Get latent representations
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X)
        latent = model.encode(X_tensor)
        print(f"Encoded {len(X)} samples into {latent.shape[1]}D latent space")
        print(f"Latent representation shape: {latent.shape}")
        
        # Reconstruct from latent
        reconstruction = model.decode(latent)
        print(f"Reconstructed back to original {reconstruction.shape[1]}D space")


def example_reconstruction_error():
    """Example: Analyzing reconstruction errors"""
    print("\n" + "="*60)
    print("Example 4: Reconstruction Error Analysis")
    print("="*60)
    
    # Generate data
    X, y = generate_synthetic_data(n_samples=100, n_features=8, contamination=0.15)
    
    # Create and train model
    model = Autoencoder(input_dim=8, hidden_dims=[6, 4], latent_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    
    X_tensor = torch.FloatTensor(X)
    model.train()
    
    # Quick training
    for _ in range(30):
        reconstruction = model(X_tensor)
        loss = criterion(reconstruction, X_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Compute reconstruction errors
    detector = AnomalyDetector(model)
    errors = detector.compute_reconstruction_error(X_tensor).cpu().detach()
    
    # Convert y to tensor for indexing
    y_tensor = torch.tensor(y.tolist())
    
    # Analyze errors
    normal_errors = errors[y_tensor == 0]
    anomaly_errors = errors[y_tensor == 1]
    
    print(f"Normal samples - Mean error: {normal_errors.mean().item():.6f}, Std: {normal_errors.std().item():.6f}")
    print(f"Anomaly samples - Mean error: {anomaly_errors.mean().item():.6f}, Std: {anomaly_errors.std().item():.6f}")
    print(f"Error ratio (anomaly/normal): {(anomaly_errors.mean() / normal_errors.mean()).item():.2f}x")


def example_data_pipeline():
    """Example: Complete data pipeline"""
    print("\n" + "="*60)
    print("Example 5: Complete Data Pipeline")
    print("="*60)
    
    # Generate data
    X, y = generate_synthetic_data(n_samples=200, n_features=10, contamination=0.1)
    print(f"Generated dataset: {X.shape}")
    
    # Prepare data with train/val/test split
    data_dict = prepare_data(X, y, train_split=0.6, val_split=0.2, normalize=True)
    
    print(f"Train set: {data_dict['train'][0].shape}")
    print(f"Val set: {data_dict['val'][0].shape}")
    print(f"Test set: {data_dict['test'][0].shape}")
    
    # Create dataloaders
    dataloaders = create_dataloaders(data_dict, batch_size=16)
    
    print(f"\nDataLoaders created:")
    print(f"  Train batches: {len(dataloaders['train'])}")
    print(f"  Val batches: {len(dataloaders['val'])}")
    print(f"  Test batches: {len(dataloaders['test'])}")


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("ENCODER-DECODER ANOMALY DETECTION EXAMPLES")
    print("="*60 + "\n")
    
    try:
        example_basic_usage()
        example_custom_architecture()
        example_latent_representation()
        example_reconstruction_error()
        example_data_pipeline()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError running examples: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
