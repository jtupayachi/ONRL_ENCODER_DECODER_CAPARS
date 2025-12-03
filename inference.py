"""
Inference script for Encoder-Decoder Anomaly Detection
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.autoencoder import Autoencoder, AnomalyDetector
from src.utils.data_utils import generate_synthetic_data, prepare_data
import config


def load_model(model_path, device):
    """Load trained model from checkpoint"""
    checkpoint = torch.load(model_path, map_location=device)
    
    model = Autoencoder(
        input_dim=checkpoint['config']['input_dim'],
        hidden_dims=checkpoint['config']['hidden_dims'],
        latent_dim=checkpoint['config']['latent_dim']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def visualize_reconstruction(model, X, y, num_samples=5, save_path=None):
    """Visualize original vs reconstructed samples"""
    model.eval()
    
    # Select samples
    normal_indices = np.where(y == 0)[0][:num_samples]
    anomaly_indices = np.where(y == 1)[0][:num_samples]
    
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    
    with torch.no_grad():
        # Normal samples
        for i, idx in enumerate(normal_indices):
            x = torch.FloatTensor(X[idx:idx+1])
            reconstruction = model(x).numpy()[0]
            
            axes[0, i].bar(range(len(X[idx])), X[idx], alpha=0.7, label='Original')
            axes[0, i].bar(range(len(reconstruction)), reconstruction, alpha=0.7, label='Reconstructed')
            axes[0, i].set_title(f'Normal {i+1}')
            axes[0, i].legend()
        
        # Anomaly samples
        for i, idx in enumerate(anomaly_indices):
            x = torch.FloatTensor(X[idx:idx+1])
            reconstruction = model(x).numpy()[0]
            
            axes[1, i].bar(range(len(X[idx])), X[idx], alpha=0.7, label='Original')
            axes[1, i].bar(range(len(reconstruction)), reconstruction, alpha=0.7, label='Reconstructed')
            axes[1, i].set_title(f'Anomaly {i+1}')
            axes[1, i].legend()
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Reconstruction visualization saved to {save_path}")
    
    plt.close()


def visualize_error_distribution(errors, labels, save_path=None):
    """Visualize reconstruction error distribution"""
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Histogram
    plt.subplot(1, 2, 1)
    plt.hist(errors[labels == 0], bins=50, alpha=0.7, label='Normal', color='blue')
    plt.hist(errors[labels == 1], bins=50, alpha=0.7, label='Anomaly', color='red')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.legend()
    plt.grid(True)
    
    # Subplot 2: Box plot
    plt.subplot(1, 2, 2)
    data = [errors[labels == 0], errors[labels == 1]]
    plt.boxplot(data, labels=['Normal', 'Anomaly'])
    plt.ylabel('Reconstruction Error')
    plt.title('Error Box Plot')
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Error distribution visualization saved to {save_path}")
    
    plt.close()


def visualize_latent_space(model, X, y, save_path=None):
    """Visualize latent space representation (only for 2D latent space)"""
    model.eval()
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X)
        latent = model.encode(X_tensor).numpy()
    
    if latent.shape[1] == 2:
        plt.figure(figsize=(10, 8))
        
        # Plot normal samples
        plt.scatter(latent[y == 0, 0], latent[y == 0, 1], 
                   c='blue', alpha=0.6, label='Normal', s=50)
        
        # Plot anomalies
        plt.scatter(latent[y == 1, 0], latent[y == 1, 1], 
                   c='red', alpha=0.6, label='Anomaly', s=50, marker='x')
        
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        plt.title('Latent Space Representation')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"Latent space visualization saved to {save_path}")
        
        plt.close()
    else:
        print(f"Latent space has {latent.shape[1]} dimensions. Visualization only available for 2D.")


def detect_anomalies(model_path, X, y=None):
    """
    Detect anomalies using trained model
    
    Args:
        model_path: Path to trained model
        X: Input data
        y: True labels (optional, for evaluation)
    
    Returns:
        predictions: Binary predictions (0=normal, 1=anomaly)
        errors: Reconstruction errors
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = load_model(model_path, device)
    
    # Create detector
    detector = AnomalyDetector(model)
    
    # For setting threshold, we need normal samples
    # If labels provided, use normal samples; otherwise use all data
    if y is not None:
        X_normal = X[y == 0]
    else:
        # Assume most data is normal
        X_normal = X
    
    X_normal_tensor = torch.FloatTensor(X_normal).to(device)
    threshold = detector.set_threshold(X_normal_tensor, percentile=config.THRESHOLD_PERCENTILE)
    
    # Predict on all data
    X_tensor = torch.FloatTensor(X).to(device)
    predictions, errors = detector.predict(X_tensor)
    
    return predictions.cpu().numpy(), errors.cpu().numpy()


def main():
    """Main inference function"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check if model exists
    if not os.path.exists(config.MODEL_SAVE_PATH):
        print(f"Model not found at {config.MODEL_SAVE_PATH}")
        print("Please train the model first by running: python train.py")
        return
    
    # Load model
    print("Loading model...")
    model = load_model(config.MODEL_SAVE_PATH, device)
    print("Model loaded successfully!")
    
    # Generate test data
    print("\nGenerating test data...")
    X, y = generate_synthetic_data(
        n_samples=500,
        n_features=config.INPUT_DIM,
        contamination=config.CONTAMINATION,
        random_state=config.RANDOM_SEED + 100
    )
    
    # Prepare data (normalize)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Detect anomalies
    print("\nDetecting anomalies...")
    predictions, errors = detect_anomalies(config.MODEL_SAVE_PATH, X, y)
    
    # Evaluate if labels are available
    if y is not None:
        from sklearn.metrics import classification_report, confusion_matrix
        
        print("\nClassification Report:")
        print(classification_report(y, predictions, target_names=['Normal', 'Anomaly']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y, predictions)
        print(cm)
    
    # Create visualizations
    print("\nCreating visualizations...")
    visualize_reconstruction(model, X, y, num_samples=5, save_path='results/reconstruction.png')
    visualize_error_distribution(errors, y, save_path='results/error_distribution.png')
    visualize_latent_space(model, X, y, save_path='results/latent_space.png')
    
    print("\nInference completed!")
    print(f"Found {predictions.sum()} anomalies out of {len(predictions)} samples")


if __name__ == '__main__':
    main()
