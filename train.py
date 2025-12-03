"""
Training script for Encoder-Decoder Anomaly Detection
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.autoencoder import Autoencoder, AnomalyDetector
from src.utils.data_utils import generate_synthetic_data, prepare_data, create_dataloaders
import config


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            data, _ = batch
        else:
            data = batch
        
        data = data.to(device)
        
        # Forward pass
        reconstruction = model(data)
        loss = criterion(reconstruction, data)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                data, _ = batch
            else:
                data = batch
            
            data = data.to(device)
            reconstruction = model(data)
            loss = criterion(reconstruction, data)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def train_autoencoder(model, train_loader, val_loader, num_epochs, learning_rate, device):
    """
    Train the autoencoder model
    
    Args:
        model: Autoencoder model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
    
    Returns:
        Dictionary with training history
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=config.WEIGHT_DECAY)
    
    train_losses = []
    val_losses = []
    
    print("Starting training...")
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    print("Training completed!")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses
    }


def plot_training_history(history, save_path=None):
    """Plot training and validation losses"""
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_losses'], label='Train Loss')
    plt.plot(history['val_losses'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")
    
    plt.close()


def main():
    """Main training function"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    
    # Generate synthetic data
    print("Generating synthetic data...")
    X, y = generate_synthetic_data(
        n_samples=1000,
        n_features=config.INPUT_DIM,
        contamination=config.CONTAMINATION,
        random_state=config.RANDOM_SEED
    )
    
    # Prepare data
    print("Preparing data...")
    data_dict = prepare_data(
        X, y,
        train_split=config.TRAIN_SPLIT,
        val_split=config.VAL_SPLIT,
        normalize=True
    )
    
    # Create dataloaders
    dataloaders = create_dataloaders(data_dict, batch_size=config.BATCH_SIZE)
    
    # Initialize model
    print("Initializing model...")
    model = Autoencoder(
        input_dim=config.INPUT_DIM,
        hidden_dims=config.HIDDEN_DIMS,
        latent_dim=config.LATENT_DIM
    ).to(device)
    
    print(f"Model architecture:\n{model}")
    
    # Train model
    history = train_autoencoder(
        model,
        dataloaders['train'],
        dataloaders['val'],
        num_epochs=config.NUM_EPOCHS,
        learning_rate=config.LEARNING_RATE,
        device=device
    )
    
    # Plot training history
    plot_training_history(history, save_path='results/training_history.png')
    
    # Save model
    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'input_dim': config.INPUT_DIM,
            'hidden_dims': config.HIDDEN_DIMS,
            'latent_dim': config.LATENT_DIM
        }
    }, config.MODEL_SAVE_PATH)
    print(f"Model saved to {config.MODEL_SAVE_PATH}")
    
    # Set threshold on validation data (normal samples only)
    print("\nSetting anomaly detection threshold...")
    detector = AnomalyDetector(model)
    
    # Get normal samples from validation set
    X_val, y_val = data_dict['val']
    X_val_normal = X_val[y_val == 0]
    X_val_normal_tensor = torch.FloatTensor(X_val_normal).to(device)
    
    threshold = detector.set_threshold(X_val_normal_tensor, percentile=config.THRESHOLD_PERCENTILE)
    print(f"Anomaly threshold set to: {threshold:.6f}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    X_test, y_test = data_dict['test']
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    predictions, errors = detector.predict(X_test_tensor)
    predictions = predictions.cpu().tolist()
    errors = errors.cpu().tolist()
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    
    print(f"\nTest Set Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))


if __name__ == '__main__':
    main()
