"""
Encoder-Decoder (Autoencoder) Architecture for Anomaly Detection
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Encoder network that compresses input to latent representation"""
    
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super(Encoder, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Create hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim
        
        # Final layer to latent space
        layers.append(nn.Linear(prev_dim, latent_dim))
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    """Decoder network that reconstructs input from latent representation"""
    
    def __init__(self, latent_dim, hidden_dims, output_dim):
        super(Decoder, self).__init__()
        
        layers = []
        prev_dim = latent_dim
        
        # Reverse the hidden dimensions for symmetric architecture
        reversed_hidden_dims = hidden_dims[::-1]
        
        # Create hidden layers
        for hidden_dim in reversed_hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim
        
        # Final layer to output space
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.decoder(x)


class Autoencoder(nn.Module):
    """Complete Autoencoder for Anomaly Detection"""
    
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super(Autoencoder, self).__init__()
        
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dims, input_dim)
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction
    
    def encode(self, x):
        """Get latent representation"""
        return self.encoder(x)
    
    def decode(self, latent):
        """Reconstruct from latent representation"""
        return self.decoder(latent)


class AnomalyDetector:
    """Wrapper class for anomaly detection using autoencoder"""
    
    def __init__(self, model, threshold=None):
        self.model = model
        self.threshold = threshold
    
    def compute_reconstruction_error(self, x, reduction='none'):
        """Compute reconstruction error (MSE)"""
        self.model.eval()
        with torch.no_grad():
            reconstruction = self.model(x)
            error = torch.mean((x - reconstruction) ** 2, dim=1)
            
            if reduction == 'mean':
                return error.mean()
            elif reduction == 'sum':
                return error.sum()
            else:
                return error
    
    def set_threshold(self, normal_data, percentile=95):
        """Set anomaly threshold based on normal data"""
        errors = self.compute_reconstruction_error(normal_data)
        self.threshold = torch.quantile(errors, percentile / 100.0)
        return self.threshold
    
    def predict(self, x):
        """Predict anomalies (1 for anomaly, 0 for normal)"""
        if self.threshold is None:
            raise ValueError("Threshold not set. Call set_threshold() first.")
        
        errors = self.compute_reconstruction_error(x)
        predictions = (errors > self.threshold).long()
        return predictions, errors
