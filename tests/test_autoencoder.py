"""
Unit tests for Encoder-Decoder Anomaly Detection
"""

import unittest
import torch
import numpy as np
import sys
import os

# Add project root to path to allow importing src modules
# This is standard practice for test files that are not installed as a package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.autoencoder import Encoder, Decoder, Autoencoder, AnomalyDetector
from src.utils.data_utils import generate_synthetic_data, prepare_data, create_dataloaders, AnomalyDataset


class TestEncoder(unittest.TestCase):
    """Test Encoder class"""
    
    def test_encoder_forward(self):
        """Test encoder forward pass"""
        encoder = Encoder(input_dim=10, hidden_dims=[8, 4], latent_dim=2)
        x = torch.randn(5, 10)
        output = encoder(x)
        
        self.assertEqual(output.shape, (5, 2))
    
    def test_encoder_dimensions(self):
        """Test encoder with different dimensions"""
        encoder = Encoder(input_dim=20, hidden_dims=[16, 12, 8], latent_dim=4)
        x = torch.randn(10, 20)
        output = encoder(x)
        
        self.assertEqual(output.shape, (10, 4))


class TestDecoder(unittest.TestCase):
    """Test Decoder class"""
    
    def test_decoder_forward(self):
        """Test decoder forward pass"""
        decoder = Decoder(latent_dim=2, hidden_dims=[8, 4], output_dim=10)
        x = torch.randn(5, 2)
        output = decoder(x)
        
        self.assertEqual(output.shape, (5, 10))
    
    def test_decoder_dimensions(self):
        """Test decoder with different dimensions"""
        decoder = Decoder(latent_dim=4, hidden_dims=[16, 12, 8], output_dim=20)
        x = torch.randn(10, 4)
        output = decoder(x)
        
        self.assertEqual(output.shape, (10, 20))


class TestAutoencoder(unittest.TestCase):
    """Test Autoencoder class"""
    
    def test_autoencoder_forward(self):
        """Test autoencoder forward pass"""
        model = Autoencoder(input_dim=10, hidden_dims=[8, 4], latent_dim=2)
        x = torch.randn(5, 10)
        output = model(x)
        
        self.assertEqual(output.shape, x.shape)
    
    def test_encode_decode(self):
        """Test encode and decode methods"""
        model = Autoencoder(input_dim=10, hidden_dims=[8, 4], latent_dim=2)
        x = torch.randn(5, 10)
        
        latent = model.encode(x)
        self.assertEqual(latent.shape, (5, 2))
        
        reconstruction = model.decode(latent)
        self.assertEqual(reconstruction.shape, x.shape)
    
    def test_reconstruction(self):
        """Test that reconstruction works end-to-end"""
        model = Autoencoder(input_dim=5, hidden_dims=[4], latent_dim=2)
        x = torch.randn(3, 5)
        
        # Forward pass should not crash
        reconstruction = model(x)
        self.assertEqual(reconstruction.shape, x.shape)


class TestAnomalyDetector(unittest.TestCase):
    """Test AnomalyDetector class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = Autoencoder(input_dim=10, hidden_dims=[8, 4], latent_dim=2)
        self.detector = AnomalyDetector(self.model)
    
    def test_reconstruction_error(self):
        """Test reconstruction error computation"""
        x = torch.randn(5, 10)
        errors = self.detector.compute_reconstruction_error(x)
        
        self.assertEqual(errors.shape, (5,))
        self.assertTrue(torch.all(errors >= 0))
    
    def test_set_threshold(self):
        """Test threshold setting"""
        x = torch.randn(100, 10)
        threshold = self.detector.set_threshold(x, percentile=95)
        
        self.assertIsNotNone(threshold)
        self.assertTrue(threshold > 0)
    
    def test_predict(self):
        """Test anomaly prediction"""
        # Set threshold first
        normal_data = torch.randn(100, 10)
        self.detector.set_threshold(normal_data, percentile=95)
        
        # Predict on test data
        test_data = torch.randn(20, 10)
        predictions, errors = self.detector.predict(test_data)
        
        self.assertEqual(predictions.shape, (20,))
        self.assertEqual(errors.shape, (20,))
        self.assertTrue(torch.all((predictions == 0) | (predictions == 1)))


class TestDataUtils(unittest.TestCase):
    """Test data utility functions"""
    
    def test_generate_synthetic_data(self):
        """Test synthetic data generation"""
        X, y = generate_synthetic_data(n_samples=100, n_features=10, contamination=0.1)
        
        self.assertEqual(X.shape, (100, 10))
        self.assertEqual(y.shape, (100,))
        self.assertAlmostEqual(y.sum() / len(y), 0.1, delta=0.01)
    
    def test_prepare_data(self):
        """Test data preparation"""
        X, y = generate_synthetic_data(n_samples=100, n_features=5)
        data_dict = prepare_data(X, y, train_split=0.7, val_split=0.15)
        
        self.assertIn('train', data_dict)
        self.assertIn('val', data_dict)
        self.assertIn('test', data_dict)
        self.assertIn('scaler', data_dict)
        
        X_train, y_train = data_dict['train']
        self.assertEqual(len(X_train), 70)
    
    def test_create_dataloaders(self):
        """Test dataloader creation"""
        X, y = generate_synthetic_data(n_samples=100, n_features=5)
        data_dict = prepare_data(X, y)
        dataloaders = create_dataloaders(data_dict, batch_size=10)
        
        self.assertIn('train', dataloaders)
        self.assertIn('val', dataloaders)
        self.assertIn('test', dataloaders)


class TestAnomalyDataset(unittest.TestCase):
    """Test AnomalyDataset class"""
    
    def test_dataset_creation(self):
        """Test dataset creation"""
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)
        
        dataset = AnomalyDataset(X, y)
        
        self.assertEqual(len(dataset), 50)
    
    def test_dataset_getitem(self):
        """Test getting items from dataset"""
        X = np.random.randn(10, 5)
        y = np.random.randint(0, 2, 10)
        
        dataset = AnomalyDataset(X, y)
        data, label = dataset[0]
        
        self.assertEqual(data.shape, (5,))
        self.assertIn(label.item(), [0, 1])


class TestEndToEnd(unittest.TestCase):
    """End-to-end integration tests"""
    
    def test_training_pipeline(self):
        """Test complete training pipeline"""
        # Generate data
        X, y = generate_synthetic_data(n_samples=100, n_features=5, contamination=0.1)
        
        # Prepare data
        data_dict = prepare_data(X, y, train_split=0.7, val_split=0.15)
        dataloaders = create_dataloaders(data_dict, batch_size=10)
        
        # Create model
        model = Autoencoder(input_dim=5, hidden_dims=[4], latent_dim=2)
        
        # Train for a few epochs
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()
        
        model.train()
        for batch in dataloaders['train']:
            data, _ = batch
            reconstruction = model(data)
            loss = criterion(reconstruction, data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            break  # Just test one batch
        
        # This test passes if no exceptions are raised
        self.assertTrue(True)
    
    def test_inference_pipeline(self):
        """Test complete inference pipeline"""
        # Generate and prepare data
        X, y = generate_synthetic_data(n_samples=100, n_features=5, contamination=0.1)
        data_dict = prepare_data(X, y)
        
        # Create and set up model
        model = Autoencoder(input_dim=5, hidden_dims=[4], latent_dim=2)
        detector = AnomalyDetector(model)
        
        # Set threshold
        X_train, y_train = data_dict['train']
        X_normal = X_train[y_train == 0]
        detector.set_threshold(torch.FloatTensor(X_normal), percentile=95)
        
        # Predict
        X_test, y_test = data_dict['test']
        predictions, errors = detector.predict(torch.FloatTensor(X_test))
        
        # Verify outputs
        self.assertEqual(len(predictions), len(X_test))
        self.assertEqual(len(errors), len(X_test))
        
        # This test passes if no exceptions are raised
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
