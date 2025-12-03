#!/bin/bash
# Quick start script for ONRL Encoder-Decoder Anomaly Detection

echo "==============================================="
echo "ONRL Encoder-Decoder Anomaly Detection"
echo "Quick Start Demo"
echo "==============================================="
echo ""

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed. Please install Python 3.7+ first."
    exit 1
fi

echo "Step 1: Installing dependencies..."
pip install -r requirements.txt --quiet

echo ""
echo "Step 2: Running unit tests..."
python -m unittest tests.test_autoencoder -v

echo ""
echo "==============================================="
echo "Step 3: Running usage examples..."
echo "==============================================="
python examples/usage_examples.py

echo ""
echo "==============================================="
echo "Step 4: Training the model..."
echo "==============================================="
echo "This will train for 100 epochs (may take a few minutes)"
python train.py

echo ""
echo "==============================================="
echo "Step 5: Running inference..."
echo "==============================================="
python inference.py

echo ""
echo "==============================================="
echo "Quick Start Complete!"
echo "==============================================="
echo ""
echo "Generated files:"
echo "  - Model: models/autoencoder.pth"
echo "  - Visualizations: results/*.png"
echo ""
echo "Check the README.md for more information and advanced usage."
