#!/usr/bin/env python3
"""
Vision Transformer (ViT) for Meteorological Data Quality Classification

This script uses google/vit-large-patch16-384 to classify meteorological 
time series images (wind speed/direction plots) into quality categories:
- Good: High quality data
- Bad: Poor quality data with issues
- Suspect: Data that needs review

The model uses pretrained ViT and fine-tunes it on station images.
"""

import os
import json
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from transformers import ViTImageProcessor, ViTForImageClassification, ViTConfig
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# Configuration
# ============================================================================

# Paths
IMAGE_BASE_DIR = Path("/home/jose/ONRL_ENCODER_DECODER_CAPARS/trimmed_images")  # Use trimmed images
METADATA_FILE = Path("/home/jose/ONRL_ENCODER_DECODER_CAPARS/station_metadata.csv")
OUTPUT_DIR = Path("/home/jose/ONRL_ENCODER_DECODER_CAPARS/model_outputs_vit")
OUTPUT_DIR.mkdir(exist_ok=True)

# Model configuration
MODEL_NAME = "google/vit-large-patch16-384"  # ViT-Large with 384x384 input
IMAGE_SIZE = 384  # Required by vit-large-patch16-384
NUM_CLASSES = 3  # good, bad, suspect

# Training configuration
BATCH_SIZE = 8  # Smaller batch size for large model
NUM_EPOCHS = 20
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
NUM_FOLDS = 3  # K-fold cross-validation
TRAIN_PORTION = 0.8  # Use 80% of data for training (20% held out for final test)

# Use both speed and direction images
USE_SPEED_IMAGES = True
USE_DIRECTION_IMAGES = True

# Random seed for reproducibility
SEED = 42

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Label mapping
LABEL_TO_ID = {"good": 0, "bad": 1, "suspect": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# Data Loading
# ============================================================================

def load_category_mapping() -> Dict[str, str]:
    """Load true station category labels from Excel file"""
    try:
        df = pd.read_excel(CATEGORY_EXCEL)
        category_map = {}
        for _, row in df.iterrows():
            station = str(row['Station']).strip()
            category = str(row['Category']).strip().lower()
            category_map[station] = category
        print(f"Loaded {len(category_map)} station categories from Excel")
        return category_map
    except Exception as e:
        print(f"Error loading category Excel: {e}")
        return {}


def get_station_name_from_image(image_path: Path) -> str:
    """Extract station name from image filename like 'MesoWest_AIRNOW_A2679_speed.png'"""
    name = image_path.stem  # Remove extension
    # Remove _speed or _dir suffix
    if name.endswith('_speed'):
        name = name[:-6]
    elif name.endswith('_dir'):
        name = name[:-4]
    return name


def collect_image_dataset(category_map: Dict[str, str]) -> List[Tuple[Path, str, str]]:
    """
    Collect all images with their true labels from Excel mapping.
    
    Returns list of (image_path, label, station_name) tuples.
    """
    dataset = []
    
    image_types = []
    if USE_SPEED_IMAGES:
        image_types.append(("speed", "_speed.png"))
    if USE_DIRECTION_IMAGES:
        image_types.append(("dir", "_dir.png"))
    
    for img_type, suffix in image_types:
        type_dir = IMAGE_BASE_DIR / img_type
        
        # Look in all subdirectories (good, bad, suspect)
        for class_dir in ['good', 'bad', 'suspect']:
            subdir = type_dir / class_dir
            if not subdir.exists():
                continue
            
            for image_path in subdir.glob("*.png"):
                station_name = get_station_name_from_image(image_path)
                
                # Get true label from Excel, fallback to directory name
                true_label = category_map.get(station_name, class_dir)
                
                if true_label in LABEL_TO_ID:
                    dataset.append((image_path, true_label, station_name))
    
    return dataset


class MeteorologicalImageDataset(Dataset):
    """PyTorch Dataset for meteorological time series images"""
    
    def __init__(self, 
                 data: List[Tuple[Path, str, str]], 
                 processor: ViTImageProcessor,
                 augment: bool = False):
        """
        Args:
            data: List of (image_path, label, station_name) tuples
            processor: ViT image processor for preprocessing
            augment: Whether to apply data augmentation
        """
        self.data = data
        self.processor = processor
        self.augment = augment
        
        # Data augmentation transforms
        if augment:
            self.aug_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomRotation(5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
            ])
        else:
            self.aug_transform = None
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path, label, station_name = self.data[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Apply augmentation if enabled
        if self.aug_transform:
            image = self.aug_transform(image)
        
        # Process image for ViT
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)
        
        # Convert label to ID
        label_id = LABEL_TO_ID[label]
        
        return {
            'pixel_values': pixel_values,
            'labels': torch.tensor(label_id),
            'station_name': station_name,
            'image_path': str(image_path)
        }


# ============================================================================
# Model
# ============================================================================

def create_model(num_classes: int, pretrained: bool = True) -> ViTForImageClassification:
    """Create ViT model for classification"""
    if pretrained:
        model = ViTForImageClassification.from_pretrained(
            MODEL_NAME,
            num_labels=num_classes,
            id2label=ID_TO_LABEL,
            label2id=LABEL_TO_ID,
            ignore_mismatched_sizes=True  # For classifier head
        )
    else:
        config = ViTConfig.from_pretrained(MODEL_NAME)
        config.num_labels = num_classes
        config.id2label = ID_TO_LABEL
        config.label2id = LABEL_TO_ID
        model = ViTForImageClassification(config)
    
    return model


# ============================================================================
# Training
# ============================================================================

def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def evaluate(model, dataloader, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_stations = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(pixel_values=pixel_values, labels=labels)
            
            total_loss += outputs.loss.item()
            
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_stations.extend(batch['station_name'])
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, accuracy, f1, all_preds, all_labels, all_stations


# ============================================================================
# Visualization
# ============================================================================

def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - ViT Classification')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_training_history(history: Dict, save_path: Path):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train Accuracy', marker='o')
    axes[1].plot(history['val_acc'], label='Val Accuracy', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("Vision Transformer (ViT) for Meteorological Data Quality Classification")
    print("=" * 70)
    print(f"\nModel: {MODEL_NAME}")
    print(f"Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Epochs: {NUM_EPOCHS}")
    
    # Set seed
    set_seed(SEED)
    
    # Load category mapping from Excel
    category_map = load_category_mapping()
    
    # Collect dataset
    print("\n--- Loading Dataset ---")
    dataset = collect_image_dataset(category_map)
    print(f"Total images: {len(dataset)}")
    
    # Count by class
    class_counts = {}
    for _, label, _ in dataset:
        class_counts[label] = class_counts.get(label, 0) + 1
    print("Class distribution:")
    for cls, count in sorted(class_counts.items()):
        print(f"  {cls}: {count}")
    
    # Split dataset
    train_data, temp_data = train_test_split(dataset, test_size=(1 - TRAIN_SPLIT), 
                                              random_state=SEED, stratify=[d[1] for d in dataset])
    val_data, test_data = train_test_split(temp_data, test_size=TEST_SPLIT/(VAL_SPLIT + TEST_SPLIT),
                                            random_state=SEED, stratify=[d[1] for d in temp_data])
    
    print(f"\nSplit: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # Load processor
    print("\n--- Loading ViT Processor ---")
    processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    
    # Create datasets
    train_dataset = MeteorologicalImageDataset(train_data, processor, augment=True)
    val_dataset = MeteorologicalImageDataset(val_data, processor, augment=False)
    test_dataset = MeteorologicalImageDataset(test_data, processor, augment=False)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Create model
    print("\n--- Loading ViT Model ---")
    model = create_model(NUM_CLASSES, pretrained=True)
    model = model.to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    total_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        total_steps=total_steps,
        pct_start=WARMUP_RATIO
    )
    
    # Training loop
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }
    
    best_val_acc = 0
    best_epoch = 0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, DEVICE)
        
        # Validate
        val_loss, val_acc, val_f1, _, _, _ = evaluate(model, val_loader, DEVICE)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), OUTPUT_DIR / "vit_best_model.pt")
            print(f"  → New best model saved!")
    
    print(f"\nBest validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    
    # Load best model for testing
    model.load_state_dict(torch.load(OUTPUT_DIR / "vit_best_model.pt"))
    
    # Final evaluation on test set
    print("\n" + "=" * 70)
    print("TEST EVALUATION")
    print("=" * 70)
    
    test_loss, test_acc, test_f1, test_preds, test_labels, test_stations = evaluate(model, test_loader, DEVICE)
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    
    # Classification report
    print("\n--- Classification Report ---")
    label_names = [ID_TO_LABEL[i] for i in range(NUM_CLASSES)]
    report = classification_report(test_labels, test_preds, target_names=label_names)
    print(report)
    
    # Save results
    print("\n--- Saving Results ---")
    
    # Save confusion matrix
    plot_confusion_matrix(test_labels, test_preds, label_names, OUTPUT_DIR / "confusion_matrix.png")
    print(f"  Saved: confusion_matrix.png")
    
    # Save training history plot
    plot_training_history(history, OUTPUT_DIR / "training_history.png")
    print(f"  Saved: training_history.png")
    
    # Save training history JSON
    with open(OUTPUT_DIR / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  Saved: training_history.json")
    
    # Save configuration
    config = {
        'model_name': MODEL_NAME,
        'image_size': IMAGE_SIZE,
        'num_classes': NUM_CLASSES,
        'batch_size': BATCH_SIZE,
        'num_epochs': NUM_EPOCHS,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'train_split': TRAIN_SPLIT,
        'val_split': VAL_SPLIT,
        'test_split': TEST_SPLIT,
        'seed': SEED,
        'use_speed_images': USE_SPEED_IMAGES,
        'use_direction_images': USE_DIRECTION_IMAGES,
        'total_samples': len(dataset),
        'train_samples': len(train_data),
        'val_samples': len(val_data),
        'test_samples': len(test_data),
        'class_distribution': class_counts,
        'label_mapping': LABEL_TO_ID,
        'best_epoch': best_epoch,
        'best_val_accuracy': best_val_acc,
        'timestamp': datetime.now().isoformat()
    }
    with open(OUTPUT_DIR / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  Saved: config.json")
    
    # Save evaluation results
    eval_results = {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'test_f1_score': test_f1,
        'classification_report': classification_report(test_labels, test_preds, 
                                                        target_names=label_names, output_dict=True),
        'confusion_matrix': confusion_matrix(test_labels, test_preds).tolist()
    }
    with open(OUTPUT_DIR / "evaluation_results.json", 'w') as f:
        json.dump(eval_results, f, indent=2)
    print(f"  Saved: evaluation_results.json")
    
    # Save predictions per station
    predictions_df = pd.DataFrame({
        'station': test_stations,
        'true_label': [ID_TO_LABEL[l] for l in test_labels],
        'predicted_label': [ID_TO_LABEL[p] for p in test_preds],
        'correct': [t == p for t, p in zip(test_labels, test_preds)]
    })
    predictions_df.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
    print(f"  Saved: test_predictions.csv")
    
    # Show misclassified examples
    print("\n--- Misclassified Stations ---")
    misclassified = predictions_df[~predictions_df['correct']]
    if len(misclassified) > 0:
        print(misclassified.to_string())
    else:
        print("No misclassified samples!")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
