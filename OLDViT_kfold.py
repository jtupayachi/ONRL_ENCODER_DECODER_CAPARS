#!/usr/bin/env python3
"""
Vision Transformer (ViT) with 3-Fold Cross-Validation
for Meteorological Data Quality Classification

Uses google/vit-large-patch16-384 with trimmed images and 3-fold CV.
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
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from transformers import ViTImageProcessor, ViTForImageClassification, ViTConfig
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# Configuration
# ============================================================================

# Paths
IMAGE_BASE_DIR = Path("/home/jose/ONRL_ENCODER_DECODER_CAPARS/trimmed_images")
METADATA_FILE = Path("/home/jose/ONRL_ENCODER_DECODER_CAPARS/station_metadata.csv")
OUTPUT_DIR = Path("/home/jose/ONRL_ENCODER_DECODER_CAPARS/model_outputs_vit_kfold")
OUTPUT_DIR.mkdir(exist_ok=True)

# Model configuration
MODEL_NAME = "google/vit-large-patch16-384"
IMAGE_SIZE = 384
NUM_CLASSES = 3  # good, bad, suspect

# Training configuration
BATCH_SIZE = 8
NUM_EPOCHS = 15  # Reduced for k-fold
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
NUM_FOLDS = 3
EARLY_STOPPING_PATIENCE = 5  # Stop if no improvement for 5 epochs

# Preprocessing configuration
USE_GRAYSCALE = True  # Convert to grayscale (better for time-series patterns)
USE_AUGMENTATION = False  # Disable augmentation (breaks temporal structure)

# Use both speed and direction images
USE_SPEED_IMAGES = True
USE_DIRECTION_IMAGES = True

# Random seed
SEED = 42
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

def load_metadata() -> pd.DataFrame:
    """Load station metadata"""
    try:
        df = pd.read_csv(METADATA_FILE)
        print(f"Loaded metadata for {len(df)} records")
        return df
    except Exception as e:
        print(f"Warning: Could not load metadata: {e}")
        return pd.DataFrame()


def collect_image_dataset() -> List[Tuple[Path, str, str]]:
    """
    Collect all trimmed images with their labels from metadata file.
    Returns list of (image_path, label, station_name) tuples.
    """
    dataset = []
    
    # Load metadata to get true categories
    metadata_df = load_metadata()
    if metadata_df.empty:
        print("ERROR: No metadata loaded!")
        return dataset
    
    # Create station -> category mapping
    station_category_map = {}
    for _, row in metadata_df.iterrows():
        station_name = row['station_name']
        category = row['category']
        if pd.notna(category) and category in ['good', 'bad', 'suspect']:
            station_category_map[station_name] = category
    
    print(f"Loaded categories for {len(station_category_map)} stations from metadata")
    
    # Scan trimmed images directory (all in one folder)
    if not IMAGE_BASE_DIR.exists():
        print(f"ERROR: Directory not found: {IMAGE_BASE_DIR}")
        return dataset
    
    for image_path in IMAGE_BASE_DIR.glob("*.png"):
        name = image_path.stem
        
        # Extract station name and type
        if name.endswith('_speed'):
            if not USE_SPEED_IMAGES:
                continue
            station_name = name[:-6]
        elif name.endswith('_dir'):
            if not USE_DIRECTION_IMAGES:
                continue
            station_name = name[:-4]
        else:
            continue
        
        # Get category from metadata
        category = station_category_map.get(station_name)
        
        if category:
            dataset.append((image_path, category, station_name))
    
    return dataset


class MeteorologicalImageDataset(Dataset):
    """PyTorch Dataset for meteorological images"""
    
    def __init__(self, 
                 data: List[Tuple[Path, str, str]], 
                 processor: ViTImageProcessor,
                 use_grayscale: bool = True,
                 augment: bool = False):
        self.data = data
        self.processor = processor
        self.use_grayscale = use_grayscale
        self.augment = augment
        
        # Disable augmentation for time-series data
        # RandomHorizontalFlip breaks temporal order
        # RandomRotation distorts trends
        self.aug_transform = None
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path, label, station_name = self.data[idx]
        
        # Load image directly as grayscale or RGB
        if self.use_grayscale:
            # Load as grayscale and replicate to 3 channels in one step
            image = Image.open(image_path).convert('L')
            image = Image.merge('RGB', [image, image, image])
        else:
            image = Image.open(image_path).convert('RGB')
        
        # No augmentation applied (preserves temporal structure)
        
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)
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

def create_model(num_classes: int) -> ViTForImageClassification:
    """Create ViT model"""
    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_classes,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        ignore_mismatched_sizes=True
    )
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
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_kfold_history(fold_histories: List[Dict], save_path: Path):
    """Plot training history across all folds"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Train loss
    for i, history in enumerate(fold_histories):
        axes[0, 0].plot(history['train_loss'], label=f'Fold {i+1}', marker='o', alpha=0.7)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss (All Folds)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Val loss
    for i, history in enumerate(fold_histories):
        axes[0, 1].plot(history['val_loss'], label=f'Fold {i+1}', marker='s', alpha=0.7)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Validation Loss (All Folds)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Train accuracy
    for i, history in enumerate(fold_histories):
        axes[1, 0].plot(history['train_acc'], label=f'Fold {i+1}', marker='o', alpha=0.7)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Training Accuracy (All Folds)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Val accuracy
    for i, history in enumerate(fold_histories):
        axes[1, 1].plot(history['val_acc'], label=f'Fold {i+1}', marker='s', alpha=0.7)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Validation Accuracy (All Folds)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ============================================================================
# K-Fold Cross-Validation
# ============================================================================

def run_kfold_training(dataset_list, processor):
    """Run k-fold cross-validation training"""
    
    # Extract labels for stratification
    labels = np.array([LABEL_TO_ID[item[1]] for item in dataset_list])
    
    # Create k-fold splitter
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    
    fold_results = []
    fold_histories = []
    all_fold_predictions = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print("\n" + "=" * 70)
        print(f"FOLD {fold + 1}/{NUM_FOLDS}")
        print("=" * 70)
        
        # Create fold directory
        fold_dir = OUTPUT_DIR / f"fold_{fold+1}"
        fold_dir.mkdir(exist_ok=True)
        
        # Split data
        train_data = [dataset_list[i] for i in train_idx]
        val_data = [dataset_list[i] for i in val_idx]
        
        print(f"Train: {len(train_data)}, Val: {len(val_data)}")
        
        # Create datasets (no augmentation, use grayscale)
        train_dataset = MeteorologicalImageDataset(train_data, processor, 
                                                   use_grayscale=USE_GRAYSCALE, 
                                                   augment=False)
        val_dataset = MeteorologicalImageDataset(val_data, processor, 
                                                 use_grayscale=USE_GRAYSCALE, 
                                                 augment=False)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        
        # Create model
        print(f"\n--- Loading Model for Fold {fold+1} ---")
        model = create_model(NUM_CLASSES)
        model = model.to(DEVICE)
        
        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        
        total_steps = len(train_loader) * NUM_EPOCHS
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=LEARNING_RATE,
            total_steps=total_steps,
            pct_start=WARMUP_RATIO
        )
        
        # Training loop with early stopping
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': []
        }
        
        best_val_loss = float('inf')
        best_val_acc = 0
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(NUM_EPOCHS):
            print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")
            
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, DEVICE)
            val_loss, val_acc, val_f1, _, _, _ = evaluate(model, val_loader, DEVICE)
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
            
            # Early stopping based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_epoch = epoch + 1
                patience_counter = 0
                torch.save(model.state_dict(), fold_dir / "best_model.pt")
                print(f"  → New best model saved! (loss improved)")
            else:
                patience_counter += 1
                print(f"  → No improvement (patience: {patience_counter}/{EARLY_STOPPING_PATIENCE})")
                
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    print(f"\n⚠ Early stopping triggered at epoch {epoch + 1}")
                    print(f"Best epoch was {best_epoch} with val_loss={best_val_loss:.4f}")
                    break
        
        # Load best model for final evaluation
        model.load_state_dict(torch.load(fold_dir / "best_model.pt"))
        
        # Final evaluation
        val_loss, val_acc, val_f1, val_preds, val_labels, val_stations = evaluate(model, val_loader, DEVICE)
        
        print(f"\nFold {fold+1} Best Results:")
        print(f"  Best Epoch: {best_epoch}")
        print(f"  Val Accuracy: {val_acc:.4f}")
        print(f"  Val F1 Score: {val_f1:.4f}")
        
        # Save fold results
        fold_result = {
            'fold': fold + 1,
            'best_epoch': best_epoch,
            'val_accuracy': val_acc,
            'val_f1_score': val_f1,
            'val_loss': val_loss
        }
        fold_results.append(fold_result)
        fold_histories.append(history)
        
        # Save predictions
        predictions_df = pd.DataFrame({
            'fold': fold + 1,
            'station': val_stations,
            'true_label': [ID_TO_LABEL[l] for l in val_labels],
            'predicted_label': [ID_TO_LABEL[p] for p in val_preds],
            'correct': [t == p for t, p in zip(val_labels, val_preds)]
        })
        all_fold_predictions.append(predictions_df)
        predictions_df.to_csv(fold_dir / "predictions.csv", index=False)
        
        # Save confusion matrix
        label_names = [ID_TO_LABEL[i] for i in range(NUM_CLASSES)]
        plot_confusion_matrix(val_labels, val_preds, label_names, fold_dir / "confusion_matrix.png")
        
        # Save fold history
        with open(fold_dir / "training_history.json", 'w') as f:
            json.dump(history, f, indent=2)
    
    return fold_results, fold_histories, all_fold_predictions


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("ViT with 3-Fold Cross-Validation (Improved Preprocessing)")
    print("=" * 70)
    print(f"\nModel: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print(f"Grayscale: {USE_GRAYSCALE} (focuses on patterns, not colors)")
    print(f"Augmentation: {USE_AUGMENTATION} (disabled for time-series)")
    print(f"K-Folds: {NUM_FOLDS}")
    print(f"Epochs per fold: {NUM_EPOCHS}")
    
    set_seed(SEED)
    
    # Load metadata
    metadata_df = load_metadata()
    
    # Collect dataset
    print("\n--- Loading Dataset ---")
    dataset = collect_image_dataset()
    print(f"Total images: {len(dataset)}")
    
    # Count by class
    class_counts = {}
    for _, label, _ in dataset:
        class_counts[label] = class_counts.get(label, 0) + 1
    print("Class distribution:")
    for cls, count in sorted(class_counts.items()):
        print(f"  {cls}: {count}")
    
    # Load processor
    print("\n--- Loading ViT Processor ---")
    processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    
    # Run k-fold training
    print("\n" + "=" * 70)
    print("K-FOLD CROSS-VALIDATION TRAINING")
    print("=" * 70)
    
    fold_results, fold_histories, all_fold_predictions = run_kfold_training(dataset, processor)
    
    # Aggregate results
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 70)
    
    results_df = pd.DataFrame(fold_results)
    print("\nPer-Fold Results:")
    print(results_df.to_string())
    
    print(f"\nMean Val Accuracy: {results_df['val_accuracy'].mean():.4f} ± {results_df['val_accuracy'].std():.4f}")
    print(f"Mean Val F1 Score: {results_df['val_f1_score'].mean():.4f} ± {results_df['val_f1_score'].std():.4f}")
    
    # Save summary
    results_df.to_csv(OUTPUT_DIR / "kfold_summary.csv", index=False)
    
    # Combine all predictions
    all_predictions = pd.concat(all_fold_predictions, ignore_index=True)
    all_predictions.to_csv(OUTPUT_DIR / "all_predictions.csv", index=False)
    
    # Plot histories
    plot_kfold_history(fold_histories, OUTPUT_DIR / "kfold_training_history.png")
    
    # Save configuration
    config = {
        'model_name': MODEL_NAME,
        'num_folds': NUM_FOLDS,
        'num_epochs': NUM_EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'total_samples': len(dataset),
        'class_distribution': class_counts,
        'mean_val_accuracy': float(results_df['val_accuracy'].mean()),
        'std_val_accuracy': float(results_df['val_accuracy'].std()),
        'mean_val_f1': float(results_df['val_f1_score'].mean()),
        'std_val_f1': float(results_df['val_f1_score'].std()),
        'timestamp': datetime.now().isoformat()
    }
    with open(OUTPUT_DIR / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "=" * 70)
    print("K-FOLD TRAINING COMPLETE!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
