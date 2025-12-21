#!/usr/bin/env python3

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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from transformers import (
    AutoImageProcessor,
    ViTForImageClassification,
    ConvNextForImageClassification,
    Swinv2ForImageClassification,
    Dinov2ForImageClassification,
)
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

# Paths
IMAGE_BASE_DIR = Path("/home/jose/ONRL_ENCODER_DECODER_CAPARS/trimmed_images")
METADATA_FILE = Path("/home/jose/ONRL_ENCODER_DECODER_CAPARS/station_metadata.csv")
STORAGE_INTERVAL_DIR = Path("/home/jose/ONRL_ENCODER_DECODER_CAPARS/lanl_met_data/data_by_storage_interval")
OUTPUT_BASE_DIR = Path("/home/jose/ONRL_ENCODER_DECODER_CAPARS/model_outputs_comparison")
OUTPUT_BASE_DIR.mkdir(exist_ok=True)

# Only use stations from these time intervals
ALLOWED_INTERVALS = [5, 10, 15, 60]

# Models to train
MODELS_CONFIG = [
    {
        'name': 'ViT-Huge',
        'model_id': 'google/vit-huge-patch14-224-in21k',
        'type': 'vit',
        'learning_rate': 1e-5,
        'color': 'purple'
    },
    {
        'name': 'ConvNeXt-XLarge',
        'model_id': 'facebook/convnext-xlarge-384-22k-1k',
        'type': 'convnext',
        'learning_rate': 5e-6,  # Lower LR for XLarge
        'color': 'blue'
    },
    {
        'name': 'Swin-V2-Large',
        'model_id': 'microsoft/swinv2-large-patch4-window12-192-22k',
        'type': 'swin',
        'learning_rate': 2e-5,
        'color': 'green'
    },
    {
        'name': 'DINOv2-Giant',
        'model_id': 'facebook/dinov2-giant',
        'type': 'dinov2',
        'learning_rate': 5e-6,  # Lower LR for Giant
        'color': 'red'
    }
]

NUM_CLASSES = 3  # good, bad, suspect

# Training configuration
BATCH_SIZE = 8
NUM_EPOCHS = 15
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


def get_stations_from_allowed_intervals() -> set:
    """Get stations that have CSV files in allowed time intervals"""
    stations = set()
    
    if not STORAGE_INTERVAL_DIR.exists():
        print(f"Warning: Storage interval directory not found: {STORAGE_INTERVAL_DIR}")
        return stations
    
    # Only scan allowed interval directories
    for interval in ALLOWED_INTERVALS:
        interval_dir = STORAGE_INTERVAL_DIR / str(interval)
        if not interval_dir.is_dir():
            continue
        
        # Look in good, bad, suspect subdirectories
        for class_dir in ['good', 'bad', 'suspect']:
            subdir = interval_dir / class_dir
            if not subdir.exists():
                continue
            
            # Find all CSV files
            for csv_file in subdir.glob("*.csv"):
                # Extract station name from filename
                name = csv_file.stem
                parts = name.split('_')
                if len(parts) >= 2:
                    # Remove year (last part)
                    station_name = '_'.join(parts[:-1])
                    stations.add(station_name)
    
    print(f"Found {len(stations)} unique stations in allowed intervals {ALLOWED_INTERVALS}")
    return stations


def collect_image_dataset() -> List[Tuple[Path, str, str]]:
    """Collect trimmed images only from stations in allowed time intervals"""
    dataset = []
    
    metadata_df = load_metadata()
    if metadata_df.empty:
        print("ERROR: No metadata loaded!")
        return dataset
    
    station_category_map = {}
    for _, row in metadata_df.iterrows():
        station_name = row['station_name']
        category = row['category']
        if pd.notna(category) and category in ['good', 'bad', 'suspect']:
            station_category_map[station_name] = category
    
    print(f"Loaded categories for {len(station_category_map)} stations from metadata")
    
    # Get stations from allowed intervals
    allowed_stations = get_stations_from_allowed_intervals()
    
    if not allowed_stations:
        print("ERROR: No stations found in allowed intervals!")
        return dataset
    
    print(f"Filtering to only use stations from intervals: {ALLOWED_INTERVALS}")
    
    if not IMAGE_BASE_DIR.exists():
        print(f"ERROR: Directory not found: {IMAGE_BASE_DIR}")
        return dataset
    
    for image_path in IMAGE_BASE_DIR.glob("*.png"):
        name = image_path.stem
        
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
        
        # Check if station is in allowed intervals
        if station_name not in allowed_stations:
            continue
        
        category = station_category_map.get(station_name)
        
        if category:
            dataset.append((image_path, category, station_name))
    
    print(f"Collected {len(dataset)} images from {len(set([s for _, _, s in dataset]))} stations")
    
    return dataset


class MeteorologicalImageDataset(Dataset):
    """PyTorch Dataset for meteorological images"""
    
    def __init__(self, data: List[Tuple[Path, str, str]], processor, 
                 use_grayscale: bool = True, augment: bool = False):
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
# Model Creation
# ============================================================================

def create_model(model_config: Dict):
    """Create model based on configuration"""
    model_type = model_config['type']
    model_id = model_config['model_id']
    
    print(f"\n  Loading {model_config['name']} ({model_id})...")
    
    if model_type == 'vit':
        model = ViTForImageClassification.from_pretrained(
            model_id,
            num_labels=NUM_CLASSES,
            id2label=ID_TO_LABEL,
            label2id=LABEL_TO_ID,
            ignore_mismatched_sizes=True
        )
    elif model_type == 'convnext':
        model = ConvNextForImageClassification.from_pretrained(
            model_id,
            num_labels=NUM_CLASSES,
            id2label=ID_TO_LABEL,
            label2id=LABEL_TO_ID,
            ignore_mismatched_sizes=True
        )
    elif model_type == 'swin':
        model = Swinv2ForImageClassification.from_pretrained(
            model_id,
            num_labels=NUM_CLASSES,
            id2label=ID_TO_LABEL,
            label2id=LABEL_TO_ID,
            ignore_mismatched_sizes=True
        )
    elif model_type == 'dinov2':
        model = Dinov2ForImageClassification.from_pretrained(
            model_id,
            num_labels=NUM_CLASSES,
            id2label=ID_TO_LABEL,
            label2id=LABEL_TO_ID,
            ignore_mismatched_sizes=True
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def load_processor(model_id: str):
    """Load appropriate processor"""
    processor = AutoImageProcessor.from_pretrained(model_id)
    return processor


# ============================================================================
# Training & Evaluation
# ============================================================================

def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc="  Training", leave=False)
    
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
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
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
        for batch in tqdm(dataloader, desc="  Evaluating", leave=False):
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
# Visualization with Interim Reporting
# ============================================================================

def plot_confusion_matrix_interim(y_true, y_pred, labels, save_path, title):
    """Plot confusion matrix with custom title"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    
    # Add accuracy info
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    plt.text(0.5, -0.15, f'Accuracy: {accuracy:.3f} | F1-Score: {f1:.3f}',
             ha='center', transform=plt.gca().transAxes, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Confusion matrix saved: {save_path.name}")


def plot_model_comparison(all_model_results: List[Dict], save_dir: Path):
    """Plot final comparison across all models"""
    
    # Prepare data
    model_names = [r['model_name'] for r in all_model_results]
    mean_accs = [r['mean_accuracy'] for r in all_model_results]
    std_accs = [r['std_accuracy'] for r in all_model_results]
    mean_f1s = [r['mean_f1'] for r in all_model_results]
    std_f1s = [r['std_f1'] for r in all_model_results]
    colors = [r['color'] for r in all_model_results]
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy comparison
    x = np.arange(len(model_names))
    axes[0].bar(x, mean_accs, yerr=std_accs, capsize=5, color=colors, alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Model Comparison - Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names, rotation=15, ha='right')
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (acc, std) in enumerate(zip(mean_accs, std_accs)):
        axes[0].text(i, acc + std + 0.02, f'{acc:.3f}±{std:.3f}', 
                     ha='center', fontsize=9, fontweight='bold')
    
    # F1-Score comparison
    axes[1].bar(x, mean_f1s, yerr=std_f1s, capsize=5, color=colors, alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('F1-Score', fontsize=12)
    axes[1].set_title('Model Comparison - F1-Score', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_names, rotation=15, ha='right')
    axes[1].set_ylim(0, 1)
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (f1, std) in enumerate(zip(mean_f1s, std_f1s)):
        axes[1].text(i, f1 + std + 0.02, f'{f1:.3f}±{std:.3f}', 
                     ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_dir / "final_model_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Final comparison saved: final_model_comparison.png")


def create_summary_table(all_model_results: List[Dict], save_dir: Path):
    """Create detailed summary table"""
    
    summary_data = []
    for result in all_model_results:
        summary_data.append({
            'Model': result['model_name'],
            'Mean Accuracy': f"{result['mean_accuracy']:.4f}",
            'Std Accuracy': f"{result['std_accuracy']:.4f}",
            'Mean F1-Score': f"{result['mean_f1']:.4f}",
            'Std F1-Score': f"{result['std_f1']:.4f}",
            'Best Fold Acc': f"{result['best_fold_accuracy']:.4f}",
            'Worst Fold Acc': f"{result['worst_fold_accuracy']:.4f}",
            'Parameters': result['parameters'],
            'Model ID': result['model_id']
        })
    
    df = pd.DataFrame(summary_data)
    
    # Save as CSV
    df.to_csv(save_dir / "final_summary_table.csv", index=False)
    
    # Print table
    print("\n" + "=" * 120)
    print("FINAL MODEL COMPARISON SUMMARY")
    print("=" * 120)
    print(df.to_string(index=False))
    print("=" * 120)


# ============================================================================
# K-Fold Training for Single Model
# ============================================================================

def train_single_model(model_config: Dict, dataset_list: List, output_dir: Path):
    """Train a single model with k-fold CV and interim reporting"""
    
    model_name = model_config['name']
    model_id = model_config['model_id']
    learning_rate = model_config['learning_rate']
    
    print("\n" + "=" * 70)
    print(f"TRAINING: {model_name}")
    print("=" * 70)
    print(f"Model ID: {model_id}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Device: {DEVICE}")
    
    # Create output directory
    model_output_dir = output_dir / model_name.replace(' ', '_').lower()
    model_output_dir.mkdir(exist_ok=True)
    
    # Load processor
    print(f"\n  Loading processor...")
    processor = load_processor(model_id)
    
    # Group images by station to prevent data leakage
    station_to_images = {}
    station_to_label = {}
    for idx, (image_path, label, station_name) in enumerate(dataset_list):
        if station_name not in station_to_images:
            station_to_images[station_name] = []
            station_to_label[station_name] = label
        station_to_images[station_name].append(idx)
    
    # Create station-level splits
    unique_stations = list(station_to_images.keys())
    station_labels = np.array([LABEL_TO_ID[station_to_label[s]] for s in unique_stations])
    
    print(f"  Total unique stations: {len(unique_stations)}")
    print(f"  Total images: {len(dataset_list)}")
    print(f"  Avg images per station: {len(dataset_list)/len(unique_stations):.1f}")
    
    # Create k-fold splitter at STATION level
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    
    fold_results = []
    all_fold_predictions = []
    
    for fold, (train_station_idx, val_station_idx) in enumerate(skf.split(np.zeros(len(unique_stations)), station_labels)):
        # Convert station indices to image indices
        train_stations = [unique_stations[i] for i in train_station_idx]
        val_stations = [unique_stations[i] for i in val_station_idx]
        
        train_idx = []
        for station in train_stations:
            train_idx.extend(station_to_images[station])
        
        val_idx = []
        for station in val_stations:
            val_idx.extend(station_to_images[station])
        print(f"\n--- Fold {fold + 1}/{NUM_FOLDS} ---")
        print(f"  Train stations: {len(train_stations)}, Val stations: {len(val_stations)}")
        
        # Create fold directory
        fold_dir = model_output_dir / f"fold_{fold+1}"
        fold_dir.mkdir(exist_ok=True)
        
        # Split data
        train_data = [dataset_list[i] for i in train_idx]
        val_data = [dataset_list[i] for i in val_idx]
        
        print(f"  Train images: {len(train_data)}, Val images: {len(val_data)}")
        
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
        model = create_model(model_config)
        model = model.to(DEVICE)
        
        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
        
        total_steps = len(train_loader) * NUM_EPOCHS
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            total_steps=total_steps,
            pct_start=WARMUP_RATIO
        )
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        best_val_acc = 0
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(NUM_EPOCHS):
            print(f"\n  Epoch {epoch + 1}/{NUM_EPOCHS}")
            
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, DEVICE)
            val_loss, val_acc, val_f1, _, _, _ = evaluate(model, val_loader, DEVICE)
            
            print(f"    Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
            print(f"    Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f}, F1={val_f1:.4f}")
            
            # Early stopping based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_epoch = epoch + 1
                patience_counter = 0
                torch.save(model.state_dict(), fold_dir / "best_model.pt")
                print(f"    → Best model saved! (loss improved)")
            else:
                patience_counter += 1
                print(f"    → No improvement (patience: {patience_counter}/{EARLY_STOPPING_PATIENCE})")
                
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    print(f"\n    ⚠ Early stopping triggered at epoch {epoch + 1}")
                    print(f"    Best epoch was {best_epoch} with val_loss={best_val_loss:.4f}")
                    break
        
        # Load best model and evaluate
        model.load_state_dict(torch.load(fold_dir / "best_model.pt"))
        val_loss, val_acc, val_f1, val_preds, val_labels, val_stations = evaluate(model, val_loader, DEVICE)
        
        print(f"\n  Fold {fold+1} Final Results:")
        print(f"    Best Epoch: {best_epoch}")
        print(f"    Val Accuracy: {val_acc:.4f}")
        print(f"    Val F1 Score: {val_f1:.4f}")
        
        # Save fold results
        fold_result = {
            'fold': fold + 1,
            'best_epoch': best_epoch,
            'val_accuracy': val_acc,
            'val_f1_score': val_f1,
            'val_loss': val_loss
        }
        fold_results.append(fold_result)
        
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
        
        # INTERIM REPORTING: Plot confusion matrix for this fold
        label_names = [ID_TO_LABEL[i] for i in range(NUM_CLASSES)]
        cm_title = f"{model_name} - Fold {fold+1}/{NUM_FOLDS}"
        plot_confusion_matrix_interim(val_labels, val_preds, label_names, 
                                     fold_dir / "confusion_matrix.png", cm_title)
        
        # Classification report
        report = classification_report(val_labels, val_preds, target_names=label_names, output_dict=True)
        with open(fold_dir / "classification_report.json", 'w') as f:
            json.dump(report, f, indent=2)
    
    # Aggregate results
    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(model_output_dir / "kfold_summary.csv", index=False)
    
    # Combine all predictions
    all_predictions = pd.concat(all_fold_predictions, ignore_index=True)
    all_predictions.to_csv(model_output_dir / "all_predictions.csv", index=False)
    
    # Calculate statistics
    mean_acc = results_df['val_accuracy'].mean()
    std_acc = results_df['val_accuracy'].std()
    mean_f1 = results_df['val_f1_score'].mean()
    std_f1 = results_df['val_f1_score'].std()
    
    print(f"\n{model_name} - Cross-Validation Summary:")
    print(f"  Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  Mean F1-Score: {mean_f1:.4f} ± {std_f1:.4f}")
    
    # Create final confusion matrix (aggregated across all folds)
    all_true = all_predictions['true_label'].map(LABEL_TO_ID).values
    all_pred = all_predictions['predicted_label'].map(LABEL_TO_ID).values
    
    label_names = [ID_TO_LABEL[i] for i in range(NUM_CLASSES)]
    final_cm_title = f"{model_name} - Final (All Folds)"
    plot_confusion_matrix_interim(all_true, all_pred, label_names,
                                 model_output_dir / "final_confusion_matrix.png", final_cm_title)
    
    # Return results for comparison
    return {
        'model_name': model_name,
        'model_id': model_id,
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'mean_f1': mean_f1,
        'std_f1': std_f1,
        'best_fold_accuracy': results_df['val_accuracy'].max(),
        'worst_fold_accuracy': results_df['val_accuracy'].min(),
        'parameters': f"{sum(p.numel() for p in create_model(model_config).parameters()):,}",
        'color': model_config['color'],
        'fold_results': fold_results
    }


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("MULTI-MODEL TRAINING WITH INTERIM REPORTING")
    print("=" * 70)
    print(f"\nModels to train:")
    for config in MODELS_CONFIG:
        print(f"  - {config['name']}")
    print(f"\nDevice: {DEVICE}")
    print(f"Grayscale: {USE_GRAYSCALE} (focuses on patterns, not colors)")
    print(f"Augmentation: {USE_AUGMENTATION} (disabled for time-series)")
    print(f"K-Folds: {NUM_FOLDS}")
    print(f"Epochs per fold: {NUM_EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    
    set_seed(SEED)
    
    # Load dataset once
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
    
    # Train each model
    all_model_results = []
    
    for model_config in MODELS_CONFIG:
        try:
            result = train_single_model(model_config, dataset, OUTPUT_BASE_DIR)
            all_model_results.append(result)
        except Exception as e:
            print(f"\n❌ ERROR training {model_config['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final comparison
    if len(all_model_results) > 0:
        print("\n" + "=" * 70)
        print("FINAL MODEL COMPARISON")
        print("=" * 70)
        
        # Plot comparison
        plot_model_comparison(all_model_results, OUTPUT_BASE_DIR)
        
        # Create summary table
        create_summary_table(all_model_results, OUTPUT_BASE_DIR)
        
        # Save complete results
        with open(OUTPUT_BASE_DIR / "complete_results.json", 'w') as f:
            # Convert to JSON-serializable format
            results_to_save = []
            for r in all_model_results:
                r_copy = r.copy()
                r_copy['fold_results'] = [dict(fr) for fr in r_copy['fold_results']]
                results_to_save.append(r_copy)
            json.dump(results_to_save, f, indent=2)
        
        print(f"\n✓ All results saved to: {OUTPUT_BASE_DIR}")
        
        # Print winner
        best_model = max(all_model_results, key=lambda x: x['mean_accuracy'])
        print(f"\n🏆 BEST MODEL: {best_model['model_name']}")
        print(f"   Accuracy: {best_model['mean_accuracy']:.4f} ± {best_model['std_accuracy']:.4f}")
        print(f"   F1-Score: {best_model['mean_f1']:.4f} ± {best_model['std_f1']:.4f}")
    
    print("\n" + "=" * 70)
    print("MULTI-MODEL TRAINING COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
