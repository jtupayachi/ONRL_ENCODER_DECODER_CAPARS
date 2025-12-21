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
    Swinv2ForImageClassification,
)
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import torch.nn.functional as F
from collections import Counter

# ============================================================================
# Configuration
# ============================================================================

# Paths
IMAGE_BASE_DIR = Path("/home/jose/ONRL_ENCODER_DECODER_CAPARS/trimmed_images")
METADATA_FILE = Path("/home/jose/ONRL_ENCODER_DECODER_CAPARS/station_metadata.csv")
STORAGE_INTERVAL_DIR = Path("/home/jose/ONRL_ENCODER_DECODER_CAPARS/lanl_met_data/data_by_storage_interval")
OUTPUT_BASE_DIR = Path("/home/jose/ONRL_ENCODER_DECODER_CAPARS/model_outputs_comparisonv3")
OUTPUT_BASE_DIR.mkdir(exist_ok=True)

# Only use stations from these time intervals
ALLOWED_INTERVALS = [5, 10, 15, 60]

# Models to train (only ViT-Huge and Swin-V2-Large)
MODELS_CONFIG = [
    {
        'name': 'ViT-Huge',
        'model_id': 'google/vit-huge-patch14-224-in21k',
        'type': 'vit',
        'learning_rate': 1e-5,
        'color': 'purple'
    },
    {
        'name': 'Swin-V2-Large',
        'model_id': 'microsoft/swinv2-large-patch4-window12-192-22k',
        'type': 'swin',
        'learning_rate': 2e-5,
        'color': 'green'
    }
]

NUM_CLASSES = 3  # good, bad, suspect

# Training configuration
BATCH_SIZE = 8
NUM_EPOCHS = 15
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
NUM_FOLDS = 3
EARLY_STOPPING_PATIENCE = 5

# Preprocessing configuration
USE_GRAYSCALE = True
USE_AUGMENTATION = False

# Use both speed and direction images
USE_SPEED_IMAGES = True
USE_DIRECTION_IMAGES = True

# ============================================================================
# IMPROVEMENTS (Configurable)
# ============================================================================

# Balance class distribution
USE_CLASS_WEIGHTS = True
USE_FOCAL_LOSS = True
FOCAL_LOSS_ALPHA = 0.25
FOCAL_LOSS_GAMMA = 2.0

# Time-series aware augmentation
USE_TIME_SERIES_AUGMENTATION = True
TIME_SERIES_AUGMENTATION = {
    'vertical_shift': 0.1,
    'horizontal_shift': 0.05,
    'amplitude_scale': 0.15,
    'add_noise': 0.02,
    'time_warp': False,
}

# ============================================================================
# ENSEMBLE CONFIGURATION
# ============================================================================

# Custom ensemble weights per class
# Format: [ViT-Huge weight, Swin-V2 weight] for each class
ENSEMBLE_WEIGHTS = {
    'good': [1.0, 0.0],      # ViT handles good class entirely
    'bad': [1.0, 0.0],       # ViT handles bad class entirely
    'suspect': [0.3, 0.7]    # Higher Swin weight for suspect class
}

print(f"\n🎯 ENSEMBLE STRATEGY:")
print(f"  Good class:    ViT-Huge={ENSEMBLE_WEIGHTS['good'][0]:.1f}, Swin={ENSEMBLE_WEIGHTS['good'][1]:.1f} (ViT-only)")
print(f"  Bad class:     ViT-Huge={ENSEMBLE_WEIGHTS['bad'][0]:.1f}, Swin={ENSEMBLE_WEIGHTS['bad'][1]:.1f} (ViT-only)")
print(f"  Suspect class: ViT-Huge={ENSEMBLE_WEIGHTS['suspect'][0]:.1f}, Swin={ENSEMBLE_WEIGHTS['suspect'][1]:.1f} (Swin-focused)")

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
    
    for interval in ALLOWED_INTERVALS:
        interval_dir = STORAGE_INTERVAL_DIR / str(interval)
        if not interval_dir.is_dir():
            continue
        
        for class_dir in ['good', 'bad', 'suspect']:
            subdir = interval_dir / class_dir
            if not subdir.exists():
                continue
            
            for csv_file in subdir.glob("*.csv"):
                name = csv_file.stem
                parts = name.split('_')
                if len(parts) >= 2:
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
                 use_grayscale: bool = True, augment: bool = False,
                 time_series_aug: Optional[Dict] = None):
        self.data = data
        self.processor = processor
        self.use_grayscale = use_grayscale
        self.augment = augment
        self.time_series_aug = time_series_aug if (augment and USE_TIME_SERIES_AUGMENTATION) else None
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path, label, station_name = self.data[idx]
        
        if self.use_grayscale:
            image = Image.open(image_path).convert('L')
            image = Image.merge('RGB', [image, image, image])
        else:
            image = Image.open(image_path).convert('RGB')
        
        if self.time_series_aug is not None:
            image = self._apply_time_series_augmentation(image)
        
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)
        label_id = LABEL_TO_ID[label]
        
        return {
            'pixel_values': pixel_values,
            'labels': torch.tensor(label_id),
            'station_name': station_name,
            'image_path': str(image_path)
        }
    
    def _apply_time_series_augmentation(self, image: Image.Image) -> Image.Image:
        """Apply time-series aware augmentations"""
        img_array = np.array(image)
        
        if random.random() < 0.5 and self.time_series_aug.get('vertical_shift', 0) > 0:
            shift = int(img_array.shape[0] * self.time_series_aug['vertical_shift'] * (random.random() - 0.5))
            img_array = np.roll(img_array, shift, axis=0)
        
        if random.random() < 0.5 and self.time_series_aug.get('horizontal_shift', 0) > 0:
            shift = int(img_array.shape[1] * self.time_series_aug['horizontal_shift'] * (random.random() - 0.5))
            img_array = np.roll(img_array, shift, axis=1)
        
        if random.random() < 0.5 and self.time_series_aug.get('amplitude_scale', 0) > 0:
            scale = 1.0 + self.time_series_aug['amplitude_scale'] * (random.random() - 0.5)
            img_array = np.clip(img_array * scale, 0, 255)
        
        if random.random() < 0.5 and self.time_series_aug.get('add_noise', 0) > 0:
            noise = np.random.normal(0, self.time_series_aug['add_noise'] * 255, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255)
        
        return Image.fromarray(img_array.astype(np.uint8))


# ============================================================================
# Loss Functions
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0, num_classes=3):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def compute_class_weights(dataset_list: List) -> torch.Tensor:
    """Compute inverse frequency class weights"""
    labels = [LABEL_TO_ID[item[1]] for item in dataset_list]
    class_counts = Counter(labels)
    total = len(labels)
    
    weights = torch.zeros(NUM_CLASSES)
    for cls_id in range(NUM_CLASSES):
        if class_counts[cls_id] > 0:
            weights[cls_id] = total / (NUM_CLASSES * class_counts[cls_id])
        else:
            weights[cls_id] = 1.0
    
    print(f"  Class weights: {weights.tolist()}")
    return weights


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
    elif model_type == 'swin':
        model = Swinv2ForImageClassification.from_pretrained(
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

def train_epoch(model, dataloader, optimizer, scheduler, device, criterion=None):
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
        
        if criterion is not None:
            loss = criterion(outputs.logits, labels)
        else:
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


def evaluate(model, dataloader, device, return_probas=False):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_stations = []
    all_probas = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  Evaluating", leave=False):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(pixel_values=pixel_values, labels=labels)
            
            total_loss += outputs.loss.item()
            
            probas = F.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probas, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_stations.extend(batch['station_name'])
            
            if return_probas:
                all_probas.extend(probas.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    if return_probas:
        return avg_loss, accuracy, f1, all_preds, all_labels, all_stations, np.array(all_probas)
    else:
        return avg_loss, accuracy, f1, all_preds, all_labels, all_stations


# ============================================================================
# Ensemble Logic
# ============================================================================

def ensemble_predictions(vit_probas: np.ndarray, swin_probas: np.ndarray) -> np.ndarray:
    """
    Ensemble predictions with class-specific weights
    
    Args:
        vit_probas: (N, 3) array of ViT probability predictions
        swin_probas: (N, 3) array of Swin probability predictions
    
    Returns:
        (N,) array of ensemble predictions
    """
    # Create weight matrix: (3 classes, 2 models)
    weight_matrix = np.array([
        ENSEMBLE_WEIGHTS['good'],    # Class 0: good
        ENSEMBLE_WEIGHTS['bad'],     # Class 1: bad
        ENSEMBLE_WEIGHTS['suspect']  # Class 2: suspect
    ])  # Shape: (3, 2)
    
    # Stack model predictions: (N, 3, 2)
    stacked_probas = np.stack([vit_probas, swin_probas], axis=2)
    
    # Apply class-specific weights
    weighted_probas = np.zeros_like(vit_probas)
    for class_idx in range(NUM_CLASSES):
        # For each class, apply its specific weights to both models
        weighted_probas[:, class_idx] = (
            stacked_probas[:, class_idx, 0] * weight_matrix[class_idx, 0] +  # ViT
            stacked_probas[:, class_idx, 1] * weight_matrix[class_idx, 1]    # Swin
        )
    
    # Get final predictions
    ensemble_preds = np.argmax(weighted_probas, axis=1)
    
    return ensemble_preds


# ============================================================================
# Visualization
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
    
    model_names = [r['model_name'] for r in all_model_results]
    mean_accs = [r['mean_accuracy'] for r in all_model_results]
    std_accs = [r['std_accuracy'] for r in all_model_results]
    mean_f1s = [r['mean_f1'] for r in all_model_results]
    std_f1s = [r['std_f1'] for r in all_model_results]
    colors = [r['color'] for r in all_model_results]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    x = np.arange(len(model_names))
    axes[0].bar(x, mean_accs, yerr=std_accs, capsize=5, color=colors, alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Model Comparison - Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names, rotation=15, ha='right')
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis='y', alpha=0.3)
    
    for i, (acc, std) in enumerate(zip(mean_accs, std_accs)):
        axes[0].text(i, acc + std + 0.02, f'{acc:.3f}±{std:.3f}', 
                     ha='center', fontsize=9, fontweight='bold')
    
    axes[1].bar(x, mean_f1s, yerr=std_f1s, capsize=5, color=colors, alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('F1-Score', fontsize=12)
    axes[1].set_title('Model Comparison - F1-Score', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_names, rotation=15, ha='right')
    axes[1].set_ylim(0, 1)
    axes[1].grid(axis='y', alpha=0.3)
    
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
    df.to_csv(save_dir / "final_summary_table.csv", index=False)
    
    print("\n" + "=" * 120)
    print("FINAL MODEL COMPARISON SUMMARY")
    print("=" * 120)
    print(df.to_string(index=False))
    print("=" * 120)


# ============================================================================
# K-Fold Training for Single Model
# ============================================================================

def train_single_model(model_config: Dict, dataset_list: List, output_dir: Path):
    """Train a single model with k-fold CV"""
    
    model_name = model_config['name']
    model_id = model_config['model_id']
    learning_rate = model_config['learning_rate']
    
    print("\n" + "=" * 70)
    print(f"TRAINING: {model_name}")
    print("=" * 70)
    print(f"Model ID: {model_id}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Device: {DEVICE}")
    
    model_output_dir = output_dir / model_name.replace(' ', '_').lower()
    model_output_dir.mkdir(exist_ok=True)
    
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
    all_fold_probas = []
    
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
        
        fold_dir = model_output_dir / f"fold_{fold+1}"
        fold_dir.mkdir(exist_ok=True)
        
        train_data = [dataset_list[i] for i in train_idx]
        val_data = [dataset_list[i] for i in val_idx]
        
        print(f"  Train images: {len(train_data)}, Val images: {len(val_data)}")
        
        train_aug_config = TIME_SERIES_AUGMENTATION if USE_TIME_SERIES_AUGMENTATION else None
        train_dataset = MeteorologicalImageDataset(train_data, processor, 
                                                   use_grayscale=USE_GRAYSCALE, 
                                                   augment=USE_TIME_SERIES_AUGMENTATION,
                                                   time_series_aug=train_aug_config)
        val_dataset = MeteorologicalImageDataset(val_data, processor, 
                                                 use_grayscale=USE_GRAYSCALE, 
                                                 augment=False,
                                                 time_series_aug=None)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        
        model = create_model(model_config)
        model = model.to(DEVICE)
        
        criterion = None
        if USE_FOCAL_LOSS:
            criterion = FocalLoss(alpha=FOCAL_LOSS_ALPHA, gamma=FOCAL_LOSS_GAMMA, num_classes=NUM_CLASSES).to(DEVICE)
            print(f"  Using Focal Loss (alpha={FOCAL_LOSS_ALPHA}, gamma={FOCAL_LOSS_GAMMA})")
        elif USE_CLASS_WEIGHTS:
            class_weights = compute_class_weights(dataset_list).to(DEVICE)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            print(f"  Using weighted CrossEntropyLoss")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
        
        total_steps = len(train_loader) * NUM_EPOCHS
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            total_steps=total_steps,
            pct_start=WARMUP_RATIO
        )
        
        best_val_loss = float('inf')
        best_val_acc = 0
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(NUM_EPOCHS):
            print(f"\n  Epoch {epoch + 1}/{NUM_EPOCHS}")
            
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, DEVICE, criterion)
            val_loss, val_acc, val_f1, _, _, _ = evaluate(model, val_loader, DEVICE)
            
            print(f"    Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
            print(f"    Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f}, F1={val_f1:.4f}")
            
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
        
        # Load best model and evaluate with probabilities
        model.load_state_dict(torch.load(fold_dir / "best_model.pt"))
        val_loss, val_acc, val_f1, val_preds, val_labels, val_stations, val_probas = evaluate(
            model, val_loader, DEVICE, return_probas=True
        )
        
        print(f"\n  Fold {fold+1} Final Results:")
        print(f"    Best Epoch: {best_epoch}")
        print(f"    Val Accuracy: {val_acc:.4f}")
        print(f"    Val F1 Score: {val_f1:.4f}")
        
        fold_result = {
            'fold': fold + 1,
            'best_epoch': best_epoch,
            'val_accuracy': val_acc,
            'val_f1_score': val_f1,
            'val_loss': val_loss
        }
        fold_results.append(fold_result)
        
        # Save predictions with probabilities
        predictions_df = pd.DataFrame({
            'fold': fold + 1,
            'station': val_stations,
            'true_label': [ID_TO_LABEL[l] for l in val_labels],
            'predicted_label': [ID_TO_LABEL[p] for p in val_preds],
            'correct': [t == p for t, p in zip(val_labels, val_preds)],
            'proba_good': val_probas[:, 0],
            'proba_bad': val_probas[:, 1],
            'proba_suspect': val_probas[:, 2]
        })
        all_fold_predictions.append(predictions_df)
        predictions_df.to_csv(fold_dir / "predictions.csv", index=False)
        
        # Store probabilities for ensemble
        all_fold_probas.append({
            'fold': fold + 1,
            'probas': val_probas,
            'labels': val_labels,
            'stations': val_stations
        })
        
        label_names = [ID_TO_LABEL[i] for i in range(NUM_CLASSES)]
        cm_title = f"{model_name} - Fold {fold+1}/{NUM_FOLDS}"
        plot_confusion_matrix_interim(val_labels, val_preds, label_names, 
                                     fold_dir / "confusion_matrix.png", cm_title)
        
        report = classification_report(val_labels, val_preds, target_names=label_names, output_dict=True)
        with open(fold_dir / "classification_report.json", 'w') as f:
            json.dump(report, f, indent=2)
    
    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(model_output_dir / "kfold_summary.csv", index=False)
    
    all_predictions = pd.concat(all_fold_predictions, ignore_index=True)
    all_predictions.to_csv(model_output_dir / "all_predictions.csv", index=False)
    
    mean_acc = results_df['val_accuracy'].mean()
    std_acc = results_df['val_accuracy'].std()
    mean_f1 = results_df['val_f1_score'].mean()
    std_f1 = results_df['val_f1_score'].std()
    
    print(f"\n{model_name} - Cross-Validation Summary:")
    print(f"  Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  Mean F1-Score: {mean_f1:.4f} ± {std_f1:.4f}")
    
    all_true = all_predictions['true_label'].map(LABEL_TO_ID).values
    all_pred = all_predictions['predicted_label'].map(LABEL_TO_ID).values
    
    label_names = [ID_TO_LABEL[i] for i in range(NUM_CLASSES)]
    final_cm_title = f"{model_name} - Final (All Folds)"
    plot_confusion_matrix_interim(all_true, all_pred, label_names,
                                 model_output_dir / "final_confusion_matrix.png", final_cm_title)
    
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
        'fold_results': fold_results,
        'fold_probas': all_fold_probas
    }


# ============================================================================
# Ensemble Evaluation
# ============================================================================

def evaluate_ensemble(vit_results: Dict, swin_results: Dict, output_dir: Path):
    """Evaluate ensemble of ViT-Huge and Swin-V2-Large with class-specific weights"""
    
    print("\n" + "=" * 70)
    print("ENSEMBLE EVALUATION (ViT-Huge + Swin-V2-Large)")
    print("=" * 70)
    print(f"Strategy: Class-specific weights")
    print(f"  Good:    ViT={ENSEMBLE_WEIGHTS['good'][0]}, Swin={ENSEMBLE_WEIGHTS['good'][1]}")
    print(f"  Bad:     ViT={ENSEMBLE_WEIGHTS['bad'][0]}, Swin={ENSEMBLE_WEIGHTS['bad'][1]}")
    print(f"  Suspect: ViT={ENSEMBLE_WEIGHTS['suspect'][0]}, Swin={ENSEMBLE_WEIGHTS['suspect'][1]} ← Swin-focused")
    
    ensemble_dir = output_dir / "ensemble"
    ensemble_dir.mkdir(exist_ok=True)
    
    fold_results = []
    all_fold_predictions = []
    
    for fold_idx in range(NUM_FOLDS):
        print(f"\n--- Fold {fold_idx + 1}/{NUM_FOLDS} ---")
        
        vit_fold = vit_results['fold_probas'][fold_idx]
        swin_fold = swin_results['fold_probas'][fold_idx]
        
        vit_probas = vit_fold['probas']
        swin_probas = swin_fold['probas']
        true_labels = vit_fold['labels']
        stations = vit_fold['stations']
        
        # Apply ensemble with class-specific weights
        ensemble_preds = ensemble_predictions(vit_probas, swin_probas)
        
        accuracy = accuracy_score(true_labels, ensemble_preds)
        f1 = f1_score(true_labels, ensemble_preds, average='weighted')
        
        print(f"  Ensemble Accuracy: {accuracy:.4f}")
        print(f"  Ensemble F1-Score: {f1:.4f}")
        
        fold_results.append({
            'fold': fold_idx + 1,
            'accuracy': accuracy,
            'f1_score': f1
        })
        
        predictions_df = pd.DataFrame({
            'fold': fold_idx + 1,
            'station': stations,
            'true_label': [ID_TO_LABEL[l] for l in true_labels],
            'predicted_label': [ID_TO_LABEL[p] for p in ensemble_preds],
            'correct': [t == p for t, p in zip(true_labels, ensemble_preds)],
            'vit_proba_good': vit_probas[:, 0],
            'vit_proba_bad': vit_probas[:, 1],
            'vit_proba_suspect': vit_probas[:, 2],
            'swin_proba_good': swin_probas[:, 0],
            'swin_proba_bad': swin_probas[:, 1],
            'swin_proba_suspect': swin_probas[:, 2]
        })
        all_fold_predictions.append(predictions_df)
        
        fold_dir = ensemble_dir / f"fold_{fold_idx+1}"
        fold_dir.mkdir(exist_ok=True)
        predictions_df.to_csv(fold_dir / "predictions.csv", index=False)
        
        label_names = [ID_TO_LABEL[i] for i in range(NUM_CLASSES)]
        cm_title = f"Ensemble - Fold {fold_idx+1}/{NUM_FOLDS}"
        plot_confusion_matrix_interim(true_labels, ensemble_preds, label_names,
                                     fold_dir / "confusion_matrix.png", cm_title)
    
    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(ensemble_dir / "kfold_summary.csv", index=False)
    
    all_predictions = pd.concat(all_fold_predictions, ignore_index=True)
    all_predictions.to_csv(ensemble_dir / "all_predictions.csv", index=False)
    
    mean_acc = results_df['accuracy'].mean()
    std_acc = results_df['accuracy'].std()
    mean_f1 = results_df['f1_score'].mean()
    std_f1 = results_df['f1_score'].std()
    
    print(f"\nEnsemble - Cross-Validation Summary:")
    print(f"  Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  Mean F1-Score: {mean_f1:.4f} ± {std_f1:.4f}")
    
    all_true = all_predictions['true_label'].map(LABEL_TO_ID).values
    all_pred = all_predictions['predicted_label'].map(LABEL_TO_ID).values
    
    label_names = [ID_TO_LABEL[i] for i in range(NUM_CLASSES)]
    final_cm_title = "Ensemble - Final (All Folds)"
    plot_confusion_matrix_interim(all_true, all_pred, label_names,
                                 ensemble_dir / "final_confusion_matrix.png", final_cm_title)
    
    return {
        'model_name': 'Ensemble (ViT+Swin)',
        'model_id': 'ensemble',
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'mean_f1': mean_f1,
        'std_f1': std_f1,
        'best_fold_accuracy': results_df['accuracy'].max(),
        'worst_fold_accuracy': results_df['accuracy'].min(),
        'parameters': 'N/A',
        'color': 'orange'
    }


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("MULTI-MODEL TRAINING V3 WITH CUSTOM ENSEMBLE")
    print("=" * 70)
    print(f"\nModels to train:")
    for config in MODELS_CONFIG:
        print(f"  - {config['name']}")
    print(f"\nDevice: {DEVICE}")
    print(f"Grayscale: {USE_GRAYSCALE}")
    print(f"K-Folds: {NUM_FOLDS}")
    print(f"Epochs per fold: {NUM_EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"\n--- IMPROVEMENTS ENABLED ---")
    print(f"Focal Loss: {USE_FOCAL_LOSS}")
    print(f"Class Weights: {USE_CLASS_WEIGHTS}")
    print(f"Time-Series Augmentation: {USE_TIME_SERIES_AUGMENTATION}")
    
    set_seed(SEED)
    
    print("\n--- Loading Dataset ---")
    dataset = collect_image_dataset()
    print(f"Total images: {len(dataset)}")
    
    class_counts = {}
    for _, label, _ in dataset:
        class_counts[label] = class_counts.get(label, 0) + 1
    print("Class distribution:")
    for cls, count in sorted(class_counts.items()):
        print(f"  {cls}: {count}")
    
    # Train both models
    all_model_results = []
    vit_results = None
    swin_results = None
    
    for model_config in MODELS_CONFIG:
        try:
            result = train_single_model(model_config, dataset, OUTPUT_BASE_DIR)
            all_model_results.append(result)
            
            if model_config['type'] == 'vit':
                vit_results = result
            elif model_config['type'] == 'swin':
                swin_results = result
                
        except Exception as e:
            print(f"\n❌ ERROR training {model_config['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Evaluate ensemble
    if vit_results is not None and swin_results is not None:
        ensemble_result = evaluate_ensemble(vit_results, swin_results, OUTPUT_BASE_DIR)
        all_model_results.append(ensemble_result)
    
    # Final comparison
    if len(all_model_results) > 0:
        print("\n" + "=" * 70)
        print("FINAL MODEL COMPARISON")
        print("=" * 70)
        
        plot_model_comparison(all_model_results, OUTPUT_BASE_DIR)
        create_summary_table(all_model_results, OUTPUT_BASE_DIR)
        
        with open(OUTPUT_BASE_DIR / "complete_results.json", 'w') as f:
            results_to_save = []
            for r in all_model_results:
                r_copy = r.copy()
                if 'fold_results' in r_copy:
                    r_copy['fold_results'] = [dict(fr) for fr in r_copy['fold_results']]
                if 'fold_probas' in r_copy:
                    del r_copy['fold_probas']  # Too large for JSON
                results_to_save.append(r_copy)
            json.dump(results_to_save, f, indent=2)
        
        print(f"\n✓ All results saved to: {OUTPUT_BASE_DIR}")
        
        best_model = max(all_model_results, key=lambda x: x['mean_accuracy'])
        print(f"\n🏆 BEST MODEL: {best_model['model_name']}")
        print(f"   Accuracy: {best_model['mean_accuracy']:.4f} ± {best_model['std_accuracy']:.4f}")
        print(f"   F1-Score: {best_model['mean_f1']:.4f} ± {best_model['std_f1']:.4f}")
    
    print("\n" + "=" * 70)
    print("MULTI-MODEL TRAINING V3 COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
