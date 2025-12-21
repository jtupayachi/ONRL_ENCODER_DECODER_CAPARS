#!/usr/bin/env python3
"""
Trim and Organize Meteorological Images

This script:
1. Trims white borders from images using OpenCV
2. Organizes images by true category from Excel
3. Saves trimmed images to a new directory structure

Directory structure created:
    trimmed_images/
        good/
        bad/
        suspect/
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import shutil

# Configuration
IMAGE_BASE_DIR = Path("/home/jose/ONRL_ENCODER_DECODER_CAPARS/lanl_met_data/images")
STORAGE_INTERVAL_DIR = Path("/home/jose/ONRL_ENCODER_DECODER_CAPARS/lanl_met_data/data_by_storage_interval")
METADATA_FILE = Path("/home/jose/ONRL_ENCODER_DECODER_CAPARS/station_metadata.csv")
OUTPUT_DIR = Path("/home/jose/ONRL_ENCODER_DECODER_CAPARS/trimmed_images")

# Trimming parameters
BORDER_THRESHOLD = 240  # Pixels above this are considered "border" (white/light)
MIN_BORDER_SIZE = 5  # Minimum pixels to trim

# Additional crop percentages to remove axis labels and legends
CROP_TOP_PERCENT = 0.122      # Remove 5% from top (title/legend)
CROP_BOTTOM_PERCENT = 0.1985  # Remove 12.5% from bottom (x-axis labels)
CROP_LEFT_PERCENT = 0.126    # Remove 12.5% from left (y-axis labels)
CROP_RIGHT_PERCENT = 0.099   # Remove 3% from right (extra margin)


def load_category_mapping() -> dict:
    """Load true station categories from metadata file"""
    try:
        df = pd.read_csv(METADATA_FILE)
        category_map = {}
        for _, row in df.iterrows():
            station = str(row['station_name']).strip()
            category = str(row['category']).strip().lower()
            if pd.notna(category) and category in ['good', 'bad', 'suspect']:
                category_map[station] = category
        print(f"Loaded {len(category_map)} station categories from metadata")
        return category_map
    except Exception as e:
        print(f"Error loading metadata file: {e}")
        return {}


def get_stations_from_storage_intervals() -> set:
    """Get set of station names that have CSV files in storage interval directories"""
    stations = set()
    
    if not STORAGE_INTERVAL_DIR.exists():
        print(f"Warning: Storage interval directory not found: {STORAGE_INTERVAL_DIR}")
        return stations
    
    # Scan all storage interval directories
    for interval_dir in STORAGE_INTERVAL_DIR.iterdir():
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
    
    print(f"Found {len(stations)} unique stations in storage interval directories")
    return stations


def get_station_name_from_image(image_path: Path) -> str:
    """Extract station name from image filename"""
    name = image_path.stem
    # Remove _speed or _dir suffix
    if name.endswith('_speed'):
        name = name[:-6]
    elif name.endswith('_dir'):
        name = name[:-4]
    return name


def trim_white_border(image: np.ndarray, threshold: int = BORDER_THRESHOLD) -> np.ndarray:
    """
    Crop meteorological plots using fixed percentage criteria
    
    Applies fixed percentage crops to remove:
    - Title/legend area from top
    - X-axis labels from bottom
    - Y-axis labels from left
    - Extra margin from right
    
    Args:
        image: Input image (BGR format from cv2)
        threshold: Unused (kept for compatibility)
        
    Returns:
        Cropped image with fixed percentage reductions applied
    """
    h, w = image.shape[:2]
    
    # Calculate crop boundaries using fixed percentages
    top = int(h * CROP_TOP_PERCENT)
    bottom = h - int(h * CROP_BOTTOM_PERCENT)
    left = int(w * CROP_LEFT_PERCENT)
    right = w - int(w * CROP_RIGHT_PERCENT)
    
    # Ensure valid boundaries
    top = max(0, min(top, h - 1))
    bottom = max(top + 1, min(bottom, h))
    left = max(0, min(left, w - 1))
    right = max(left + 1, min(right, w))
    
    # Crop image
    cropped = image[top:bottom, left:right]
    
    return cropped


def process_images(category_map: dict, valid_stations: set):
    """Process images only for stations in storage interval directories"""
    
    # Create single output directory (no category subdirectories)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    stats = {
        'total': 0,
        'processed': 0,
        'skipped_not_in_storage': 0,
        'skipped_no_category': 0,
        'by_category': {'good': 0, 'bad': 0, 'suspect': 0},
    }
    
    # Process both speed and direction images
    for img_type in ['speed', 'dir']:
        type_dir = IMAGE_BASE_DIR / img_type
        
        if not type_dir.exists():
            print(f"Warning: Directory not found: {type_dir}")
            continue
        
        print(f"\n--- Processing {img_type} images ---")
        
        # Collect all images from all subdirectories
        all_images = []
        for class_dir in ['good', 'bad', 'suspect']:
            subdir = type_dir / class_dir
            if subdir.exists():
                all_images.extend(list(subdir.glob("*.png")))
        
        print(f"Found {len(all_images)} images")
        
        # Process each image
        for image_path in tqdm(all_images, desc=f"Trimming {img_type}"):
            stats['total'] += 1
            
            # Get station name
            station_name = get_station_name_from_image(image_path)
            
            # Check if station is in storage interval directories
            if station_name not in valid_stations:
                stats['skipped_not_in_storage'] += 1
                continue
            
            # Get true category from metadata
            true_category = category_map.get(station_name)
            
            if not true_category or true_category not in ['good', 'bad', 'suspect']:
                stats['skipped_no_category'] += 1
                continue
            
            try:
                # Read image
                image = cv2.imread(str(image_path))
                
                if image is None:
                    print(f"Warning: Could not read {image_path}")
                    stats['skipped'] += 1
                    continue
                
                # Trim borders
                trimmed = trim_white_border(image)
                
                # Create output filename
                suffix = '_speed.png' if img_type == 'speed' else '_dir.png'
                output_filename = station_name + suffix
                output_path = OUTPUT_DIR / output_filename
                
                # Save trimmed image
                cv2.imwrite(str(output_path), trimmed)
                
                stats['processed'] += 1
                stats['by_category'][true_category] += 1
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                stats['skipped_no_category'] += 1
    
    return stats


def verify_trimmed_images():
    """Verify trimmed images and show statistics"""
    print("\n--- Verifying Trimmed Images ---")
    
    if OUTPUT_DIR.exists():
        images = list(OUTPUT_DIR.glob("*.png"))
        
        if images:
            # Sample one image to show size
            sample = cv2.imread(str(images[0]))
            print(f"Total trimmed images: {len(images)}")
            print(f"Sample size: {sample.shape[1]}x{sample.shape[0]}")
            
            # Count by type
            speed_count = len([img for img in images if img.stem.endswith('_speed')])
            dir_count = len([img for img in images if img.stem.endswith('_dir')])
            print(f"  Speed images: {speed_count}")
            print(f"  Direction images: {dir_count}")
        else:
            print("No images found")


def main():
    print("=" * 70)
    print("Trim and Organize Meteorological Images")
    print("=" * 70)
    print(f"\nInput: {IMAGE_BASE_DIR}")
    print(f"Storage Intervals: {STORAGE_INTERVAL_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Metadata from: {METADATA_FILE}")
    
    # Get valid stations from storage interval directories
    valid_stations = get_stations_from_storage_intervals()
    
    if not valid_stations:
        print("ERROR: No stations found in storage interval directories. Exiting.")
        return
    
    # Load category mapping from metadata
    category_map = load_category_mapping()
    
    if not category_map:
        print("ERROR: No categories loaded from metadata. Exiting.")
        return
    
    # Show overlap
    stations_with_categories = set(category_map.keys())
    overlap = valid_stations & stations_with_categories
    print(f"\nStations with both CSV data and categories: {len(overlap)}")
    
    # Process images
    print("\n" + "=" * 70)
    print("Processing Images")
    print("=" * 70)
    
    stats = process_images(category_map, valid_stations)
    
    # Print statistics
    print("\n" + "=" * 70)
    print("Processing Statistics")
    print("=" * 70)
    print(f"Total images found: {stats['total']}")
    print(f"Successfully processed: {stats['processed']}")
    print(f"Skipped (not in storage intervals): {stats['skipped_not_in_storage']}")
    print(f"Skipped (no category in metadata): {stats['skipped_no_category']}")
    
    print("\n--- Images by Category ---")
    for category, count in stats['by_category'].items():
        print(f"{category.upper()}: {count}")
    
    # Verify output
    verify_trimmed_images()
    
    print("\n" + "=" * 70)
    print("Image Trimming Complete!")
    print(f"Trimmed images saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
