#!/usr/bin/env python3
"""
Create Metadata File for Meteorological Stations

This script creates a comprehensive metadata file combining:
1. Station names from CSV files
2. Storage intervals from directory structure
3. True categories from LANL_categories.xlsx

Output: station_metadata.csv with columns:
- station_name: Full station identifier (e.g., MesoWest_AIRNOW_A2679)
- storage_interval: Time interval in minutes (5, 10, 15, 60, etc.)
- category: True category from Excel (good, bad, suspect)
- directory_class: Original directory classification (for reference)
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Set
import glob

# Configuration
BASE_DIR = Path("/home/jose/ONRL_ENCODER_DECODER_CAPARS/lanl_met_data/data_by_storage_interval")
CATEGORY_EXCEL = Path("/home/jose/ONRL_ENCODER_DECODER_CAPARS/LANL_categories.xlsx")
OUTPUT_FILE = Path("/home/jose/ONRL_ENCODER_DECODER_CAPARS/station_metadata.csv")

def load_category_mapping() -> Dict[str, str]:
    """Load true station categories from Excel file"""
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
        print(f"Error loading Excel file: {e}")
        return {}


def extract_station_name_from_file(filepath: Path) -> str:
    """Extract full station name from CSV filename"""
    # Example: MesoWest_ASOS-AWOS_KABQ_2023.csv -> MesoWest_ASOS-AWOS_KABQ
    name = filepath.stem  # Remove .csv
    parts = name.split('_')
    if len(parts) >= 2:
        # Remove year (last part)
        return '_'.join(parts[:-1])
    return name


def scan_storage_intervals() -> pd.DataFrame:
    """Scan all storage interval directories and collect station information"""
    
    all_records = []
    
    # Get all storage interval directories
    interval_dirs = [d for d in BASE_DIR.iterdir() if d.is_dir() and d.name.isdigit()]
    
    print(f"\nFound {len(interval_dirs)} storage interval directories")
    
    for interval_dir in sorted(interval_dirs, key=lambda x: int(x.name)):
        interval = int(interval_dir.name)
        
        # Look in good, bad, suspect subdirectories
        for class_dir in ['good', 'bad', 'suspect']:
            subdir = interval_dir / class_dir
            
            if not subdir.exists():
                continue
            
            # Find all CSV files
            csv_files = list(subdir.glob("*.csv"))
            
            for csv_file in csv_files:
                station_name = extract_station_name_from_file(csv_file)
                
                all_records.append({
                    'station_name': station_name,
                    'storage_interval': interval,
                    'directory_class': class_dir,
                    'source_file': csv_file.name,
                    'file_path': str(csv_file)
                })
        
        print(f"  Interval {interval} min: {len([r for r in all_records if r['storage_interval'] == interval])} stations")
    
    return pd.DataFrame(all_records)


def merge_with_true_categories(df: pd.DataFrame, category_map: Dict[str, str]) -> pd.DataFrame:
    """Add true categories from Excel to the dataframe"""
    
    # Map true categories
    df['category'] = df['station_name'].map(category_map)
    
    # For stations not in Excel, use directory class as fallback
    df['category'] = df['category'].fillna(df['directory_class'])
    
    # Count matches and mismatches
    matches = (df['category'] == df['directory_class']).sum()
    mismatches = (df['category'] != df['directory_class']).sum()
    
    print(f"\n--- Category Mapping ---")
    print(f"Total stations: {len(df)}")
    print(f"Matches (Excel == Directory): {matches}")
    print(f"Mismatches (Excel != Directory): {mismatches}")
    print(f"Not in Excel (using directory): {df['category'].isna().sum()}")
    
    return df


def main():
    print("=" * 70)
    print("Creating Station Metadata File")
    print("=" * 70)
    
    # Load true categories from Excel
    category_map = load_category_mapping()
    
    # Scan storage interval directories
    print("\n--- Scanning Storage Interval Directories ---")
    metadata_df = scan_storage_intervals()
    
    print(f"\nTotal records collected: {len(metadata_df)}")
    print(f"Unique stations: {metadata_df['station_name'].nunique()}")
    
    # Show distribution by storage interval
    print("\n--- Distribution by Storage Interval ---")
    print(metadata_df['storage_interval'].value_counts().sort_index())
    
    # Show distribution by directory class
    print("\n--- Distribution by Directory Class ---")
    print(metadata_df['directory_class'].value_counts())
    
    # Merge with true categories from Excel
    metadata_df = merge_with_true_categories(metadata_df, category_map)
    
    # Show distribution by true category
    print("\n--- Distribution by True Category (from Excel) ---")
    print(metadata_df['category'].value_counts())
    
    # Show stations where category changed
    changed = metadata_df[metadata_df['category'] != metadata_df['directory_class']]
    if len(changed) > 0:
        print(f"\n--- Stations with Category Corrections ({len(changed)}) ---")
        correction_summary = changed.groupby(['station_name', 'directory_class', 'category']).size().reset_index(name='count')
        print(correction_summary.head(20).to_string())
    
    # Remove duplicate stations (keep first occurrence by interval)
    # Some stations may appear in multiple intervals
    print("\n--- Handling Duplicates ---")
    print(f"Total records: {len(metadata_df)}")
    
    # Keep all records (station may have different intervals)
    # But create a summary with unique stations
    station_summary = metadata_df.groupby('station_name').agg({
        'storage_interval': lambda x: ','.join(map(str, sorted(x.unique()))),
        'category': 'first',
        'directory_class': 'first'
    }).reset_index()
    
    print(f"Unique stations: {len(station_summary)}")
    
    # Save full metadata (all interval records)
    output_cols = ['station_name', 'storage_interval', 'category', 'directory_class', 'source_file']
    metadata_df[output_cols].to_csv(OUTPUT_FILE, index=False)
    print(f"\n✓ Full metadata saved to: {OUTPUT_FILE}")
    
    # Save station summary (one row per station)
    summary_file = OUTPUT_FILE.parent / "station_summary.csv"
    station_summary.to_csv(summary_file, index=False)
    print(f"✓ Station summary saved to: {summary_file}")
    
    # Display sample
    print("\n--- Sample Records (first 10) ---")
    print(metadata_df[output_cols].head(10).to_string())
    
    print("\n--- Sample Station Summary (first 10) ---")
    print(station_summary.head(10).to_string())
    
    print("\n" + "=" * 70)
    print("Metadata Creation Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
