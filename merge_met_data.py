#!/usr/bin/env python3
"""
Merge LANL Meteorological CSV Data

This script reads CSV files from storage interval directories (5, 10, 15, 60 minutes)
and merges them into a single dataset with added metadata columns:
- storage_interval: The time interval (5, 10, 15, or 60 minutes)
- class: Data quality class (good, bad, suspect)
- station: Station ID extracted from filename

Supports timestamp alignment via resampling to a common interval.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import glob
from multiprocessing import Pool, cpu_count
from functools import partial

# Configuration
BASE_DIR = Path("/home/jose/ONRL_ENCODER_DECODER_CAPARS/lanl_met_data/data_by_storage_interval")
CATEGORY_EXCEL = Path("/home/jose/ONRL_ENCODER_DECODER_CAPARS/LANL_categories.xlsx")
STORAGE_INTERVALS = [5, 10, 15]  # minutes ,60 #HERE WE CAN TRY DIFFERENT CONVINATIONS AND INTERVALS
QUALITY_CLASSES = ["good", "bad", "suspect"]  # All classes from Excel file
OUTPUT_FILE = Path("/home/jose/ONRL_ENCODER_DECODER_CAPARS/merged_met_data.parquet")

# Alignment configuration
ALIGN_TIMESTAMPS = True  # Set to True to align all data to common interval # WE CAN TRY TRUE OR FALSE
TARGET_INTERVAL_MINUTES = 5  # Resample everything to this interval (use LCM or common interval) EITHER 5 OR 15 OR 60
AGGREGATION_METHOD = 'mean'  # 'mean', 'median', 'first', 'last', 'max', 'min' # NO OTHER PARAEMTER THAN MEAN
INTERPOLATE_UPSAMPLING = True  # If True, interpolate when upsampling; if False, leave NaN HERE TRUE OR NAN

# Multiprocessing configuration
NUM_WORKERS = cpu_count()  # Use all available CPU cores


def load_category_mapping() -> Dict[str, str]:
    """Load station category mapping from Excel file"""
    try:
        df = pd.read_excel(CATEGORY_EXCEL)
        # Create mapping: station name -> category (lowercase)
        category_map = {}
        for _, row in df.iterrows():
            station = str(row['Station']).strip()
            category = str(row['Category']).strip().lower()
            category_map[station] = category
        print(f"Loaded {len(category_map)} station categories from {CATEGORY_EXCEL}")
        return category_map
    except Exception as e:
        print(f"Error loading category Excel file: {e}")
        return {}


def extract_station_id(filename: str) -> str:
    """Extract station ID from filename like 'MesoWest_ASOS-AWOS_KABQ_2023.csv'"""
    # Remove extension and split by underscore
    name = filename.replace('.csv', '')
    parts = name.split('_')
    # Station ID is typically the third part
    if len(parts) >= 3:
        return parts[-2]  # Second to last (before year)
    return name


def get_full_station_name(filename: str) -> str:
    """Extract full station name (without year) from filename"""
    # Remove extension and year
    name = filename.replace('.csv', '')
    parts = name.split('_')
    # Remove the last part (year)
    if len(parts) >= 2:
        return '_'.join(parts[:-1])
    return name


def parse_timestamp(ts_str) -> pd.Timestamp:
    """Parse timestamp string like '202301010000' to datetime"""
    try:
        return pd.to_datetime(str(ts_str), format='%Y%m%d%H%M')
    except:
        return pd.NaT


def process_alignment_group(args: Tuple[Tuple[str, str, int], pd.DataFrame, int, str]) -> pd.DataFrame:
    """
    Process a single station/class/interval group for alignment.
    Used for parallel processing.
    """
    (station, quality_class, orig_interval), group_df, target_minutes, agg_method = args
    
    group_df = group_df.set_index('datetime').sort_index()
    
    # Resample to target interval
    resample_rule = f'{target_minutes}min'
    
    # Check if downsampling or upsampling
    is_downsampling = orig_interval < target_minutes
    is_upsampling = orig_interval > target_minutes
    
    if is_downsampling:
        # Downsampling: Aggregate multiple readings into one
        resampled = group_df[['wind direction', 'wind speed']].resample(resample_rule).agg(agg_method)
    elif is_upsampling:
        # Upsampling: Need to interpolate or forward-fill
        resampled = group_df[['wind direction', 'wind speed']].resample(resample_rule).asfreq()
        
        if INTERPOLATE_UPSAMPLING:
            # Linear interpolation for wind speed
            resampled['wind speed'] = resampled['wind speed'].interpolate(method='linear')
            # For wind direction, use nearest (circular interpolation is complex)
            resampled['wind direction'] = resampled['wind direction'].interpolate(method='nearest')
    else:
        # Same interval - no resampling needed
        resampled = group_df[['wind direction', 'wind speed']]
    
    # Drop rows where all values are NaN
    resampled = resampled.dropna(how='all')
    
    # Add metadata back
    resampled['station'] = station
    resampled['class'] = quality_class
    resampled['original_interval'] = orig_interval
    resampled['aligned_interval'] = target_minutes
    
    # Reset index to get datetime as column
    resampled = resampled.reset_index()
    resampled['timestamp string'] = resampled['datetime'].dt.strftime('%Y%m%d%H%M')
    
    return resampled


def align_timestamps_to_interval(df: pd.DataFrame, target_minutes: int, 
                                  agg_method: str = 'mean') -> pd.DataFrame:
    """
    Align timestamps to a common interval by resampling using parallel processing.
    
    For each station+class combination, resample to target interval.
    - Downsampling (e.g., 5min → 60min): Aggregates using agg_method
    - Upsampling (e.g., 60min → 5min): Interpolates or leaves NaN based on INTERPOLATE_UPSAMPLING
    """
    if df.empty:
        return df
    
    # Parse timestamp to datetime
    df = df.copy()
    df['datetime'] = df['timestamp string'].apply(parse_timestamp)
    df = df.dropna(subset=['datetime'])
    
    # Prepare arguments for parallel processing
    groups = df.groupby(['station', 'class', 'storage_interval'])
    group_args = [
        (group_key, group_df.copy(), target_minutes, agg_method)
        for group_key, group_df in groups
    ]
    
    print(f"Processing {len(group_args)} station/class/interval groups using {NUM_WORKERS} cores...")
    
    # Process groups in parallel
    with Pool(processes=NUM_WORKERS) as pool:
        aligned_dfs = pool.map(process_alignment_group, group_args)
    
    # Filter out empty dataframes
    aligned_dfs = [df for df in aligned_dfs if not df.empty]
    
    if aligned_dfs:
        return pd.concat(aligned_dfs, ignore_index=True)
    return pd.DataFrame()


def create_wide_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create wide format where each timestamp has columns for each station.
    Useful for comparing stations at the same time points.
    """
    if df.empty:
        return df
    
    # Pivot to wide format - one row per timestamp, columns per station
    wide_df = df.pivot_table(
        index='datetime',
        columns=['station', 'class'],
        values=['wind direction', 'wind speed'],
        aggfunc='mean'
    )
    
    # Flatten column names
    wide_df.columns = ['_'.join(map(str, col)).strip() for col in wide_df.columns.values]
    wide_df = wide_df.reset_index()
    
    return wide_df


def read_and_process_csv(filepath: Path, storage_interval: int, quality_class: str, category_map: Dict[str, str]) -> pd.DataFrame:
    """Read a CSV file and add metadata columns with true category from Excel"""
    try:
        df = pd.read_csv(filepath)
        
        # Get station names
        station_id = extract_station_id(filepath.name)
        full_station_name = get_full_station_name(filepath.name)
        
        # Use true category from Excel mapping, fall back to directory-based class if not found
        true_category = category_map.get(full_station_name, quality_class)
        
        # Add metadata columns
        df['storage_interval'] = storage_interval
        df['class'] = true_category  # Use true category from Excel
        df['directory_class'] = quality_class  # Keep original directory class for reference
        df['station'] = station_id
        df['full_station_name'] = full_station_name
        df['source_file'] = filepath.name
        
        return df
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return pd.DataFrame()


def process_csv_file(args: Tuple[Path, int, str, Dict[str, str]]) -> pd.DataFrame:
    """Wrapper function for multiprocessing - processes a single CSV file"""
    filepath, storage_interval, quality_class, category_map = args
    return read_and_process_csv(filepath, storage_interval, quality_class, category_map)


def merge_all_data() -> pd.DataFrame:
    """Merge all CSV files from specified directories using multiprocessing"""
    # Load category mapping from Excel
    category_map = load_category_mapping()
    
    file_args: List[Tuple[Path, int, str, Dict[str, str]]] = []
    
    # Collect all files to process
    for interval in STORAGE_INTERVALS:
        interval_dir = BASE_DIR / str(interval)
        
        if not interval_dir.exists():
            print(f"Warning: Directory not found: {interval_dir}")
            continue
            
        for quality_class in QUALITY_CLASSES:
            class_dir = interval_dir / quality_class
            
            if not class_dir.exists():
                print(f"Warning: Directory not found: {class_dir}")
                continue
            
            # Find all CSV files
            csv_files = list(class_dir.glob("*.csv"))
            print(f"Found {len(csv_files)} files in {interval}min/{quality_class}")
            
            # Add to processing queue with category_map
            for csv_file in csv_files:
                file_args.append((csv_file, interval, quality_class, category_map))
    
    print(f"\nTotal files to process: {len(file_args)}")
    print(f"Using {NUM_WORKERS} CPU cores for parallel processing...")
    
    if not file_args:
        return pd.DataFrame()
    
    # Process files in parallel
    with Pool(processes=NUM_WORKERS) as pool:
        all_dataframes = pool.map(process_csv_file, file_args)
    
    # Filter out empty dataframes
    all_dataframes = [df for df in all_dataframes if not df.empty]
    print(f"Successfully processed: {len(all_dataframes)} files")
    
    if all_dataframes:
        # Concatenate all dataframes
        merged_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Reorder columns
        column_order = [
            'timestamp string', 
            'wind direction', 
            'wind speed',
            'storage_interval',
            'class',
            'directory_class',
            'station',
            'full_station_name',
            'source_file'
        ]
        
        # Keep only columns that exist
        final_columns = [col for col in column_order if col in merged_df.columns]
        merged_df = merged_df[final_columns]
        
        return merged_df
    
    return pd.DataFrame()


def main():
    print("=" * 60)
    print("LANL Meteorological Data Merger")
    print("=" * 60)
    print(f"\nBase directory: {BASE_DIR}")
    print(f"Storage intervals: {STORAGE_INTERVALS} minutes")
    print(f"Quality classes: {QUALITY_CLASSES}")
    print(f"Parallel workers: {NUM_WORKERS} CPU cores")
    print()
    
    # Merge data
    merged_df = merge_all_data()
    
    if merged_df.empty:
        print("No data was merged!")
        return
    
    # Display summary
    print("\n" + "=" * 60)
    print("MERGE SUMMARY")
    print("=" * 60)
    print(f"Total rows: {len(merged_df):,}")
    print(f"Columns: {list(merged_df.columns)}")
    
    print("\nRows by storage interval:")
    print(merged_df['storage_interval'].value_counts().sort_index())
    
    print("\nRows by quality class (from Excel):")
    print(merged_df['class'].value_counts())
    
    print("\nRows by directory class (original):")
    print(merged_df['directory_class'].value_counts())
    
    print("\n--- Category Mapping Comparison ---")
    comparison = merged_df.groupby(['directory_class', 'class']).size().reset_index(name='count')
    print(comparison.to_string())
    
    # Show stations where directory class != Excel class
    mismatches = merged_df[merged_df['directory_class'] != merged_df['class']]
    if len(mismatches) > 0:
        print(f"\n--- Stations with category corrections: {mismatches['full_station_name'].nunique()} ---")
        mismatch_summary = mismatches.groupby(['full_station_name', 'directory_class', 'class']).size().reset_index(name='records')
        print(mismatch_summary.to_string())
    
    # Timestamp alignment
    if ALIGN_TIMESTAMPS:
        print(f"\n{'=' * 60}")
        print(f"ALIGNING TIMESTAMPS TO {TARGET_INTERVAL_MINUTES}-MINUTE INTERVALS")
        print(f"Aggregation method: {AGGREGATION_METHOD}")
        print("=" * 60)
        
        aligned_df = align_timestamps_to_interval(
            merged_df, 
            TARGET_INTERVAL_MINUTES, 
            AGGREGATION_METHOD
        )
        
        print(f"\nAligned rows: {len(aligned_df):,}")
        print(f"Unique timestamps: {aligned_df['datetime'].nunique():,}")
        print(f"Date range: {aligned_df['datetime'].min()} to {aligned_df['datetime'].max()}")
        
        print("\nAligned data sample (first 10 rows):")
        print(aligned_df.head(10).to_string())
        
        # Save aligned data
        aligned_output = OUTPUT_FILE.parent / f"aligned_{TARGET_INTERVAL_MINUTES}min_met_data.parquet"
        aligned_df.to_parquet(aligned_output, index=False, engine='pyarrow')
        print(f"\nAligned data saved to: {aligned_output}")
        
        # # Optional: Create wide format for easy station comparison
        # print("\nCreating wide format (stations as columns)...")
        # wide_df = create_wide_format(aligned_df)
        # if not wide_df.empty:
        #     wide_output = OUTPUT_FILE.parent / f"wide_{TARGET_INTERVAL_MINUTES}min_met_data.parquet"
        #     wide_df.to_parquet(wide_output, index=False, engine='pyarrow')
        #     print(f"Wide format saved to: {wide_output}")
        #     print(f"Wide format shape: {wide_df.shape}")

    print("\nSample data (first 10 rows):")
    print(merged_df.head(10).to_string())
    
    # # Save to Parquet (more efficient than CSV)
    # merged_df.to_parquet(OUTPUT_FILE, index=False, engine='pyarrow')
    # print(f"\nMerged data saved to: {OUTPUT_FILE}")
    # print(f"File size: {OUTPUT_FILE.stat().st_size / (1024*1024):.2f} MB")


if __name__ == "__main__":
    main()
