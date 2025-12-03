#!/usr/bin/env python3
"""
Exploratory Data Analysis (EDA) for LANL Meteorological Data

This script performs comprehensive EDA on the merged meteorological dataset including:
- Data overview and statistics
- Missing data analysis
- Temporal patterns analysis
- Distribution analysis by class and storage interval
- Correlation analysis
- Visualization generation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_FILE = Path("/home/jose/ONRL_ENCODER_DECODER_CAPARS/merged_met_data.parquet")
OUTPUT_DIR = Path("/home/jose/ONRL_ENCODER_DECODER_CAPARS/eda_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_data() -> pd.DataFrame:
    """Load the merged parquet data"""
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    
    df = pd.read_parquet(DATA_FILE)
    
    # Parse timestamp
    df['datetime'] = pd.to_datetime(df['timestamp string'].astype(str), format='%Y%m%d%H%M', errors='coerce')
    
    # Extract temporal features
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['dayofyear'] = df['datetime'].dt.dayofyear
    
    print(f"Loaded {len(df):,} rows from {DATA_FILE}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df


def basic_info(df: pd.DataFrame):
    """Display basic dataset information"""
    print("\n" + "=" * 70)
    print("BASIC DATASET INFORMATION")
    print("=" * 70)
    
    print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\nColumns: {list(df.columns)}")
    
    print("\n--- Data Types ---")
    print(df.dtypes)
    
    print("\n--- First 5 Rows ---")
    print(df.head().to_string())
    
    print("\n--- Last 5 Rows ---")
    print(df.tail().to_string())
    
    print("\n--- Numeric Summary Statistics ---")
    print(df[['wind direction', 'wind speed']].describe().to_string())
    
    # Date range
    print(f"\n--- Temporal Range ---")
    print(f"Start: {df['datetime'].min()}")
    print(f"End: {df['datetime'].max()}")
    print(f"Duration: {df['datetime'].max() - df['datetime'].min()}")


def categorical_analysis(df: pd.DataFrame):
    """Analyze categorical variables"""
    print("\n" + "=" * 70)
    print("CATEGORICAL ANALYSIS")
    print("=" * 70)
    
    # Class distribution
    print("\n--- Quality Class Distribution ---")
    class_counts = df['class'].value_counts()
    class_pct = df['class'].value_counts(normalize=True) * 100
    class_summary = pd.DataFrame({'Count': class_counts, 'Percentage': class_pct.round(2)})
    print(class_summary.to_string())
    
    # Storage interval distribution
    print("\n--- Storage Interval Distribution ---")
    interval_counts = df['storage_interval'].value_counts().sort_index()
    interval_pct = df['storage_interval'].value_counts(normalize=True).sort_index() * 100
    interval_summary = pd.DataFrame({'Count': interval_counts, 'Percentage': interval_pct.round(2)})
    print(interval_summary.to_string())
    
    # Station statistics
    print("\n--- Station Statistics ---")
    print(f"Total unique stations: {df['station'].nunique()}")
    print(f"\nTop 10 stations by record count:")
    print(df['station'].value_counts().head(10).to_string())
    
    print(f"\nBottom 10 stations by record count:")
    print(df['station'].value_counts().tail(10).to_string())
    
    # Cross-tabulation: class vs storage_interval
    print("\n--- Class × Storage Interval Cross-Tab ---")
    crosstab = pd.crosstab(df['class'], df['storage_interval'], margins=True)
    print(crosstab.to_string())
    
    return class_counts, interval_counts


def missing_data_analysis(df: pd.DataFrame):
    """Analyze missing data patterns"""
    print("\n" + "=" * 70)
    print("MISSING DATA ANALYSIS")
    print("=" * 70)
    
    # Overall missing counts
    missing = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
    missing_summary = pd.DataFrame({
        'Missing Count': missing,
        'Missing %': missing_pct,
        'Non-Missing': len(df) - missing
    })
    print("\n--- Missing Values by Column ---")
    print(missing_summary.to_string())
    
    # Missing by class
    print("\n--- Missing Wind Direction by Class ---")
    missing_by_class = df.groupby('class')['wind direction'].apply(lambda x: x.isnull().sum())
    total_by_class = df.groupby('class').size()
    pct_by_class = (missing_by_class / total_by_class * 100).round(2)
    print(pd.DataFrame({'Missing': missing_by_class, 'Total': total_by_class, 'Missing %': pct_by_class}).to_string())
    
    print("\n--- Missing Wind Speed by Class ---")
    missing_speed = df.groupby('class')['wind speed'].apply(lambda x: x.isnull().sum())
    pct_speed = (missing_speed / total_by_class * 100).round(2)
    print(pd.DataFrame({'Missing': missing_speed, 'Total': total_by_class, 'Missing %': pct_speed}).to_string())
    
    # Missing by storage interval
    print("\n--- Missing Wind Direction by Storage Interval ---")
    missing_by_interval = df.groupby('storage_interval')['wind direction'].apply(lambda x: x.isnull().sum())
    total_by_interval = df.groupby('storage_interval').size()
    pct_by_interval = (missing_by_interval / total_by_interval * 100).round(2)
    print(pd.DataFrame({'Missing': missing_by_interval, 'Total': total_by_interval, 'Missing %': pct_by_interval}).to_string())
    
    # Rows with any missing
    rows_any_missing = df[['wind direction', 'wind speed']].isnull().any(axis=1).sum()
    rows_all_missing = df[['wind direction', 'wind speed']].isnull().all(axis=1).sum()
    print(f"\n--- Row-Level Missing Summary ---")
    print(f"Rows with any missing (direction or speed): {rows_any_missing:,} ({rows_any_missing/len(df)*100:.2f}%)")
    print(f"Rows with both missing: {rows_all_missing:,} ({rows_all_missing/len(df)*100:.2f}%)")
    
    return missing_summary


def temporal_analysis(df: pd.DataFrame):
    """Analyze temporal patterns"""
    print("\n" + "=" * 70)
    print("TEMPORAL ANALYSIS")
    print("=" * 70)
    
    # Records per month
    print("\n--- Records by Month ---")
    monthly = df.groupby('month').size()
    print(monthly.to_string())
    
    # Records per hour
    print("\n--- Records by Hour of Day ---")
    hourly = df.groupby('hour').size()
    print(hourly.to_string())
    
    # Records per day of week
    print("\n--- Records by Day of Week (0=Monday) ---")
    daily = df.groupby('dayofweek').size()
    print(daily.to_string())
    
    # Average wind speed by hour
    print("\n--- Average Wind Speed by Hour ---")
    hourly_speed = df.groupby('hour')['wind speed'].mean().round(2)
    print(hourly_speed.to_string())
    
    # Average wind speed by month
    print("\n--- Average Wind Speed by Month ---")
    monthly_speed = df.groupby('month')['wind speed'].mean().round(2)
    print(monthly_speed.to_string())
    
    return monthly, hourly


def distribution_analysis(df: pd.DataFrame):
    """Analyze distributions of numeric variables"""
    print("\n" + "=" * 70)
    print("DISTRIBUTION ANALYSIS")
    print("=" * 70)
    
    # Wind direction statistics
    print("\n--- Wind Direction Statistics ---")
    wd = df['wind direction'].dropna()
    print(f"Count: {len(wd):,}")
    print(f"Mean: {wd.mean():.2f}°")
    print(f"Std: {wd.std():.2f}°")
    print(f"Min: {wd.min():.2f}°")
    print(f"25%: {wd.quantile(0.25):.2f}°")
    print(f"50% (Median): {wd.median():.2f}°")
    print(f"75%: {wd.quantile(0.75):.2f}°")
    print(f"Max: {wd.max():.2f}°")
    
    # Wind speed statistics
    print("\n--- Wind Speed Statistics ---")
    ws = df['wind speed'].dropna()
    print(f"Count: {len(ws):,}")
    print(f"Mean: {ws.mean():.2f}")
    print(f"Std: {ws.std():.2f}")
    print(f"Min: {ws.min():.2f}")
    print(f"25%: {ws.quantile(0.25):.2f}")
    print(f"50% (Median): {ws.median():.2f}")
    print(f"75%: {ws.quantile(0.75):.2f}")
    print(f"Max: {ws.max():.2f}")
    print(f"99%: {ws.quantile(0.99):.2f}")
    
    # Wind speed by class
    print("\n--- Wind Speed Statistics by Class ---")
    speed_by_class = df.groupby('class')['wind speed'].agg(['mean', 'std', 'min', 'max', 'median']).round(2)
    print(speed_by_class.to_string())
    
    # Wind speed by storage interval
    print("\n--- Wind Speed Statistics by Storage Interval ---")
    speed_by_interval = df.groupby('storage_interval')['wind speed'].agg(['mean', 'std', 'min', 'max', 'median']).round(2)
    print(speed_by_interval.to_string())
    
    # Outlier detection (IQR method)
    Q1 = ws.quantile(0.25)
    Q3 = ws.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = ws[(ws < lower_bound) | (ws > upper_bound)]
    
    print(f"\n--- Outlier Detection (Wind Speed, IQR Method) ---")
    print(f"IQR: {IQR:.2f}")
    print(f"Lower bound: {lower_bound:.2f}")
    print(f"Upper bound: {upper_bound:.2f}")
    print(f"Number of outliers: {len(outliers):,} ({len(outliers)/len(ws)*100:.2f}%)")
    
    return wd, ws


def correlation_analysis(df: pd.DataFrame):
    """Analyze correlations"""
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)
    
    # Numeric columns correlation
    numeric_cols = ['wind direction', 'wind speed', 'hour', 'month', 'dayofweek', 'storage_interval']
    corr_matrix = df[numeric_cols].corr().round(3)
    
    print("\n--- Correlation Matrix ---")
    print(corr_matrix.to_string())
    
    return corr_matrix


def station_analysis(df: pd.DataFrame):
    """Analyze data by station"""
    print("\n" + "=" * 70)
    print("STATION-LEVEL ANALYSIS")
    print("=" * 70)
    
    # Station summary
    station_summary = df.groupby('station').agg({
        'wind speed': ['count', 'mean', 'std', 'min', 'max'],
        'wind direction': lambda x: x.isnull().sum() / len(x) * 100,
        'class': 'first',
        'storage_interval': 'first'
    }).round(2)
    
    station_summary.columns = ['count', 'mean_speed', 'std_speed', 'min_speed', 'max_speed', 
                                'missing_dir_pct', 'class', 'interval']
    station_summary = station_summary.sort_values('count', ascending=False)
    
    print("\n--- Top 20 Stations by Record Count ---")
    print(station_summary.head(20).to_string())
    
    print("\n--- Station Count by Class ---")
    stations_by_class = df.groupby('class')['station'].nunique()
    print(stations_by_class.to_string())
    
    print("\n--- Station Count by Storage Interval ---")
    stations_by_interval = df.groupby('storage_interval')['station'].nunique()
    print(stations_by_interval.to_string())
    
    return station_summary


def generate_visualizations(df: pd.DataFrame, class_counts, interval_counts, corr_matrix):
    """Generate and save visualization plots"""
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    # 1. Class distribution pie chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', 
                colors=sns.color_palette("Set2", len(class_counts)))
    axes[0].set_title('Distribution by Quality Class', fontsize=12, fontweight='bold')
    
    axes[1].bar(interval_counts.index.astype(str), interval_counts.values, color=sns.color_palette("Set2", len(interval_counts)))
    axes[1].set_xlabel('Storage Interval (minutes)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Distribution by Storage Interval', fontsize=12, fontweight='bold')
    for i, v in enumerate(interval_counts.values):
        axes[1].text(i, v + v*0.01, f'{v:,}', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_class_interval_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 01_class_interval_distribution.png")
    
    # 2. Wind speed distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    ws = df['wind speed'].dropna()
    axes[0].hist(ws, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Wind Speed')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Wind Speed Distribution (All Data)', fontsize=12, fontweight='bold')
    axes[0].axvline(ws.mean(), color='red', linestyle='--', label=f'Mean: {ws.mean():.2f}')
    axes[0].axvline(ws.median(), color='green', linestyle='--', label=f'Median: {ws.median():.2f}')
    axes[0].legend()
    
    # Box plot by class
    df.boxplot(column='wind speed', by='class', ax=axes[1])
    axes[1].set_title('Wind Speed by Class', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Wind Speed')
    plt.suptitle('')
    
    # Box plot by interval
    df.boxplot(column='wind speed', by='storage_interval', ax=axes[2])
    axes[2].set_title('Wind Speed by Storage Interval', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Storage Interval (min)')
    axes[2].set_ylabel('Wind Speed')
    plt.suptitle('')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_wind_speed_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 02_wind_speed_distribution.png")
    
    # 3. Wind direction distribution (wind rose style histogram)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    wd = df['wind direction'].dropna()
    axes[0].hist(wd, bins=36, edgecolor='black', alpha=0.7)  # 36 bins = 10 degree intervals
    axes[0].set_xlabel('Wind Direction (degrees)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Wind Direction Distribution', fontsize=12, fontweight='bold')
    axes[0].set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
    axes[0].set_xticklabels(['N(0)', 'NE', 'E(90)', 'SE', 'S(180)', 'SW', 'W(270)', 'NW', 'N(360)'])
    
    # Polar plot for wind direction
    ax_polar = fig.add_subplot(1, 2, 2, projection='polar')
    theta = np.deg2rad(wd)
    hist, bin_edges = np.histogram(theta, bins=36)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax_polar.bar(bin_centers, hist, width=np.deg2rad(10), alpha=0.7, edgecolor='black')
    ax_polar.set_theta_zero_location('N')
    ax_polar.set_theta_direction(-1)
    ax_polar.set_title('Wind Direction (Polar)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03_wind_direction_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 03_wind_direction_distribution.png")
    
    # 4. Temporal patterns
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Hourly pattern
    hourly_speed = df.groupby('hour')['wind speed'].mean()
    axes[0, 0].plot(hourly_speed.index, hourly_speed.values, marker='o', linewidth=2)
    axes[0, 0].fill_between(hourly_speed.index, hourly_speed.values, alpha=0.3)
    axes[0, 0].set_xlabel('Hour of Day')
    axes[0, 0].set_ylabel('Average Wind Speed')
    axes[0, 0].set_title('Average Wind Speed by Hour', fontsize=12, fontweight='bold')
    axes[0, 0].set_xticks(range(0, 24, 2))
    
    # Monthly pattern
    monthly_speed = df.groupby('month')['wind speed'].mean()
    axes[0, 1].bar(monthly_speed.index, monthly_speed.values, color='steelblue', edgecolor='black')
    axes[0, 1].set_xlabel('Month')
    axes[0, 1].set_ylabel('Average Wind Speed')
    axes[0, 1].set_title('Average Wind Speed by Month', fontsize=12, fontweight='bold')
    axes[0, 1].set_xticks(range(1, 13))
    
    # Day of week pattern
    dow_speed = df.groupby('dayofweek')['wind speed'].mean()
    dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    axes[1, 0].bar(dow_speed.index, dow_speed.values, color='coral', edgecolor='black')
    axes[1, 0].set_xlabel('Day of Week')
    axes[1, 0].set_ylabel('Average Wind Speed')
    axes[1, 0].set_title('Average Wind Speed by Day of Week', fontsize=12, fontweight='bold')
    axes[1, 0].set_xticks(range(7))
    axes[1, 0].set_xticklabels(dow_labels)
    
    # Records over time (daily counts)
    daily_counts = df.groupby(df['datetime'].dt.date).size()
    axes[1, 1].plot(daily_counts.index, daily_counts.values, linewidth=0.5, alpha=0.7)
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel('Number of Records')
    axes[1, 1].set_title('Daily Record Counts Over Time', fontsize=12, fontweight='bold')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '04_temporal_patterns.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 04_temporal_patterns.png")
    
    # 5. Missing data heatmap
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Missing by class and variable
    missing_matrix = df.groupby('class')[['wind direction', 'wind speed']].apply(
        lambda x: x.isnull().sum() / len(x) * 100
    ).round(2)
    sns.heatmap(missing_matrix, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[0])
    axes[0].set_title('Missing Data % by Class', fontsize=12, fontweight='bold')
    
    # Missing by interval and variable  
    missing_by_interval = df.groupby('storage_interval')[['wind direction', 'wind speed']].apply(
        lambda x: x.isnull().sum() / len(x) * 100
    ).round(2)
    sns.heatmap(missing_by_interval, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[1])
    axes[1].set_title('Missing Data % by Storage Interval', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '05_missing_data_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 05_missing_data_heatmap.png")
    
    # 6. Correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                square=True, linewidths=0.5, ax=ax, fmt='.2f')
    ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '06_correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 06_correlation_heatmap.png")
    
    # 7. Wind speed vs direction scatter (sample for performance)
    fig, ax = plt.subplots(figsize=(10, 6))
    sample = df[['wind direction', 'wind speed', 'class']].dropna().sample(min(10000, len(df)))
    colors = {'good': 'green', 'bad': 'red', 'suspect': 'orange'}
    for cls in sample['class'].unique():
        subset = sample[sample['class'] == cls]
        ax.scatter(subset['wind direction'], subset['wind speed'], 
                   alpha=0.3, label=cls, c=colors.get(cls, 'blue'), s=10)
    ax.set_xlabel('Wind Direction (degrees)')
    ax.set_ylabel('Wind Speed')
    ax.set_title('Wind Speed vs Direction (Sample)', fontsize=12, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '07_speed_vs_direction.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 07_speed_vs_direction.png")
    
    # 8. Hourly heatmap by month
    fig, ax = plt.subplots(figsize=(14, 6))
    hourly_monthly = df.pivot_table(values='wind speed', index='hour', columns='month', aggfunc='mean')
    sns.heatmap(hourly_monthly, cmap='YlOrRd', annot=False, ax=ax)
    ax.set_xlabel('Month')
    ax.set_ylabel('Hour of Day')
    ax.set_title('Average Wind Speed: Hour × Month Heatmap', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '08_hourly_monthly_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 08_hourly_monthly_heatmap.png")
    
    print(f"\nAll visualizations saved to: {OUTPUT_DIR}")


def generate_report(df: pd.DataFrame, class_counts, interval_counts, missing_summary, 
                    corr_matrix, station_summary):
    """Generate a comprehensive summary report with ALL analysis results"""
    print("\n" + "=" * 70)
    print("GENERATING COMPREHENSIVE EDA REPORT")
    print("=" * 70)
    
    # Calculate all statistics for the report
    ws = df['wind speed'].dropna()
    wd = df['wind direction'].dropna()
    
    # Outlier detection
    Q1 = ws.quantile(0.25)
    Q3 = ws.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = ws[(ws < lower_bound) | (ws > upper_bound)]
    
    # Missing by class
    missing_dir_by_class = df.groupby('class')['wind direction'].apply(lambda x: x.isnull().sum())
    missing_speed_by_class = df.groupby('class')['wind speed'].apply(lambda x: x.isnull().sum())
    total_by_class = df.groupby('class').size()
    
    # Missing by interval
    missing_dir_by_interval = df.groupby('storage_interval')['wind direction'].apply(lambda x: x.isnull().sum())
    missing_speed_by_interval = df.groupby('storage_interval')['wind speed'].apply(lambda x: x.isnull().sum())
    total_by_interval = df.groupby('storage_interval').size()
    
    # Temporal stats
    hourly_speed = df.groupby('hour')['wind speed'].mean().round(2)
    monthly_speed = df.groupby('month')['wind speed'].mean().round(2)
    hourly_counts = df.groupby('hour').size()
    monthly_counts = df.groupby('month').size()
    dow_counts = df.groupby('dayofweek').size()
    
    # Speed by class and interval
    speed_by_class = df.groupby('class')['wind speed'].agg(['mean', 'std', 'min', 'max', 'median']).round(2)
    speed_by_interval = df.groupby('storage_interval')['wind speed'].agg(['mean', 'std', 'min', 'max', 'median']).round(2)
    
    # Cross-tabulation
    crosstab = pd.crosstab(df['class'], df['storage_interval'], margins=True)
    
    # Rows with missing
    rows_any_missing = df[['wind direction', 'wind speed']].isnull().any(axis=1).sum()
    rows_all_missing = df[['wind direction', 'wind speed']].isnull().all(axis=1).sum()
    
    report = f"""
{'='*80}
LANL METEOROLOGICAL DATA - COMPREHENSIVE EDA SUMMARY REPORT
{'='*80}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Data File: {DATA_FILE}

{'='*80}
1. DATASET OVERVIEW
{'='*80}
Shape: {df.shape[0]:,} rows × {df.shape[1]} columns
Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB

Columns: {list(df.columns)}

Data Types:
{df.dtypes.to_string()}

Temporal Range:
  Start: {df['datetime'].min()}
  End: {df['datetime'].max()}
  Duration: {df['datetime'].max() - df['datetime'].min()}
  Total Days: {(df['datetime'].max() - df['datetime'].min()).days}

Unique Values:
  - Unique Stations: {df['station'].nunique()}
  - Unique Timestamps: {df['datetime'].nunique():,}
  - Unique Source Files: {df['source_file'].nunique()}

{'='*80}
2. QUALITY CLASS DISTRIBUTION
{'='*80}
{pd.DataFrame({
    'Count': class_counts,
    'Percentage': (class_counts / len(df) * 100).round(2)
}).to_string()}

{'='*80}
3. STORAGE INTERVAL DISTRIBUTION
{'='*80}
{pd.DataFrame({
    'Count': interval_counts,
    'Percentage': (interval_counts / len(df) * 100).round(2)
}).to_string()}

{'='*80}
4. CLASS × STORAGE INTERVAL CROSS-TABULATION
{'='*80}
{crosstab.to_string()}

{'='*80}
5. MISSING DATA ANALYSIS
{'='*80}

Overall Missing Values:
{missing_summary.to_string()}

Rows with Any Missing (direction or speed): {rows_any_missing:,} ({rows_any_missing/len(df)*100:.2f}%)
Rows with Both Missing: {rows_all_missing:,} ({rows_all_missing/len(df)*100:.2f}%)

Missing Wind Direction by Class:
{pd.DataFrame({
    'Missing': missing_dir_by_class,
    'Total': total_by_class,
    'Missing %': (missing_dir_by_class / total_by_class * 100).round(2)
}).to_string()}

Missing Wind Speed by Class:
{pd.DataFrame({
    'Missing': missing_speed_by_class,
    'Total': total_by_class,
    'Missing %': (missing_speed_by_class / total_by_class * 100).round(2)
}).to_string()}

Missing Wind Direction by Storage Interval:
{pd.DataFrame({
    'Missing': missing_dir_by_interval,
    'Total': total_by_interval,
    'Missing %': (missing_dir_by_interval / total_by_interval * 100).round(2)
}).to_string()}

Missing Wind Speed by Storage Interval:
{pd.DataFrame({
    'Missing': missing_speed_by_interval,
    'Total': total_by_interval,
    'Missing %': (missing_speed_by_interval / total_by_interval * 100).round(2)
}).to_string()}

{'='*80}
6. WIND SPEED STATISTICS
{'='*80}
Count: {len(ws):,}
Mean: {ws.mean():.4f}
Std: {ws.std():.4f}
Min: {ws.min():.4f}
25%: {ws.quantile(0.25):.4f}
50% (Median): {ws.median():.4f}
75%: {ws.quantile(0.75):.4f}
Max: {ws.max():.4f}
90%: {ws.quantile(0.90):.4f}
95%: {ws.quantile(0.95):.4f}
99%: {ws.quantile(0.99):.4f}

Outlier Detection (IQR Method):
  IQR: {IQR:.4f}
  Lower Bound: {lower_bound:.4f}
  Upper Bound: {upper_bound:.4f}
  Number of Outliers: {len(outliers):,} ({len(outliers)/len(ws)*100:.2f}%)

Wind Speed by Class:
{speed_by_class.to_string()}

Wind Speed by Storage Interval:
{speed_by_interval.to_string()}

{'='*80}
7. WIND DIRECTION STATISTICS
{'='*80}
Count: {len(wd):,}
Mean: {wd.mean():.2f}°
Std: {wd.std():.2f}°
Min: {wd.min():.2f}°
25%: {wd.quantile(0.25):.2f}°
50% (Median): {wd.median():.2f}°
75%: {wd.quantile(0.75):.2f}°
Max: {wd.max():.2f}°

Direction Bins (Cardinal):
  N (337.5-22.5°): {len(wd[(wd >= 337.5) | (wd < 22.5)]):,}
  NE (22.5-67.5°): {len(wd[(wd >= 22.5) & (wd < 67.5)]):,}
  E (67.5-112.5°): {len(wd[(wd >= 67.5) & (wd < 112.5)]):,}
  SE (112.5-157.5°): {len(wd[(wd >= 112.5) & (wd < 157.5)]):,}
  S (157.5-202.5°): {len(wd[(wd >= 157.5) & (wd < 202.5)]):,}
  SW (202.5-247.5°): {len(wd[(wd >= 202.5) & (wd < 247.5)]):,}
  W (247.5-292.5°): {len(wd[(wd >= 247.5) & (wd < 292.5)]):,}
  NW (292.5-337.5°): {len(wd[(wd >= 292.5) & (wd < 337.5)]):,}

{'='*80}
8. TEMPORAL ANALYSIS
{'='*80}

Records by Month:
{monthly_counts.to_string()}

Average Wind Speed by Month:
{monthly_speed.to_string()}

Records by Hour of Day:
{hourly_counts.to_string()}

Average Wind Speed by Hour:
{hourly_speed.to_string()}

Records by Day of Week (0=Monday):
{dow_counts.to_string()}

{'='*80}
9. CORRELATION MATRIX
{'='*80}
{corr_matrix.to_string()}

{'='*80}
10. STATION ANALYSIS
{'='*80}

Total Unique Stations: {df['station'].nunique()}

Stations by Class:
{df.groupby('class')['station'].nunique().to_string()}

Stations by Storage Interval:
{df.groupby('storage_interval')['station'].nunique().to_string()}

Top 20 Stations by Record Count:
{station_summary.head(20).to_string()}

Bottom 10 Stations by Record Count:
{station_summary.tail(10).to_string()}

Station Record Count Statistics:
  Mean records per station: {station_summary['count'].mean():.0f}
  Median records per station: {station_summary['count'].median():.0f}
  Min records: {station_summary['count'].min():,}
  Max records: {station_summary['count'].max():,}

{'='*80}
11. DATA SAMPLE
{'='*80}

First 10 Rows:
{df.head(10).to_string()}

Last 10 Rows:
{df.tail(10).to_string()}

{'='*80}
12. KEY FINDINGS & RECOMMENDATIONS
{'='*80}

Data Quality:
  - {df[df['class']=='good'].shape[0]/len(df)*100:.1f}% of records are classified as 'good'
  - {df[df['class']=='bad'].shape[0]/len(df)*100:.1f}% of records are classified as 'bad'
  - {df[df['class']=='suspect'].shape[0]/len(df)*100:.1f}% of records are classified as 'suspect'

Missing Data:
  - Wind direction has {df['wind direction'].isnull().sum()/len(df)*100:.1f}% missing values
  - Wind speed has {df['wind speed'].isnull().sum()/len(df)*100:.1f}% missing values
  - 'bad' class has highest missing rate

Temporal Coverage:
  - Data spans {(df['datetime'].max() - df['datetime'].min()).days} days
  - Most common storage interval: {interval_counts.idxmax()} minutes ({interval_counts.max():,} records)

Wind Patterns:
  - Average wind speed: {ws.mean():.2f} (std: {ws.std():.2f})
  - {len(outliers)/len(ws)*100:.1f}% of wind speed values are outliers (IQR method)

{'='*80}
GENERATED VISUALIZATION FILES
{'='*80}
- 01_class_interval_distribution.png
- 02_wind_speed_distribution.png
- 03_wind_direction_distribution.png
- 04_temporal_patterns.png
- 05_missing_data_heatmap.png
- 06_correlation_heatmap.png
- 07_speed_vs_direction.png
- 08_hourly_monthly_heatmap.png

All files saved in: {OUTPUT_DIR}

{'='*80}
END OF REPORT
{'='*80}
"""
    
    # Save report
    report_path = OUTPUT_DIR / 'eda_summary_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(report)
    print(f"\nFull report saved to: {report_path}")


def main():
    print("\n" + "=" * 70)
    print("LANL METEOROLOGICAL DATA - EXPLORATORY DATA ANALYSIS")
    print("=" * 70)
    
    # Load data
    df = load_data()
    
    # Run analyses
    basic_info(df)
    class_counts, interval_counts = categorical_analysis(df)
    missing_summary = missing_data_analysis(df)
    temporal_analysis(df)
    distribution_analysis(df)
    corr_matrix = correlation_analysis(df)
    station_summary = station_analysis(df)
    
    # Generate visualizations
    generate_visualizations(df, class_counts, interval_counts, corr_matrix)
    
    # Generate comprehensive summary report (includes ALL shell output)
    generate_report(df, class_counts, interval_counts, missing_summary, 
                    corr_matrix, station_summary)
    
    print("\n" + "=" * 70)
    print("EDA COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
