#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import json
import h5py
from collections import defaultdict
import time
from scipy import stats


def load_crowd_behavior_file(file_path):
    """Loads crowd behavior data from text file"""
    try:
        data = pd.read_csv(file_path, header=None, 
                         names=['frame_id', 'density', 'centroid_x', 'centroid_y', 'convex_hull_area'])
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def trimmed_mean(values, trim_percent=5):
    """Calculates trimmed mean by removing extreme values from both ends"""
    if len(values) == 0:
        return 0.0
    
    n = len(values)
    trim_count = int(n * trim_percent / 100)
    
    if trim_count * 2 >= n:
        return np.mean(values)
    
    sorted_values = np.sort(values)
    if trim_count > 0:
        trimmed_values = sorted_values[trim_count:-trim_count]
    else:
        trimmed_values = sorted_values
    
    return np.mean(trimmed_values)


def winsorized_mean(values, winsor_percent=5):
    """Calculates winsorized mean by replacing extreme values with percentile cutoffs"""
    if len(values) == 0:
        return 0.0
    
    lower_percentile = winsor_percent
    upper_percentile = 100 - winsor_percent
    
    lower_bound = np.percentile(values, lower_percentile)
    upper_bound = np.percentile(values, upper_percentile)
    
    winsorized_values = np.clip(values, lower_bound, upper_bound)
    return np.mean(winsorized_values)


def median_absolute_deviation(values):
    """Calculates Median Absolute Deviation"""
    if len(values) == 0:
        return 0.0
    
    median_val = np.median(values)
    return np.median(np.abs(values - median_val))


def coefficient_of_variation(values):
    """Calculates coefficient of variation (std/mean)"""
    if len(values) == 0:
        return 0.0
    
    mean_val = np.mean(values)
    if mean_val == 0:
        return 0.0
    
    return np.std(values) / np.abs(mean_val)


def max_absolute_consecutive_diff(values):
    """Calculates maximum absolute difference between consecutive values"""
    if len(values) <= 1:
        return 0.0
    
    diffs = np.abs(np.diff(values))
    return np.max(diffs)


def longest_consecutive_run(values, increasing=True):
    """Calculates longest run of consecutive increases or decreases"""
    if len(values) <= 1:
        return 0
    
    diffs = np.diff(values)
    if increasing:
        conditions = diffs > 0
    else:
        conditions = diffs < 0
    
    max_run = 0
    current_run = 0
    
    for condition in conditions:
        if condition:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    
    return max_run


def count_values_above_median(values):
    """Counts number of values above the median"""
    if len(values) == 0:
        return 0
    
    median_val = np.median(values)
    return np.sum(values > median_val)


def count_outliers(values):
    """Counts outliers using IQR method"""
    if len(values) == 0:
        return 0
    
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = (values < lower_bound) | (values > upper_bound)
    return np.sum(outliers)


def calculate_feature_statistics(values):
    """Calculates all 22 statistics for a single feature"""
    if len(values) == 0:
        return np.zeros(22)
    
    stats_list = []
    
    # Location (3)
    stats_list.append(np.mean(values))
    stats_list.append(np.median(values))
    stats_list.append(trimmed_mean(values))
    
    # Dispersion (6)
    stats_list.append(np.std(values))
    stats_list.append(np.var(values))
    stats_list.append(np.max(values) - np.min(values))
    stats_list.append(np.percentile(values, 75) - np.percentile(values, 25))
    stats_list.append(median_absolute_deviation(values))
    stats_list.append(coefficient_of_variation(values))
    
    # Shape (2)
    try:
        stats_list.append(stats.skew(values))
        stats_list.append(stats.kurtosis(values))
    except:
        stats_list.extend([0.0, 0.0])
    
    # Percentiles (2)
    stats_list.append(np.percentile(values, 25))
    stats_list.append(np.percentile(values, 75))
    
    # Temporal (4)
    stats_list.append(values[0])
    stats_list.append(values[-1])
    stats_list.append(values[-1] - values[0])
    
    if len(values) > 1:
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        stats_list.append(slope)
    else:
        stats_list.append(0.0)
    
    # Robust (1)
    stats_list.append(winsorized_mean(values))
    
    # Sequential (2)
    stats_list.append(max_absolute_consecutive_diff(values))
    max_inc = longest_consecutive_run(values, increasing=True)
    max_dec = longest_consecutive_run(values, increasing=False)
    stats_list.append(max(max_inc, max_dec))
    
    # Frequency (2)
    stats_list.append(count_values_above_median(values))
    stats_list.append(count_outliers(values))
    
    return np.array(stats_list)


def calculate_correlations(data):
    """Calculates Pearson correlations between all feature pairs"""
    if data is None or len(data) == 0:
        return np.zeros(6)
    
    feature_cols = ['density', 'centroid_x', 'centroid_y', 'convex_hull_area']
    correlations = []
    
    try:
        for i in range(len(feature_cols)):
            for j in range(i+1, len(feature_cols)):
                corr = np.corrcoef(data[feature_cols[i]], data[feature_cols[j]])[0, 1]
                if np.isnan(corr):
                    corr = 0.0
                correlations.append(corr)
    except:
        correlations = [0.0] * 6
    
    return np.array(correlations)


def encode_crowd_behavior(data):
    """Complete encoding: 22 statistics per feature (4 features) + 6 correlations = 94 dimensions"""
    if data is None or len(data) == 0:
        return np.zeros(94)
    
    feature_stats = []
    feature_cols = ['density', 'centroid_x', 'centroid_y', 'convex_hull_area']
    
    for col in feature_cols:
        values = data[col].values
        stats_vector = calculate_feature_statistics(values)
        feature_stats.extend(stats_vector)
    
    correlations = calculate_correlations(data)
    all_features = np.concatenate([feature_stats, correlations])
    
    return all_features


def process_all_crowd_data(input_dir, output_dir):
    """Processes all crowd behavior files and creates encoded features"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    feature_dim = 94
    
    print(f"Creating {feature_dim}-dimensional statistical feature vectors")
    print("Feature breakdown: 88 per-feature statistics + 6 correlations")
    
    stats = {
        'total_videos': 0,
        'successful': 0,
        'failed': 0,
        'feature_dimension': feature_dim,
        'encoding_method': 'comprehensive_statistics',
        'splits': {}
    }
    
    for split_dir in input_path.iterdir():
        if split_dir.is_dir() and split_dir.name in ['train', 'val', 'test']:
            split_name = split_dir.name
            print(f"\nProcessing {split_name} split...")
            
            split_output_dir = output_path / split_name
            split_output_dir.mkdir(parents=True, exist_ok=True)
            
            split_stats = {
                'categories': {},
                'total': 0,
                'successful': 0
            }
            
            categories = ['background', 'before_goal', 'free_kicks_goals', 'penalties', 'shots_no_goals']
            
            for category in categories:
                category_dir = split_dir / category
                if not category_dir.exists():
                    print(f"  Category not found: {category}")
                    continue
                
                print(f"  Processing category: {category}")
                
                h5_file_path = split_output_dir / f"{category}_crowd_features.h5"
                
                category_results = []
                successful_count = 0
                
                with h5py.File(h5_file_path, 'w') as hf:
                    crowd_files = list(category_dir.glob("*/*_crowd_behavior.txt"))
                    
                    for crowd_file in crowd_files:
                        video_name = crowd_file.parent.name
                        
                        try:
                            data = load_crowd_behavior_file(crowd_file)
                            features = encode_crowd_behavior(data)
                            
                            assert len(features) == feature_dim, f"Expected {feature_dim} features, got {len(features)}"
                            
                            hf.create_dataset(video_name, data=features)
                            
                            category_results.append({
                                'video': video_name,
                                'feature_shape': list(features.shape),
                                'original_frames': len(data) if data is not None else 0,
                                'status': 'success'
                            })
                            
                            successful_count += 1
                            stats['successful'] += 1
                            
                        except Exception as e:
                            print(f"      ERROR processing {video_name}: {e}")
                            
                            zero_features = np.zeros(feature_dim)
                            hf.create_dataset(video_name, data=zero_features)
                            
                            category_results.append({
                                'video': video_name,
                                'error': str(e),
                                'status': 'failed'
                            })
                            stats['failed'] += 1
                        
                        stats['total_videos'] += 1
                
                metadata_path = split_output_dir / f"{category}_crowd_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump({
                        'split': split_name,
                        'category': category,
                        'encoding_method': 'comprehensive_statistics',
                        'feature_dimension': feature_dim,
                        'total_videos': len(crowd_files),
                        'successful': successful_count,
                        'failed': len(crowd_files) - successful_count,
                        'h5_file': str(h5_file_path),
                        'results': category_results
                    }, f, indent=2)
                
                split_stats['categories'][category] = {
                    'total': len(crowd_files),
                    'successful': successful_count,
                    'h5_file': str(h5_file_path)
                }
                
                split_stats['total'] += len(crowd_files)
                split_stats['successful'] += successful_count
                
                print(f"    {category}: {successful_count}/{len(crowd_files)} successful")
            
            stats['splits'][split_name] = split_stats
            print(f"  {split_name} total: {split_stats['successful']}/{split_stats['total']} successful")
    
    overall_metadata = output_path / "crowd_statistical_metadata.json"
    with open(overall_metadata, 'w') as f:
        json.dump({
            'input_directory': str(input_dir),
            'output_directory': str(output_dir),
            'encoding_method': 'comprehensive_statistics',
            'feature_dimension': feature_dim,
            'total_statistics': stats,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }, f, indent=2)
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Create 94-dimensional statistical features from crowd behavior data")
    parser.add_argument("input_dir", help="Input directory containing crowd behavior features")
    parser.add_argument("output_dir", help="Output directory for statistical features")
    
    args = parser.parse_args()
    
    print("="*60)
    print("CROWD BEHAVIOR STATISTICAL ENCODING")
    print("="*60)
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print("Feature vector: 94-dimensional")
    print("- 22 statistics per feature × 4 features = 88 dimensions")
    print("- 6 pairwise correlations = 6 dimensions")
    print()
    
    start_time = time.time()
    stats = process_all_crowd_data(args.input_dir, args.output_dir)
    total_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("STATISTICAL ENCODING COMPLETE")
    print("="*60)
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Videos processed: {stats['total_videos']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Success rate: {100 * stats['successful'] / stats['total_videos']:.1f}%")
    print(f"Feature dimension: {stats['feature_dimension']}")
    print(f"\nStatistical features saved to: {args.output_dir}")


if __name__ == "__main__":
    main()