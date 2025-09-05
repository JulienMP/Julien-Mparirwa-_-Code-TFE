#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import ConvexHull
import argparse
import json
from collections import defaultdict
import time


def parse_tracking_line(line):
    """Parses a line from tracking file"""
    parts = line.strip().split(',')
    if len(parts) < 7:
        return None
    
    try:
        frame_id = int(parts[0])
        track_id = int(parts[1])
        x1 = float(parts[2])
        y1 = float(parts[3])
        w = float(parts[4])
        h = float(parts[5])
        score = float(parts[6])
        
        centroid_x = x1 + w / 2.0
        centroid_y = y1 + h / 2.0
        
        return {
            'frame_id': frame_id,
            'track_id': track_id,
            'x1': x1, 'y1': y1, 'w': w, 'h': h,
            'centroid_x': centroid_x,
            'centroid_y': centroid_y,
            'score': score
        }
    except (ValueError, IndexError):
        return None


def calculate_convex_hull_area(points):
    """Calculates convex hull area from a set of 2D points"""
    if len(points) < 3:
        return 0.0
    
    try:
        points = np.array(points)
        unique_points = np.unique(points, axis=0)
        
        if len(unique_points) < 3:
            return 0.0
            
        hull = ConvexHull(unique_points)
        return hull.volume
    except Exception:
        return 0.0


def process_tracking_file(tracking_file_path):
    """Processes a single tracking file and extracts crowd behavior features"""
    if not os.path.exists(tracking_file_path):
        print(f"Warning: Tracking file not found: {tracking_file_path}")
        return None
    
    frame_data = defaultdict(list)
    
    try:
        with open(tracking_file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    parsed = parse_tracking_line(line)
                    if parsed:
                        frame_data[parsed['frame_id']].append(parsed)
                    else:
                        print(f"Warning: Could not parse line {line_num} in {tracking_file_path}")
    except Exception as e:
        print(f"Error reading {tracking_file_path}: {e}")
        return None
    
    if not frame_data:
        print(f"Warning: No valid tracking data found in {tracking_file_path}")
        return None
    
    crowd_features = []
    
    for frame_id in sorted(frame_data.keys()):
        detections = frame_data[frame_id]
        
        if not detections:
            continue
            
        centroids = [(det['centroid_x'], det['centroid_y']) for det in detections]
        centroid_array = np.array(centroids)
        
        density = len(detections)
        
        if density > 0:
            avg_centroid_x = np.mean(centroid_array[:, 0])
            avg_centroid_y = np.mean(centroid_array[:, 1])
        else:
            avg_centroid_x = 0.0
            avg_centroid_y = 0.0
        
        convex_hull_area = calculate_convex_hull_area(centroids)
        
        crowd_features.append({
            'frame_id': frame_id,
            'density': density,
            'centroid_x': avg_centroid_x,
            'centroid_y': avg_centroid_y,
            'convex_hull_area': convex_hull_area
        })
    
    return crowd_features


def process_video_directory(video_dir_path, output_dir):
    """Processes a single video directory and saves crowd behavior features"""
    video_name = os.path.basename(video_dir_path)
    tracking_file = os.path.join(video_dir_path, f"{video_name}_tracking.txt")
    
    print(f"Processing: {video_name}")
    
    crowd_features = process_tracking_file(tracking_file)
    
    if crowd_features is None or len(crowd_features) == 0:
        print(f"  No crowd features extracted for {video_name}")
        return False
    
    relative_path = os.path.relpath(video_dir_path, 
                                   os.path.join(os.path.dirname(video_dir_path), '..', '..'))
    
    output_video_dir = os.path.join(output_dir, relative_path)
    os.makedirs(output_video_dir, exist_ok=True)
    
    output_file = os.path.join(output_video_dir, f"{video_name}_crowd_behavior.txt")
    
    try:
        with open(output_file, 'w') as f:
            f.write("frame_id,density,centroid_x,centroid_y,convex_hull_area\n")
            
            for feature in crowd_features:
                f.write(f"{feature['frame_id']},{feature['density']},"
                       f"{feature['centroid_x']:.2f},{feature['centroid_y']:.2f},"
                       f"{feature['convex_hull_area']:.2f}\n")
        
        print(f"  Saved: {output_file}")
        print(f"  Frames processed: {len(crowd_features)}")
        return True
        
    except Exception as e:
        print(f"  Error saving {output_file}: {e}")
        return False


def process_all_tracking_data(input_dir, output_dir):
    """Processes all tracking data in the directory structure"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    stats = {
        'total_videos': 0,
        'successful': 0,
        'failed': 0,
        'processing_times': [],
        'splits': {}
    }
    
    for split_dir in input_path.iterdir():
        if split_dir.is_dir() and split_dir.name in ['train', 'val', 'test']:
            split_name = split_dir.name
            print(f"\nProcessing {split_name} split...")
            
            split_stats = {
                'categories': {},
                'total': 0,
                'successful': 0
            }
            
            for category_dir in split_dir.iterdir():
                if category_dir.is_dir():
                    category_name = category_dir.name
                    print(f"  Processing category: {category_name}")
                    
                    category_stats = {
                        'total': 0,
                        'successful': 0,
                        'failed': 0
                    }
                    
                    for video_dir in category_dir.iterdir():
                        if video_dir.is_dir():
                            start_time = time.time()
                            
                            success = process_video_directory(str(video_dir), str(output_path))
                            
                            elapsed = time.time() - start_time
                            stats['processing_times'].append(elapsed)
                            
                            category_stats['total'] += 1
                            stats['total_videos'] += 1
                            
                            if success:
                                category_stats['successful'] += 1
                                stats['successful'] += 1
                            else:
                                category_stats['failed'] += 1
                                stats['failed'] += 1
                    
                    split_stats['categories'][category_name] = category_stats
                    split_stats['total'] += category_stats['total']
                    split_stats['successful'] += category_stats['successful']
                    
                    print(f"    {category_name}: {category_stats['successful']}/{category_stats['total']} successful")
            
            stats['splits'][split_name] = split_stats
            print(f"  {split_name} total: {split_stats['successful']}/{split_stats['total']} successful")
    
    stats_file = output_path / "crowd_behavior_processing_stats.json"
    with open(stats_file, 'w') as f:
        json.dump({
            'input_directory': str(input_dir),
            'output_directory': str(output_dir),
            'total_videos': stats['total_videos'],
            'successful': stats['successful'],
            'failed': stats['failed'],
            'success_rate': stats['successful'] / stats['total_videos'] if stats['total_videos'] > 0 else 0,
            'avg_processing_time': np.mean(stats['processing_times']) if stats['processing_times'] else 0,
            'splits': stats['splits'],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }, f, indent=2)
    
    print("\n" + "="*60)
    print("CROWD BEHAVIOR ANALYSIS COMPLETE")
    print("="*60)
    print(f"Total videos processed: {stats['total_videos']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Success rate: {100 * stats['successful'] / stats['total_videos']:.1f}%")
    
    if stats['processing_times']:
        print(f"Average processing time: {np.mean(stats['processing_times']):.2f}s per video")
    
    print(f"\nOutput saved to: {output_dir}")
    print(f"Statistics saved to: {stats_file}")


def main():
    parser = argparse.ArgumentParser(description="Extract crowd behavior features from tracking data")
    parser.add_argument("input_dir", help="Input directory containing tracking results")
    parser.add_argument("output_dir", help="Output directory for crowd behavior features")
    
    args = parser.parse_args()
    
    print("="*60)
    print("CROWD BEHAVIOR FEATURE EXTRACTION")
    print("="*60)
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print()
    
    process_all_tracking_data(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()