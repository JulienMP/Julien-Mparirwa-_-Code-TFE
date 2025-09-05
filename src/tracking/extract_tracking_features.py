#!/usr/bin/env python3

import os
import time
import json
import torch
import numpy as np
import h5py
from pathlib import Path
from glob import glob
from collections import defaultdict
import argparse

from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F


class STGCNFeatureExtractor(nn.Module):
    def __init__(self, in_channels, num_classes=512):
        super(STGCNFeatureExtractor, self).__init__()
        self.gcn1 = GCNConv(in_channels, 256)
        self.gcn2 = GCNConv(256, 512)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x, edge_index):
        num_nodes, num_frames, _ = x.shape
        x_reshaped = x.view(num_nodes * num_frames, -1)
        
        edge_index_expanded = edge_index.repeat(1, num_frames)
        offsets = torch.arange(num_frames).repeat_interleave(edge_index.shape[1]) * num_nodes
        edge_index_expanded[0, :] += offsets
        edge_index_expanded[1, :] += offsets

        x = self.gcn1(x_reshaped, edge_index_expanded)
        x = F.relu(x)
        x = self.gcn2(x, edge_index_expanded)
        x = F.relu(x)

        x = x.view(num_nodes, num_frames, -1).mean(dim=0).mean(dim=0)
        x = self.fc(x)

        return x


def create_graph_data(tracking_data, num_players=22):
    """Creates a temporal graph from tracking data with fully connected nodes"""
    frames = sorted(tracking_data.keys())
    
    nodes = list(range(num_players + 1))
    edge_index = []
    for i in nodes:
        for j in nodes:
            if i != j:
                edge_index.append([i, j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    all_trajectories = []
    
    for frame_idx in frames:
        frame_data = tracking_data[frame_idx]
        frame_trajectories = []
        for i in range(num_players):
            if i in frame_data:
                x, y, w, h = frame_data[i]
                center_x = x + w / 2
                center_y = y + h / 2
                frame_trajectories.append([center_x, center_y])
            else:
                frame_trajectories.append([0, 0])
        all_trajectories.append(frame_trajectories)
    
    trajectories_tensor = torch.tensor(all_trajectories, dtype=torch.float32).permute(1, 0, 2)
    
    return trajectories_tensor, edge_index


def process_tracking_data(dataset_dir, output_dir, device='cuda'):
    """Processes all tracking files in the dataset directory structure"""
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)

    print("Loading ST-GCN model...")
    feature_extractor = STGCNFeatureExtractor(in_channels=2)
    feature_extractor.eval()

    if torch.cuda.is_available() and device == 'cuda':
        feature_extractor = feature_extractor.cuda()
        print("Using GPU")
    else:
        print("Using CPU")

    splits = ['train', 'val', 'test']
    categories = ['background', 'before_goal', 'free_kicks_goals', 'penalties', 'shots_no_goals']
    
    for split in splits:
        for category in categories:
            category_dir = dataset_path / split / category
            if not category_dir.exists():
                print(f"Skipping: {category_dir} not found.")
                continue

            split_output_dir = output_path / split
            split_output_dir.mkdir(parents=True, exist_ok=True)
            h5_file_path = split_output_dir / f"{category}_tracking_features.h5"
            
            video_dirs = glob(str(category_dir / '*'))
            
            with h5py.File(h5_file_path, 'w') as hf:
                for video_dir in video_dirs:
                    video_name = Path(video_dir).name
                    tracking_file_path = Path(video_dir) / f"{video_name}_tracking.txt"
                    
                    if not tracking_file_path.exists():
                        print(f"  Tracking file not found: {tracking_file_path}. Skipping.")
                        continue
                    
                    print(f"  Processing: {video_name}")
                    
                    tracking_data = defaultdict(dict)
                    with open(tracking_file_path, 'r') as f:
                        for line in f:
                            frame, id, x1, y1, w, h, score, _, _, _ = map(float, line.strip().split(','))
                            tracking_data[int(frame)][int(id)] = [x1, y1, w, h]
                    
                    trajectories_tensor, edge_index = create_graph_data(tracking_data)
                    
                    if device == 'cuda' and torch.cuda.is_available():
                        trajectories_tensor = trajectories_tensor.cuda()
                        edge_index = edge_index.cuda()
                    
                    with torch.no_grad():
                        features = feature_extractor(trajectories_tensor, edge_index)
                        if device == 'cuda':
                            features = features.cpu()

                    hf.create_dataset(video_name, data=features.numpy())
                    
            print(f"Completed {category}. Features saved to {h5_file_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract tracking features using ST-GCN")
    parser.add_argument("input_dir", help="Path to tracking data directory")
    parser.add_argument("output_dir", help="Output directory for features")
    parser.add_argument("--device", choices=['cuda', 'cpu'], default='cuda',
                       help="Device to use for processing")
    
    args = parser.parse_args()

    print("="*60)
    print("ST-GCN Tracking Feature Extraction")
    print("="*60)
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {args.device}")

    if not Path(args.input_dir).exists():
        print(f"ERROR: Input directory not found: {args.input_dir}")
        return
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    process_tracking_data(args.input_dir, args.output_dir, args.device)
    total_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Features saved to: {args.output_dir}")


if __name__ == "__main__":
    main()