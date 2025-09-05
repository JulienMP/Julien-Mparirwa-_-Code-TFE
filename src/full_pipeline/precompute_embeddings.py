#!/usr/bin/env python3

import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import argparse
from pathlib import Path


def log_message(message):
    """Prints timestamped log message"""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}")


class SimpleFusionLayer(nn.Module):
    """Fusion layer that combines visual and crowd features intelligently"""
    def __init__(self, visual_dim=400, crowd_dim=94, fusion_dim=256):
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, fusion_dim)
        self.crowd_proj = nn.Linear(crowd_dim, fusion_dim)
        self.attention = nn.Sequential(
            nn.Linear(fusion_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(fusion_dim)
        )
    
    def forward(self, visual_feat, crowd_feat):
        visual_proj = F.relu(self.visual_proj(visual_feat))
        crowd_proj = F.relu(self.crowd_proj(crowd_feat))
        combined = torch.cat([visual_proj, crowd_proj], dim=1)
        weights = self.attention(combined)
        visual_weight = weights[:, 0:1]
        crowd_weight = weights[:, 1:2]
        fused = visual_weight * visual_proj + crowd_weight * crowd_proj
        output = self.fusion(fused)
        return output, weights


class MultiTaskSoccerModel(nn.Module):
    """Multi-task model"""
    def __init__(self, visual_dim=400, crowd_dim=94, fusion_dim=256, num_clusters=5):
        super().__init__()
        self.fusion = SimpleFusionLayer(visual_dim, crowd_dim, fusion_dim)
        self.cluster_head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_clusters)
        )
        self.goal_head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
    
    def forward(self, visual_feat, crowd_feat):
        fused, attention_weights = self.fusion(visual_feat, crowd_feat)
        cluster_logits = self.cluster_head(fused)
        goal_logits = self.goal_head(fused)
        return cluster_logits, goal_logits, fused, attention_weights


def load_trained_model(model_path, device):
    """Loads the trained soccer model"""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Trained model not found at: {model_path}")
    
    log_message(f"Loading trained model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model_config = checkpoint['model_config']
    
    model = MultiTaskSoccerModel(
        visual_dim=model_config['visual_dim'],
        crowd_dim=model_config['crowd_dim'],
        fusion_dim=model_config['fusion_dim'],
        num_clusters=model_config['num_clusters']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(device)
    
    log_message("Trained model loaded successfully")
    return model


def process_split(split_name, visual_dir, crowd_dir, model, device, output_dir):
    """Processes a single split to compute the learned embeddings"""
    log_message(f"Processing {split_name} split...")
    
    visual_path = Path(visual_dir) / split_name
    crowd_path = Path(crowd_dir) / split_name
    output_path = Path(output_dir) / split_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    categories = ['background', 'before_goal', 'free_kicks_goals', 'penalties', 'shots_no_goals']
    split_stats = {
        'total_videos': 0,
        'successful': 0,
        'categories': {}
    }
    
    for category in categories:
        log_message(f"  Processing {category}...")
        
        visual_h5 = visual_path / f"{category}_features.h5"
        crowd_h5 = crowd_path / f"{category}_crowd_features.h5"
        
        if not visual_h5.exists() or not crowd_h5.exists():
            log_message(f"    Skipping {category} - missing files")
            continue
        
        output_h5 = output_path / f"{category}_learned_features.h5"
        
        category_stats = {
            'total_videos': 0,
            'successful': 0,
            'failed': 0
        }
        
        with h5py.File(visual_h5, 'r') as vf, h5py.File(crowd_h5, 'r') as cf, h5py.File(output_h5, 'w') as of:
            visual_videos = set(vf.keys())
            crowd_videos = set(cf.keys())
            common_videos = visual_videos.intersection(crowd_videos)
            
            log_message(f"    Found {len(common_videos)} common videos")
            
            for video_name in common_videos:
                try:
                    visual_feat = vf[video_name][:]
                    crowd_feat = cf[video_name][:]
                    
                    if visual_feat.shape != (400,) or crowd_feat.shape != (94,):
                        log_message(f"    Skipping {video_name} - invalid dimensions")
                        category_stats['failed'] += 1
                        continue
                    
                    visual_tensor = torch.FloatTensor(visual_feat).unsqueeze(0).to(device)
                    crowd_tensor = torch.FloatTensor(crowd_feat).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        _, _, fused_features, _ = model(visual_tensor, crowd_tensor)
                        learned_features = fused_features.cpu().numpy()[0]
                    
                    of.create_dataset(video_name, data=learned_features)
                    category_stats['successful'] += 1
                    
                except Exception as e:
                    log_message(f"    Error processing {video_name}: {e}")
                    category_stats['failed'] += 1
                
                category_stats['total_videos'] += 1
        
        split_stats['categories'][category] = category_stats
        split_stats['total_videos'] += category_stats['total_videos']
        split_stats['successful'] += category_stats['successful']
        
        log_message(f"    {category}: {category_stats['successful']}/{category_stats['total_videos']} successful")
    
    return split_stats


def create_metadata(output_dir, train_stats, val_stats, config):
    """Creates the metadata file with processing statistics"""
    metadata = {
        'preprocessing_info': {
            'description': 'Learned 256D embeddings from trained multi-task soccer model',
            'model_path': config['model_path'],
            'feature_dimension': 256,
            'splits_processed': ['train', 'val'],
            'exclusions': 'test split excluded to prevent data leakage'
        },
        'processing_statistics': {
            'train': train_stats,
            'val': val_stats,
            'total_videos': train_stats['total_videos'] + val_stats['total_videos'],
            'total_successful': train_stats['successful'] + val_stats['successful']
        },
        'configuration': config,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    metadata_path = Path(output_dir) / 'learned_embeddings_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    log_message(f"Metadata saved to: {metadata_path}")
    return metadata_path


def main():
    parser = argparse.ArgumentParser(description="Precompute learned embeddings from trained model")
    parser.add_argument("model_path", help="Path to trained model")
    parser.add_argument("visual_dir", help="Directory containing visual features")
    parser.add_argument("crowd_dir", help="Directory containing crowd features")
    parser.add_argument("output_dir", help="Output directory for learned embeddings")
    parser.add_argument("--device", choices=['cuda', 'cpu'], default='cuda', help="Device to use")
    
    args = parser.parse_args()
    
    config = {
        'model_path': args.model_path,
        'visual_dir': args.visual_dir,
        'crowd_dir': args.crowd_dir,
        'output_dir': args.output_dir,
        'device': args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    }
    
    log_message("Starting preprocessing of learned embeddings...")
    log_message(f"Device: {config['device']}")
    log_message(f"Output directory: {config['output_dir']}")
    
    Path(config['output_dir']).mkdir(parents=True, exist_ok=True)
    
    device = torch.device(config['device'])
    model = load_trained_model(config['model_path'], device)
    
    start_time = time.time()
    
    train_stats = process_split('train', config['visual_dir'], config['crowd_dir'], 
                               model, device, config['output_dir'])
    
    val_stats = process_split('val', config['visual_dir'], config['crowd_dir'], 
                             model, device, config['output_dir'])
    
    create_metadata(config['output_dir'], train_stats, val_stats, config)
    
    total_time = time.time() - start_time
    total_videos = train_stats['total_videos'] + val_stats['total_videos']
    total_successful = train_stats['successful'] + val_stats['successful']
    
    log_message(f"Preprocessing completed in {total_time:.2f} seconds")
    log_message(f"Total videos processed: {total_videos}")
    log_message(f"Total successful: {total_successful}")
    log_message(f"Success rate: {100 * total_successful / total_videos:.1f}%")
    log_message(f"Learned embeddings saved to: {config['output_dir']}")


if __name__ == "__main__":
    main()