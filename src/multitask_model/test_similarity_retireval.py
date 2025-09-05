#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np
import h5py
import json
import argparse
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


class SimpleFusionLayer(nn.Module):
    """Same fusion layer as in training"""
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
        visual_proj = torch.relu(self.visual_proj(visual_feat))
        crowd_proj = torch.relu(self.crowd_proj(crowd_feat))
        
        combined = torch.cat([visual_proj, crowd_proj], dim=1)
        weights = self.attention(combined)
        
        visual_weight = weights[:, 0:1]
        crowd_weight = weights[:, 1:2]
        
        fused = visual_weight * visual_proj + crowd_weight * crowd_proj
        output = self.fusion(fused)
        return output, weights


class MultiTaskSoccerModel(nn.Module):
    """Same model as in training"""
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


def load_test_data(visual_dir, crowd_dir, split='test'):
    """Loads test data with full file paths"""
    visual_path = Path(visual_dir) / split
    crowd_path = Path(crowd_dir) / split
    
    categories = ['background', 'before_goal', 'free_kicks_goals', 'penalties', 'shots_no_goals']
    goal_mapping = {
        'background': 0,
        'before_goal': 1, 
        'free_kicks_goals': 1,
        'penalties': 1,
        'shots_no_goals': 0
    }
    
    visual_features = []
    crowd_features = []
    goal_labels = []
    video_names = []
    file_paths = []
    categories_list = []
    
    for category in categories:
        visual_h5 = visual_path / f"{category}_features.h5"
        crowd_h5 = crowd_path / f"{category}_crowd_features.h5"
        
        if visual_h5.exists() and crowd_h5.exists():
            print(f"Loading {category} test data...")
            with h5py.File(visual_h5, 'r') as vf, h5py.File(crowd_h5, 'r') as cf:
                visual_videos = set(vf.keys())
                crowd_videos = set(cf.keys())
                common_videos = visual_videos.intersection(crowd_videos)
                
                print(f"  Found {len(common_videos)} common videos")
                
                for video_name in common_videos:
                    visual_feat = vf[video_name][:]
                    crowd_feat = cf[video_name][:]
                    
                    if visual_feat.shape == (400,) and crowd_feat.shape == (94,):
                        visual_features.append(visual_feat)
                        crowd_features.append(crowd_feat)
                        goal_labels.append(goal_mapping[category])
                        video_names.append(video_name)
                        categories_list.append(category)
                        
                        file_paths.append({
                            'visual': str(visual_h5),
                            'crowd': str(crowd_h5),
                            'video_name': video_name,
                            'category': category
                        })
    
    visual_features = np.array(visual_features)
    crowd_features = np.array(crowd_features)
    goal_labels = np.array(goal_labels)
    
    visual_scaler = StandardScaler()
    crowd_scaler = StandardScaler()
    
    visual_features = visual_scaler.fit_transform(visual_features)
    crowd_features = crowd_scaler.fit_transform(crowd_features)
    
    print(f"Test dataset: {len(visual_features)} samples")
    print(f"Goal distribution: {np.bincount(goal_labels.astype(int))}")
    
    return visual_features, crowd_features, goal_labels, video_names, categories_list, file_paths


def load_trained_model(model_path, device='cuda'):
    """Loads the trained model"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_config = checkpoint['model_config']
    
    model = MultiTaskSoccerModel(
        visual_dim=model_config['visual_dim'],
        crowd_dim=model_config['crowd_dim'],
        fusion_dim=model_config['fusion_dim'],
        num_clusters=model_config['num_clusters']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model with {sum(p.numel() for p in model.parameters())} parameters")
    
    return model


def extract_test_features(model, visual_features, crowd_features, device='cuda'):
    """Extracts features from test data"""
    model.eval()
    all_features = []
    all_attention_weights = []
    all_goal_probs = []
    
    batch_size = 32
    n_samples = len(visual_features)
    
    print("Extracting features from test data...")
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            
            visual_batch = torch.FloatTensor(visual_features[i:end_idx]).to(device)
            crowd_batch = torch.FloatTensor(crowd_features[i:end_idx]).to(device)
            
            cluster_logits, goal_logits, fused, attention_weights = model(visual_batch, crowd_batch)
            
            all_features.append(fused.cpu().numpy())
            all_attention_weights.append(attention_weights.cpu().numpy())
            all_goal_probs.append(torch.sigmoid(goal_logits).cpu().numpy())
    
    features = np.vstack(all_features)
    attention_weights = np.vstack(all_attention_weights)
    goal_probs = np.vstack(all_goal_probs).flatten()
    
    print(f"Extracted features shape: {features.shape}")
    print(f"Average attention weights - Visual: {attention_weights[:, 0].mean():.3f}, Crowd: {attention_weights[:, 1].mean():.3f}")
    
    return features, attention_weights, goal_probs


def build_similarity_index(features, metric='cosine'):
    """Builds nearest neighbor index for similarity search"""
    if metric == 'cosine':
        nn_model = NearestNeighbors(metric='cosine', algorithm='brute')
    else:
        nn_model = NearestNeighbors(metric='euclidean', algorithm='auto')
    
    nn_model.fit(features)
    return nn_model


def find_similar_clips(query_idx, nn_model, features, video_names, categories, file_paths, goal_labels, goal_probs, k=5):
    """Finds k most similar clips to query"""
    query_features = features[query_idx:query_idx+1]
    distances, indices = nn_model.kneighbors(query_features, n_neighbors=k+1)
    
    similar_indices = indices[0][1:]
    similar_distances = distances[0][1:]
    
    query_info = {
        'index': query_idx,
        'video_name': video_names[query_idx],
        'category': categories[query_idx],
        'goal_label': goal_labels[query_idx],
        'goal_prob': goal_probs[query_idx],
        'file_paths': file_paths[query_idx]
    }
    
    similar_clips = []
    for idx, dist in zip(similar_indices, similar_distances):
        similar_clips.append({
            'index': int(idx),
            'video_name': video_names[idx],
            'category': categories[idx],
            'goal_label': int(goal_labels[idx]),
            'goal_prob': float(goal_probs[idx]),
            'distance': float(dist),
            'file_paths': file_paths[idx]
        })
    
    return query_info, similar_clips


def create_test_tsne_visualization(features, categories, goal_labels, output_dir):
    """Creates t-SNE visualization of test features"""
    print("Computing t-SNE embedding...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)//4))
    tsne_embedding = tsne.fit_transform(features)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    unique_categories = list(set(categories))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_categories)))
    
    for i, category in enumerate(unique_categories):
        mask = np.array(categories) == category
        axes[0].scatter(tsne_embedding[mask, 0], tsne_embedding[mask, 1], 
                       c=[colors[i]], alpha=0.7, s=20, label=category)
    
    axes[0].set_xlabel('t-SNE Component 1')
    axes[0].set_ylabel('t-SNE Component 2')
    axes[0].set_title('Test Data t-SNE by Category')
    axes[0].legend()
    
    goal_colors = ['blue', 'red']
    goal_labels_str = ['No Goal', 'Goal']
    
    for i, (color, label) in enumerate(zip(goal_colors, goal_labels_str)):
        mask = goal_labels == i
        axes[1].scatter(tsne_embedding[mask, 0], tsne_embedding[mask, 1], 
                       c=color, alpha=0.7, s=20, label=label)
    
    axes[1].set_xlabel('t-SNE Component 1')
    axes[1].set_ylabel('t-SNE Component 2')
    axes[1].set_title('Test Data t-SNE by Goal Labels')
    axes[1].legend()
    
    plt.tight_layout()
    
    plot_path = Path(output_dir) / 'test_tsne_visualization.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"t-SNE visualization saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Test similarity retrieval with trained model")
    parser.add_argument("model_path", help="Path to trained model")
    parser.add_argument("visual_dir", help="Directory containing visual features")
    parser.add_argument("crowd_dir", help="Directory containing crowd features")
    parser.add_argument("output_dir", help="Output directory for results")
    parser.add_argument("--device", choices=['cuda', 'cpu'], default='cuda', help="Device to use")
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("Loading trained model...")
    model = load_trained_model(args.model_path, device)
    
    print("Loading test data...")
    visual_test, crowd_test, goal_test, video_names, categories, file_paths = load_test_data(
        args.visual_dir, args.crowd_dir, 'test'
    )
    
    if len(video_names) == 0:
        print("ERROR: No test data found!")
        return
    
    print("Extracting test features...")
    features, attention_weights, goal_probs = extract_test_features(
        model, visual_test, crowd_test, device
    )
    
    print("Building similarity indices...")
    cosine_index = build_similarity_index(features, 'cosine')
    euclidean_index = build_similarity_index(features, 'euclidean')
    
    n_test_samples = len(video_names)
    
    test_indices = [
        0,
        n_test_samples // 2,
        min(n_test_samples - 1, 100)
    ]
    
    results = {
        'model_info': {
            'model_path': str(args.model_path),
            'total_test_samples': n_test_samples,
            'average_attention_weights': {
                'visual': float(attention_weights[:, 0].mean()),
                'crowd': float(attention_weights[:, 1].mean())
            }
        },
        'similarity_tests': []
    }
    
    print(f"\nTesting similarity retrieval on {len(test_indices)} example clips...")
    print("="*80)
    
    for i, query_idx in enumerate(test_indices):
        if query_idx >= n_test_samples:
            continue
            
        print(f"\nExample {i+1}: Query clip index {query_idx}")
        print("-" * 50)
        
        for metric, nn_index in [('cosine', cosine_index), ('euclidean', euclidean_index)]:
            print(f"\n{metric.upper()} SIMILARITY:")
            
            query_info, similar_clips = find_similar_clips(
                query_idx, nn_index, features, video_names, categories, 
                file_paths, goal_test, goal_probs, k=5
            )
            
            goal_str = 'Goal' if query_info['goal_label'] == 1 else 'No Goal'
            print(f"Query: {query_info['video_name']}")
            print(f"Category: {query_info['category']}, True: {goal_str}, Pred: {query_info['goal_prob']:.3f}")
            
            print(f"\nTop-5 most similar clips:")
            for j, clip in enumerate(similar_clips, 1):
                goal_str = 'Goal' if clip['goal_label'] == 1 else 'No Goal'
                print(f"{j}. {clip['video_name']}")
                print(f"   Category: {clip['category']}, True: {goal_str}, Pred: {clip['goal_prob']:.3f}")
                print(f"   Distance: {clip['distance']:.4f}")
                print()
            
            results['similarity_tests'].append({
                'query_index': query_idx,
                'metric': metric,
                'query_info': query_info,
                'similar_clips': similar_clips
            })
    
    results_path = Path(args.output_dir) / 'test_similarity_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    print("Creating t-SNE visualization of test data...")
    create_test_tsne_visualization(features, categories, goal_test, args.output_dir)
    
    print("\nTesting completed successfully!")
    print(f"Results directory: {args.output_dir}")


if __name__ == "__main__":
    main()