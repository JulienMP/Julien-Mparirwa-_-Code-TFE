#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import h5py
import json
import time
import argparse
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class SimpleFusionLayer(nn.Module):
    """Simple fusion layer that combines visual and crowd features intelligently"""
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
    """Multi-task model for soccer clip analysis"""
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


class SoccerDataset(torch.utils.data.Dataset):
    """Dataset for soccer clips with visual and crowd features"""
    
    def __init__(self, visual_features, crowd_features, goal_labels):
        self.visual_features = torch.FloatTensor(visual_features)
        self.crowd_features = torch.FloatTensor(crowd_features)
        self.goal_labels = torch.FloatTensor(goal_labels)
    
    def __len__(self):
        return len(self.visual_features)
    
    def __getitem__(self, idx):
        return self.visual_features[idx], self.crowd_features[idx], self.goal_labels[idx]


def load_soccer_data(visual_dir, crowd_dir, split='train'):
    """Loads and combines visual and crowd features from H5 files"""
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
    
    for category in categories:
        visual_h5 = visual_path / f"{category}_features.h5"
        crowd_h5 = crowd_path / f"{category}_crowd_features.h5"
        
        if visual_h5.exists() and crowd_h5.exists():
            print(f"  Loading {category}...")
            with h5py.File(visual_h5, 'r') as vf, h5py.File(crowd_h5, 'r') as cf:
                visual_videos = set(vf.keys())
                crowd_videos = set(cf.keys())
                common_videos = visual_videos.intersection(crowd_videos)
                
                print(f"    Visual: {len(visual_videos)}, Crowd: {len(crowd_videos)}, Common: {len(common_videos)}")
                
                for video_name in common_videos:
                    visual_feat = vf[video_name][:]
                    crowd_feat = cf[video_name][:]
                    
                    if visual_feat.shape == (400,) and crowd_feat.shape == (94,):
                        visual_features.append(visual_feat)
                        crowd_features.append(crowd_feat)
                        goal_labels.append(goal_mapping[category])
        else:
            print(f"  WARNING: Missing files for {category}")
    
    if not visual_features:
        raise ValueError("No valid features found! Check file paths.")
    
    visual_features = np.array(visual_features)
    crowd_features = np.array(crowd_features)
    goal_labels = np.array(goal_labels)
    
    visual_scaler = StandardScaler()
    crowd_scaler = StandardScaler()
    
    visual_features = visual_scaler.fit_transform(visual_features)
    crowd_features = crowd_scaler.fit_transform(crowd_features)
    
    print(f"  Final dataset: {len(visual_features)} samples")
    print(f"  Goal distribution: {np.bincount(goal_labels.astype(int))}")
    
    return visual_features, crowd_features, goal_labels


def train_multitask_model(model, train_loader, val_loader, num_epochs=50, device='cuda'):
    """Trains the multi-task model"""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5)
    
    kmeans = KMeans(n_clusters=model.cluster_head[-1].out_features, random_state=42, n_init=10)
    
    history = {
        'train_loss': [], 'val_loss': [], 'val_goal_acc': [], 'val_silhouette': [],
        'attention_weights': []
    }
    
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        all_features = []
        with torch.no_grad():
            for visual, crowd, _ in train_loader:
                visual, crowd = visual.to(device), crowd.to(device)
                _, _, fused, _ = model(visual, crowd)
                all_features.append(fused.cpu().numpy())
        
        all_features = np.vstack(all_features)
        cluster_labels = kmeans.fit_predict(all_features)
        cluster_idx = 0
        
        epoch_attention_weights = []
        for visual, crowd, goal_labels in train_loader:
            visual, crowd, goal_labels = visual.to(device), crowd.to(device), goal_labels.to(device)
            
            batch_size = visual.size(0)
            batch_cluster_labels = torch.LongTensor(
                cluster_labels[cluster_idx:cluster_idx+batch_size]
            ).to(device)
            cluster_idx += batch_size
            
            optimizer.zero_grad()
            
            cluster_logits, goal_logits, _, attention_weights = model(visual, crowd)
            
            epoch_attention_weights.append(attention_weights.detach().cpu().numpy())
            
            cluster_loss = F.cross_entropy(cluster_logits, batch_cluster_labels)
            goal_loss = F.binary_cross_entropy_with_logits(goal_logits.squeeze(), goal_labels)
            
            total_loss = 0.5 * cluster_loss + 0.5 * goal_loss
            
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
        
        epoch_attention_weights = np.vstack(epoch_attention_weights)
        avg_attention = np.mean(epoch_attention_weights, axis=0)
        history['attention_weights'].append(avg_attention)
        
        model.eval()
        val_loss = 0
        goal_preds = []
        goal_true = []
        val_features = []
        
        with torch.no_grad():
            for visual, crowd, goal_labels in val_loader:
                visual, crowd, goal_labels = visual.to(device), crowd.to(device), goal_labels.to(device)
                
                cluster_logits, goal_logits, fused, _ = model(visual, crowd)
                
                val_cluster_labels = kmeans.predict(fused.cpu().numpy())
                val_cluster_labels = torch.LongTensor(val_cluster_labels).to(device)
                
                cluster_loss = F.cross_entropy(cluster_logits, val_cluster_labels)
                goal_loss = F.binary_cross_entropy_with_logits(goal_logits.squeeze(), goal_labels)
                
                val_loss += (0.5 * cluster_loss + 0.5 * goal_loss).item()
                
                goal_preds.extend(torch.sigmoid(goal_logits).cpu().numpy())
                goal_true.extend(goal_labels.cpu().numpy())
                val_features.append(fused.cpu().numpy())
        
        val_features = np.vstack(val_features)
        goal_preds = np.array(goal_preds)
        goal_true = np.array(goal_true)
        
        goal_acc = accuracy_score(goal_true, goal_preds > 0.5)
        
        val_cluster_labels = kmeans.predict(val_features)
        if len(np.unique(val_cluster_labels)) > 1:
            silhouette = silhouette_score(val_features, val_cluster_labels)
        else:
            silhouette = 0
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_goal_acc'].append(goal_acc)
        history['val_silhouette'].append(silhouette)
        
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            visual_att = avg_attention[0]
            crowd_att = avg_attention[1]
            print(f"Epoch {epoch:3d}: Train: {train_loss/len(train_loader):.4f}, "
                  f"Val: {val_loss/len(val_loader):.4f}, "
                  f"Goal Acc: {goal_acc:.4f}, Sil: {silhouette:.4f}, "
                  f"Att: V={visual_att:.3f}/C={crowd_att:.3f}")
    
    return history, kmeans


def evaluate_clustering_quality(model, data_loader, k_range, device='cuda'):
    """Evaluates clustering quality for different k values"""
    model.eval()
    
    all_features = []
    with torch.no_grad():
        for visual, crowd, _ in data_loader:
            visual, crowd = visual.to(device), crowd.to(device)
            _, _, fused, _ = model(visual, crowd)
            all_features.append(fused.cpu().numpy())
    
    features = np.vstack(all_features)
    
    silhouette_scores = []
    inertias = []
    
    for k in k_range:
        if k <= len(features) and k >= 2:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features)
            
            if len(np.unique(labels)) > 1:
                sil_score = silhouette_score(features, labels)
                silhouette_scores.append(sil_score)
                inertias.append(kmeans.inertia_)
            else:
                silhouette_scores.append(-1)
                inertias.append(float('inf'))
        else:
            silhouette_scores.append(-1)
            inertias.append(float('inf'))
    
    return list(k_range), silhouette_scores, inertias


def plot_results(history, k_range, silhouette_scores, inertias, model, val_loader, device, save_dir):
    """Plots training history, clustering evaluation, and t-SNE visualization"""
    print("Extracting features for t-SNE visualization...")
    model.eval()
    all_features = []
    all_goal_labels = []
    
    valid_scores = [score for score in silhouette_scores if score != -1]
    if valid_scores:
        best_k_idx = np.argmax(valid_scores)
        best_k = k_range[best_k_idx]
    else:
        best_k = 2
    
    with torch.no_grad():
        for visual, crowd, goal_labels in val_loader:
            visual, crowd, goal_labels = visual.to(device), crowd.to(device), goal_labels.to(device)
            _, _, fused, _ = model(visual, crowd)
            all_features.append(fused.cpu().numpy())
            all_goal_labels.extend(goal_labels.cpu().numpy())
    
    features = np.vstack(all_features)
    goal_labels = np.array(all_goal_labels)
    
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features)
    
    print("Computing t-SNE embedding...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)//4))
    tsne_embedding = tsne.fit_transform(features)
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    epochs = range(len(history['train_loss']))
    
    axes[0,0].plot(epochs, history['train_loss'], label='Train Loss')
    axes[0,0].plot(epochs, history['val_loss'], label='Val Loss')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].set_title('Training History')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].plot(epochs, history['val_goal_acc'])
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Goal Prediction Accuracy')
    axes[0,1].set_title('Goal Prediction Performance')
    axes[0,1].grid(True, alpha=0.3)
    
    axes[0,2].plot(epochs, history['val_silhouette'])
    axes[0,2].set_xlabel('Epoch')
    axes[0,2].set_ylabel('Silhouette Score')
    axes[0,2].set_title('Clustering Quality During Training')
    axes[0,2].grid(True, alpha=0.3)
    
    if 'attention_weights' in history and history['attention_weights']:
        attention_weights = np.array(history['attention_weights'])
        axes[0,3].plot(epochs, attention_weights[:, 0], label='Visual Attention')
        axes[0,3].plot(epochs, attention_weights[:, 1], label='Crowd Attention')
        axes[0,3].set_xlabel('Epoch')
        axes[0,3].set_ylabel('Attention Weight')
        axes[0,3].set_title('Attention Weights Over Time')
        axes[0,3].legend()
        axes[0,3].grid(True, alpha=0.3)
    
    valid_k = [k for k, score in zip(k_range, silhouette_scores) if score != -1]
    valid_scores = [score for score in silhouette_scores if score != -1]
    
    if valid_k:
        axes[1,0].plot(valid_k, valid_scores, 'bo-')
        axes[1,0].set_xlabel('Number of Clusters (k)')
        axes[1,0].set_ylabel('Silhouette Score')
        axes[1,0].set_title('Clustering Quality vs k')
        axes[1,0].grid(True, alpha=0.3)
        
        best_idx = np.argmax(valid_scores)
        axes[1,0].scatter(valid_k[best_idx], valid_scores[best_idx], 
                         color='red', s=100, zorder=5, label=f'Best k={valid_k[best_idx]}')
        axes[1,0].legend()
    
    valid_inertias = [inertia for inertia in inertias if inertia != float('inf')]
    if valid_k and valid_inertias:
        axes[1,1].plot(valid_k, valid_inertias, 'ro-')
        axes[1,1].set_xlabel('Number of Clusters (k)')
        axes[1,1].set_ylabel('Inertia')
        axes[1,1].set_title('Elbow Method for Optimal k')
        axes[1,1].grid(True, alpha=0.3)
    
    scatter = axes[1,2].scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], 
                               c=cluster_labels, cmap='tab10', alpha=0.7, s=20)
    axes[1,2].set_xlabel('t-SNE Component 1')
    axes[1,2].set_ylabel('t-SNE Component 2')
    axes[1,2].set_title(f't-SNE Visualization by Cluster (k={best_k})')
    plt.colorbar(scatter, ax=axes[1,2])
    
    colors = ['blue', 'red']
    labels = ['No Goal', 'Goal']
    for i, (color, label) in enumerate(zip(colors, labels)):
        mask = goal_labels == i
        axes[1,3].scatter(tsne_embedding[mask, 0], tsne_embedding[mask, 1], 
                         c=color, alpha=0.7, s=20, label=label)
    axes[1,3].set_xlabel('t-SNE Component 1')
    axes[1,3].set_ylabel('t-SNE Component 2')
    axes[1,3].set_title('t-SNE Visualization by Goal Labels')
    axes[1,3].legend()
    
    plt.tight_layout()
    
    plot_path = Path(save_dir) / 'training_results.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Results plot saved to: {plot_path}")
    plt.close()
    
    return tsne_embedding, cluster_labels


def create_summary_report(results, output_dir):
    """Creates a text summary report"""
    report_path = Path(output_dir) / 'TRAINING_SUMMARY.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SIMPLE SOCCER MULTI-TASK MODEL - TRAINING SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        config = results['config']
        f.write("CONFIGURATION:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Visual feature dimension: {400}\n")
        f.write(f"Crowd feature dimension: {94}\n")
        f.write(f"Fusion dimension: {config['fusion_dim']}\n")
        f.write(f"Number of clusters: {config['num_clusters']}\n")
        f.write(f"Training epochs: {config['epochs']}\n")
        f.write(f"Batch size: {config['batch_size']}\n\n")
        
        data = results['data_summary']
        f.write("DATA SUMMARY:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Training samples: {data['train_samples']}\n")
        f.write(f"Validation samples: {data['val_samples']}\n")
        f.write(f"Train goal distribution: {data['train_goal_dist']}\n")
        f.write(f"Validation goal distribution: {data['val_goal_dist']}\n\n")
        
        metrics = results['final_metrics']
        f.write("FINAL PERFORMANCE:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Goal prediction accuracy: {metrics['goal_accuracy']:.4f}\n")
        f.write(f"Best silhouette score: {metrics['best_silhouette']:.4f}\n")
        if metrics['best_k']:
            f.write(f"Optimal number of clusters: {metrics['best_k']}\n")
            f.write(f"Best clustering score: {metrics['best_k_score']:.4f}\n\n")
        
        clustering = results['clustering_evaluation']
        f.write("CLUSTERING EVALUATION:\n")
        f.write("-" * 30 + "\n")
        f.write(f"{'k':<3} {'Silhouette Score':<15}\n")
        f.write("-" * 25 + "\n")
        for k, score in zip(clustering['k_range'], clustering['silhouette_scores']):
            if score != -1:
                f.write(f"{k:<3} {score:<15.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write(f"Report generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n")
    
    print(f"Summary report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Train multi-task soccer model")
    parser.add_argument("visual_dir", help="Directory containing visual features")
    parser.add_argument("crowd_dir", help="Directory containing crowd features")
    parser.add_argument("output_dir", help="Output directory for results")
    parser.add_argument("--fusion-dim", type=int, default=256, help="Fusion layer dimension")
    parser.add_argument("--num-clusters", type=int, default=5, help="Number of clusters")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", choices=['cuda', 'cpu'], default='cuda', help="Device to use")
    
    args = parser.parse_args()
    
    config = {
        'visual_dir': args.visual_dir,
        'crowd_dir': args.crowd_dir,
        'output_dir': args.output_dir,
        'fusion_dim': args.fusion_dim,
        'num_clusters': args.num_clusters,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'k_range': list(range(2, 11))
    }
    
    Path(config['output_dir']).mkdir(parents=True, exist_ok=True)
    
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    print("\nLoading training data...")
    visual_train, crowd_train, goal_train = load_soccer_data(
        config['visual_dir'], config['crowd_dir'], 'train'
    )
    
    print("\nLoading validation data...")
    visual_val, crowd_val, goal_val = load_soccer_data(
        config['visual_dir'], config['crowd_dir'], 'val'
    )
    
    train_dataset = SoccerDataset(visual_train, crowd_train, goal_train)
    val_dataset = SoccerDataset(visual_val, crowd_val, goal_val)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4
    )
    
    model = MultiTaskSoccerModel(
        visual_dim=400,
        crowd_dim=94,
        fusion_dim=config['fusion_dim'],
        num_clusters=config['num_clusters']
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel initialized with {total_params:,} parameters")
    
    print(f"\nTraining multi-task model for {config['epochs']} epochs...")
    history, trained_kmeans = train_multitask_model(
        model, train_loader, val_loader, 
        num_epochs=config['epochs'], device=device
    )
    
    print("\nEvaluating clustering quality for different k values...")
    k_range, silhouette_scores, inertias = evaluate_clustering_quality(
        model, val_loader, config['k_range'], device
    )
    
    print("Generating plots...")
    tsne_embedding, cluster_labels = plot_results(
        history, k_range, silhouette_scores, inertias, 
        model, val_loader, device, config['output_dir']
    )
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Goal Prediction Accuracy: {history['val_goal_acc'][-1]:.4f}")
    print(f"Best Silhouette Score: {max(history['val_silhouette']):.4f}")
    
    if history['attention_weights']:
        final_attention = history['attention_weights'][-1]
        print(f"Final Attention Weights - Visual: {final_attention[0]:.3f}, Crowd: {final_attention[1]:.3f}")
    
    print("\nClustering Evaluation:")
    best_k = None
    best_score = -1
    for k, score in zip(k_range, silhouette_scores):
        if score != -1:
            print(f"k={k}: Silhouette Score = {score:.4f}")
            if score > best_score:
                best_score = score
                best_k = k
    
    if best_k:
        print(f"\nBest clustering configuration: k={best_k} (score={best_score:.4f})")
    
    model_path = Path(config['output_dir']) / 'simple_soccer_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'visual_dim': 400,
            'crowd_dim': 94,
            'fusion_dim': config['fusion_dim'],
            'num_clusters': config['num_clusters']
        },
        'training_config': config,
        'final_metrics': {
            'goal_accuracy': float(history['val_goal_acc'][-1]),
            'best_silhouette': float(max(history['val_silhouette'])),
            'best_k': int(best_k) if best_k else None,
            'best_k_score': float(best_score) if best_score > -1 else None
        }
    }, model_path)
    print(f"Model saved to: {model_path}")
    
    results = {
        'config': {k: (list(v) if isinstance(v, range) else v) for k, v in config.items()},
        'training_history': {k: v for k, v in history.items() if k != 'attention_weights'},
        'final_metrics': {
            'goal_accuracy': float(history['val_goal_acc'][-1]),
            'best_silhouette': float(max(history['val_silhouette'])),
            'best_k': int(best_k) if best_k else None,
            'best_k_score': float(best_score) if best_score > -1 else None
        },
        'clustering_evaluation': {
            'k_range': list(k_range),
            'silhouette_scores': silhouette_scores,
            'inertias': inertias
        },
        'data_summary': {
            'train_samples': len(visual_train),
            'val_samples': len(visual_val),
            'train_goal_dist': np.bincount(goal_train.astype(int)).tolist(),
            'val_goal_dist': np.bincount(goal_val.astype(int)).tolist()
        }
    }
    
    results_path = Path(config['output_dir']) / 'training_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    create_summary_report(results, config['output_dir'])
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)


if __name__ == "__main__":
    main()