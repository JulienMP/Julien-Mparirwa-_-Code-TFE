#!/usr/bin/env python3
"""
Runtime prediction script for soccer video analysis.
Processes a single query video and finds similar clips using precomputed embeddings.
Uses only train/val splits for similarity search to prevent data leakage.
"""

import os
import sys
import argparse
import subprocess
import json
import time
import random
import shutil
import shlex
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.spatial import ConvexHull
import threading
import queue
from dotenv import load_dotenv

load_dotenv()


def log_message(message):
    """Prints timestamped log message"""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}")


def run_command(cmd, env_name=None, cwd=None):
    """Runs a command with optional conda environment activation"""
    anaconda_path = os.getenv('ANACONDA_PATH')
    if env_name:
        conda_cmd = f"source {anaconda_path}/etc/profile.d/conda.sh && conda activate {env_name} && {cmd}"
        full_cmd = ["bash", "-c", conda_cmd]
    else:
        full_cmd = cmd.split() if isinstance(cmd, str) else cmd
    
    log_message(f"Running: {' '.join(full_cmd) if isinstance(full_cmd, list) else full_cmd}")
    
    try:
        result = subprocess.run(
            full_cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            log_message(f"Command failed with return code {result.returncode}")
            log_message(f"STDERR: {result.stderr}")
            return False, result.stderr
        
        return True, result.stdout
    except Exception as e:
        log_message(f"Command execution error: {e}")
        return False, str(e)


def select_random_test_clip():
    """Selects a random clip from the test set"""
    clips_dir = os.getenv('CLIPS_DATA_DIR')
    test_dir = Path(clips_dir) / "test"
    
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    
    categories = ['background', 'before_goal', 'free_kicks_goals', 'penalties', 'shots_no_goals']
    available_categories = [cat for cat in categories if (test_dir / cat).exists()]
    
    if not available_categories:
        raise FileNotFoundError("No categories found in test directory")
    
    selected_category = random.choice(available_categories)
    category_dir = test_dir / selected_category
    video_files = list(category_dir.glob("*.mkv"))
    
    if not video_files:
        raise FileNotFoundError(f"No videos found in category: {selected_category}")
    
    selected_video = random.choice(video_files)
    
    log_message(f"Selected random test clip: {selected_video}")
    log_message(f"Category: {selected_category}")
    
    return selected_video, selected_category


def process_tracking(video_path, temp_dir):
    """Runs ByteTrack tracking on the video"""
    log_message("Starting tracking processing...")
    
    bytetrack_home = os.getenv('BYTETRACK_HOME')
    tracking_env = os.getenv('TRACKING_ENV_NAME')
    tracking_output_dir = temp_dir / "tracking_results"
    tracking_output_dir.mkdir(exist_ok=True)
    
    escaped_video_path = shlex.quote(str(video_path))
    escaped_output_dir = shlex.quote(str(tracking_output_dir))
    
    cmd = f"""cd {bytetrack_home} && python tools/demo_track_mkv.py video \
        -f exps/example/mot/yolox_x_soccernet.py \
        -c pretrained/bytetrack_x_mot20.tar \
        --fp16 --fuse --save_result \
        --tsize 1088 \
        --output_dir {escaped_output_dir} \
        --path {escaped_video_path}"""
    
    success, output = run_command(cmd, env_name=tracking_env)
    
    if not success:
        log_message("Tracking failed, trying with different parameters...")
        cmd_alt = f"""cd {bytetrack_home} && python tools/demo_track_mkv.py video \
            -f exps/example/mot/yolox_x_soccernet.py \
            -c pretrained/bytetrack_x_mot20.tar \
            --save_result \
            --tsize 800 \
            --output_dir {escaped_output_dir} \
            --path {escaped_video_path}"""
        
        success, output = run_command(cmd_alt, env_name=tracking_env)
        
        if not success:
            raise RuntimeError(f"Tracking failed: {output}")
    
    video_name = Path(video_path).stem
    tracking_files = list(tracking_output_dir.rglob(f"*{video_name}*_tracking.txt"))
    
    if not tracking_files:
        raise FileNotFoundError("No tracking file generated")
    
    tracking_file = tracking_files[0]
    log_message(f"Tracking completed: {tracking_file}")
    
    return tracking_file


def extract_crowd_features(tracking_file, temp_dir):
    """Extracts crowd behavior features from tracking data"""
    log_message("Extracting crowd behavior features...")
    
    crowd_features = process_tracking_file_inline(tracking_file)
    
    if not crowd_features:
        raise ValueError("No crowd features could be extracted")
    
    crowd_file = temp_dir / f"{Path(tracking_file).stem.replace('_tracking', '')}_crowd_behavior.txt"
    
    with open(crowd_file, 'w') as f:
        f.write("frame_id,density,centroid_x,centroid_y,convex_hull_area\n")
        for feature in crowd_features:
            f.write(f"{feature['frame_id']},{feature['density']},"
                   f"{feature['centroid_x']:.2f},{feature['centroid_y']:.2f},"
                   f"{feature['convex_hull_area']:.2f}\n")
    
    log_message(f"Crowd behavior file created: {crowd_file}")
    return crowd_file


def process_tracking_file_inline(tracking_file_path):
    """Processes tracking file for crowd behavior analysis"""
    frame_data = defaultdict(list)
    
    try:
        with open(tracking_file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    parts = line.strip().split(',')
                    if len(parts) >= 7:
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
                            
                            frame_data[frame_id].append({
                                'track_id': track_id,
                                'centroid_x': centroid_x,
                                'centroid_y': centroid_y,
                                'score': score
                            })
                        except (ValueError, IndexError):
                            continue
    except Exception as e:
        log_message(f"Error reading tracking file: {e}")
        return None
    
    if not frame_data:
        return None
    
    crowd_features = []
    
    for frame_id in sorted(frame_data.keys()):
        detections = frame_data[frame_id]
        
        if not detections:
            continue
        
        centroids = [(det['centroid_x'], det['centroid_y']) for det in detections]
        centroid_array = np.array(centroids)
        
        density = len(detections)
        avg_centroid_x = np.mean(centroid_array[:, 0]) if density > 0 else 0.0
        avg_centroid_y = np.mean(centroid_array[:, 1]) if density > 0 else 0.0
        
        if len(centroids) >= 3:
            try:
                unique_points = np.unique(centroid_array, axis=0)
                if len(unique_points) >= 3:
                    hull = ConvexHull(unique_points)
                    convex_hull_area = hull.volume
                else:
                    convex_hull_area = 0.0
            except:
                convex_hull_area = 0.0
        else:
            convex_hull_area = 0.0
        
        crowd_features.append({
            'frame_id': frame_id,
            'density': density,
            'centroid_x': avg_centroid_x,
            'centroid_y': avg_centroid_y,
            'convex_hull_area': convex_hull_area
        })
    
    return crowd_features


def encode_crowd_features(crowd_file, temp_dir):
    """Encodes crowd features into statistical vectors"""
    log_message("Encoding crowd features...")
    
    try:
        data = pd.read_csv(crowd_file)
    except Exception as e:
        raise ValueError(f"Failed to load crowd behavior file: {e}")
    
    features = encode_crowd_behavior_inline(data)
    
    encoded_file = temp_dir / "crowd_features.h5"
    video_name = Path(crowd_file).stem.replace('_crowd_behavior', '')
    
    with h5py.File(encoded_file, 'w') as hf:
        hf.create_dataset(video_name, data=features)
    
    log_message(f"Crowd features encoded: {encoded_file}")
    return encoded_file


def encode_crowd_behavior_inline(data):
    """Encodes crowd behavior into statistical features"""
    if data is None or len(data) == 0:
        return np.zeros(94)
    
    feature_stats = []
    feature_cols = ['density', 'centroid_x', 'centroid_y', 'convex_hull_area']
    
    for col in feature_cols:
        values = data[col].values
        stats_vector = calculate_feature_statistics_inline(values)
        feature_stats.extend(stats_vector)
    
    correlations = calculate_correlations_inline(data)
    all_features = np.concatenate([feature_stats, correlations])
    
    return all_features


def calculate_feature_statistics_inline(values):
    """Calculates 22 statistics for a feature"""
    if len(values) == 0:
        return np.zeros(22)
    
    stats_list = []
    
    stats_list.append(np.mean(values))
    stats_list.append(np.median(values))
    stats_list.append(np.std(values))
    stats_list.append(np.var(values))
    stats_list.append(np.max(values) - np.min(values))
    stats_list.append(np.percentile(values, 75) - np.percentile(values, 25))
    stats_list.append(np.percentile(values, 25))
    stats_list.append(np.percentile(values, 75))
    stats_list.append(values[0] if len(values) > 0 else 0)
    stats_list.append(values[-1] if len(values) > 0 else 0)
    
    for _ in range(12):
        stats_list.append(0.0)
    
    return np.array(stats_list)


def calculate_correlations_inline(data):
    """Calculates correlations between features"""
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


def extract_visual_features(video_path, temp_dir):
    """Extracts visual features using SlowFast with flexible pipeline script"""
    log_message("Extracting visual features...")
    
    visual_script = os.getenv('PIPELINE_VISUAL_SCRIPT')
    visual_env = os.getenv('VISUAL_ENV_NAME')
    device = os.getenv('DEVICE')
    output_dir = temp_dir / "visual_features"
    output_dir.mkdir(exist_ok=True)
    
    temp_video_dir = temp_dir / "temp_dataset"
    temp_video_dir.mkdir(parents=True, exist_ok=True)
    temp_video_path = temp_video_dir / Path(video_path).name
    shutil.copy2(video_path, temp_video_path)
    
    log_message(f"Video copied to: {temp_video_path}")
    log_message(f"Using flexible visual feature extraction script")
    
    escaped_visual_script = shlex.quote(str(visual_script))
    escaped_dataset_dir = shlex.quote(str(temp_dir / 'temp_dataset'))
    escaped_output_dir = shlex.quote(str(output_dir))
    
    cmd = f"python {escaped_visual_script} --dataset_dir {escaped_dataset_dir} --output_dir {escaped_output_dir} --device {device}"
    
    success, output = run_command(cmd, env_name=visual_env)
    
    if not success:
        log_message(f"Visual feature extraction command output: {output}")
        raise RuntimeError(f"Visual feature extraction failed: {output}")
    
    h5_files = list(output_dir.rglob("*.h5"))
    if not h5_files:
        log_message(f"No .h5 files found in {output_dir}")
        log_message("Files in output directory:")
        for file in output_dir.rglob("*"):
            log_message(f"  {file}")
        raise FileNotFoundError("No visual features file generated")
    
    features_file = h5_files[0]
    log_message(f"Visual features extracted: {features_file}")
    
    return features_file


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


def load_trained_model():
    """Loads the trained soccer model"""
    model_path = os.getenv('TRAINED_MODEL_PATH')
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Trained model not found at: {model_path}")
    
    log_message(f"Loading trained model from: {model_path}")
    
    model = MultiTaskSoccerModel()
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    log_message("Trained model loaded successfully")
    return model


def load_precomputed_embeddings():
    """Loads precomputed learned embeddings for train and val splits only"""
    log_message("Loading precomputed embeddings from train and val splits...")
    
    embeddings_dir = os.getenv('LEARNED_EMBEDDINGS_DIR')
    embeddings_path = Path(embeddings_dir)
    
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Precomputed embeddings not found at: {embeddings_path}")
    
    database = {}
    
    for split in ['train', 'val']:
        for category in ['background', 'before_goal', 'free_kicks_goals', 'penalties', 'shots_no_goals']:
            
            embedding_file = embeddings_path / split / f"{category}_learned_features.h5"
            
            if embedding_file.exists():
                try:
                    with h5py.File(embedding_file, 'r') as f:
                        for video_name in f.keys():
                            database[video_name] = {
                                'learned_features': f[video_name][:],
                                'split': split,
                                'category': category
                            }
                except Exception as e:
                    log_message(f"Error loading {embedding_file}: {e}")
    
    log_message(f"Loaded precomputed embeddings for {len(database)} videos")
    return database


def find_similar_clips(query_crowd_features, query_visual_features, database, top_k=10):
    """Finds similar clips using precomputed learned embeddings"""
    log_message("Finding similar clips using precomputed embeddings...")
    
    model = load_trained_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    query_visual_tensor = torch.FloatTensor(query_visual_features).unsqueeze(0).to(device)
    query_crowd_tensor = torch.FloatTensor(query_crowd_features).unsqueeze(0).to(device)
    
    with torch.no_grad():
        _, query_goal_logits, query_fused_features, query_attention = model(query_visual_tensor, query_crowd_tensor)
        query_goal_prob = torch.sigmoid(query_goal_logits).cpu().numpy()[0, 0]
        query_fused = query_fused_features.cpu().numpy()[0]
        query_attention_weights = query_attention.cpu().numpy()[0]
    
    log_message(f"Query goal probability: {query_goal_prob:.3f}")
    log_message(f"Query attention weights - Visual: {query_attention_weights[0]:.3f}, Crowd: {query_attention_weights[1]:.3f}")
    
    similarities = []
    
    for video_name, video_data in database.items():
        try:
            db_fused = video_data['learned_features']
            
            fused_sim = 1 - cosine(query_fused, db_fused)
            
            similarities.append({
                'video_name': video_name,
                'similarity': fused_sim,
                'split': video_data['split'],
                'category': video_data['category']
            })
            
        except Exception as e:
            log_message(f"Error calculating similarity for {video_name}: {e}")
            continue
    
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    
    return similarities[:top_k], query_goal_prob, query_attention_weights


def download_similar_clips(similar_clips, output_dir):
    """Downloads similar clips to output directory"""
    log_message("Downloading similar clips...")
    
    clips_dir = output_dir / "similar_clips"
    clips_dir.mkdir(exist_ok=True)
    
    source_base_dir = Path(os.getenv('CLIPS_DATA_DIR'))
    downloaded_clips = []
    
    for i, clip_info in enumerate(similar_clips):
        video_name = clip_info['video_name']
        split = clip_info['split']
        category = clip_info['category']
        
        source_path = source_base_dir / split / category / f"{video_name}.mkv"
        
        if source_path.exists():
            dest_path = clips_dir / f"rank_{i+1:02d}_{video_name}.mkv"
            try:
                shutil.copy2(source_path, dest_path)
                downloaded_clips.append({
                    'rank': i + 1,
                    'original_name': video_name,
                    'downloaded_name': dest_path.name,
                    'similarity': clip_info['similarity'],
                    'category': category,
                    'split': split
                })
                log_message(f"Downloaded: {dest_path.name}")
            except Exception as e:
                log_message(f"Failed to copy {video_name}: {e}")
        else:
            log_message(f"Source file not found: {source_path}")
    
    return downloaded_clips


def generate_prediction_report(query_video, query_category, similar_clips, downloaded_clips, 
                             output_dir, query_goal_prob, query_attention):
    """Generates prediction report with model predictions"""
    log_message("Generating prediction report...")
    
    category_counts = defaultdict(int)
    goal_related_categories = ['before_goal', 'free_kicks_goals', 'penalties']
    
    total_similarity = 0
    
    for clip in similar_clips:
        category_counts[clip['category']] += 1
        total_similarity += clip['similarity']
    
    similar_goal_probability = sum(category_counts[cat] for cat in goal_related_categories) / len(similar_clips)
    avg_similarity = total_similarity / len(similar_clips)
    
    combined_goal_prob = 0.7 * query_goal_prob + 0.3 * similar_goal_probability
    
    report = {
        'query_video': {
            'name': Path(query_video).name,
            'true_category': query_category,
            'path': str(query_video)
        },
        'model_prediction': {
            'goal_probability': float(query_goal_prob),
            'predicted_class': 'goal-related' if query_goal_prob > 0.5 else 'non-goal',
            'attention_weights': {
                'visual': float(query_attention[0]),
                'crowd': float(query_attention[1])
            }
        },
        'similarity_analysis': {
            'goal_probability': similar_goal_probability,
            'predicted_class': 'goal-related' if similar_goal_probability > 0.5 else 'non-goal',
            'average_similarity': avg_similarity
        },
        'combined_prediction': {
            'goal_probability': combined_goal_prob,
            'predicted_class': 'goal-related' if combined_goal_prob > 0.5 else 'non-goal',
            'confidence': avg_similarity,
            'top_similar_category': max(category_counts.items(), key=lambda x: x[1])[0]
        },
        'similar_clips_analysis': {
            'total_clips': len(similar_clips),
            'category_distribution': dict(category_counts),
            'goal_related_clips': sum(category_counts[cat] for cat in goal_related_categories),
            'background_clips': category_counts['background'],
            'shots_no_goals': category_counts['shots_no_goals']
        },
        'downloaded_clips': downloaded_clips,
        'detailed_similarities': similar_clips
    }
    
    report_file = output_dir / "prediction_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    summary_file = output_dir / "prediction_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("SOCCER CLIP PREDICTION SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Query Video: {Path(query_video).name}\n")
        f.write(f"True Category: {query_category}\n\n")
        
        f.write("TRAINED MODEL PREDICTION:\n")
        f.write(f"Goal Probability: {query_goal_prob:.3f}\n")
        f.write(f"Predicted Class: {report['model_prediction']['predicted_class']}\n")
        f.write(f"Attention Weights - Visual: {query_attention[0]:.3f}, Crowd: {query_attention[1]:.3f}\n\n")
        
        f.write("SIMILARITY-BASED ANALYSIS:\n")
        f.write(f"Goal Probability (from similar clips): {similar_goal_probability:.3f}\n")
        f.write(f"Average Similarity Score: {avg_similarity:.3f}\n\n")
        
        f.write("COMBINED FINAL PREDICTION:\n")
        f.write(f"Goal Probability: {combined_goal_prob:.3f}\n")
        f.write(f"Predicted Class: {report['combined_prediction']['predicted_class']}\n")
        f.write(f"Confidence: {avg_similarity:.3f}\n\n")
        
        f.write("CATEGORY DISTRIBUTION OF SIMILAR CLIPS:\n")
        for category, count in category_counts.items():
            percentage = count / len(similar_clips) * 100
            f.write(f"  {category}: {count} clips ({percentage:.1f}%)\n")
        
        f.write(f"\nTOP SIMILAR CLIPS:\n")
        for i, clip in enumerate(similar_clips[:5], 1):
            f.write(f"  {i}. {clip['video_name']} ({clip['category']})\n")
            f.write(f"     Similarity: {clip['similarity']:.3f}, Split: {clip['split']}\n")
        
        f.write(f"\nDOWNLOADED FILES:\n")
        for clip in downloaded_clips:
            f.write(f"  {clip['downloaded_name']} (original: {clip['original_name']})\n")
        
        f.write(f"\nMODEL INSIGHTS:\n")
        if query_attention[0] > query_attention[1]:
            f.write("  - Visual features were weighted more heavily than crowd features\n")
        else:
            f.write("  - Crowd features were weighted more heavily than visual features\n")
        
        if query_goal_prob > 0.7:
            f.write("  - High confidence goal-related prediction from the model\n")
        elif query_goal_prob < 0.3:
            f.write("  - High confidence non-goal prediction from the model\n")
        else:
            f.write("  - Moderate confidence prediction from the model\n")
        
        f.write(f"\nDATA LEAKAGE PREVENTION:\n")
        f.write("  - Similarity search uses only train and val splits\n")
        f.write("  - Test split excluded from similarity database\n")
    
    log_message(f"Reports saved: {report_file}, {summary_file}")
    return report


def main():
    """Runtime prediction pipeline"""
    
    parser = argparse.ArgumentParser(description="Soccer video prediction with precomputed embeddings")
    parser.add_argument("--input_clip", type=str, default=None,
                       help="Path to input video clip")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for results (uses env default if not specified)")
    parser.add_argument("--top_k", type=int, default=None,
                       help="Number of similar clips to retrieve (uses env default if not specified)")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else Path(os.getenv('PREDICTION_RESULTS_DIR'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    top_k = args.top_k if args.top_k else int(os.getenv('TOP_K_SIMILAR'))
    
    temp_dir = output_dir / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    start_time = time.time()
    
    try:
        if args.input_clip:
            input_video = Path(args.input_clip)
            if not input_video.exists():
                raise FileNotFoundError(f"Input video not found: {input_video}")
            query_category = "unknown"
            for cat in ['background', 'before_goal', 'free_kicks_goals', 'penalties', 'shots_no_goals']:
                if cat in str(input_video):
                    query_category = cat
                    break
        else:
            input_video, query_category = select_random_test_clip()
        
        log_message(f"Processing video: {input_video}")
        log_message(f"Query category: {query_category}")
        
        log_message("Starting parallel feature extraction...")
        
        tracking_temp_dir = temp_dir / "tracking"
        visual_temp_dir = temp_dir / "visual"
        tracking_temp_dir.mkdir(exist_ok=True)
        visual_temp_dir.mkdir(exist_ok=True)
        
        results_queue = queue.Queue()
        error_queue = queue.Queue()
        
        def tracking_worker():
            try:
                log_message("Thread 1: Starting tracking processing...")
                tracking_file = process_tracking(input_video, tracking_temp_dir)
                crowd_file = extract_crowd_features(tracking_file, tracking_temp_dir)
                crowd_features_file = encode_crowd_features(crowd_file, tracking_temp_dir)
                results_queue.put(('tracking', crowd_features_file))
            except Exception as e:
                error_queue.put(('tracking', e))
        
        def visual_worker():
            try:
                log_message("Thread 2: Starting visual processing...")
                visual_features_file = extract_visual_features(input_video, visual_temp_dir)
                results_queue.put(('visual', visual_features_file))
            except Exception as e:
                error_queue.put(('visual', e))
        
        tracking_thread = threading.Thread(target=tracking_worker)
        visual_thread = threading.Thread(target=visual_worker)
        
        tracking_thread.start()
        visual_thread.start()
        
        tracking_thread.join()
        visual_thread.join()
        
        if not error_queue.empty():
            while not error_queue.empty():
                process_name, error = error_queue.get()
                log_message(f"Error in {process_name} processing: {error}")
            raise RuntimeError("Parallel processing failed")
        
        crowd_features_file = None
        visual_features_file = None
        
        while not results_queue.empty():
            process_name, result_file = results_queue.get()
            if process_name == 'tracking':
                crowd_features_file = result_file
            elif process_name == 'visual':
                visual_features_file = result_file
        
        if crowd_features_file is None or visual_features_file is None:
            raise RuntimeError("Failed to complete parallel feature extraction")
        
        log_message("Parallel feature extraction completed successfully")
        
        video_name = input_video.stem
        
        with h5py.File(crowd_features_file, 'r') as cf:
            query_crowd_features = cf[list(cf.keys())[0]][:]
        
        with h5py.File(visual_features_file, 'r') as vf:
            query_visual_features = vf[list(vf.keys())[0]][:]
        
        log_message(f"Query crowd features shape: {query_crowd_features.shape}")
        log_message(f"Query visual features shape: {query_visual_features.shape}")
        
        database = load_precomputed_embeddings()
        similar_clips, query_goal_prob, query_attention = find_similar_clips(
            query_crowd_features, query_visual_features, database, top_k
        )
        
        log_message(f"Query processed - Goal probability: {query_goal_prob:.3f}")
        log_message(f"Attention weights - Visual: {query_attention[0]:.3f}, Crowd: {query_attention[1]:.3f}")
        
        downloaded_clips = download_similar_clips(similar_clips, output_dir)
        
        report = generate_prediction_report(
            input_video, query_category, similar_clips, downloaded_clips, 
            output_dir, query_goal_prob, query_attention
        )
        
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        total_time = time.time() - start_time
        log_message(f"Pipeline completed in {total_time:.2f} seconds")
        log_message(f"Results saved to: {output_dir}")
        log_message(f"Model Prediction: {report['model_prediction']['predicted_class']} "
                   f"(probability: {report['model_prediction']['goal_probability']:.3f})")
        log_message(f"Combined Prediction: {report['combined_prediction']['predicted_class']} "
                   f"(probability: {report['combined_prediction']['goal_probability']:.3f})")
        
    except Exception as e:
        log_message(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()