#!/usr/bin/env python3

import os
import time
import json
import torch
import cv2
import argparse
from pathlib import Path
import numpy as np
import h5py
from pytorchvideo.models.hub import slowfast_r50


class SlowFastFeatureExtractor(torch.nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.model = original_model

    def forward(self, x):
        """Extracts features using hook-based approach"""
        features = {}

        def hook_fn(name):
            def hook(module, input, output):
                features[name] = output
            return hook

        handle = None
        try:
            if hasattr(self.model, 'head') and hasattr(self.model.head, 'proj'):
                handle = self.model.head.proj.register_forward_hook(hook_fn('pre_proj'))
            elif hasattr(self.model, 'head') and hasattr(self.model.head, 'projection'):
                handle = self.model.head.projection.register_forward_hook(hook_fn('pre_proj'))
        except:
            pass

        with torch.no_grad():
            output = self.model(x)

        if handle:
            handle.remove()

        if 'pre_proj' in features:
            extracted_features = features['pre_proj']
            if isinstance(extracted_features, (list, tuple)):
                if len(extracted_features) == 2:
                    slow_feat = torch.nn.functional.adaptive_avg_pool3d(extracted_features[0], (1, 1, 1))
                    fast_feat = torch.nn.functional.adaptive_avg_pool3d(extracted_features[1], (1, 1, 1))
                    slow_feat = slow_feat.flatten(start_dim=1)
                    fast_feat = fast_feat.flatten(start_dim=1)
                    return torch.cat([slow_feat, fast_feat], dim=1)
                else:
                    feat = extracted_features[0]
                    if len(feat.shape) > 2:
                        feat = torch.nn.functional.adaptive_avg_pool3d(feat, (1, 1, 1))
                    return feat.flatten(start_dim=1)
            else:
                if len(extracted_features.shape) > 2:
                    extracted_features = torch.nn.functional.adaptive_avg_pool3d(extracted_features, (1, 1, 1))
                return extracted_features.flatten(start_dim=1)
        else:
            if hasattr(output, 'logits'):
                return output.logits
            return output


def load_video_opencv(video_path, max_frames=64):
    """Loads video frames using OpenCV with uniform sampling"""
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        raise ValueError("Video has no frames")

    if total_frames <= max_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = [int(i) for i in np.linspace(0, total_frames - 1, max_frames)]

    frames = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

    cap.release()

    if len(frames) == 0:
        raise ValueError("No frames could be read")

    return np.array(frames, dtype=np.uint8)


def preprocess_frames(frames_np, target_size=224, target_frames=32):
    """Preprocesses frames for SlowFast model"""
    T, H, W, C = frames_np.shape
    processed_frames = []

    for t in range(T):
        frame = frames_np[t]
        if H != 256 or W != 256:
            frame = cv2.resize(frame, (256, 256))

        if target_size != 256:
            h, w = frame.shape[:2]
            start_h = (h - target_size) // 2
            start_w = (w - target_size) // 2
            frame = frame[start_h:start_h + target_size, start_w:start_w + target_size]

        processed_frames.append(frame)

    frames_np = np.array(processed_frames)
    frames_tensor = torch.from_numpy(frames_np).float() / 255.0
    frames_tensor = frames_tensor.permute(3, 0, 1, 2)

    T = frames_tensor.shape[1]
    if T > target_frames:
        indices = torch.linspace(0, T - 1, target_frames).long()
        frames_tensor = torch.index_select(frames_tensor, 1, indices)
    elif T < target_frames:
        repeat_factor = (target_frames + T - 1) // T
        frames_tensor = frames_tensor.repeat(1, repeat_factor, 1, 1)[:, :target_frames]

    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]
    for c in range(3):
        frames_tensor[c] = (frames_tensor[c] - mean[c]) / std[c]

    return frames_tensor


def pack_pathway_output(frames):
    """Packs frames for SlowFast dual pathways"""
    fast_pathway = frames

    T = frames.shape[2]
    if T >= 4:
        slow_indices = torch.linspace(0, T - 1, max(1, T // 4)).long()
        if frames.is_cuda:
            slow_indices = slow_indices.cuda()
        slow_pathway = torch.index_select(frames, 2, slow_indices)
    else:
        slow_pathway = frames

    return [slow_pathway, fast_pathway]


def process_videos(dataset_dir, output_dir, device='cuda'):
    """Processes all videos in the dataset directory structure"""
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)

    print("Loading SlowFast model...")
    model = slowfast_r50(pretrained=True)
    feature_extractor = SlowFastFeatureExtractor(model)
    feature_extractor.eval()

    if torch.cuda.is_available() and device == 'cuda':
        feature_extractor = feature_extractor.cuda()
        print("Using GPU")
    else:
        print("Using CPU")

    splits = ['train', 'val', 'test']
    categories = ['background', 'before_goal', 'free_kicks_goals', 'penalties', 'shots_no_goals']

    total_stats = {
        'total_videos': 0,
        'successful': 0,
        'failed': 0,
        'processing_times': [],
        'feature_shapes': []
    }

    for split in splits:
        split_dir = dataset_path / split
        if not split_dir.exists():
            print(f"Split directory not found: {split_dir}")
            continue

        print(f"\nProcessing {split} split...")

        split_output_dir = output_path / split
        split_output_dir.mkdir(parents=True, exist_ok=True)

        for category in categories:
            category_dir = split_dir / category
            if not category_dir.exists():
                print(f"  Category not found: {category}")
                continue

            print(f"  Processing {category}...")

            video_files = list(category_dir.glob("*.mkv"))
            if not video_files:
                print(f"    No videos found in {category}")
                continue

            print(f"    Found {len(video_files)} videos")

            h5_file_path = split_output_dir / f"{category}_features.h5"

            category_results = []
            successful_count = 0

            with h5py.File(h5_file_path, 'w') as hf:
                for idx, video_path in enumerate(video_files, 1):
                    start_time = time.time()
                    video_name = video_path.stem

                    try:
                        print(f"    [{idx}/{len(video_files)}] {video_name}")

                        frames_np = load_video_opencv(video_path, max_frames=64)
                        frames_tensor = preprocess_frames(frames_np, target_size=224, target_frames=32)
                        frames_batch = frames_tensor.unsqueeze(0)

                        if device == 'cuda' and torch.cuda.is_available():
                            frames_batch = frames_batch.cuda()

                        frames_list = pack_pathway_output(frames_batch)
                        if device == 'cuda' and torch.cuda.is_available():
                            frames_list = [pathway.cuda() for pathway in frames_list]

                        with torch.no_grad():
                            features = feature_extractor(frames_list).squeeze()
                            if device == 'cuda':
                                features = features.cpu()

                        hf.create_dataset(video_name, data=features.numpy())

                        elapsed = time.time() - start_time

                        category_results.append({
                            'video': video_name,
                            'feature_shape': list(features.shape),
                            'processing_time': elapsed,
                            'status': 'success'
                        })

                        successful_count += 1
                        total_stats['successful'] += 1
                        total_stats['processing_times'].append(elapsed)
                        total_stats['feature_shapes'].append(list(features.shape))

                    except Exception as e:
                        elapsed = time.time() - start_time
                        print(f"      ERROR: {e}")

                        category_results.append({
                            'video': video_name,
                            'error': str(e),
                            'processing_time': elapsed,
                            'status': 'failed'
                        })

                        total_stats['failed'] += 1

                    total_stats['total_videos'] += 1

            metadata_path = split_output_dir / f"{category}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump({
                    'split': split,
                    'category': category,
                    'total_videos': len(video_files),
                    'successful': successful_count,
                    'failed': len(video_files) - successful_count,
                    'success_rate': successful_count / len(video_files),
                    'h5_file': str(h5_file_path),
                    'results': category_results
                }, f, indent=2)

            print(f"    Completed: {successful_count}/{len(video_files)} successful")

    overall_metadata = output_path / "overall_metadata.json"
    with open(overall_metadata, 'w') as f:
        json.dump({
            'dataset_directory': str(dataset_dir),
            'output_directory': str(output_dir),
            'total_statistics': total_stats,
            'avg_processing_time': np.mean(total_stats['processing_times']) if total_stats['processing_times'] else 0,
            'feature_dimension': total_stats['feature_shapes'][0] if total_stats['feature_shapes'] else None,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }, f, indent=2)

    return total_stats


def main():
    parser = argparse.ArgumentParser(description="Extract visual features using SlowFast")
    parser.add_argument("dataset_dir", help="Path to dataset directory")
    parser.add_argument("output_dir", help="Output directory for features")
    parser.add_argument("--device", choices=['cuda', 'cpu'], default='cuda',
                       help="Device to use for processing")

    args = parser.parse_args()

    print("="*60)
    print("SlowFast Visual Feature Extraction")
    print("="*60)
    print(f"Dataset: {args.dataset_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {args.device}")

    if not Path(args.dataset_dir).exists():
        print(f"ERROR: Dataset directory not found: {args.dataset_dir}")
        return

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    stats = process_videos(args.dataset_dir, args.output_dir, args.device)
    total_time = time.time() - start_time

    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Videos processed: {stats['total_videos']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Success rate: {100 * stats['successful'] / stats['total_videos']:.1f}%")

    if stats['processing_times']:
        print(f"Average time per video: {np.mean(stats['processing_times']):.2f}s")

    if stats['feature_shapes']:
        print(f"Feature dimension: {stats['feature_shapes'][0]}")

    print(f"\nFeatures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()