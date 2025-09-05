#!/usr/bin/env python3

import os
import shutil
from pathlib import Path
import argparse
from collections import defaultdict


def systematic_sample(files, target_count):
    """Uses systematic sampling to select videos with regular intervals maintaining temporal distribution"""
    if len(files) <= target_count:
        return files

    step = len(files) / target_count
    selected = []
    for i in range(target_count):
        index = int(i * step)
        if index < len(files):
            selected.append(files[index])

    return selected


def create_subset(source_dir, target_dir, target_total, proportions):
    """Creates a subset of the dataset using systematic sampling"""
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    if abs(sum(proportions.values()) - 1.0) > 1e-6:
        raise ValueError("Proportions must sum to 1.0")

    categories = list(proportions.keys())
    splits = ['train', 'val', 'test']
    stats = defaultdict(lambda: defaultdict(int))
    total_copied = 0

    print(f"Creating subset with {target_total} total videos")
    print("Target distribution:")
    for category, prop in proportions.items():
        target_count = int(target_total * prop)
        print(f"  {category:20}: {target_count:4d} videos ({prop:.1%})")
    print()

    for category in categories:
        target_category_count = int(target_total * proportions[category])
        print(f"Processing {category} (target: {target_category_count} videos)")

        all_category_videos = []
        split_info = []

        for split in splits:
            split_category_dir = source_path / split / category
            if split_category_dir.exists():
                videos = list(split_category_dir.glob("*.mkv"))
                print(f"  Found {len(videos)} videos in {split}/{category}")

                for video in videos:
                    all_category_videos.append(video)
                    split_info.append(split)

        if len(all_category_videos) == 0:
            print(f"  Warning: No videos found for {category}")
            continue

        selected_videos = systematic_sample(all_category_videos, target_category_count)
        print(f"  Selected {len(selected_videos)} videos using systematic sampling")

        for i, video_path in enumerate(selected_videos):
            original_split = split_info[all_category_videos.index(video_path)]

            target_split_category_dir = target_path / original_split / category
            target_split_category_dir.mkdir(parents=True, exist_ok=True)

            target_file = target_split_category_dir / video_path.name
            print(f"    Copying {video_path.name} to {original_split}/{category}/")
            shutil.copy2(video_path, target_file)

            stats[original_split][category] += 1
            total_copied += 1

        print()

    print("="*60)
    print("SUBSET CREATION SUMMARY")
    print("="*60)

    for split_name in ['train', 'val', 'test']:
        print(f"\n{split_name.upper()}:")
        split_total = 0
        for category in categories:
            count = stats[split_name][category]
            split_total += count
            print(f"  {category:20}: {count:4d} videos")
        print(f"  {'TOTAL':20}: {split_total:4d} videos")

    print(f"\nGRAND TOTAL: {total_copied} videos")
    print(f"Target was: {target_total} videos")
    print(f"Efficiency: {total_copied/target_total:.1%}")

    print(f"\nACTUAL PROPORTIONS ACHIEVED:")
    for category in categories:
        category_total = sum(stats[split][category] for split in stats)
        actual_prop = category_total / total_copied if total_copied > 0 else 0
        target_prop = proportions[category]
        print(f"  {category:20}: {actual_prop:.1%} (target: {target_prop:.1%})")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Create subset of organized dataset using systematic sampling")
    parser.add_argument("source_dir", help="Source directory containing organized clips")
    parser.add_argument("target_dir", help="Target directory for subset")
    parser.add_argument("total_videos", type=int, help="Total number of videos in subset")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without copying files")

    parser.add_argument("--background-prop", type=float, default=0.25, help="Proportion for background videos")
    parser.add_argument("--before-goal-prop", type=float, default=0.28, help="Proportion for before_goal videos")
    parser.add_argument("--free-kicks-prop", type=float, default=0.025, help="Proportion for free_kicks_goals videos")
    parser.add_argument("--penalties-prop", type=float, default=0.028, help="Proportion for penalties videos")
    parser.add_argument("--shots-no-goals-prop", type=float, default=0.417, help="Proportion for shots_no_goals videos")

    args = parser.parse_args()

    proportions = {
        'background': args.background_prop,
        'before_goal': args.before_goal_prop,
        'free_kicks_goals': args.free_kicks_prop,
        'penalties': args.penalties_prop,
        'shots_no_goals': args.shots_no_goals_prop
    }

    prop_sum = sum(proportions.values())
    if abs(prop_sum - 1.0) > 1e-6:
        print(f"Error: Proportions sum to {prop_sum:.3f}, must equal 1.0")
        return

    if args.dry_run:
        print("DRY RUN MODE - No files will be copied")
        print(f"Source: {args.source_dir}")
        print(f"Target: {args.target_dir}")
        print(f"Total videos: {args.total_videos}")

        source_path = Path(args.source_dir)
        for category, prop in proportions.items():
            target_count = int(args.total_videos * prop)

            total_existing = 0
            for split in ['train', 'val', 'test']:
                split_dir = source_path / split / category
                if split_dir.exists():
                    total_existing += len(list(split_dir.glob("*.mkv")))

            print(f"{category}: {total_existing} available -> {target_count} would be selected")
    else:
        print("Creating subset dataset...")
        stats = create_subset(args.source_dir, args.target_dir, args.total_videos, proportions)
        print(f"\nSubset creation complete! Check {args.target_dir}")


if __name__ == "__main__":
    main()