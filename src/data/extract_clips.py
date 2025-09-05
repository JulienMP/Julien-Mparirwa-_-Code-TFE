#!/usr/bin/env python3

import json
import os
import cv2
import subprocess
import glob
import argparse
import random
import math
from pathlib import Path


def get_event_time_seconds(event):
    """Converts event time to period and total seconds for easier processing"""
    try:
        period = int(event['gameTime'].split(' - ')[0])
        time_str = event['gameTime'].split(' - ')[1]
        minutes, seconds = map(int, time_str.split(':'))
        total_seconds = (minutes * 60) + seconds
        return period, total_seconds
    except (KeyError, ValueError, IndexError, AttributeError):
        return None, None


def get_video_info(video_path):
    """Gets FPS and duration information from video file"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds = total_frames / fps if fps > 0 else 0
    cap.release()
    
    return fps, total_frames, duration_seconds


def extract_clip_with_ffmpeg(video_path, start_time, duration, output_path):
    """Extracts video clip using ffmpeg with fallback methods"""
    ffmpeg_cmd = "ffmpeg"
    
    cmd = [
        ffmpeg_cmd,
        "-ss", f"{start_time:.3f}",
        "-i", video_path,
        "-t", f"{duration:.3f}",
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        "-y",
        output_path
    ]
    
    try:
        result = subprocess.run(cmd, check=False, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            return True
        
        print(f"Copy method failed. Trying re-encoding...")
        fallback_cmd = [
            ffmpeg_cmd,
            "-ss", f"{start_time:.3f}",
            "-i", video_path,
            "-t", f"{duration:.3f}",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-y",
            output_path
        ]
        
        fallback_result = subprocess.run(fallback_cmd, check=False, stderr=subprocess.PIPE, text=True)
        if fallback_result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            return True
        
        print(f"Extraction failed: {fallback_result.stderr}")
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass
        return False
        
    except Exception as e:
        print(f"Error during extraction: {e}")
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass
        return False


def extract_goal_clips(json_file, game_dir, output_dir, game_name):
    """Extracts 15-second clips before goals from football match videos"""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading JSON file {json_file}: {e}")
        return

    annotations = data.get('annotations', [])
    goal_events = [event for event in annotations if event.get('label') == 'Goal']
    
    print(f"Game: {game_name} - Found {len(goal_events)} goal events")
    if len(goal_events) == 0:
        return

    video_files = [
        os.path.join(game_dir, "1_224p.mkv"),
        os.path.join(game_dir, "2_224p.mkv")
    ]
    
    for i, file in enumerate(video_files):
        if not os.path.exists(file):
            print(f"Warning: Video file for half {i+1} not found: {file}")
            video_files[i] = None

    for i, goal in enumerate(goal_events):
        try:
            period = int(goal['gameTime'].split(' - ')[0])
            time_str = goal['gameTime'].split(' - ')[1]
            minutes, seconds = map(int, time_str.split(':'))
            team = goal.get('team', 'unknown')
            
            if period <= 0 or period > 2:
                print(f"Error: Invalid period {period} for goal at {goal['gameTime']}")
                continue
                
            video_path = video_files[period - 1]
            if video_path is None:
                print(f"Error: Video file for period {period} not available")
                continue
            
            clip_name = f"{game_name}_goal_{i+1}_period{period}_{time_str.replace(':', 'm')}s_{team}.mkv"
            output_path = os.path.join(output_dir, clip_name)
            
            if os.path.exists(output_path):
                print(f"Clip already exists: {output_path}. Skipping.")
                continue
            
            fps, _, _ = get_video_info(video_path)
            if fps is None:
                print(f"Error: Could not get video info for {video_path}")
                continue
            
            total_seconds = minutes * 60 + seconds
            goal_frame = int(total_seconds * fps)
            start_frame = max(0, goal_frame - (15 * int(fps)))
            start_time = start_frame / fps
            duration = 15.0
            
            print(f"Extracting goal {i+1}: {goal['gameTime']} - {team} team")
            
            if extract_clip_with_ffmpeg(video_path, start_time, duration, output_path):
                print(f"Successfully saved clip to {output_path}")
            else:
                print(f"Failed to extract clip for goal {i+1}")
                
        except (KeyError, ValueError, IndexError) as e:
            print(f"Error processing goal event: {e}")
            continue


def extract_background_clips(json_file, game_dir, output_dir, game_name, clips_per_game=3):
    """Extracts 15-second background clips avoiding goal regions"""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading JSON file {json_file}: {e}")
        return

    annotations = data.get('annotations', [])
    goal_events = [event for event in annotations if event.get('label') == 'Goal']
    
    print(f"Game: {game_name} - Found {len(goal_events)} goal events to avoid")
    
    video_files = [
        os.path.join(game_dir, "1_224p.mkv"),
        os.path.join(game_dir, "2_224p.mkv")
    ]
    
    for i, file in enumerate(video_files):
        if not os.path.exists(file):
            print(f"Warning: Video file for half {i+1} not found: {file}")
            video_files[i] = None

    danger_zones = []
    for goal in goal_events:
        try:
            period = int(goal['gameTime'].split(' - ')[0])
            time_str = goal['gameTime'].split(' - ')[1]
            minutes, seconds = map(int, time_str.split(':'))
            total_seconds = minutes * 60 + seconds
            danger_zones.append((period, max(0, total_seconds - 30), total_seconds + 30))
        except (KeyError, ValueError, IndexError):
            continue
    
    clips_extracted = 0
    attempts = 0
    max_attempts = 150
    
    for period, video_path in enumerate(video_files, 1):
        if video_path is None:
            continue
        
        fps, _, duration_seconds = get_video_info(video_path)
        if fps is None:
            print(f"Error: Could not open video file {video_path}")
            continue
        
        while clips_extracted < clips_per_game and attempts < max_attempts:
            attempts += 1
            
            min_time = 20
            max_time = max(min_time, duration_seconds - 20)
            
            if max_time <= min_time:
                print(f"Video too short for period {period}")
                break
                
            random_time = random.uniform(min_time, max_time)
            random_minute = int(random_time // 60)
            random_second = int(random_time % 60)
            
            is_in_danger_zone = False
            for danger_period, start_time, end_time in danger_zones:
                if period == danger_period:
                    clip_time = random_minute * 60 + random_second
                    if start_time <= clip_time <= end_time:
                        is_in_danger_zone = True
                        break
            
            if is_in_danger_zone:
                continue
            
            clip_name = f"{game_name}_background_{clips_extracted+1}_period{period}_{random_minute}m{random_second}s.mkv"
            output_path = os.path.join(output_dir, clip_name)
            
            if os.path.exists(output_path):
                continue
            
            print(f"Extracting background clip {clips_extracted+1}: period {period} - {random_minute}:{random_second}")
            
            if extract_clip_with_ffmpeg(video_path, random_time, 15.0, output_path):
                print(f"Successfully saved clip to {output_path}")
                clips_extracted += 1
            
        if clips_extracted >= clips_per_game:
            break


def extract_freekick_goal_clips(json_file, game_dir, output_dir, game_name, freekick_window=10):
    """Extracts clips ending just before goals that resulted from recent free kicks"""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading JSON file {json_file}: {e}")
        return

    annotations = data.get('annotations', [])
    if not annotations:
        return

    events_with_time = []
    for event in annotations:
        period, total_seconds = get_event_time_seconds(event)
        label = event.get('label')
        if period is not None and label:
            events_with_time.append({
                'period': period,
                'time_seconds': total_seconds,
                'label': label,
                'team': event.get('team', 'unknown'),
                'gameTime': event.get('gameTime')
            })

    events_with_time.sort(key=lambda x: (x['period'], x['time_seconds']))

    freekick_goal_pairs = []
    for i, current_event in enumerate(events_with_time):
        if current_event['label'] == 'Goal':
            goal_period = current_event['period']
            goal_time = current_event['time_seconds']

            found_freekick = None
            for j in range(i - 1, -1, -1):
                prev_event = events_with_time[j]
                if prev_event['period'] != goal_period or (goal_time - prev_event['time_seconds']) > freekick_window:
                    break

                if prev_event['label'] in ["Direct free-kick", "Indirect free-kick"]:
                    found_freekick = prev_event
                    break

            if found_freekick:
                freekick_goal_pairs.append({
                    'goal_event': current_event,
                    'freekick_event': found_freekick
                })

    print(f"Game: {game_name} - Found {len(freekick_goal_pairs)} free kick → goal sequences")
    if len(freekick_goal_pairs) == 0:
        return

    video_files = [
        os.path.join(game_dir, "1_224p.mkv"),
        os.path.join(game_dir, "2_224p.mkv")
    ]
    
    for i, file in enumerate(video_files):
        if not os.path.exists(file):
            video_files[i] = None

    for idx, pair in enumerate(freekick_goal_pairs):
        goal_event = pair['goal_event']
        freekick_event = pair['freekick_event']
        period = goal_event['period']
        goal_time_sec = goal_event['time_seconds']

        try:
            if period <= 0 or period > 2:
                continue
            video_path = video_files[period - 1]
            if video_path is None:
                continue

            goal_game_time_str = goal_event['gameTime'].split(' - ')[1].replace(':', 'm') + 's'
            fk_game_time_str = freekick_event['gameTime'].split(' - ')[1].replace(':', 'm') + 's'
            
            clip_name = f"{game_name}_freekick_goal_{idx+1}_period{period}_fk{fk_game_time_str}_goal{goal_game_time_str}.mkv"
            output_path = os.path.join(output_dir, clip_name)

            if os.path.exists(output_path):
                continue

            fps, _, _ = get_video_info(video_path)
            if fps is None:
                continue

            clip_duration_sec = 15
            clip_end_time_sec = math.nextafter(goal_time_sec, goal_time_sec - 1)
            clip_start_time_sec = max(0, clip_end_time_sec - clip_duration_sec)
            actual_duration = clip_end_time_sec - clip_start_time_sec

            if actual_duration < 1:
                continue

            print(f"Extracting free kick→goal sequence {idx+1}")
            
            if extract_clip_with_ffmpeg(video_path, clip_start_time_sec, actual_duration, output_path):
                print(f"Successfully saved clip to {output_path}")

        except Exception as e:
            print(f"Error processing freekick→goal pair {idx+1}: {e}")
            continue


def extract_penalty_clips(json_file, game_dir, output_dir, game_name, trigger_window=120):
    """Extracts clips leading up to fouls/cards that cause penalties"""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading JSON file {json_file}: {e}")
        return

    annotations = data.get('annotations', [])
    if not annotations:
        return

    events_with_time = []
    trigger_labels = {"Foul", "Yellow card", "Red card", "Yellow->red card"}
    
    for event in annotations:
        period, total_seconds = get_event_time_seconds(event)
        label = event.get('label')
        if period is not None and label:
            events_with_time.append({
                'period': period,
                'time_seconds': total_seconds,
                'label': label,
                'team': event.get('team', 'unknown'),
                'gameTime': event.get('gameTime')
            })

    events_with_time.sort(key=lambda x: (x['period'], x['time_seconds']))

    penalty_trigger_pairs = []
    for i, current_event in enumerate(events_with_time):
        if current_event['label'] == 'Penalty':
            penalty_period = current_event['period']
            penalty_time = current_event['time_seconds']

            found_trigger = None
            for j in range(i - 1, -1, -1):
                prev_event = events_with_time[j]
                if prev_event['period'] != penalty_period or (penalty_time - prev_event['time_seconds']) > trigger_window:
                    break

                if prev_event['label'] in trigger_labels:
                    found_trigger = prev_event
                    break

            if found_trigger:
                penalty_trigger_pairs.append({
                    'penalty_event': current_event,
                    'trigger_event': found_trigger
                })

    print(f"Game: {game_name} - Found {len(penalty_trigger_pairs)} penalty sequences")
    if len(penalty_trigger_pairs) == 0:
        return

    video_files = [
        os.path.join(game_dir, "1_224p.mkv"),
        os.path.join(game_dir, "2_224p.mkv")
    ]
    
    for i, file in enumerate(video_files):
        if not os.path.exists(file):
            video_files[i] = None

    for idx, pair in enumerate(penalty_trigger_pairs):
        penalty_event = pair['penalty_event']
        trigger_event = pair['trigger_event']
        period = trigger_event['period']
        trigger_time_sec = trigger_event['time_seconds']

        try:
            if period <= 0 or period > 2:
                continue
            video_path = video_files[period - 1]
            if video_path is None:
                continue

            trigger_game_time_str = trigger_event['gameTime'].split(' - ')[1].replace(':', 'm') + 's'
            penalty_game_time_str = penalty_event['gameTime'].split(' - ')[1].replace(':', 'm') + 's'
            
            clip_name = f"{game_name}_penalty_clip_{idx+1}_period{period}_trigger{trigger_game_time_str}_penalty{penalty_game_time_str}.mkv"
            output_path = os.path.join(output_dir, clip_name)

            if os.path.exists(output_path):
                continue

            fps, _, duration_video_sec = get_video_info(video_path)
            if fps is None:
                continue

            clip_duration_sec = 15
            clip_start_time_sec = max(0, trigger_time_sec - clip_duration_sec)
            clip_end_time_sec = min(duration_video_sec, trigger_time_sec)
            actual_duration = clip_end_time_sec - clip_start_time_sec

            if actual_duration < 1:
                continue

            print(f"Extracting penalty sequence {idx+1}")
            
            if extract_clip_with_ffmpeg(video_path, clip_start_time_sec, actual_duration, output_path):
                print(f"Successfully saved clip to {output_path}")

        except Exception as e:
            print(f"Error processing penalty pair {idx+1}: {e}")
            continue


def extract_shot_clips(json_file, game_dir, output_dir, game_name):
    """Extracts clips of shots on target that didn't immediately result in goals"""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading JSON file {json_file}: {e}")
        return

    annotations = data.get('annotations', [])

    goal_event_times = []
    for event in annotations:
        if event.get('label') == 'Goal':
            try:
                period = int(event['gameTime'].split(' - ')[0])
                time_str = event['gameTime'].split(' - ')[1]
                minutes, seconds = map(int, time_str.split(':'))
                total_seconds = minutes * 60 + seconds
                goal_event_times.append((period, total_seconds))
            except (KeyError, ValueError, IndexError):
                continue

    shot_events = []
    for event in annotations:
        if event.get('label') == 'Shots on target':
            try:
                period = int(event['gameTime'].split(' - ')[0])
                time_str = event['gameTime'].split(' - ')[1]
                minutes, seconds = map(int, time_str.split(':'))
                team = event.get('team', 'unknown')

                is_near_goal = False
                shot_time = minutes * 60 + seconds

                for goal_period, goal_time in goal_event_times:
                    if period == goal_period:
                        time_diff = goal_time - shot_time
                        if 0 <= time_diff < 2:
                            is_near_goal = True
                            break

                if not is_near_goal:
                    shot_events.append({
                        'period': period,
                        'minutes': minutes,
                        'seconds': seconds,
                        'team': team,
                        'gameTime': event['gameTime']
                    })
            except (KeyError, ValueError, IndexError):
                continue

    print(f"Game: {game_name} - Found {len(shot_events)} shots on target (excluding those followed by goals)")
    if len(shot_events) == 0:
        return

    video_files = [
        os.path.join(game_dir, "1_224p.mkv"),
        os.path.join(game_dir, "2_224p.mkv")
    ]
    
    for i, file in enumerate(video_files):
        if not os.path.exists(file):
            video_files[i] = None

    for i, shot in enumerate(shot_events):
        try:
            period = shot['period']
            minutes = shot['minutes']
            seconds = shot['seconds']
            team = shot['team']

            if period <= 0 or period > 2:
                continue
            video_path = video_files[period - 1]
            if video_path is None:
                continue

            time_str = f"{minutes:02d}:{seconds:02d}"
            clip_name = f"{game_name}_shot_{i+1}_period{period}_{time_str.replace(':', 'm')}s_{team}.mkv"
            output_path = os.path.join(output_dir, clip_name)

            if os.path.exists(output_path):
                continue

            fps, _, _ = get_video_info(video_path)
            if fps is None:
                continue

            total_seconds = (minutes * 60) + seconds
            shot_frame = int(total_seconds * fps)
            start_frame = max(0, shot_frame - (15 * int(fps)))
            start_time = start_frame / fps

            print(f"Extracting shot {i+1}: {shot['gameTime']} - {team} team")
            
            if extract_clip_with_ffmpeg(video_path, start_time, 15.0, output_path):
                print(f"Successfully saved clip to {output_path}")

        except Exception as e:
            print(f"Error processing shot event {i+1}: {e}")
            continue


def find_all_games(data_dir):
    """Finds all game directories containing Labels-v2.json and video files"""
    all_games = []
    for root, dirs, files in os.walk(data_dir):
        if "Labels-v2.json" in files and any(f.endswith("_224p.mkv") for f in files):
            all_games.append(root)
    return all_games


def process_all_games(data_dir, output_dir, extraction_type, **kwargs):
    """Processes all games and extracts clips based on extraction type"""
    os.makedirs(output_dir, exist_ok=True)
    
    all_games = find_all_games(data_dir)
    print(f"Found {len(all_games)} potential game directories")
    
    extraction_functions = {
        'goals': extract_goal_clips,
        'background': extract_background_clips,
        'freekicks': extract_freekick_goal_clips,
        'penalties': extract_penalty_clips,
        'shots': extract_shot_clips
    }
    
    extract_func = extraction_functions.get(extraction_type)
    if not extract_func:
        print(f"Error: Unknown extraction type '{extraction_type}'")
        return
    
    for i, game_dir in enumerate(all_games):
        print(f"\nProcessing game {i+1}/{len(all_games)}: {game_dir}")
        
        game_name = os.path.basename(game_dir)
        json_file = os.path.join(game_dir, "Labels-v2.json")
        
        if os.path.exists(json_file):
            extract_func(json_file, game_dir, output_dir, game_name, **kwargs)
        else:
            print(f"Warning: No Labels-v2.json file found in {game_dir}")


def main():
    parser = argparse.ArgumentParser(description='Extract clips from SoccerNet football games')
    parser.add_argument('data_directory', help='Directory containing the game data')
    parser.add_argument('output_directory', help='Directory to save the extracted clips')
    parser.add_argument('--type', choices=['goals', 'background', 'freekicks', 'penalties', 'shots'],
                        required=True, help='Type of clips to extract')
    parser.add_argument('--clips-per-game', type=int, default=3,
                        help='Number of background clips per game (background only)')
    parser.add_argument('--freekick-window', type=int, default=10,
                        help='Max seconds before goal to look for free kick (freekicks only)')
    parser.add_argument('--penalty-window', type=int, default=120,
                        help='Max seconds before penalty to look for trigger event (penalties only)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_directory):
        print(f"Error: Data directory {args.data_directory} does not exist")
        return
    
    print(f"Data directory: {args.data_directory}")
    print(f"Output directory: {args.output_directory}")
    print(f"Extraction type: {args.type}")
    
    random.seed(42)
    
    kwargs = {}
    if args.type == 'background':
        kwargs['clips_per_game'] = args.clips_per_game
    elif args.type == 'freekicks':
        kwargs['freekick_window'] = args.freekick_window
    elif args.type == 'penalties':
        kwargs['trigger_window'] = args.penalty_window
    
    process_all_games(args.data_directory, args.output_directory, args.type, **kwargs)


if __name__ == "__main__":
    main()