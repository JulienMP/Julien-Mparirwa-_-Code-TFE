#!/usr/bin/env python3

import os
import sys
import argparse
from SoccerNet.Downloader import SoccerNetDownloader


def download_labels(downloader):
    """Downloads the Labels-v2.json files for all splits"""
    downloader.downloadGames(files=["Labels-v2.json"], split=["train", "valid", "test"])


def download_videos(downloader, password):
    """Downloads the 224p video files for all splits"""
    downloader.password = password
    downloader.downloadGames(files=["1_224p.mkv", "2_224p.mkv"], split=["train", "valid", "test", "challenge"])


def main():
    parser = argparse.ArgumentParser(description="Download SoccerNet dataset")
    parser.add_argument("data_directory", help="Directory to download the data to")
    parser.add_argument("--password", default="s0cc3rn3t", help="Password for video downloads")
    parser.add_argument("--labels-only", action="store_true", help="Download only labels, skip videos")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_directory):
        os.makedirs(args.data_directory, exist_ok=True)
        print(f"Created directory: {args.data_directory}")
    
    print(f"Downloading data to: {args.data_directory}")
    
    downloader = SoccerNetDownloader(LocalDirectory=args.data_directory)
    
    try:
        print("Downloading labels...")
        download_labels(downloader)
        print("Labels download completed")
        
        if not args.labels_only:
            print("Downloading videos...")
            download_videos(downloader, args.password)
            print("Videos download completed")
        
        print("Download process completed successfully")
        
    except Exception as e:
        print(f"Error during download: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()