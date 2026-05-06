#!/usr/bin/env python
"""
Script to download VoxTell model checkpoint from Hugging Face Hub.

This script downloads the VoxTell model and supporting files from the 
Hugging Face model repository.

Usage:
    python download_checkpoint.py [--model-name MODEL_NAME] [--download-dir DIRECTORY]

Example:
    python download_checkpoint.py --download-dir ./models
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import snapshot_download


def download_voxtell_checkpoint(model_name: str, download_dir: str) -> str:
    """
    Download VoxTell model checkpoint from Hugging Face Hub.
    
    Args:
        model_name: Name of the model version (e.g., "voxtell_v1.1")
        download_dir: Directory where the model will be downloaded
        
    Returns:
        Path to the downloaded model directory
    """
    # Create download directory if it doesn't exist
    os.makedirs(download_dir, exist_ok=True)
    
    print(f"Downloading {model_name} from Hugging Face Hub...")
    print(f"Download directory: {download_dir}")
    
    download_path = snapshot_download(
        repo_id="mrokuss/VoxTell",
        allow_patterns=[f"{model_name}/*", "*.json"],
        local_dir=download_dir
    )
    
    # Path to model directory
    model_path = os.path.join(download_path, model_name)
    
    print(f"✓ Model downloaded successfully!")
    print(f"Model path: {model_path}")
    
    return model_path


def main():
    parser = argparse.ArgumentParser(
        description="Download VoxTell model checkpoint from Hugging Face Hub"
    )
    parser.add_argument(
        "--model-name",
        default="voxtell_v1.1",
        help="Name of the model version (default: voxtell_v1.1)"
    )
    parser.add_argument(
        "--download-dir",
        default="./models",
        help="Directory to download the model (default: ./models)"
    )
    
    args = parser.parse_args()
    
    model_path = download_voxtell_checkpoint(
        model_name=args.model_name,
        download_dir=args.download_dir
    )
    
    print(f"\nTo use this model in your code:")
    print(f"  model_path = '{model_path}'")


if __name__ == "__main__":
    main()
