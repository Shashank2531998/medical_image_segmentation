#!/usr/bin/env python
"""
Script to visualize VoxTell prediction results using Napari viewer.

This script loads a medical image and its corresponding segmentation masks
and displays them in an interactive Napari viewer.

Usage:
    # Interactive (on login node with display):
    python visualize.py --image scan.nii.gz
    
    # Offscreen rendering (on compute node, saves screenshots):
    python visualize.py --image scan.nii.gz --offscreen

Example:
    python visualize.py --image scan.nii.gz
    python visualize.py --image scan.nii.gz --offscreen --output-dir ./screenshots
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

# Set offscreen rendering BEFORE importing napari if --offscreen flag is present
if "--offscreen" in sys.argv:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = ""

import numpy as np
import nibabel as nib
import napari
from src.utils.logging import get_logger


# ============================================================================
# DIRECTORY CONFIGURATION
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.absolute()

# Input data directory
DATA_DIR = PROJECT_ROOT / "data"

# Predictions directory
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"
# ============================================================================


logger = get_logger(__name__)


def visualize_predictions(
    image_path: str,
    predictions_dir: str,
    image_name: Optional[str] = None,
    offscreen: bool = False,
    output_dir: Optional[str] = None,
) -> None:
    """
    Visualize medical image and segmentation predictions in Napari.
    
    Args:
        image_path: Path to the original medical image
        predictions_dir: Directory containing segmentation masks
        image_name: Optional name for the image layer
        offscreen: If True, save screenshots instead of showing interactive viewer
        output_dir: Directory to save screenshots (required if offscreen=True)
        
    Raises:
        FileNotFoundError: If image or predictions directory not found
    """
    # Validate paths
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    if not Path(predictions_dir).exists():
        raise FileNotFoundError(f"Predictions directory not found: {predictions_dir}")
    
    logger.info("%s", "=" * 70)
    logger.info("VoxTell Prediction Visualizer")
    logger.info("%s", "=" * 70)
    
    # Load original image
    logger.info("Loading image: %s", image_path)
    nifti_img = nib.load(image_path)
    img_data = nifti_img.get_fdata()
    
    # Normalize image to 0-1 range for better visualization
    img_min = img_data.min()
    img_max = img_data.max()
    if img_max > img_min:
        img_normalized = (img_data - img_min) / (img_max - img_min)
    else:
        img_normalized = img_data
    
    logger.info("Image shape: %s", img_data.shape)
    logger.info("Image value range: [%.2f, %.2f]", img_data.min(), img_data.max())
    
    # Create Napari viewer
    logger.info("Initializing Napari viewer...")
    viewer = napari.Viewer()
    
    # Add original image
    image_layer_name = image_name or Path(image_path).stem
    viewer.add_image(
        img_normalized,
        name=image_layer_name,
        colormap="gray",
    )
    
    # Find and load segmentation masks
    predictions_path = Path(predictions_dir)
    seg_files = sorted(predictions_path.glob("seg_*.nii.gz")) + sorted(
        predictions_path.glob("seg_*.nii")
    )
    
    if not seg_files:
        logger.warning("No segmentation files found in %s", predictions_dir)
        logger.info("Expected files like: seg_liver.nii.gz, seg_kidney.nii.gz, etc.")
    else:
        logger.info("Found %d segmentation masks", len(seg_files))
        
        # Color palette for different structures
        colors = [
            "#FF0000",  # Red
            "#00FF00",  # Green
            "#0000FF",  # Blue
            "#FFFF00",  # Yellow
            "#FF00FF",  # Magenta
            "#00FFFF",  # Cyan
            "#FFA500",  # Orange
            "#800080",  # Purple
        ]
        
        for idx, seg_file in enumerate(seg_files):
            logger.info("  [%d] %s", idx + 1, seg_file.name)
            
            # Load segmentation mask
            seg_nifti = nib.load(seg_file)
            seg_data = seg_nifti.get_fdata().astype(np.uint8)
            
            # Extract structure name from filename
            # e.g., "seg_liver.nii.gz" -> "liver"
            seg_name = seg_file.stem.replace("seg_", "").replace(".nii", "")
            
            # Add as label layer with custom color
            color = colors[idx % len(colors)]
            viewer.add_labels(
                seg_data,
                name=seg_name,
                color=color,
                opacity=0.5,
            )
    
    # Add combined mask if it exists
    combined_file = predictions_path / "seg_combined.nii.gz"
    if not combined_file.exists():
        combined_file = predictions_path / "seg_combined.nii"
    
    if combined_file.exists():
        logger.info("Loading combined segmentation: %s", combined_file.name)
        combined_nifti = nib.load(combined_file)
        combined_data = combined_nifti.get_fdata().astype(np.uint8)
        
        viewer.add_labels(
            combined_data,
            name="combined",
            color="#FFFFFF",
            opacity=0.3,
        )
    
    logger.info("%s", "=" * 70)
    
    if offscreen:
        if not output_dir:
            raise ValueError("output_dir must be specified for offscreen rendering")
        
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving screenshots to: %s", output_dir)
        
        # Save screenshot at current slice
        viewer.camera.zoom = 1.5
        screenshot_path = os.path.join(output_dir, "visualization.png")
        screenshot = viewer.screenshot()
        screenshot.save(screenshot_path)
        logger.info("Saved: %s", screenshot_path)
        
        logger.info("%s", "=" * 70)
    else:
        logger.info("Napari Viewer Controls:")
        logger.info("%s", "=" * 70)
        logger.info("  - Left panel: Toggle layers on/off")
        logger.info("  - Scroll wheel: Zoom in/out")
        logger.info("  - Mouse drag: Pan")
        logger.info("  - Arrow keys: Navigate slices (for 3D data)")
        logger.info("  - L: Toggle selected layer visibility")
        logger.info("  - D: Delete selected layer")
        logger.info("  - R: Reset view")
        logger.info("%s", "=" * 70)
        logger.info("Starting Napari viewer...")
        
        # Run interactive viewer
        napari.run()


def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(
        description="Visualize VoxTell predictions in Napari viewer"
    )
    parser.add_argument(
        "--image",
        default="imaging.nii.gz",
        help="Image filename in data directory (default: imaging.nii.gz)"
    )
    parser.add_argument(
        "--offscreen",
        action="store_true",
        help="Use offscreen rendering (for compute nodes without display)"
    )
    parser.add_argument(
        "--output-dir",
        default="./screenshots",
        help="Output directory for offscreen screenshots (default: ./screenshots)"
    )
    
    args = parser.parse_args()
    
    # Construct full image path
    image_path = DATA_DIR / args.image
    
    # Validate directories
    if not DATA_DIR.exists():
        logger.error("Data directory not found: %s", DATA_DIR)
        logger.info("Please create the directory and place your medical image there:")
        logger.info("  mkdir -p %s", DATA_DIR)
        logger.info("  cp /path/to/your/image.nii.gz %s/%s", DATA_DIR, args.image)
        sys.exit(1)
    
    if not image_path.exists():
        logger.error("Image not found: %s", image_path)
        logger.info("Please place your medical image in the data directory:")
        logger.info("  cp /path/to/your/image.nii.gz %s", image_path)
        logger.info("Available files in %s:", DATA_DIR)
        files = list(DATA_DIR.glob("*"))
        if files:
            for f in files:
                if f.is_file():
                    logger.info("  - %s", f.name)
        else:
            logger.info("  (directory is empty)")
        sys.exit(1)
    
    if not PREDICTIONS_DIR.exists():
        logger.error("Predictions directory not found: %s", PREDICTIONS_DIR)
        logger.info("Please run prediction first:")
        logger.info("  python predict.py --image %s --prompts 'liver' 'kidney'", args.image)
        sys.exit(1)
    
    try:
        visualize_predictions(
            image_path=str(image_path),
            predictions_dir=str(PREDICTIONS_DIR),
            image_name=Path(image_path).stem,
            offscreen=args.offscreen,
            output_dir=args.output_dir if args.offscreen else None,
        )
        
    except FileNotFoundError as e:
        logger.error("Error: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.exception("Error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
