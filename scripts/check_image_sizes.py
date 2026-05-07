#!/usr/bin/env python
"""
Script to check and compare sizes of data images and prediction masks.

This script displays:
- Input image shape and file size
- Prediction mask shape and file size
- Verification that shapes match
"""

import os
from pathlib import Path
import nibabel as nib
from src.utils.logging import get_logger


logger = get_logger(__name__)

# ============================================================================
# DIRECTORY CONFIGURATION
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"
# ============================================================================


def get_file_size_mb(file_path):
    """Get file size in MB."""
    size_bytes = os.path.getsize(file_path)
    size_mb = size_bytes / (1024 * 1024)
    return size_mb


def check_image_shapes():
    """Check and compare data image and prediction mask shapes."""
    logger.info("%s", "=" * 80)
    logger.info("IMAGE SIZE CHECKER")
    logger.info("%s", "=" * 80)
    
    # Find input images
    logger.info("Data directory: %s", DATA_DIR)
    image_files = sorted(DATA_DIR.glob("*.nii.gz")) + sorted(DATA_DIR.glob("*.nii"))
    
    if not image_files:
        logger.error("No image files found in data directory")
        return
    
    logger.info("Found %d image file(s)", len(image_files))
    
    # Check input images
    logger.info("%s", "-" * 80)
    logger.info("INPUT IMAGES")
    logger.info("%s", "-" * 80)
    
    image_info = {}
    for img_file in image_files:
        nifti = nib.load(img_file)
        shape = nifti.shape
        size_mb = get_file_size_mb(img_file)
        
        image_info[img_file.stem] = shape
        
        logger.info("File: %s", img_file.name)
        logger.info("  Shape: %s", shape)
        logger.info("  Size: %.2f MB", size_mb)
        logger.info("  Data type: %s", nifti.get_data_dtype())
        logger.info("  Affine:\n%s", nifti.affine)
    
    # Check prediction masks
    logger.info("%s", "-" * 80)
    logger.info("PREDICTION MASKS")
    logger.info("%s", "-" * 80)
    
    if not PREDICTIONS_DIR.exists():
        logger.error("Predictions directory not found: %s", PREDICTIONS_DIR)
        return
    
    pred_files = sorted(PREDICTIONS_DIR.glob("seg_*.nii.gz")) + sorted(
        PREDICTIONS_DIR.glob("seg_*.nii")
    )
    
    if not pred_files:
        logger.error("No prediction files found in predictions directory")
        return
    
    logger.info("Found %d prediction file(s)", len(pred_files))
    
    for pred_file in pred_files:
        nifti = nib.load(pred_file)
        shape = nifti.shape
        size_mb = get_file_size_mb(pred_file)
        
        logger.info("File: %s", pred_file.name)
        logger.info("  Shape: %s", shape)
        logger.info("  Size: %.2f MB", size_mb)
        logger.info("  Data type: %s", nifti.get_data_dtype())
    
    # Verify shape matching
    logger.info("%s", "-" * 80)
    logger.info("SHAPE VERIFICATION")
    logger.info("%s", "-" * 80)
    
    if image_info:
        # Get the first (or only) image shape
        first_image_name = list(image_info.keys())[0]
        image_shape = image_info[first_image_name]
        
        logger.info("Reference image: %s", first_image_name)
        logger.info("Reference shape: %s", image_shape)
        
        all_match = True
        for pred_file in pred_files:
            nifti = nib.load(pred_file)
            pred_shape = nifti.shape
            
            # Compare shapes (exact match)
            if pred_shape == image_shape:
                status = "✓ MATCH"
            # Check if it's just axis permutation
            elif sorted(pred_shape) == sorted(image_shape):
                status = f"⚠️  AXIS PERMUTATION (needs reordering)"
                all_match = False
            else:
                status = "❌ MISMATCH (different dimensions)"
                all_match = False
            
            logger.info("%s", pred_file.name)
            logger.info("  Predicted shape: %s", pred_shape)
            logger.info("  Status: %s", status)
        
        logger.info("%s", "=" * 80)
        if all_match:
            logger.info("All predictions match the input image shape!")
        else:
            logger.warning("Shape issues detected!")
            logger.info("To fix axis ordering issues, run:")
            logger.info("  python fix_prediction_axes.py")
        logger.info("%s", "=" * 80)


if __name__ == "__main__":
    check_image_shapes()
