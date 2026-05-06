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
    print("=" * 80)
    print("IMAGE SIZE CHECKER")
    print("=" * 80)
    
    # Find input images
    print(f"\n📁 Data directory: {DATA_DIR}")
    image_files = sorted(DATA_DIR.glob("*.nii.gz")) + sorted(DATA_DIR.glob("*.nii"))
    
    if not image_files:
        print("❌ No image files found in data directory")
        return
    
    print(f"✓ Found {len(image_files)} image file(s)\n")
    
    # Check input images
    print("-" * 80)
    print("INPUT IMAGES")
    print("-" * 80)
    
    image_info = {}
    for img_file in image_files:
        nifti = nib.load(img_file)
        shape = nifti.shape
        size_mb = get_file_size_mb(img_file)
        
        image_info[img_file.stem] = shape
        
        print(f"\nFile: {img_file.name}")
        print(f"  Shape: {shape}")
        print(f"  Size: {size_mb:.2f} MB")
        print(f"  Data type: {nifti.get_data_dtype()}")
        print(f"  Affine:\n{nifti.affine}")
    
    # Check prediction masks
    print("\n" + "-" * 80)
    print("PREDICTION MASKS")
    print("-" * 80)
    
    if not PREDICTIONS_DIR.exists():
        print(f"❌ Predictions directory not found: {PREDICTIONS_DIR}")
        return
    
    pred_files = sorted(PREDICTIONS_DIR.glob("seg_*.nii.gz")) + sorted(
        PREDICTIONS_DIR.glob("seg_*.nii")
    )
    
    if not pred_files:
        print("❌ No prediction files found in predictions directory")
        return
    
    print(f"✓ Found {len(pred_files)} prediction file(s)\n")
    
    for pred_file in pred_files:
        nifti = nib.load(pred_file)
        shape = nifti.shape
        size_mb = get_file_size_mb(pred_file)
        
        print(f"\nFile: {pred_file.name}")
        print(f"  Shape: {shape}")
        print(f"  Size: {size_mb:.2f} MB")
        print(f"  Data type: {nifti.get_data_dtype()}")
    
    # Verify shape matching
    print("\n" + "-" * 80)
    print("SHAPE VERIFICATION")
    print("-" * 80)
    
    if image_info:
        # Get the first (or only) image shape
        first_image_name = list(image_info.keys())[0]
        image_shape = image_info[first_image_name]
        
        print(f"\nReference image: {first_image_name}")
        print(f"Reference shape: {image_shape}")
        
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
            
            print(f"\n{pred_file.name}")
            print(f"  Predicted shape: {pred_shape}")
            print(f"  Status: {status}")
        
        print("\n" + "=" * 80)
        if all_match:
            print("✓ All predictions match the input image shape!")
        else:
            print("⚠️  Shape issues detected!")
            print("\nTo fix axis ordering issues, run:")
            print("  python fix_prediction_axes.py")
        print("=" * 80)


if __name__ == "__main__":
    check_image_shapes()
