#!/usr/bin/env python
"""
Script to run predictions using the VoxTell model.

This script loads a medical image and uses text prompts to generate
segmentation masks for specified anatomical structures.

Usage:
    python predict.py --image scan.nii.gz --prompts "liver" "kidney"

Example:
    python predict.py \\
        --image scan.nii.gz \\
        --prompts "liver" "right kidney" "left kidney" "spleen" \\
        --device cuda:0
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

# Set Hugging Face Hub environment variables for ma
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import numpy as np
import nibabel as nib
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient

from src.inference.predictor import VoxTellPredictor


# ============================================================================
# DIRECTORY CONFIGURATION - Edit these if needed
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.absolute()

# Input data directory
DATA_DIR = PROJECT_ROOT / "data"

# Model directory (downloaded from Hugging Face)
MODEL_DIR = PROJECT_ROOT / "models" / "voxtell_v1.1"

# Output predictions directory
OUTPUT_DIR = PROJECT_ROOT / "predictions"
# ============================================================================


def reorient_seg_to_original(seg_data: np.ndarray, properties: dict) -> nib.Nifti1Image:
    """
    Reorient segmentation from RAS (processing) space back to original image orientation.
    
    Uses the same approach as nnUNetv2's NibabelIOWithReorient.write_seg()
    
    Args:
        seg_data: Segmentation array (single volume, not batched)
        properties: Metadata from NibabelIOWithReorient.read_images()
        
    Returns:
        Reoriented NIfTI image matching original image orientation
    """
    nibabel_stuff = properties['nibabel_stuff']
    original_affine = nibabel_stuff['original_affine']
    reoriented_affine = nibabel_stuff['reoriented_affine']
    
    # CRITICAL: Reverse the transpose that was applied during read_images()
    # NibabelIOWithReorient transposes as (2, 1, 0) for SimpleITK compatibility
    seg_data = seg_data.transpose((2, 1, 0))
    
    # Create temporary NIfTI with reoriented affine
    seg_nib = nib.Nifti1Image(seg_data, affine=reoriented_affine)
    
    # Compute reorientation transform back to original orientation
    img_ornt = io_orientation(original_affine)
    ras_ornt = axcodes2ornt("RAS")
    from_canonical = ornt_transform(ras_ornt, img_ornt)
    
    # Apply inverse reorientation
    seg_nib_reoriented = seg_nib.as_reoriented(from_canonical)
    
    # Verify affine matches
    if not np.allclose(original_affine, seg_nib_reoriented.affine):
        print(f'WARNING: Restored affine does not match original affine')
        print(f'Original affine:\n{original_affine}')
        print(f'Restored affine:\n{seg_nib_reoriented.affine}')
    
    return seg_nib_reoriented


def predict_with_voxtell(
    image_path: str,
    model_dir: str,
    text_prompts: List[str],
    device: Optional[torch.device] = None,
    output_dir: Optional[str] = None,
) -> dict:
    """
    Run VoxTell prediction on a medical image.
    
    Args:
        image_path: Path to the input medical image (NIfTI format)
        model_dir: Path to the VoxTell model directory
        text_prompts: List of text prompts (anatomical structures to segment)
        device: Torch device (default: cuda if available, else cpu)
        output_dir: Directory to save prediction results (optional)
        
    Returns:
        Dictionary containing segmentation masks and metadata
        
    Raises:
        FileNotFoundError: If image or model directory not found
        ValueError: If no text prompts provided
    """
    # Validate inputs
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    if not Path(model_dir).exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    if not text_prompts:
        raise ValueError("At least one text prompt must be provided")
    
    # Set device
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print(f"Device: {device}")
    print(f"Image: {image_path}")
    print(f"Model: {model_dir}")
    print(f"Text prompts: {text_prompts}")
    
    # Load image
    print("\nLoading image...")
    img, image_metadata = NibabelIOWithReorient().read_images([image_path])
    print(f"Image shape (reoriented to RAS): {img.shape}")
    
    # Initialize predictor
    print("\nInitializing predictor...")
    predictor = VoxTellPredictor(
        model_dir=model_dir,
        device=device,
    )
    
    # Run prediction
    print("\nRunning prediction...")
    voxtell_seg = predictor.predict_single_image(img, text_prompts)
    print(f"Prediction shape: {voxtell_seg.shape}")
    
    # Save results if output directory is specified
    results = {
        "segmentations": voxtell_seg,
        "image_path": image_path,
        "text_prompts": text_prompts,
        "device": str(device),
    }
    
    if output_dir:
        # Create subdirectory named after the input image
        image_name = Path(image_path).stem  # Get filename without extension
        output_subdir = os.path.join(output_dir, image_name)
        os.makedirs(output_subdir, exist_ok=True)
        
        print(f"\nSaving predictions to: {output_subdir}")
        
        # Save each segmentation mask
        for idx, prompt in enumerate(text_prompts):
            # Get segmentation data and reorient back to original space
            seg_data = voxtell_seg[idx].astype("uint8")
            
            # Reorient from RAS back to original orientation
            seg_nib_reoriented = reorient_seg_to_original(seg_data, image_metadata)
            
            # Save with sanitized prompt name
            safe_prompt = prompt.lower().replace(" ", "_").replace("/", "_")
            output_path = os.path.join(output_subdir, f"seg_{safe_prompt}.nii.gz")
            nib.save(seg_nib_reoriented, output_path)
            print(f"✓ Saved: {output_path}")
        
        # Save combined segmentation with unique labels for each structure
        # Background = 0, Structure 1 = 1, Structure 2 = 2, etc.
        combined_seg = np.zeros_like(voxtell_seg[0], dtype="uint8")
        for idx in range(voxtell_seg.shape[0]):
            # Assign label value = idx + 1 (to avoid overwriting with 0)
            combined_seg[voxtell_seg[idx] > 0] = idx + 1
        
        seg_combined_reoriented = reorient_seg_to_original(combined_seg, image_metadata)
        combined_path = os.path.join(output_subdir, "seg_combined.nii.gz")
        nib.save(seg_combined_reoriented, combined_path)
        print(f"✓ Saved: {combined_path}")
        
        # Also save a text file documenting the label mapping
        label_map_path = os.path.join(output_subdir, "label_mapping.txt")
        with open(label_map_path, "w") as f:
            f.write("Label mapping for combined segmentation:\n")
            f.write("=" * 50 + "\n")
            f.write("0 = Background\n")
            for idx, prompt in enumerate(text_prompts):
                f.write(f"{idx + 1} = {prompt}\n")
        print(f"✓ Saved: {label_map_path}")
        
        results["output_dir"] = output_subdir
    
    print("\n✓ Prediction completed!")
    return results


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(
        description="Run VoxTell predictions on medical images"
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Image filename in data directory (e.g., scan.nii.gz)"
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        required=True,
        help="Text prompts for segmentation (e.g., 'liver' 'kidney')"
    )
    parser.add_argument(
        "--device",
        choices=["gpu", "cpu"],
        default="gpu",
        help="Device to use: 'gpu' or 'cpu' (default: gpu)"
    )
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "gpu":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    
    # Construct full image path
    image_path = DATA_DIR / args.image
    
    # Validate directories and files
    print("=" * 70)
    print("VoxTell Prediction Script")
    print("=" * 70)
    
    if not DATA_DIR.exists():
        print(f"\n❌ Error: Data directory not found: {DATA_DIR}")
        print(f"Please create the directory and place your medical image there:")
        print(f"  mkdir -p {DATA_DIR}")
        print(f"  cp /path/to/your/image.nii.gz {DATA_DIR}/{args.image}")
        sys.exit(1)
    
    if not image_path.exists():
        print(f"\n❌ Error: Image not found: {image_path}")
        print(f"Please place your medical image in the data directory:")
        print(f"  cp /path/to/your/image.nii.gz {image_path}")
        print(f"\nAvailable files in {DATA_DIR}:")
        files = list(DATA_DIR.glob("*"))
        if files:
            for f in files:
                if f.is_file():
                    print(f"  - {f.name}")
        else:
            print("  (directory is empty)")
        sys.exit(1)
    
    if not MODEL_DIR.exists():
        print(f"\n❌ Error: Model directory not found: {MODEL_DIR}")
        print(f"Please download the model first:")
        print(f"  python download_checkpoint.py --download-dir {MODEL_DIR.parent}")
        sys.exit(1)
    
    # Print configuration
    print(f"\nConfiguration:")
    print(f"  Device:       {device}")
    print(f"  Image:        {image_path}")
    print(f"  Model:        {MODEL_DIR}")
    print(f"  Text Prompts: {args.prompts}")
    print(f"  Output Dir:   {OUTPUT_DIR}")
    
    try:
        results = predict_with_voxtell(
            image_path=str(image_path),
            model_dir=str(MODEL_DIR),
            text_prompts=args.prompts,
            device=device,
            output_dir=str(OUTPUT_DIR),
        )
        
        print(f"\n✓ Results saved to: {results['output_dir']}")
        print("=" * 70)
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
