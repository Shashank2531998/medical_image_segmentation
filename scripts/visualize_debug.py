#!/usr/bin/env python3
"""
Script to visualize debug artifacts saved during VoxTell training.

This script loads debug artifacts (.pt files) from a training experiment
and visualizes the data using napari for 3D visualization.

Usage:
    python visualize_debug.py --experiment experiments/exp_debug --epoch 1 --step 1

Arguments:
    --experiment: Path to experiment directory containing debug_artifacts/
    --epoch: Epoch number (default: 1)
    --step: Step number (default: 1)
    --backend: Visualization backend ('napari' or 'sitk', default: napari)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_debug_artifact(experiment_dir: Path, epoch: int, step: int) -> dict:
    """Load debug artifact for specific epoch and step."""
    debug_dir = experiment_dir / "debug_artifacts"
    if not debug_dir.exists():
        raise FileNotFoundError(f"Debug artifacts directory not found: {debug_dir}")

    filename = f"epoch_{epoch:02d}_step_{step:03d}.pt"
    artifact_path = debug_dir / filename

    if not artifact_path.exists():
        raise FileNotFoundError(f"Debug artifact not found: {artifact_path}")

    logger.info("Loading debug artifact: %s", artifact_path)
    data = torch.load(artifact_path, map_location='cpu')
    return data


def visualize_with_napari(data: dict, epoch: int, step: int, experiment_dir: Path):
    """Visualize debug data using napari."""
    try:
        import napari
    except ImportError:
        logger.error("napari not available. Install with: pip install napari[all]")
        return

    images = data['images'].numpy()  # (B, C, D, H, W)
    masks = data['masks'].numpy()    # (B, N, D, H, W)
    preds = data['preds'].numpy()    # (B, N, D, H, W)
    prompts = data['prompts']        # List of lists

    # Assume batch size 1 for simplicity
    if images.shape[0] != 1:
        logger.warning("Batch size > 1, visualizing only first sample")
    b = 0

    # Handle multi-channel images (take first channel)
    if images.shape[1] > 1:
        img = images[b, 0]  # First channel
    else:
        img = images[b, 0]

    # Normalize image to 0-1 range for better visualization
    img_min = img.min()
    img_max = img.max()
    if img_max > img_min:
        img_normalized = (img - img_min) / (img_max - img_min)
    else:
        img_normalized = img

    num_prompts = masks.shape[1]

    # Create napari viewer
    viewer = napari.Viewer()
    viewer.add_image(img_normalized, name='image', colormap='gray')

    for p in range(num_prompts):
        prompt = prompts[b][p] if b < len(prompts) and p < len(prompts[b]) else f"prompt_{p}"

        # Get data for this prompt
        mask = masks[b, p]
        pred = preds[b, p]

        # Add to viewer
        viewer.add_labels(mask.astype(np.uint8), name=f'gt_{prompt}')
        viewer.add_labels(pred.astype(np.uint8), name=f'pred_{prompt}')

    logger.info("Starting napari viewer...")
    logger.info("Layers: image (gray), gt_* (ground truth masks), pred_* (predictions)")
    napari.run()


def visualize_with_sitk(data: dict, epoch: int, step: int, experiment_dir: Path):
    """
    Save debug artifacts as NIfTI files for viewing in ITK-SNAP or 3D Slicer.
    """
    try:
        import SimpleITK as sitk
    except ImportError:
        logger.error("SimpleITK not available. Install with: pip install SimpleITK")
        return

    from pathlib import Path

    images = data['images'].numpy()   # (B, C, D, H, W)
    masks = data['masks'].numpy()     # (B, N, D, H, W)
    preds = data['preds'].numpy()     # (B, N, D, H, W)
    prompts = data['prompts']

    logger.info("Saving visualization volumes...")

    output_dir = experiment_dir / "debug_artifacts" / f"debug_vis_epoch_{epoch:02d}_step_{step:03d}"
    output_dir.mkdir(exist_ok=True, parents=True)

    # Assume batch size 1
    b = 0

    # Use first image channel
    image = images[b, 0]

    # Normalize image for visualization
    image = image.astype(np.float32)

    img_sitk = sitk.GetImageFromArray(image)

    image_path = output_dir / "image.nii.gz"
    sitk.WriteImage(img_sitk, str(image_path))

    logger.info("Saved image: %s", image_path)

    num_prompts = masks.shape[1]

    for p in range(num_prompts):

        prompt = (
            prompts[b][p]
            if b < len(prompts) and p < len(prompts[b])
            else f"prompt_{p}"
        )

        prompt = str(prompt).replace(" ", "_")

        gt = masks[b, p].astype(np.uint8)
        pred = preds[b, p].astype(np.uint8)

        gt_sitk = sitk.GetImageFromArray(gt)
        pred_sitk = sitk.GetImageFromArray(pred)

        gt_path = output_dir / f"gt_{p}_{prompt}.nii.gz"
        pred_path = output_dir / f"pred_{p}_{prompt}.nii.gz"

        sitk.WriteImage(gt_sitk, str(gt_path))
        sitk.WriteImage(pred_sitk, str(pred_path))

        logger.info("Saved GT: %s", gt_path)
        logger.info("Saved prediction: %s", pred_path)

        logger.info(
            "Prompt '%s': GT voxels=%d, Pred voxels=%d",
            prompt,
            gt.sum(),
            pred.sum(),
        )

    logger.info("Done.")
    logger.info("Open .nii.gz files in ITK-SNAP or 3D Slicer.")


def visualize_debug_artifact(
    experiment_dir: Path,
    data: dict,
    epoch: int,
    step: int,
    backend: str = 'napari'
):
    """Visualize debug artifact using specified backend."""
    logger.info("Data shapes:")
    logger.info("  Images: %s", data['images'].shape)
    logger.info("  Masks: %s", data['masks'].shape)
    logger.info("  Logits: %s", data['logits'].shape)
    logger.info("  Probs: %s", data['probs'].shape)
    logger.info("  Preds: %s", data['preds'].shape)
    logger.info("  Prompts: %s", data['prompts'])

    # Print statistics
    images = data['images'].numpy()
    masks = data['masks'].numpy()
    preds = data['preds'].numpy()

    logger.info("Image stats: min=%.4f, max=%.4f, mean=%.4f, std=%.4f",
                images.min(), images.max(), images.mean(), images.std())
    logger.info("Mask unique values: %s", np.unique(masks))
    logger.info("Preds unique values: %s", np.unique(preds))

    if backend == 'napari':
        visualize_with_napari(data, epoch, step, experiment_dir)
    elif backend == 'sitk':
        visualize_with_sitk(data, epoch, step, experiment_dir)
    else:
        logger.error("Unknown backend: %s. Use 'napari' or 'sitk'", backend)


def main():
    parser = argparse.ArgumentParser(description="Visualize VoxTell debug artifacts")
    parser.add_argument("--experiment", required=True, help="Path to experiment directory")
    parser.add_argument("--epoch", type=int, default=1, help="Epoch number")
    parser.add_argument("--step", type=int, default=1, help="Step number")
    parser.add_argument("--backend", choices=['napari', 'sitk'], default='napari',
                       help="Visualization backend (default: napari). Install napari with: pip install napari[all]")

    args = parser.parse_args()

    experiment_dir = Path(args.experiment)
    if not experiment_dir.exists():
        logger.error("Experiment directory not found: %s", experiment_dir)
        sys.exit(1)

    try:
        data = load_debug_artifact(experiment_dir, args.epoch, args.step)
        visualize_debug_artifact(experiment_dir, data, args.epoch, args.step, args.backend)
    except Exception as e:
        logger.error("Error processing debug artifact: %s", e)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()