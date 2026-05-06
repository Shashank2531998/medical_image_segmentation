from pathlib import Path
from typing import List

import numpy as np
import nibabel
import torch

from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient

from src.inference.predictor import VoxTellPredictor
from src.utils.reorientation import reorient_seg_from_props


# ============================================================================
# IO HELPERS
# ============================================================================

def get_reader_writer(file_path: str):
    suffix = Path(file_path).suffix.lower()

    if suffix in [".nii", ".gz"]:
        return NibabelIOWithReorient()

    raise ValueError(
        f"Unsupported file format: {suffix}. Only NIfTI supported."
    )


def save_segmentation(segmentation, output_file: Path):
    """
    NOTE: assumes segmentation already has correct affine if needed.
    """
    nibabel.save(segmentation, output_file)


def save_all_segmentations(
    segmentations,
    output_folder: Path,
    input_path: Path,
    prompts: List[str],
    save_combined: bool = False,
    verbose: bool = False,
):

    output_folder.mkdir(parents=True, exist_ok=True)

    input_filename = input_path.stem
    if input_filename.endswith(".nii"):
        input_filename = input_filename[:-4]

    suffix = ".nii.gz" if input_path.suffix == ".gz" else input_path.suffix

    output_files = []

    if save_combined:
        if len(prompts) > 1 and verbose:
            print("WARNING: combining multi-label segmentation")

        if len(prompts) == 1:
            out_file = output_folder / f"{input_filename}{suffix}"
            save_segmentation(segmentations[0], out_file)
        else:
            combined = np.zeros_like(segmentations[0], dtype=np.uint8)

            for i, seg in enumerate(segmentations):
                combined[seg > 0] = i + 1

            out_file = output_folder / f"{input_filename}{suffix}"
            save_segmentation(combined, out_file)

        output_files.append(out_file)

    else:
        for i, prompt in enumerate(prompts):
            safe = "".join(
                c if c.isalnum() or c in (" ", "_") else "_"
                for c in prompt
            ).replace(" ", "_")

            out_file = output_folder / f"{input_filename}_{safe}{suffix}"
            save_segmentation(segmentations[i], out_file)
            output_files.append(out_file)

    return output_files


# ============================================================================
# INFERENCE CORE
# ============================================================================

def predict_image(
    predictor,
    input_path: str | Path,
    prompts: List[str],
    verbose: bool = False,
):

    input_path = Path(input_path)

    if verbose:
        print(f"Loading image: {input_path}")

    reader = get_reader_writer(str(input_path))
    img, props = reader.read_images([str(input_path)])

    if verbose:
        print(f"Image shape: {img.shape}")
        print(f"Prompts: {prompts}")

    if verbose:
        print("Running prediction...")

    segmentations = predictor.predict_single_image(img, prompts)

    segmentations = [
        reorient_seg_from_props(seg, props)
        for seg in segmentations
    ]

    if verbose:
        print("Prediction completed")

    return segmentations


def get_predictor(model_path, device):
    model_path = Path(model_path)

    if not (model_path / "plans.json").exists():
        raise FileNotFoundError("plans.json missing")

    if not (model_path / "fold_0" / "checkpoint_final.pth").exists():
        raise FileNotFoundError("checkpoint missing")

    print(f"Loading model from {model_path}")

    return VoxTellPredictor(
        model_dir=str(model_path),
        device=device,
    )
