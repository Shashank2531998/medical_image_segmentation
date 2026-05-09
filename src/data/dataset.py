from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient

from src.data.preprocess import preprocess_image
from src.data.utils import (
    build_prompt_mask_stack,
    crop_mask_to_bbox,
    extract_random_patch,
    sample_prompt_triplet,
)


@dataclass
class TrainingSample:
    image_path: Path
    prompts: Optional[List[str]] = None
    negatives: Optional[List[str]] = None
    mask_path: Path | None = None


class VoxTellDataset(Dataset):
    """Dataset for prompt-conditioned VoxTell training.

    Each sample provides:
    - preprocessed image tensor (C, X, Y, Z)
    - binary mask tensor (X, Y, Z)
    - text prompt string for embedding
    """

    def __init__(self, samples: List[TrainingSample], patch_size: Tuple[int, int, int] = (192, 192, 192)):
        self.samples = samples
        self.reader = NibabelIOWithReorient()
        self.patch_size = patch_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img, _ = self.reader.read_images([str(sample.image_path)])
        print(f"Image Path: {str(sample.image_path)}")
        print(f"Original Image Shape: {img.shape}")
        img_tensor, bbox, _ = preprocess_image(img)
        if sample.mask_path is not None:
            mask_arr, _ = self.reader.read_images([str(sample.mask_path)])
            print(f"Original Mask shape: {mask_arr.shape}")
            mask = mask_arr[0].astype(np.uint8)
            mask = crop_mask_to_bbox(mask, bbox)
        else:
            mask = np.zeros_like(img_tensor[0].numpy(), dtype=np.uint8)

        mask_tensor = torch.from_numpy(mask)

        prompts_out, prompt_is_positive = sample_prompt_triplet(sample.prompts, sample.negatives)
        print(f"Sample Negatives - {sample.negatives}")
        img_patch, mask_patch, (z0, y0, x0) = extract_random_patch(img_tensor, mask_tensor, self.patch_size)
        mask_patches = build_prompt_mask_stack(mask_patch, prompt_is_positive)

        print("===== DATASET DEBUG =====")
        print("Num samples:", len(self.samples))
        print("Patch size:", self.patch_size)

        print("Image tensor shape:", img_tensor.shape)
        print("Mask shape:", mask.shape)

        print("Mask unique values:", np.unique(mask))

        print("Prompts example:", prompts_out)

        print("Patch crop origin:", (z0, y0, x0))
        print("Patch shape image:", img_patch.shape)
        print("Patch shape mask:", mask_patch.shape)

        return {
            "image": img_patch,
            # list of masks aligned with `prompts` (positive, positive, negative)
            "mask": mask_patches,
            "prompts": prompts_out,
        }
