from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient

from src.data.preprocess import preprocess_image
from src.data.utils import (
    crop_mask_to_bbox,
    extract_foreground_image_patch,
    extract_random_image_patch,
    extract_mask_patch
)


@dataclass
class TrainingSample:
    image_path: Path
    prompts: Optional[List[str]] = None
    mask_paths: list[Path | None] = field(default_factory=list)


class VoxTellDataset(Dataset):
    """Dataset for prompt-conditioned VoxTell training.

    Each sample provides:
    - preprocessed image tensor (C, X, Y, Z)
    - binary mask tensor (X, Y, Z)
    - text prompt string for embedding
    """

    def __init__(
        self,
        samples: List[TrainingSample],
        mode: str = "train",
        patch_size: Tuple[int, int, int] = (192, 192, 192),
        seed=None,
        foreground_patch_fraction: float = 0.85,
    ):
        self.samples = samples
        self.reader = NibabelIOWithReorient()
        self.patch_size = patch_size
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.foreground_patch_fraction = foreground_patch_fraction

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img, _ = self.reader.read_images([str(sample.image_path)])
        img_tensor, bbox, _ = preprocess_image(img)
        prompts = sample.prompts

        masks = []
        for mask_path in sample.mask_paths:
            if mask_path is None:
                mask = torch.zeros_like(img_tensor[0], dtype=torch.float32)
            else:
                mask_arr, _ = self.reader.read_images([str(mask_path)])
                mask = mask_arr[0].astype(np.float32)
                mask = torch.from_numpy(crop_mask_to_bbox(mask, bbox))

            masks.append(mask)

        patch_start = None
        if self.mode == "train":
            foreground_masks = [mask for mask in masks if bool(mask.any())]
            use_foreground_patch = (
                self.foreground_patch_fraction > 0.0
                and len(foreground_masks) > 0
                and float(self.rng.random()) < self.foreground_patch_fraction
            )
            if use_foreground_patch:
                img_tensor, patch_start = extract_foreground_image_patch(
                    img_tensor,
                    foreground_masks,
                    self.patch_size,
                    rng=self.rng,
                )
            else:
                img_tensor, patch_start = extract_random_image_patch(
                    img_tensor,
                    self.patch_size,
                    rng=self.rng,
                )

            masks = [extract_mask_patch(mask, self.patch_size, patch_start) for mask in masks]

        mask_tensor_list = torch.stack(masks)

        return {
            "image": img_tensor,
            "masks": mask_tensor_list,
            "prompts": prompts,
            "img_path": str(sample.image_path)
        }
