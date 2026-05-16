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

    def __init__(self, samples: List[TrainingSample], mode: str = "train", patch_size: Tuple[int, int, int] = (192, 192, 192), seed=None):
        self.samples = samples
        self.reader = NibabelIOWithReorient()
        self.patch_size = patch_size
        self.mode = mode
        self.seed = seed

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img, _ = self.reader.read_images([str(sample.image_path)])
        img_tensor, bbox, _ = preprocess_image(img)
        prompts = sample.prompts

        if self.mode == "train":
            img_tensor, patch_center = extract_random_image_patch(img_tensor, self.patch_size, self.seed)

        masks = []
        for mask_path in sample.mask_paths:
            
            if mask_path is None:
                mask = torch.zeros_like(img_tensor[0])
            else:
                mask_arr, _ = self.reader.read_images([str(mask_path)])
                mask = mask_arr[0].astype(np.uint8)
                mask = torch.from_numpy(crop_mask_to_bbox(mask, bbox))
                if self.mode == "train":
                    mask = extract_mask_patch(mask, self.patch_size, patch_center)

            masks.append(mask)

        mask_tensor_list = torch.stack(masks)

        return {
            "image": img_tensor,
            "masks": mask_tensor_list,
            "prompts": prompts,
            "img_path": str(sample.image_path)
        }
