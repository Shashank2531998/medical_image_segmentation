from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import numpy as np
import torch
from torch.utils.data import Dataset

from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient

from src.data.preprocess import preprocess_image


@dataclass(frozen=True)
class TrainingSample:
    image_path: Path
    prompt: str
    mask_path: Path | None = None


class VoxTellDataset(Dataset):
    """Dataset for prompt-conditioned VoxTell training.

    Each sample provides:
    - preprocessed image tensor (C, X, Y, Z)
    - binary mask tensor (X, Y, Z)
    - text prompt string for embedding
    """

    def __init__(self, samples: List[TrainingSample]):
        self.samples = samples
        self.reader = NibabelIOWithReorient()

    def __len__(self) -> int:
        return len(self.samples)

    def _crop_mask_to_bbox(self, mask: np.ndarray, bbox: Any) -> np.ndarray:
        if isinstance(bbox, list) and len(bbox) == 3:
            slices = tuple(slice(int(start), int(end)) for start, end in bbox)
            return mask[slices]
        if isinstance(bbox, tuple):
            return mask[bbox]
        return mask

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img, _ = self.reader.read_images([str(sample.image_path)])
        img_tensor, bbox, _ = preprocess_image(img)

        if sample.mask_path is not None:
            mask_arr, _ = self.reader.read_images([str(sample.mask_path)])
            mask = mask_arr[0].astype(np.uint8)
            mask = self._crop_mask_to_bbox(mask, bbox)
        else:
            mask = np.zeros_like(img_tensor[0].numpy(), dtype=np.uint8)

        mask_tensor = torch.from_numpy(mask)
        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "prompt": sample.prompt,
        }
