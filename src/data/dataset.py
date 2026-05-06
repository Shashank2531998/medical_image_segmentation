from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient


class VoxTellDataset(Dataset):
    """Minimal dataset wrapper that yields preprocessed images and dummy targets.

    This is a lightweight placeholder intended to be replaced by a patch-based
    training dataset. It reads NIfTI images and returns the image volume as a
    float32 tensor and a zero mask as a placeholder target.
    """

    def __init__(self, image_paths: List[Path]):
        self.paths = [Path(p) for p in image_paths]
        self.reader = NibabelIOWithReorient()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img, props = self.reader.read_images([str(self.paths[idx])])
        # img: shape (C, X, Y, Z) or (1, X, Y, Z)
        img = img.astype('float32')
        # return image and dummy mask
        mask = np.zeros_like(img[0], dtype=np.uint8)
        img_tensor = torch.from_numpy(img)
        mask_tensor = torch.from_numpy(mask)
        return img_tensor, mask_tensor
