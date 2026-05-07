from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient

from src.data.preprocess import preprocess_image


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

        # Prompt sampling: support single prompt or lists provided by TrainingSample
        # Desired behavior: return 2 positive + 1 negative prompts per sample when possible
        prompts_out: List[str]
        if sample.prompts and isinstance(sample.prompts, (list, tuple)):
            positives = list(sample.prompts)
        else:
            positives = ["organ"]

        # choose two positives (with replacement if needed)
        if len(positives) >= 2:
            pos_choices = list(np.random.choice(positives, size=2, replace=False))
        else:
            pos_choices = [positives[0], positives[0]]

        # negative prompt: use provided negatives or synthesize
        if sample.negatives and len(sample.negatives) > 0:
            neg = str(np.random.choice(sample.negatives))
        else:
            neg = f"not {pos_choices[0]}"

        prompts_out = [pos_choices[0], pos_choices[1], neg]

        # Patch sampling: extract random patch of size self.patch_size from image and mask
        # img_tensor: (C, D, H, W); mask_tensor: (D, H, W)
        c, d, h, w = img_tensor.shape
        pz, py, px = self.patch_size

        def _pad_if_needed(t: torch.Tensor, target: Tuple[int, int, int]):
            _, dz, hy, wx = t.shape
            pad_z = max(0, target[0] - dz)
            pad_y = max(0, target[1] - hy)
            pad_x = max(0, target[2] - wx)
            if pad_z or pad_y or pad_x:
                pad = (0, pad_x, 0, pad_y, 0, pad_z)
                return torch.nn.functional.pad(t, pad)
            return t

        img_tensor = _pad_if_needed(img_tensor, (pz, py, px))
        mask_tensor = _pad_if_needed(mask_tensor.unsqueeze(0), (pz, py, px)).squeeze(0)

        _, d2, h2, w2 = img_tensor.shape
        z0 = int(np.random.randint(0, max(1, d2 - pz + 1)))
        y0 = int(np.random.randint(0, max(1, h2 - py + 1)))
        x0 = int(np.random.randint(0, max(1, w2 - px + 1)))

        img_patch = img_tensor[:, z0 : z0 + pz, y0 : y0 + py, x0 : x0 + px]
        mask_patch = mask_tensor[z0 : z0 + pz, y0 : y0 + py, x0 : x0 + px]

        return {
            "image": img_patch,
            "mask": mask_patch,
            "prompts": prompts_out,
        }
