from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def crop_mask_to_bbox(mask: np.ndarray, bbox: Any) -> np.ndarray:
    if isinstance(bbox, list) and len(bbox) == 3:
        slices = tuple(slice(int(start), int(end)) for start, end in bbox)
        return mask[slices]
    if isinstance(bbox, tuple):
        return mask[bbox]
    return mask


def pad_spatial_tensor(tensor: torch.Tensor, target_size: Tuple[int, int, int]) -> torch.Tensor:
    spatial_shape = tensor.shape[-3:]
    pad_z = max(0, target_size[0] - spatial_shape[0])
    pad_y = max(0, target_size[1] - spatial_shape[1])
    pad_x = max(0, target_size[2] - spatial_shape[2])
    if pad_z or pad_y or pad_x:
        pad = (0, pad_x, 0, pad_y, 0, pad_z)
        return F.pad(tensor, pad)
    return tensor

def extract_random_image_patch(
    image_tensor: torch.Tensor,
    patch_size: Tuple[int, int, int],
    seed: None
) -> tuple[torch.Tensor, tuple[int, int, int]]:
    image_tensor = pad_spatial_tensor(image_tensor, patch_size)

    depth, height, width = image_tensor.shape[-3:]
    rng = np.random.default_rng(seed) if seed else np.random.default_rng()
    z0 = int(rng.integers(0, max(1, depth - patch_size[0] + 1)))
    y0 = int(rng.integers(0, max(1, height - patch_size[1] + 1)))
    x0 = int(rng.integers(0, max(1, width - patch_size[2] + 1)))

    img_patch = image_tensor[:, z0 : z0 + patch_size[0], y0 : y0 + patch_size[1], x0 : x0 + patch_size[2]]
    return img_patch, (z0, y0, x0)

def extract_mask_patch(
    mask_tensor: torch.Tensor,
    patch_size: Tuple[int, int, int],
    patch_center: Tuple[int, int, int],
) -> torch.Tensor:
    z0, y0, x0 = patch_center
    mask_tensor = pad_spatial_tensor(mask_tensor, patch_size)
    mask_patch = mask_tensor[z0 : z0 + patch_size[0], y0 : y0 + patch_size[1], x0 : x0 + patch_size[2]]
    return mask_patch
