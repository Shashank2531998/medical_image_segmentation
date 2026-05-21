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


def _sample_patch_start(
    spatial_shape: tuple[int, int, int],
    patch_size: Tuple[int, int, int],
    center: tuple[int, int, int],
) -> tuple[int, int, int]:
    depth, height, width = spatial_shape
    center_z, center_y, center_x = center
    z0 = min(max(0, center_z - patch_size[0] // 2), depth - patch_size[0])
    y0 = min(max(0, center_y - patch_size[1] // 2), height - patch_size[1])
    x0 = min(max(0, center_x - patch_size[2] // 2), width - patch_size[2])
    return z0, y0, x0

def extract_random_image_patch(
    image_tensor: torch.Tensor,
    patch_size: Tuple[int, int, int],
    rng: np.random.Generator | None = None,
) -> tuple[torch.Tensor, tuple[int, int, int]]:
    image_tensor = pad_spatial_tensor(image_tensor, patch_size)

    depth, height, width = image_tensor.shape[-3:]
    rng = rng if rng is not None else np.random.default_rng()
    z0 = int(rng.integers(0, max(1, depth - patch_size[0] + 1)))
    y0 = int(rng.integers(0, max(1, height - patch_size[1] + 1)))
    x0 = int(rng.integers(0, max(1, width - patch_size[2] + 1)))

    img_patch = image_tensor[:, z0 : z0 + patch_size[0], y0 : y0 + patch_size[1], x0 : x0 + patch_size[2]]
    return img_patch, (z0, y0, x0)


def extract_foreground_image_patch(
    image_tensor: torch.Tensor,
    foreground_masks: Sequence[torch.Tensor],
    patch_size: Tuple[int, int, int],
    rng: np.random.Generator | None = None,
) -> tuple[torch.Tensor, tuple[int, int, int]]:
    image_tensor = pad_spatial_tensor(image_tensor, patch_size)

    eligible_masks = []
    for mask in foreground_masks:
        mask = pad_spatial_tensor(mask, patch_size)
        if bool(mask.any()):
            eligible_masks.append(mask)

    if len(eligible_masks) == 0:
        return extract_random_image_patch(image_tensor, patch_size, rng=rng)

    rng = rng if rng is not None else np.random.default_rng()
    selected_mask = eligible_masks[int(rng.integers(0, len(eligible_masks)))]
    foreground_voxels = torch.nonzero(selected_mask > 0, as_tuple=False)
    voxel_index = int(rng.integers(0, foreground_voxels.shape[0]))
    center = tuple(int(value) for value in foreground_voxels[voxel_index].tolist())
    patch_start = _sample_patch_start(image_tensor.shape[-3:], patch_size, center)

    img_patch = image_tensor[
        :,
        patch_start[0] : patch_start[0] + patch_size[0],
        patch_start[1] : patch_start[1] + patch_size[1],
        patch_start[2] : patch_start[2] + patch_size[2],
    ]
    return img_patch, patch_start

def extract_mask_patch(
    mask_tensor: torch.Tensor,
    patch_size: Tuple[int, int, int],
    patch_start: Tuple[int, int, int],
) -> torch.Tensor:
    z0, y0, x0 = patch_start
    mask_tensor = pad_spatial_tensor(mask_tensor, patch_size)
    mask_patch = mask_tensor[z0 : z0 + patch_size[0], y0 : y0 + patch_size[1], x0 : x0 + patch_size[2]]
    return mask_patch
