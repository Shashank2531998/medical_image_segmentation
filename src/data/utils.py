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


def sample_prompt_triplet(
    prompts: Optional[Sequence[str]],
    negatives: Optional[Sequence[str]],
    default_positive: str = "organ",
) -> tuple[list[str], list[bool]]:
    if prompts:
        positive_pool = list(prompts)
    else:
        positive_pool = [default_positive]

    if len(positive_pool) >= 2:
        positive_prompts = list(np.random.choice(positive_pool, size=2, replace=False))
    else:
        positive_prompts = [positive_pool[0], positive_pool[0]]

    if negatives:
        negative_prompt = str(np.random.choice(negatives))
    else:
        negative_prompt = f"not {positive_prompts[0]}"

    prompt_texts = [positive_prompts[0], positive_prompts[1], negative_prompt]
    prompt_is_positive = [True, True, False]
    return prompt_texts, prompt_is_positive


def pad_spatial_tensor(tensor: torch.Tensor, target_size: Tuple[int, int, int]) -> torch.Tensor:
    spatial_shape = tensor.shape[-3:]
    pad_z = max(0, target_size[0] - spatial_shape[0])
    pad_y = max(0, target_size[1] - spatial_shape[1])
    pad_x = max(0, target_size[2] - spatial_shape[2])
    if pad_z or pad_y or pad_x:
        pad = (0, pad_x, 0, pad_y, 0, pad_z)
        return F.pad(tensor, pad)
    return tensor


def extract_random_patch(
    image_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
    patch_size: Tuple[int, int, int],
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int, int]]:
    image_tensor = pad_spatial_tensor(image_tensor, patch_size)
    mask_tensor = pad_spatial_tensor(mask_tensor, patch_size)

    depth, height, width = image_tensor.shape[-3:]
    rng = np.random.default_rng(seed)
    z0 = int(rng.integers(0, max(1, depth - patch_size[0] + 1)))
    y0 = int(rng.integers(0, max(1, height - patch_size[1] + 1)))
    x0 = int(rng.integers(0, max(1, width - patch_size[2] + 1)))

    img_patch = image_tensor[:, z0 : z0 + patch_size[0], y0 : y0 + patch_size[1], x0 : x0 + patch_size[2]]
    mask_patch = mask_tensor[z0 : z0 + patch_size[0], y0 : y0 + patch_size[1], x0 : x0 + patch_size[2]]
    return img_patch, mask_patch, (z0, y0, x0)


def extract_random_image_patch(
    image_tensor: torch.Tensor,
    patch_size: Tuple[int, int, int],
    seed: int = 42,
) -> tuple[torch.Tensor, tuple[int, int, int]]:
    image_tensor = pad_spatial_tensor(image_tensor, patch_size)

    depth, height, width = image_tensor.shape[-3:]
    rng = np.random.default_rng(seed)
    z0 = int(rng.integers(0, max(1, depth - patch_size[0] + 1)))
    y0 = int(rng.integers(0, max(1, height - patch_size[1] + 1)))
    x0 = int(rng.integers(0, max(1, width - patch_size[2] + 1)))

    img_patch = image_tensor[:, z0 : z0 + patch_size[0], y0 : y0 + patch_size[1], x0 : x0 + patch_size[2]]
    return img_patch, (z0, y0, x0)


def build_prompt_mask_stack(mask_patch: torch.Tensor, prompt_is_positive: Sequence[bool]) -> torch.Tensor:
    positive_mask = mask_patch.clone()
    empty_mask = torch.zeros_like(mask_patch)
    masks = [positive_mask if is_positive else empty_mask for is_positive in prompt_is_positive]
    return torch.stack(masks)