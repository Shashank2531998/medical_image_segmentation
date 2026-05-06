from __future__ import annotations

import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image


def logits_to_segmentation(
    logits: torch.Tensor,
    bbox: tuple,
    orig_shape: tuple[int, ...],
    threshold: float = 0.5,
) -> np.ndarray:
    with torch.no_grad():
        prediction = torch.sigmoid(logits.float()) > threshold

    prediction = prediction.to("cpu").numpy().astype(np.uint8, copy=False)
    segmentation = np.zeros([prediction.shape[0], *orig_shape], dtype=np.uint8)
    return insert_crop_into_image(segmentation, prediction, bbox)
