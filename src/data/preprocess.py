from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from nnunetv2.preprocessing.normalization.default_normalization_schemes import ZScoreNormalization


_normalizer = ZScoreNormalization(intensityproperties={})


def preprocess_image(image: np.ndarray) -> Tuple[torch.Tensor, Tuple, Tuple[int, ...]]:
    if image.ndim == 3:
        image = image[None]
    image = image.astype('float32')
    original_shape = image.shape[1:]
    image, _, bbox = crop_to_nonzero(image, None)
    image = _normalizer.run(image, None)
    return torch.from_numpy(image), bbox, original_shape
