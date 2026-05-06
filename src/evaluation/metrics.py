from __future__ import annotations

from typing import Iterable, Union

import numpy as np


def _to_bool_array(array: np.ndarray | Iterable) -> np.ndarray:
    """Convert input to boolean numpy array."""
    arr = np.asarray(array)
    if arr.dtype == bool:
        return arr
    return arr > 0


def dice_coefficient(
    prediction: Union[np.ndarray, Iterable],
    target: Union[np.ndarray, Iterable],
    empty_score: float = 1.0,
) -> float:
    """
    Compute the Dice coefficient between prediction and target segmentation masks.
    
    The Dice coefficient is defined as:
        Dice = 2 * |A ∩ B| / (|A| + |B|)
    
    where A is the prediction and B is the target.
    
    Args:
        prediction: Predicted segmentation mask (binary or with values > 0 indicating foreground)
        target: Ground truth segmentation mask
        empty_score: Score to return when both prediction and target are empty
        
    Returns:
        Dice coefficient (1.0 if both are empty, otherwise 0 to 1)
    """
    pred = _to_bool_array(prediction)
    ref = _to_bool_array(target)

    pred_sum = int(pred.sum())
    ref_sum = int(ref.sum())

    # Handle empty case: both prediction and target are empty
    if pred_sum == 0 and ref_sum == 0:
        return float(empty_score)

    # Calculate intersection
    intersection = int(np.logical_and(pred, ref).sum())
    
    # Calculate denominator
    denominator = pred_sum + ref_sum
    
    if denominator == 0:
        return float(empty_score)
    
    # Dice coefficient formula: 2 * |A ∩ B| / (|A| + |B|)
    return float(2.0 * intersection / denominator)


def iou_score(
    prediction: Union[np.ndarray, Iterable],
    target: Union[np.ndarray, Iterable],
    empty_score: float = 1.0,
) -> float:
    """
    Compute the Intersection over Union (IoU) score between prediction and target.
    
    IoU is defined as:
        IoU = |A ∩ B| / |A ∪ B|
    
    Args:
        prediction: Predicted segmentation mask
        target: Ground truth segmentation mask
        empty_score: Score to return when both prediction and target are empty
        
    Returns:
        IoU score (1.0 if both are empty, otherwise 0 to 1)
    """
    pred = _to_bool_array(prediction)
    ref = _to_bool_array(target)

    pred_sum = int(pred.sum())
    ref_sum = int(ref.sum())

    # Handle empty case
    if pred_sum == 0 and ref_sum == 0:
        return float(empty_score)

    # Calculate intersection and union
    intersection = int(np.logical_and(pred, ref).sum())
    union = int(np.logical_or(pred, ref).sum())
    
    if union == 0:
        return float(empty_score)
    
    return float(intersection / union)
