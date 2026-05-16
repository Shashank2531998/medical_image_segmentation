import torch
import torch.nn.functional as F
from typing import List, Tuple

import monai


def _stack_target_if_list(target):
    # support a list of per-prompt masks (for a single sample)
    if isinstance(target, (list, tuple)) and all(isinstance(t, torch.Tensor) for t in target):
        stacked = torch.stack(target, dim=1)  # (N, D, H, W) -> (1, N, D, H, W) if needed
        if stacked.ndim == 4:
            stacked = stacked.unsqueeze(0)
        return stacked
    return target


def _align_and_resize_target(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return target tensor shaped like pred for elementwise loss.

    pred: (B, N, ... ) or (B, ...)
    target: (B, N, ...), (B, ...), or list -> handled by _stack_target_if_list
    """
    target = _stack_target_if_list(target)

    # ensure tensors
    if not isinstance(target, torch.Tensor):
        target = torch.tensor(target, dtype=pred.dtype, device=pred.device)

    # if pred has prompt dim but target doesn't, expand target per-prompt
    if pred.ndim == target.ndim + 1:
        # pred: (B, N, D, H, W), target: (B, D, H, W)
        target = target.unsqueeze(1).expand(-1, pred.shape[1], *target.shape[1:])

    # if shapes differ in spatial dims, resize target to match pred spatial size
    if pred.ndim == target.ndim:
        # either both (B, N, D, H, W) or both (B, D, H, W)
        if pred.shape[1:] != target.shape[1:]:
            # handle both cases by reshaping to (B*N, 1, D, H, W) for interpolate
            if pred.ndim == 5:
                B, N = target.shape[0], target.shape[1]
                t = target.reshape(B * N, 1, *target.shape[-3:]).float()
                t = F.interpolate(t, size=pred.shape[2:], mode="nearest")
                target = t.reshape(B, N, *pred.shape[2:]).to(pred.dtype)
            else:
                t = target.unsqueeze(1).float()
                t = F.interpolate(t, size=pred.shape[1:], mode="nearest")
                target = t.squeeze(1).to(pred.dtype)

    return target.float()


def bce_loss_logits(pred_logits: torch.Tensor, target) -> torch.Tensor:
    """Binary cross-entropy for logits supporting prompt dimension.

    pred_logits: (B, N, D, H, W) or (B, D, H, W)
    target: tensor or list aligned with prompts; can be (B, N, D, H, W) or (B, D, H, W)
    """
    pred = pred_logits
    target = _align_and_resize_target(pred, target)

    loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")

    # average over spatial dims then over prompts and batch
    if pred.ndim == 5:
        spatial_dims = tuple(range(2, pred.ndim))
    else:
        spatial_dims = tuple(range(1, pred.ndim))

    loss = loss.mean(dim=spatial_dims)
    return loss.mean()


def dice_coefficient(pred: torch.Tensor,
                     target: torch.Tensor) -> torch.Tensor:
    """
    Computes Dice coefficient between prediction and target.
    """
    if pred.ndim != target.ndim or pred.ndim != 4:
        raise ValueError(f"pred ndim must be 4 (B, H, W, D), got {pred.ndim}")
    return monai.metrics.compute_meandice(pred, target)


def dice_loss_logits(pred_logits: torch.Tensor, target, eps: float = 1e-6) -> torch.Tensor:
    target = _align_and_resize_target(pred_logits, target)
    if pred_logits.ndim != target.ndim or pred_logits.ndim != 5:
        raise ValueError(f"pred ndim must be 5 (B, N, H, W, D), got {pred_logits.ndim}")
    return monai.losses.DiceLoss(include_background=True, sigmoid=True, reduction="mean")(
        pred_logits.float(), target.float()
    )


def combined_seg_loss_logits(
    pred_logits: torch.Tensor,
    target,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
) -> torch.Tensor:
    # Do not reduce prompt dimension here; align/resize target accordingly
    tgt = _align_and_resize_target(pred_logits, target)

    bce = F.binary_cross_entropy_with_logits(pred_logits, tgt, reduction="none")
    # average spatial dims
    if pred_logits.ndim == 5:
        spatial_dims = tuple(range(2, pred_logits.ndim))
    else:
        spatial_dims = tuple(range(1, pred_logits.ndim))
    bce = bce.mean(dim=spatial_dims).mean()

    dice = dice_loss_logits(pred_logits, tgt)
    return bce_weight * bce + dice_weight * dice


def deep_supervision_loss(
    predictions,
    target: torch.Tensor,
    weights: list[float] | None = None,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
) -> torch.Tensor:
    if not isinstance(predictions, (list, tuple)):
        return combined_seg_loss_logits(predictions, target, bce_weight=bce_weight, dice_weight=dice_weight)

    if len(predictions) == 0:
        raise ValueError("predictions list is empty")

    if weights is None:
        weights = [1.0 / (2 ** i) for i in range(len(predictions))]

    if len(weights) != len(predictions):
        raise ValueError("deep supervision weights length must match number of predictions")

    total_weight = float(sum(weights))
    total_loss = predictions[0].new_tensor(0.0)
    for pred, weight in zip(predictions, weights):
        total_loss = total_loss + float(weight) * combined_seg_loss_logits(
            pred,
            target,
            bce_weight=bce_weight,
            dice_weight=dice_weight,
        )

    return total_loss / max(total_weight, 1e-8)
