import torch
import torch.nn.functional as F


def bce_loss_logits(pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # pred_logits: (B, N, D, H, W) or (B, C, ...); target: (B, D, H, W)
    # Simplified for single-mask per-sample targets
    if pred_logits.ndim > target.ndim:
        # reduce prompts dimension if present
        pred = pred_logits[:, 0]
    else:
        pred = pred_logits
    target = target.float()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, target)
    return loss


def dice_loss_logits(pred_logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if pred_logits.ndim > target.ndim:
        pred_logits = pred_logits[:, 0]

    pred = torch.sigmoid(pred_logits).float()
    target = target.float()

    if pred.shape != target.shape:
        target = F.interpolate(target.unsqueeze(1), size=pred.shape[1:], mode="nearest").squeeze(1)

    reduce_dims = tuple(range(1, pred.ndim))
    intersection = torch.sum(pred * target, dim=reduce_dims)
    denominator = torch.sum(pred, dim=reduce_dims) + torch.sum(target, dim=reduce_dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def combined_seg_loss_logits(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
) -> torch.Tensor:
    if pred_logits.ndim > target.ndim:
        pred = pred_logits[:, 0]
    else:
        pred = pred_logits

    tgt = target.float()
    if pred.shape != tgt.shape:
        tgt = F.interpolate(tgt.unsqueeze(1), size=pred.shape[1:], mode="nearest").squeeze(1)

    bce = F.binary_cross_entropy_with_logits(pred, tgt)
    dice = dice_loss_logits(pred_logits, target)
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
