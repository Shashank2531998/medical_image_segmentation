import torch


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
