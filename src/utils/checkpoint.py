from __future__ import annotations

from pathlib import Path
from typing import Any
import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    path: Path,
    *,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "network_weights": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    if metadata:
        payload.update(metadata)
    torch.save(payload, str(path))


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(str(path), map_location=map_location, weights_only=False)

