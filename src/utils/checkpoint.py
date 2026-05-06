from __future__ import annotations

from pathlib import Path
import torch


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "network_weights": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, str(path))


def load_checkpoint(path: Path):
    data = torch.load(str(path), map_location=torch.device('cpu'))
    return data
