from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import torch
from torch.utils.data import DataLoader

from src.data.dataset import VoxTellDataset


class VoxTellDataModule:
    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.batch_size = cfg.get("batch_size", 1)
        self.num_workers = cfg.get("num_workers", 0)
        self.train_paths = cfg.get("train_paths", [])
        self.val_paths = cfg.get("val_paths", [])

    def train_dataloader(self) -> DataLoader:
        ds = VoxTellDataset(self.train_paths)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        ds = VoxTellDataset(self.val_paths)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
