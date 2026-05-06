from __future__ import annotations

from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader

from src.data.dataset import TrainingSample, VoxTellDataset


class VoxTellDataModule:
    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.batch_size = cfg.get("batch_size", 1)
        self.num_workers = cfg.get("num_workers", 0)
        self.train_samples = self._build_samples(cfg.get("train_samples", []), cfg.get("train_paths", []))
        self.val_samples = self._build_samples(cfg.get("val_samples", []), cfg.get("val_paths", []))

    def _build_samples(self, configured_samples: list[dict], legacy_paths: list[str]) -> List[TrainingSample]:
        samples: List[TrainingSample] = []

        for entry in configured_samples:
            samples.append(
                TrainingSample(
                    image_path=Path(entry["image"]),
                    mask_path=Path(entry["mask"]) if entry.get("mask") else None,
                    prompt=entry.get("prompt", "organ"),
                )
            )

        for path in legacy_paths:
            samples.append(TrainingSample(image_path=Path(path), prompt="organ", mask_path=None))

        return samples

    def train_dataloader(self) -> DataLoader:
        ds = VoxTellDataset(self.train_samples)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        ds = VoxTellDataset(self.val_samples)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
