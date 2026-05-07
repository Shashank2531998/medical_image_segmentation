from __future__ import annotations

from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader

from src.data.adapters import get_dataset_adapter
from src.data.dataset import TrainingSample, VoxTellDataset


class VoxTellDataModule:
    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.batch_size = cfg.get("batch_size", 1)
        self.patch_size = tuple(cfg.get("patch_size", (192, 192, 192)))
        self.dataset_name = cfg.get("name", "aeropath")
        self.train_root = cfg.get("train_root")
        self.val_root = cfg.get("val_root")
        self.val_fraction = float(cfg.get("val_fraction", 0.2))
        self.seed = int(cfg.get("seed", 42))
        self.train_max_cases = int(cfg.get("train_max_cases"))
        self.val_max_cases = int(cfg.get("val_max_cases"))

        if not self.train_root:
            raise ValueError("dataset.train_root (or dataset.root) must be provided")

        self.train_samples, self.val_samples = self._build_splits()

    def _build_splits(self) -> tuple[List[TrainingSample], List[TrainingSample]]:
        train_adapter = get_dataset_adapter(self.dataset_name, Path(self.train_root))

        if self.val_root:
            val_adapter = get_dataset_adapter(self.dataset_name, Path(self.val_root))
            val_items = val_adapter.build_training_samples()
        else:
            train_cases, val_cases = train_adapter.split_cases(val_fraction=self.val_fraction, seed=self.seed)
            train_items = train_adapter.build_training_samples(train_cases)
            val_items = train_adapter.build_training_samples(val_cases)

        train_items = train_items[:self.train_max_cases] if self.train_max_cases else train_items
        val_items = val_items[:self.val_max_cases] if self.val_max_cases else val_items
        return train_items, val_items

    def _collate_batch(self, batch):
        images = torch.stack([item["image"] for item in batch], dim=0)
        masks = torch.stack([item["mask"] for item in batch], dim=0)
        prompts = [item["prompts"] for item in batch]
        return {
            "image": images,
            "mask": masks,
            "prompts": prompts,
        }

    def train_dataloader(self) -> DataLoader:
        ds = VoxTellDataset(self.train_samples, patch_size=self.patch_size)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=True, collate_fn=self._collate_batch)

    def val_dataloader(self) -> DataLoader:
        ds = VoxTellDataset(self.val_samples, patch_size=self.patch_size)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False, collate_fn=self._collate_batch)
