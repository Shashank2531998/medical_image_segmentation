from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.adapters import get_dataset_adapter
from src.data.dataset import VoxTellDataset

from src.utils.logging import get_logger

logger = get_logger(__name__)


class VoxTellDataModule:
    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.batch_size = cfg.get("batch_size", 1)
        self.patch_size = tuple(cfg.get("patch_size", (192, 192, 192)))

        self.dataset_name = cfg.get("name", "aeropath")
        self.train_root = cfg.get("root")
        # keep compatibility: `root` attribute expected by other code
        self.root = self.train_root

        # fractions for split: train_fraction and val_fraction
        self.train_fraction = float(cfg.get("train_fraction", 0.7))
        self.val_fraction = float(cfg.get("val_fraction", 0.15))
        self.foreground_patch_fraction = float(cfg.get("foreground_patch_fraction", 0.85))

        self.seed = cfg.get("seed")

        self.train_max_cases = cfg.get("train_max_cases")
        self.val_max_cases = cfg.get("val_max_cases")
        self.test_max_cases = cfg.get("test_max_cases")
        
        self.filter_img_list = cfg.get("filter_img_list")

        if not (self.train_root):
            raise ValueError("dataset.root must be provided")

        self.train_samples, self.val_samples, self.test_samples = self._build_splits()

    def _build_splits(self):
        adapter = get_dataset_adapter(
            self.dataset_name,
            Path(self.root),
        )

        all_cases = adapter.cases()

        rng = random.Random(self.seed)
        rng.shuffle(all_cases)

        n = len(all_cases)

        n_train = round(n * self.train_fraction)
        n_val = round(n * self.val_fraction)
        n_test  = n - n_train - n_val 

        train_cases = all_cases[:n_train]
        val_cases = all_cases[n_train:n_train + n_val]
        test_cases = all_cases[n_train + n_val:n_train + n_val + n_test]

        logger.info(f"Train/Val/Test Split: {len(train_cases)}/{len(val_cases)}/{len(test_cases)}")

        train_items = adapter.build_training_samples(
            train_cases,
            max_cases=self.train_max_cases,
        )

        val_items = adapter.build_training_samples(
            val_cases,
            max_cases=self.val_max_cases,
        )

        test_items = adapter.build_training_samples(
            test_cases,
            max_cases=self.test_max_cases,
            filter_img_list=self.filter_img_list,
        )

        return train_items, val_items, test_items

    def _collate_batch(self, batch):
        images = torch.stack([item["image"] for item in batch], dim=0)
        masks = torch.stack([item["masks"] for item in batch], dim=0)
        prompts = [item["prompts"] for item in batch]
        return {
            "image": images,
            "masks": masks,
            "prompts": prompts,
            "img_path": [item["img_path"] for item in batch]
        }

    def train_dataloader(self) -> DataLoader:
        ds = VoxTellDataset(
            self.train_samples,
            patch_size=self.patch_size,
            seed=self.seed,
            foreground_patch_fraction=self.foreground_patch_fraction,
        )
        return DataLoader(ds, batch_size=self.batch_size, shuffle=True, collate_fn=self._collate_batch)

    def val_dataloader(self) -> DataLoader:
        ds = VoxTellDataset(
            self.val_samples,
            patch_size=self.patch_size,
            seed=self.seed,
        )
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False, collate_fn=self._collate_batch)
    
    def test_dataloader(self) -> DataLoader:
        ds = VoxTellDataset(self.test_samples, mode="test", seed=self.seed)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False, collate_fn=self._collate_batch)
