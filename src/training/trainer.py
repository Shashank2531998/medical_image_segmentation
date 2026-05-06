from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from src.model.builder import load_voxtell_model
from src.training.optimizer import build_optimizer_and_scheduler
from src.training.losses import bce_loss_logits
from src.utils.checkpoint import save_checkpoint


class Trainer:
    def __init__(self, train_cfg: dict | None = None, model_cfg: dict | None = None):
        self.train_cfg = train_cfg or {}
        self.model_cfg = model_cfg or {}
        self.device = torch.device(self.train_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")

        # lazy model load: actual weights loaded when fit() is called with model_dir
        self.model = None
        self.optimizer = None
        self.scheduler = None

    def _prepare_model(self, model_dir: str):
        model, _ = load_voxtell_model(model_dir)
        self.model = model.to(self.device)
        self.optimizer, self.scheduler = build_optimizer_and_scheduler(self.model, self.train_cfg.get("optimizer", {}))

    def fit(self, datamodule, model_dir: Optional[str] = None):
        if model_dir is None:
            model_dir = self.model_cfg.get("dir", None)
        if model_dir is None:
            raise ValueError("model_dir must be provided in model_cfg or as argument")

        self._prepare_model(model_dir)

        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()

        epochs = int(self.train_cfg.get("epochs", 1))
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            for batch_idx, (imgs, masks) in enumerate(train_loader):
                imgs = imgs.to(self.device)
                masks = masks.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(imgs)  # model expected signature may vary
                loss = bce_loss_logits(outputs, masks)
                loss.backward()
                self.optimizer.step()

                running_loss += float(loss.item())

            if self.scheduler is not None:
                self.scheduler.step()

            # simple checkpoint per epoch
            out_dir = Path(self.train_cfg.get("output_dir", "experiments/exp_debug"))
            out_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = out_dir / f"checkpoint_epoch_{epoch}.pt"
            save_checkpoint(self.model, self.optimizer, ckpt_path)

            print(f"Epoch {epoch+1}/{epochs} - loss: {running_loss:.4f}")

            # run light validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for imgs, masks in val_loader:
                    imgs = imgs.to(self.device)
                    masks = masks.to(self.device)
                    preds = self.model(imgs)
                    val_loss += float(bce_loss_logits(preds, masks).item())
            print(f"Validation loss: {val_loss:.4f}")



if __name__ == '__main__':
    print("Trainer module")
