from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from src.model.builder import load_voxtell_model
from src.text.encoder import TextPromptEncoder
from src.training.optimizer import build_optimizer_and_scheduler
from src.training.losses import deep_supervision_loss
from src.utils.checkpoint import save_checkpoint
from src.utils.logging import get_logger


class Trainer:
    def __init__(self, train_cfg: dict | None = None, model_cfg: dict | None = None):
        self.train_cfg = train_cfg or {}
        self.model_cfg = model_cfg or {}
        self.device = torch.device(self.train_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")

        # lazy model load: actual weights loaded when fit() is called with model_dir
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.text_encoder = TextPromptEncoder(
            self.model_cfg.get("text_encoding_model", "Qwen/Qwen3-Embedding-4B"),
            device=self.device,
        )

    def _prepare_model(self, model_dir: str, max_epochs: int):
        model, _ = load_voxtell_model(
            model_dir,
            deep_supervision=self.train_cfg.get("deep_supervision", True),
            model_overrides=self.model_cfg,
        )
        self.model = model.to(self.device)
        self.optimizer, self.scheduler = build_optimizer_and_scheduler(
            self.model,
            self.train_cfg.get("optimizer", {}),
            max_epochs=max_epochs,
        )

    def fit(self, datamodule, model_dir: Optional[str] = None):
        if model_dir is None:
            model_dir = self.model_cfg.get("dir", None)
        if model_dir is None:
            raise ValueError("model_dir must be provided in model_cfg or as argument")

        epochs = int(self.train_cfg.get("epochs", 1))
        self._prepare_model(model_dir, max_epochs=epochs)

        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()

        out_root = Path(self.train_cfg.get("output_dir", "experiments/exp_debug"))
        (out_root / "logs").mkdir(parents=True, exist_ok=True)
        logger = get_logger(__name__, log_file=out_root / "logs" / "run.log")
        ds_weights = self.train_cfg.get("deep_supervision_weights", [1, 0.5, 0.25, 0.125, 0.0625])

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            for batch in train_loader:
                imgs = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)
                prompts = batch["prompt"]

                if imgs.shape[0] != 1:
                    raise ValueError("Current trainer supports batch_size=1 for prompt-conditioned training")

                text_embeddings = self.text_encoder.embed([prompts[0]])

                self.optimizer.zero_grad()
                outputs = self.model(imgs, text_embeddings)
                loss = deep_supervision_loss(outputs, masks, weights=ds_weights)
                loss.backward()
                self.optimizer.step()

                running_loss += float(loss.item())

            if self.scheduler is not None:
                self.scheduler.step()

            # simple checkpoint per epoch into experiment checkpoints
            checkpoints_dir = out_root / "checkpoints"
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = checkpoints_dir / f"checkpoint_epoch_{epoch}.pt"
            save_checkpoint(self.model, self.optimizer, ckpt_path)

            logger.info(f"Epoch {epoch+1}/{epochs} - loss: {running_loss:.4f}")

            # run light validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    imgs = batch["image"].to(self.device)
                    masks = batch["mask"].to(self.device)
                    prompts = batch["prompt"]

                    if imgs.shape[0] != 1:
                        raise ValueError("Current trainer supports batch_size=1 for prompt-conditioned validation")

                    text_embeddings = self.text_encoder.embed([prompts[0]])
                    preds = self.model(imgs, text_embeddings)
                    val_loss += float(deep_supervision_loss(preds, masks, weights=ds_weights).item())
            logger.info(f"Validation loss: {val_loss:.4f}")



if __name__ == '__main__':
    print("Trainer module")
