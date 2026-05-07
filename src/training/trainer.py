from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import numpy as np
import random

from src.model.builder import load_voxtell_model
from src.text.encoder import TextPromptEncoder
from src.training.optimizer import build_optimizer_and_scheduler
from src.training.losses import deep_supervision_loss
from src.utils.checkpoint import save_checkpoint
from src.utils.logging import get_logger


logger = get_logger(__name__)


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
        # Optionally reinitialize weights deterministically before optimizer setup
        if self.train_cfg.get("reinit_weights", False):
            try:
                from src.model.voxtell_model import VoxTellModel

                model.apply(VoxTellModel.initialize)
            except Exception:
                pass

        self.model = model.to(self.device)
        iters = getattr(self, "_iters_per_epoch", None)
        opt_ret = build_optimizer_and_scheduler(
            self.model,
            self.train_cfg.get("optimizer", {}),
            max_epochs=max_epochs,
            iters_per_epoch=iters,
        )
        if isinstance(opt_ret, tuple) and len(opt_ret) == 3:
            self.optimizer, self.scheduler, self._scheduler_per_iteration = opt_ret
        else:
            self.optimizer, self.scheduler = opt_ret
            self._scheduler_per_iteration = False

    def fit(self, datamodule, model_dir: Optional[str] = None):
        if model_dir is None:
            model_dir = self.model_cfg.get("dir", None)
        if model_dir is None:
            raise ValueError("model_dir must be provided in model_cfg or as argument")

        epochs = int(self.train_cfg.get("epochs", 1))

        # Determinism / seeding
        seed = int(self.train_cfg.get("seed", 42))
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass

        # Optional stricter deterministic algorithms (may raise on unsupported ops)
        if self.train_cfg.get("use_deterministic_algorithms", False):
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass

        # build dataloaders first to compute iterations/epoch if needed
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()

        try:
            self._iters_per_epoch = len(train_loader)
        except Exception:
            self._iters_per_epoch = None

        self._prepare_model(model_dir, max_epochs=epochs)

        out_root = Path(self.train_cfg.get("output_dir", "experiments/exp_debug"))
        (out_root / "logs").mkdir(parents=True, exist_ok=True)
        run_logger = get_logger(__name__, log_file=out_root / "logs" / "run.log")
        ds_weights = self.train_cfg.get("deep_supervision_weights", [1, 0.5, 0.25, 0.125, 0.0625])

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            for i, batch in enumerate(train_loader):
                imgs = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)
                prompts = batch["prompts"]

                # prompts is expected to be a list-of-lists: [[p1,p2,p3], [p1,p2,p3], ...]
                text_embeddings = self.text_encoder.embed(prompts).clone()

                self.optimizer.zero_grad()
                outputs = self.model(imgs, text_embeddings)
                loss = deep_supervision_loss(outputs, masks, weights=ds_weights)
                loss.backward()
                self.optimizer.step()

                # step scheduler per-iteration if configured
                if getattr(self, "_scheduler_per_iteration", False) and self.scheduler is not None:
                    self.scheduler.step()

                running_loss += float(loss.item())

                if i % 10 == 0:
                    run_logger.info(
                        "Epoch %d [%d/%d] - loss: %.4f",
                        epoch + 1, i, len(train_loader), loss.item()
                    )

            # step scheduler per-epoch if configured that way
            if getattr(self, "_scheduler_per_iteration", False) is False and self.scheduler is not None:
                self.scheduler.step()
            
            run_logger.info(
                "Epoch %d/%d - train_loss: %.4f",
                epoch + 1, epochs, running_loss
            )

            # simple checkpoint per 10 epoch into experiment checkpoints
            if epoch % 10 == 0:
                checkpoints_dir = out_root / "checkpoints"
                checkpoints_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = checkpoints_dir / f"checkpoint_epoch_{epoch}.pt"
                save_checkpoint(self.model, self.optimizer, ckpt_path)


            # run light validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    imgs = batch["image"].to(self.device)
                    masks = batch["mask"].to(self.device)
                    prompts = batch["prompts"]
                    text_embeddings = self.text_encoder.embed(prompts)
                    preds = self.model(imgs, text_embeddings)
                    val_loss += float(deep_supervision_loss(preds, masks, weights=ds_weights).item())
            run_logger.info("Validation loss: %.4f", val_loss)



if __name__ == '__main__':
    logger.info("Trainer module")
