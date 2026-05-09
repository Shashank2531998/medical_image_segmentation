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

    def _save_debug_artifacts(self, batch, outputs, epoch: int, step: int, debug_dir: Path):
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]

        outputs = outputs.detach().cpu()
        images = batch["image"].detach().cpu()
        masks = batch["mask"].detach().cpu()

        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).to(torch.uint8)

        debug_path = debug_dir / f"epoch_{epoch + 1:02d}_step_{step + 1:03d}.pt"
        torch.save({
            "images": images,
            "masks": masks,
            "logits": outputs,
            "probs": probs,
            "preds": preds,
            "prompts": batch["prompts"],
        }, str(debug_path))

    def _prepare_model(self, model_dir: str, max_epochs: int):
        model, _ = load_voxtell_model(
            model_dir,
            deep_supervision=self.train_cfg.get("deep_supervision", False),
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
        # Set to train mode early to enable batch statistics and dropout
        self.model.train()
        
        deep_sup_enabled = self.train_cfg.get("deep_supervision", False)
        logger.info(f"Loading model with deep_supervision={deep_sup_enabled}")
        logger.info("Model set to training mode - BatchNorm will accumulate batch statistics")
        
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
        debug_save_artifacts = bool(self.train_cfg.get("debug_save_artifacts", False))
        debug_save_interval = int(self.train_cfg.get("debug_save_interval", 1))
        if debug_save_artifacts:
            debug_dir = out_root / "debug_artifacts"
            debug_dir.mkdir(parents=True, exist_ok=True)

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

                if debug_save_artifacts and i % debug_save_interval == 0:
                    self._save_debug_artifacts(batch, outputs, epoch, i, debug_dir)

                if epoch == 0 and i == 0:
                    run_logger.info(
                        f"""
                    ======== FIRST BATCH DEBUG ========
                    Images: shape={tuple(imgs.shape)} dtype={imgs.dtype}
                    Masks: shape={tuple(masks.shape)} unique={torch.unique(masks).cpu().tolist()}
                    Prompts: {prompts}
                    Text embeddings: shape={tuple(text_embeddings.shape)} dtype={text_embeddings.dtype}
                    Text embeddings range: min={text_embeddings.min():.4f} max={text_embeddings.max():.4f}

                    Outputs:
                    {[tuple(o.shape) for o in outputs] if isinstance(outputs, (list, tuple)) else tuple(outputs.shape)}

                    Image stats:
                    min={imgs.min():.4f} max={imgs.max():.4f}
                    mean={imgs.mean():.4f} std={imgs.std():.4f}
                    
                    Model training mode: {self.model.training}
                    ===================================
                    """
                        )
                    
                    if isinstance(outputs, (list, tuple)):
                        o = outputs[0]
                    else:
                        o = outputs

                    print("LOGITS:")
                    print(o.min().item(), o.max().item(), o.mean().item())
                    print("LOGITS per prompt:")
                    for idx in range(o.shape[1]):
                        logits = o[0, idx]
                        print(f"  Prompt {idx}: min={logits.min():.2f} max={logits.max():.2f} mean={logits.mean():.2f}")

                    probs = torch.sigmoid(o)
                    print("PROBS:")
                    print(probs.min().item(), probs.max().item(), probs.mean().item())
                    
                # At a patch level -
                # outputs will have length equal to 5, if deep supervision is enabled, otherwise 1 (e.g. - (5, 1, 3, 192, 192, 192))
                # For each sample of the batch, there will be 3 masks, 2 for positive and 1 for negative (e.g. - (1, 3, 192, 192, 192))

                loss = deep_supervision_loss(outputs, masks, weights=ds_weights)
                loss.backward()
                self.optimizer.step()

                # step scheduler per-iteration if configured
                if getattr(self, "_scheduler_per_iteration", False) and self.scheduler is not None:
                    self.scheduler.step()

                running_loss += float(loss.item())

                if i % 50 == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=1e9
                    )

                    run_logger.info(
                        "Gradient norm: %.4f",
                        grad_norm
                    )

            # step scheduler per-epoch if configured that way
            if getattr(self, "_scheduler_per_iteration", False) is False and self.scheduler is not None:
                self.scheduler.step()

            run_logger.info(
                "Epoch %d/%d completed | running_loss=%.4f",
                epoch + 1,
                epochs,
                running_loss,
            )

            # simple checkpoint per 10 epoch into experiment checkpoints
            if (epoch + 1) % 10 == 0:
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
