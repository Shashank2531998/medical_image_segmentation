from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.training.utils import seed_everything
import torch
import numpy as np
import monai

from src.training.losses import deep_supervision_loss
from src.training.optimizer import build_optimizer_and_scheduler
from src.utils.checkpoint import save_checkpoint
from src.utils.logging import get_logger

logger = get_logger(__name__)


class Trainer:
    def __init__(self, engine, train_cfg: dict | None = None):
        self.engine = engine
        self.train_cfg = train_cfg or {}
        self.device = self.engine.device

        self.epochs = int(self.train_cfg.get("epochs", 1))

        self.optimizer = None
        self.scheduler = None
        self._scheduler_per_iteration = False

        # Set output dir and logging
        self.out_root = Path(self.train_cfg.get("output_dir", "experiments/exp_debug"))
        (self.out_root / "logs").mkdir(parents=True, exist_ok=True)
        logger_name = f"{__name__}.{self.out_root.as_posix().replace('/', '_')}"
        self.run_logger = get_logger(logger_name, log_file=self.out_root / "logs" / "run.log")

        # Save Debug Artifacts
        self.debug_save_artifacts = bool(self.train_cfg.get("debug_save_artifacts", False))
        self.debug_save_interval = int(self.train_cfg.get("debug_save_interval", 1))
        if self.debug_save_artifacts:
            self.debug_dir = self.out_root / "debug_artifacts"
            self.debug_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_loss = float("inf")
        self.dice_metric = monai.metrics.DiceMetric(include_background=True, reduction="none", ignore_empty=False)

    def _build_optimizer_and_scheduler(self, iters_per_epoch=None):
        self.optimizer, self.scheduler, self._scheduler_per_iteration = build_optimizer_and_scheduler(
            self.engine.model,
            self.train_cfg.get("optimizer", {}),
            max_epochs=self.epochs,
            iters_per_epoch=iters_per_epoch,
        )

    def _save_debug_artifacts(self, batch, outputs, epoch: int, step: int, debug_dir: Path):
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]

        outputs = outputs.detach().cpu()
        images = batch["image"].detach().cpu()
        masks = batch["masks"].detach().cpu()

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

    def train_one_batch(self, batch_idx, batch_item, epoch):

        img_paths = batch_item["img_path"]
        prompts = batch_item["prompts"]
        prompt_labels = prompts[0]
        self.run_logger.info("    [DATA] img=%s prompts=%s", img_paths, ", ".join(prompt_labels))

        outputs = self.engine.forward(batch_item)

        masks = batch_item["masks"].to(self.device)
        loss = deep_supervision_loss(outputs, masks, weights=self.engine.ds_weights)

        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        dice = self.dice_metric(preds, masks)
        dice = torch.nan_to_num(dice, nan=0.0)
        dice_vals = dice.detach().cpu().view(-1).tolist()
        batch_dice_mean = np.nanmean(dice_vals)
        per_prompt_val = { p: d for p, d in zip(prompt_labels, dice_vals)}
        per_prompt_str = ", ".join(
            f"{p}: {d:.4f}"
            for p, d in per_prompt_val.items()
        )

        self.optimizer.zero_grad()

        if self.debug_save_artifacts and batch_idx % self.debug_save_interval == 0:
            self._save_debug_artifacts(batch_item, outputs, epoch, batch_idx, self.debug_dir)
            
        loss.backward()
        self.optimizer.step()

        # step scheduler per-iteration if configured
        if getattr(self, "_scheduler_per_iteration", False) and self.scheduler is not None:
            self.scheduler.step()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.engine.model.parameters(),
            max_norm=1e9
        )

        # Logging
        current_lrs = [pg["lr"] for pg in self.optimizer.param_groups]
        self.run_logger.info(
            "    Loss: %.6f | LR(s): %s | Grad Norm (clipped): %.6f | Dice(mean): %.6f | Per-prompt: %s",
            loss.item(),
            current_lrs,
            grad_norm,
            batch_dice_mean,
            per_prompt_str,
        )

        return loss

    def fit(self, datamodule):

        # Determinism / seeding
        seed = int(self.train_cfg.get("seed", 42))
        seed_everything(
            seed,
            self.train_cfg.get("use_deterministic_algorithms", False)
        )

        # build dataloaders first to compute iterations/epoch if needed
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()

        self.run_logger.info(
            "Training started | epochs=%d | device=%s | train_batches=%d | val_batches=%d",
            self.epochs,
            self.device,
            len(train_loader),
            len(val_loader)
        )

        # Prepare model, optimizer and scheduler
        self._build_optimizer_and_scheduler(iters_per_epoch=len(train_loader))
        
        # Log optimizer and model info
        optimizer_state = self.optimizer.state_dict()
        self.run_logger.info("Optimizer: %s | LR: %s", 
                            optimizer_state.get('param_groups', [{}])[0].get('name', 'unknown'),
                            [pg['lr'] for pg in self.optimizer.param_groups])

        for epoch in range(self.epochs):
            self.run_logger.info(f"[EPOCH {epoch}/{self.epochs}]")
            self.engine.model.train()
            total_training_loss = 0.0
            for batch_idx, batch in enumerate(train_loader):
                self.run_logger.info(f"  [BATCH {batch_idx}]")
                loss = self.train_one_batch(batch_idx, batch, epoch)
                total_training_loss += float(loss.item())

            train_loss = total_training_loss / len(train_loader)

            # step scheduler per-epoch if configured that way
            if getattr(self, "_scheduler_per_iteration", False) is False and self.scheduler is not None:
                self.scheduler.step()

            # run light validation
            self.engine.model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    masks = batch["masks"].to(self.device)
                    outputs = self.engine.forward(batch)
                    loss = deep_supervision_loss(outputs, masks, weights=self.engine.ds_weights)
                    total_val_loss += float(loss.item())

            val_loss = total_val_loss / len(val_loader)
            self.run_logger.info(
                "Epoch %d/%d completed | Training Loss=%.4f, | Validation Loss=%.4f",
                epoch + 1, self.epochs, train_loss, val_loss
            )

            # Experiment Checkpoints
            checkpoints_dir = self.out_root / "checkpoints"
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_path = checkpoints_dir / f"best_model_epoch_{epoch}.pt"
                save_checkpoint(self.engine.model, self.optimizer, best_path)
                self.run_logger.info("New best model saved (val_loss=%.6f) at epoch=%d", val_loss, epoch)        
            elif (epoch + 1) % 10 == 0:
                ckpt_path = checkpoints_dir / f"checkpoint_epoch_{epoch}.pt"
                save_checkpoint(self.engine.model, self.optimizer, ckpt_path)
                self.run_logger.info("Checkpoint saved at epoch=%d", epoch)
        

        final_ckpt_path = checkpoints_dir / "final_checkpoint.pt"
        save_checkpoint(self.engine.model, self.optimizer, final_ckpt_path)
        self.run_logger.info("Training completed successfully | Best validation loss: %.6f", self.best_val_loss)


if __name__ == '__main__':
    logger.info("Trainer module")
