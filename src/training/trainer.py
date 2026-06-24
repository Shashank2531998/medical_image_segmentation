from __future__ import annotations

from pathlib import Path

from src.training.utils import seed_everything
from src.utils.training_plot import plot_training_loss
import torch
import numpy as np
import monai

from src.training.losses import deep_supervision_loss
from src.training.optimizer import build_optimizer_and_scheduler
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.logging import get_logger

logger = get_logger(__name__)


class Trainer:
    def __init__(self, engine, train_cfg: dict | None = None, hooks=None, task=None):
        self.engine = engine
        self.train_cfg = train_cfg or {}
        self.hooks = hooks
        self.task = task
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
        
        # Early stopping settings
        es_cfg = self.train_cfg.get("early_stopping", {}) or {}
        self.early_stopping_enabled = bool(es_cfg.get("enabled", False))
        self.early_stopping_patience = int(es_cfg.get("patience", 5))
        self.early_stopping_min_delta = float(es_cfg.get("min_delta", 0.0))
        self._no_improve_epochs = 0

        # Loss tracking for plotting
        self.train_losses = []
        self.val_losses = []

        # Recovery checkpointing for crash-safe resume.
        self.resume_checkpoint_interval = int(self.train_cfg.get("resume_checkpoint_interval", 1))
        self.recovery_checkpoint_path = self.out_root / "checkpoints" / "latest_recovery.pt"

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

        if self.engine.return_features:
            out_logits, mask_embedding, img_feature = outputs
        else:
            out_logits = outputs

        masks = batch_item["masks"].to(self.device)
        loss = deep_supervision_loss(out_logits, masks, weights=self.engine.ds_weights)

        if self.hooks is not None and hasattr(self.hooks, "compute_loss"):
            loss = self.hooks.compute_loss(
                task=self.task,
                batch=batch_item,
                outputs=outputs,
                base_loss=loss,
            )

        probs = torch.sigmoid(out_logits)
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
            self._save_debug_artifacts(batch_item, out_logits, epoch, batch_idx, self.debug_dir)
            
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
        self.run_logger.info(
            "    Loss: %.6f | Grad Norm (clipped): %.6f | Dice(mean): %.6f | Per-prompt: %s",
            loss.item(),
            grad_norm,
            batch_dice_mean,
            per_prompt_str,
        )

        return loss

    def _notify_batch_end(self, *, batch_idx: int, epoch: int, batch_loss: float) -> None:
        if self.hooks is None:
            return

        callback = getattr(self.hooks, "on_train_batch_end", None)
        if callable(callback):
            callback(task=self.task, batch_idx=batch_idx, epoch=epoch, batch_loss=batch_loss)

    def _notify_recovery_checkpoint(self, *, checkpoint_path: Path, epoch: int, batch_idx: int, stage: str) -> None:
        if self.hooks is None:
            return

        callback = getattr(self.hooks, "on_recovery_checkpoint", None)
        if callable(callback):
            callback(
                task=self.task,
                checkpoint_path=checkpoint_path,
                epoch=epoch,
                batch_idx=batch_idx,
                stage=stage,
            )

    def _save_recovery_checkpoint(self, *, epoch: int, batch_idx: int, stage: str) -> None:
        metadata = {
            "epoch": int(epoch),
            "batch_idx": int(batch_idx),
            "stage": str(stage),
            "best_val_loss": float(self.best_val_loss),
            "no_improve_epochs": int(self._no_improve_epochs),
            "train_losses": list(self.train_losses),
            "val_losses": list(self.val_losses),
        }
        save_checkpoint(
            self.engine.model,
            self.optimizer,
            self.recovery_checkpoint_path,
            scheduler=self.scheduler,
            metadata=metadata,
        )
        self._notify_recovery_checkpoint(
            checkpoint_path=self.recovery_checkpoint_path,
            epoch=epoch,
            batch_idx=batch_idx,
            stage=stage,
        )

    def _load_recovery_checkpoint(self, resume_from: str | Path) -> tuple[int, int]:
        checkpoint = load_checkpoint(resume_from, map_location="cpu")
        self.engine.model.load_state_dict(checkpoint["network_weights"])

        optimizer_state = checkpoint.get("optimizer_state")
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)

        scheduler_state = checkpoint.get("scheduler_state")
        if scheduler_state is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(scheduler_state)

        self.best_val_loss = float(checkpoint.get("best_val_loss", self.best_val_loss))
        self._no_improve_epochs = int(checkpoint.get("no_improve_epochs", self._no_improve_epochs))
        self.train_losses = list(checkpoint.get("train_losses", self.train_losses))
        self.val_losses = list(checkpoint.get("val_losses", self.val_losses))

        epoch = int(checkpoint.get("epoch", 0))
        batch_idx = int(checkpoint.get("batch_idx", -1))
        stage = str(checkpoint.get("stage", "train_batch"))

        if stage == "epoch_end":
            start_epoch = epoch + 1
            start_batch = 0
        else:
            start_epoch = epoch
            start_batch = batch_idx + 1

        self.run_logger.info(
            "Resuming from checkpoint=%s | stage=%s | start_epoch=%d | start_batch=%d",
            resume_from,
            stage,
            start_epoch,
            start_batch,
        )
        return start_epoch, start_batch

    def fit(self, datamodule, resume_from: str | Path | None = None):

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

        checkpoints_dir = self.out_root / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        start_epoch = 0
        start_batch = 0
        if resume_from is not None:
            start_epoch, start_batch = self._load_recovery_checkpoint(resume_from)

        if start_epoch >= self.epochs:
            self.run_logger.info(
                "Resume checkpoint indicates training already finished for configured epochs=%d",
                self.epochs,
            )
            final_train_loss = self.train_losses[-1] if self.train_losses else float("nan")
            final_val_loss = self.val_losses[-1] if self.val_losses else float("nan")
            return {
                "train_loss": final_train_loss,
                "val_loss": final_val_loss,
                "best_val_loss": self.best_val_loss,
                "epochs_ran": len(self.train_losses),
                "train_losses": list(self.train_losses),
                "val_losses": list(self.val_losses),
            }

        for epoch in range(start_epoch, self.epochs):
            self.run_logger.info(f"[EPOCH {epoch}/{self.epochs}]")
            self.engine.model.train()
            total_training_loss = 0.0
            batches_processed = 0
            for batch_idx, batch in enumerate(train_loader):
                if epoch == start_epoch and batch_idx < start_batch:
                    continue
                self.run_logger.info(f"  [BATCH {batch_idx}]")
                self.run_logger.info("     Optimizer LR: %s", [pg['lr'] for pg in self.optimizer.param_groups])
                loss = self.train_one_batch(batch_idx, batch, epoch)
                total_training_loss += float(loss.item())
                batches_processed += 1
                self._notify_batch_end(batch_idx=batch_idx, epoch=epoch, batch_loss=float(loss.item()))
                if self.resume_checkpoint_interval > 0 and (batch_idx + 1) % self.resume_checkpoint_interval == 0:
                    self._save_recovery_checkpoint(epoch=epoch, batch_idx=batch_idx, stage="train_batch")

            if batches_processed == 0:
                self.run_logger.info("No train batches processed for epoch=%d, skipping loss/scheduler/validation", epoch)
                start_batch = 0
                continue

            train_loss = total_training_loss / batches_processed

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

                    if self.engine.return_features:
                        out_logits, mask_embedding, img_feature = outputs
                    else:
                        out_logits = outputs

                    loss = deep_supervision_loss(out_logits, masks, weights=self.engine.ds_weights)
                    total_val_loss += float(loss.item())

            val_loss = total_val_loss / len(val_loader)
            self.run_logger.info(
                "Epoch %d/%d completed | Training Loss=%.4f, | Validation Loss=%.4f",
                epoch + 1, self.epochs, train_loss, val_loss
            )

            # Collect losses for plotting
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self._save_recovery_checkpoint(epoch=epoch, batch_idx=-1, stage="epoch_end")

            # Experiment Checkpoints + Early Stopping
            improved = val_loss < (self.best_val_loss - self.early_stopping_min_delta)
            if improved:
                self.best_val_loss = val_loss
                best_path = checkpoints_dir / f"best_model.pt"
                save_checkpoint(self.engine.model, self.optimizer, best_path)
                self.run_logger.info("New best model saved (val_loss=%.6f) at epoch=%d", val_loss, epoch)
                self._no_improve_epochs = 0
            else:
                self._no_improve_epochs += 1
                if self.early_stopping_enabled:
                    self.run_logger.info(
                        "No improvement for %d/%d epochs (min_delta=%.6f)",
                        self._no_improve_epochs,
                        self.early_stopping_patience,
                        self.early_stopping_min_delta,
                    )

            # Trigger early stopping if enabled
            if self.early_stopping_enabled and self._no_improve_epochs >= self.early_stopping_patience:
                self.run_logger.info(
                    "Early stopping triggered after %d epochs without improvement",
                    self._no_improve_epochs,
                )
                break

            # Reset resume offset after the first resumed epoch.
            start_batch = 0
        
        self.run_logger.info("Training completed successfully | Best validation loss: %.6f", self.best_val_loss)
        
        # Plot training loss
        plot_training_loss(
            self.train_losses, 
            self.val_losses, 
            self.out_root / "plots"
        )

        final_train_loss = self.train_losses[-1] if self.train_losses else float("nan")
        final_val_loss = self.val_losses[-1] if self.val_losses else float("nan")

        return {
            "train_loss": final_train_loss,
            "val_loss": final_val_loss,
            "best_val_loss": self.best_val_loss,
            "epochs_ran": len(self.train_losses),
            "train_losses": list(self.train_losses),
            "val_losses": list(self.val_losses),
        }


if __name__ == '__main__':
    logger.info("Trainer module")
