from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from src.continual.task_manager import ContinualTask, ContinualTaskManager
from src.continual.strategies.base import BaseContinualStrategy
from src.continual.strategies.registry import register_strategy


@register_strategy
class ZSCLStrategy(BaseContinualStrategy):
    """ZSCL-style continual learning using teacher distillation and weight averaging.

    This implementation follows the paper's core ideas in a model-agnostic way:
    1. keep a frozen teacher snapshot from the initial task state;
    2. add a distillation regularizer on current logits to preserve old feature-space behavior;
    3. maintain an exponential weight-ensemble snapshot that can be saved for reuse.
    """

    strategy_name = "zscl"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.teacher_model: torch.nn.Module | None = None
        self.ensemble_model: torch.nn.Module | None = None
        self._ensemble_steps = 0

        continual_cfg = dict(self.cfg.get("continual", {}))
        zscl_cfg = dict(continual_cfg.get("zscl", {}))
        self.distillation_weight = float(zscl_cfg.get("distillation_weight", 0.1))
        self.weight_ensemble_interval = int(zscl_cfg.get("weight_ensemble_interval", 0))

    def build_engine(self):
        engine = super().build_engine()

        self.teacher_model = copy.deepcopy(engine.model)
        self.teacher_model.to(engine.device).eval()
        for parameter in self.teacher_model.parameters():
            parameter.requires_grad = False

        self.ensemble_model = copy.deepcopy(engine.model)
        self.ensemble_model.to(engine.device).eval()
        for parameter in self.ensemble_model.parameters():
            parameter.requires_grad = False

        self.logger.info(
            "ZSCL strategy initialized | distillation_weight=%.4f | weight_ensemble_interval=%d",
            self.distillation_weight,
            self.weight_ensemble_interval,
        )
        return engine

    def compute_loss(
        self,
        *,
        task: ContinualTask,
        batch: dict[str, Any],
        outputs: Any,
        base_loss: Any,
    ) -> Any:
        if self.teacher_model is None or self.distillation_weight <= 0.0:
            return base_loss

        student_logits = outputs
        if isinstance(student_logits, (list, tuple)):
            student_logits = student_logits[0]

        teacher_logits = self._teacher_logits(batch)
        if teacher_logits is None:
            return base_loss

        student_logits = student_logits.float()
        teacher_logits = teacher_logits.float()

        student_log_probs = F.log_softmax(student_logits, dim=1)
        teacher_probs = F.softmax(teacher_logits, dim=1).detach()
        distillation_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean", log_target=False)
        return base_loss + self.distillation_weight * distillation_loss

    def _teacher_logits(self, batch: dict[str, Any]) -> torch.Tensor | None:
        if self.teacher_model is None:
            return None

        engine = self.require_engine()
        imgs = batch["image"].to(engine.device)
        text_embeddings = engine.text_encoder.embed(batch["prompts"]).clone()

        with torch.no_grad():
            teacher_outputs = self.teacher_model(imgs, text_embeddings)

        if isinstance(teacher_outputs, (list, tuple)):
            teacher_outputs = teacher_outputs[0]
        return teacher_outputs

    def on_train_batch_end(
        self,
        task: ContinualTask,
        batch_idx: int,
        epoch: int,
        batch_loss: float,
    ) -> None:
        if self.weight_ensemble_interval <= 0 or self.ensemble_model is None:
            return

        if (batch_idx + 1) % self.weight_ensemble_interval != 0:
            return

        engine = self.require_engine()
        self._ensemble_steps += 1

        current_state = engine.model.state_dict()
        ensemble_state = self.ensemble_model.state_dict()

        alpha = 1.0 / float(self._ensemble_steps + 1)
        blended = {
            name: (alpha * current_state[name]) + ((1.0 - alpha) * ensemble_state[name])
            for name in current_state
        }
        self.ensemble_model.load_state_dict(blended, strict=True)

    def after_task(
        self,
        task: ContinualTask,
        task_dir: Path,
        task_training_cfg: dict[str, Any],
        task_metrics: dict[str, Any] | None = None,
    ) -> None:
        if self.ensemble_model is None:
            return

        ensemble_path = task_dir / "zscl_weight_ensemble.pt"
        torch.save({"network_weights": self.ensemble_model.state_dict()}, str(ensemble_path))
        self.logger.info("Saved ZSCL weight ensemble snapshot to %s", ensemble_path)


def run_zscl_strategy(
    *,
    cfg: dict[str, Any],
    task_manager: ContinualTaskManager,
    tasks: list[ContinualTask],
    dirs: dict[str, Path],
    logger,
) -> None:
    ZSCLStrategy(
        cfg=cfg,
        task_manager=task_manager,
        tasks=tasks,
        dirs=dirs,
        logger=logger,
    ).run()
