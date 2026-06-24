from __future__ import annotations

from pathlib import Path
from typing import Any

from importlib import import_module

from src.continual.strategies.base import BaseContinualStrategy, EvaluationSpec
from src.continual.strategies.registry import register_strategy
from src.continual.task_manager import ContinualTask, ContinualTaskManager
from src.engine.model_engine import VoxTellEngine
from src.continual.strategies.cpe_clip.model import CPECLIPPromptedVoxTell, configure_loaded_cpe_clip_model


@register_strategy
class CPECLIPStrategy(BaseContinualStrategy):
    strategy_name = "cpe_clip"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cpe_cfg = dict(self.task_manager.cpe_clip_cfg)
        self.regularization_method = str(self.cpe_cfg.get("regularization_method", "balance")).lower()

    def build_engine(self) -> VoxTellEngine:
        engine = super().build_engine()
        engine.model = configure_loaded_cpe_clip_model(
            engine.model,
            self.cpe_cfg,
            mark_trainable=True,
        )
        return engine

    def configure_engine(self) -> None:
        engine = self.require_engine()
        if not isinstance(engine.model, CPECLIPPromptedVoxTell):
            engine.model = configure_loaded_cpe_clip_model(engine.model, self.cpe_cfg, mark_trainable=True)

        task_index = int(getattr(self, "_active_task_index", 0))
        if self.regularization_method == "freeze" and task_index > 0:
            engine.model.base_model.transformer_decoder.set_trainable(False)
            alpha = 1.0
        elif self.regularization_method == "balance":
            engine.model.base_model.transformer_decoder.set_trainable(True)
            alpha = engine.model.scale_alpha(task_index)
        else:
            engine.model.base_model.transformer_decoder.set_trainable(True)
            alpha = 1.0

        engine.model.base_model.transformer_decoder.set_session_alpha(alpha)
        self.logger.info(
            "CPE-CLIP session scaling | task_index=%d | alpha_t=%.4f | regularization=%s",
            task_index + 1,
            alpha,
            self.regularization_method,
        )

    def after_task(
        self,
        task: ContinualTask,
        task_dir: Path,
        task_training_cfg: dict[str, Any],
        task_metrics: dict[str, Any] | None = None,
    ) -> None:
        self.logger.info("Completed CPE-CLIP task %s", task.name)

    @classmethod
    def build_evaluation_spec(
        cls,
        *,
        task_manager: ContinualTaskManager,
        task: ContinualTask,
        task_dir: Path,
        trained_model_cfg: dict[str, Any],
        checkpoint_name: str,
    ) -> EvaluationSpec:
        cpe_cfg = dict(task_manager.cpe_clip_cfg)
        cpe_cfg["trainable"] = False

        return EvaluationSpec(
            model_cfg={
                **trained_model_cfg,
                "cpe_clip_cfg": cpe_cfg,
                "checkpoint_path": str(task_dir / "checkpoints" / checkpoint_name),
            },
        )


def run_cpe_clip_strategy(
    *,
    cfg: dict[str, Any],
    task_manager: ContinualTaskManager,
    tasks: list[ContinualTask],
    dirs: dict[str, Path],
    logger,
) -> None:
    CPECLIPStrategy(
        cfg=cfg,
        task_manager=task_manager,
        tasks=tasks,
        dirs=dirs,
        logger=logger,
    ).run()