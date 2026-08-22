from __future__ import annotations

from pathlib import Path
from typing import Any

from src.continual.strategies.lora.loading import configure_loaded_lora_model
from src.continual.strategies.lora.common.utils import save_lora_adapter
from src.continual.task_manager import ContinualTask, ContinualTaskManager
from src.continual.strategies.base import BaseContinualStrategy, EvaluationSpec
from src.continual.strategies.registry import register_strategy


@register_strategy
class LoRAStrategy(BaseContinualStrategy):
    strategy_name = "lora"

    def configure_engine(self) -> None:
        self.engine = self.build_engine()
        self.engine.model = configure_loaded_lora_model(
            self.engine.model,
            lora_cfg=self.task_manager.lora_cfg
        )

    def after_task(
        self,
        task: ContinualTask,
        task_dir: Path,
        task_training_cfg: dict[str, Any],
        task_metrics: dict[str, Any] | None = None,
    ) -> None:
        engine = self.require_engine()
        lora_bias = str(self.task_manager.lora_cfg.get("bias", "none"))
        save_lora_adapter(engine.model, task_dir / "lora_adapter.pt", bias=lora_bias)
        self.logger.info("Saved LoRA adapter for task %s", task.name)

    @classmethod
    def build_evaluation_spec(
        cls,
        *,
        task_manager: ContinualTaskManager,
        task: ContinualTask,
        task_dir: Path,
        trained_model_cfg: dict[str, Any],
        checkpoint_name: str,
        evaluation_task: ContinualTask | None = None,
    ) -> EvaluationSpec:
        adapter_task = task

        if evaluation_task.is_retention:
            return EvaluationSpec(model_cfg=trained_model_cfg)

        if (
            evaluation_task is not None
            and evaluation_task.index <= task.index
        ):
            adapter_task = evaluation_task

        adapter_task_dir = task_dir.parent.parent / "tasks" / task_manager.task_dir_name(adapter_task)
        adapter_path = adapter_task_dir / "lora_adapter.pt"
        if not adapter_path.exists():
            raise FileNotFoundError(f"Expected LoRA adapter at {adapter_path}")

        return EvaluationSpec(
            model_cfg={
                **trained_model_cfg,
                "lora_cfg": dict(task_manager.lora_cfg),
                "lora_adapter_path": str(adapter_path),
            },
        )


def run_lora_strategy(
    *,
    cfg: dict[str, Any],
    task_manager: ContinualTaskManager,
    tasks: list[ContinualTask],
    dirs: dict[str, Path],
    logger,
) -> None:
    LoRAStrategy(
        cfg=cfg,
        task_manager=task_manager,
        tasks=tasks,
        dirs=dirs,
        logger=logger,
    ).run()