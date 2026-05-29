from __future__ import annotations

from pathlib import Path
from typing import Any

from src.continual.strategies.lora.loralib_lora import apply_loralib_lora, save_lora_adapter
from src.continual.task_manager import ContinualTask, ContinualTaskManager

from ..base import BaseContinualStrategy
from ..registry import register_strategy


@register_strategy
class LoRAStrategy(BaseContinualStrategy):
    strategy_name = "lora"

    def configure_engine(self) -> None:
        engine = self.require_engine()

        self.logger.info("Applying LoRA adaptation to base model...")
        engine.model = apply_loralib_lora(engine.model, self.task_manager.lora_cfg)
        self.logger.info("LoRA adaptation complete. Model ready for continual learning.")

    def after_task(
        self,
        task: ContinualTask,
        task_dir: Path,
        task_training_cfg: dict[str, Any],
    ) -> None:
        engine = self.require_engine()
        lora_bias = str(self.task_manager.lora_cfg.get("bias", "none"))
        save_lora_adapter(engine.model, task_dir / "lora_adapter.pt", bias=lora_bias)
        self.logger.info("Saved LoRA adapter for task %s", task.name)

    def completion_message(self) -> str:
        return "LoRA continual strategy completed successfully"

    @classmethod
    def build_model_cfg_for_evaluation(
        cls,
        *,
        task_manager: ContinualTaskManager,
        task: ContinualTask,
        task_dir: Path,
        trained_model_cfg: dict[str, Any],
        checkpoint_name: str,
    ) -> dict[str, Any]:
        adapter_path = task_dir / "lora_adapter.pt"
        if not adapter_path.exists():
            raise FileNotFoundError(f"Expected LoRA adapter for task {task.name} at {adapter_path}")

        return {
            **trained_model_cfg,
            "lora_cfg": dict(task_manager.lora_cfg),
            "lora_adapter_path": str(adapter_path),
        }


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