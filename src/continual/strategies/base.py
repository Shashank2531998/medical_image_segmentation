from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import Any

from src.continual.utils import build_base_model_cfg, save_task_snapshot
from src.continual.task_manager import ContinualTask, ContinualTaskManager, merge_dicts
from src.data.datamodule import VoxTellDataModule
from src.engine.model_engine import VoxTellEngine
from src.training.trainer import Trainer


class BaseContinualStrategy(ABC):
    strategy_name: str = ""
    aliases: tuple[str, ...] = ()

    def __init__(
        self,
        *,
        cfg: dict[str, Any],
        task_manager: ContinualTaskManager,
        tasks: list[ContinualTask],
        dirs: dict[str, Path],
        logger,
    ) -> None:
        self.cfg = cfg
        self.task_manager = task_manager
        self.tasks = tasks
        self.dirs = dirs
        self.logger = logger
        self.base_model_cfg = build_base_model_cfg(task_manager, tasks)
        self.engine: VoxTellEngine | None = None

    def run(self) -> None:
        self.engine = self.build_engine()
        self.configure_engine()
        self.log_model_summary()

        for task in self.tasks:
            task_dir = self.task_dir(task)
            task_dir.mkdir(parents=True, exist_ok=True)

            task_training_cfg = self.build_task_training_cfg(task, task_dir)
            save_task_snapshot(self.cfg, self.task_manager, task, task_training_cfg, task_dir, self.base_model_cfg)

            self.logger.info("Starting task %s (%s)", task.index + 1, task.name)
            self.logger.info("  Dataset: %s", task.dataset_cfg.get("name", "unknown"))
            self.logger.info("  Output: %s", task_dir)

            datamodule = VoxTellDataModule(task.dataset_cfg)
            trainer = Trainer(self.engine, task_training_cfg)
            self.logger.info("Training task %d/%d: %s", task.index + 1, len(self.tasks), task.name)
            trainer.fit(datamodule)
            self.logger.info("Task %s training completed", task.name)

            self.after_task(task, task_dir, task_training_cfg)

        self.logger.info(self.completion_message())

    def build_engine(self) -> VoxTellEngine:
        return VoxTellEngine(self.base_model_cfg)

    def configure_engine(self) -> None:
        return None

    def log_model_summary(self) -> None:
        engine = self.require_engine()
        total_params = sum(p.numel() for p in engine.model.parameters())
        trainable_params = sum(p.numel() for p in engine.model.parameters() if p.requires_grad)
        self.logger.info(
            "Model parameters - Total: %d | Trainable: %d | Frozen: %d",
            total_params,
            trainable_params,
            total_params - trainable_params,
        )

    def build_task_training_cfg(self, task: ContinualTask, task_dir: Path) -> dict[str, Any]:
        return merge_dicts(task.training_cfg, {"output_dir": str(task_dir)})

    def task_dir(self, task: ContinualTask) -> Path:
        return self.dirs["root"] / "tasks" / self.task_manager.task_dir_name(task)

    def require_engine(self) -> VoxTellEngine:
        if self.engine is None:
            raise RuntimeError("Strategy engine has not been initialized")
        return self.engine

    def completion_message(self) -> str:
        return f"{self.strategy_name} strategy completed successfully"

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
        return {
            **trained_model_cfg,
            "checkpoint_path": str(task_dir / "checkpoints" / checkpoint_name),
        }

    def after_task(
        self,
        task: ContinualTask,
        task_dir: Path,
        task_training_cfg: dict[str, Any],
    ) -> None:
        return None
