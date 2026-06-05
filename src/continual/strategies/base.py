from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from src.continual.utils import build_base_model_cfg, save_task_snapshot
from src.continual.task_manager import ContinualTask, ContinualTaskManager, merge_dicts
from src.data.datamodule import VoxTellDataModule
from src.engine.model_engine import VoxTellEngine
from src.training.trainer import Trainer


@dataclass(frozen=True)
class EvaluationSpec:
    model_cfg: dict[str, Any]
    eval_cfg: dict[str, Any] = field(default_factory=dict)


class BaseContinualStrategy(ABC):
    strategy_name: str = ""

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
        self.state_path = self.dirs["root"] / "continual_state.json"

    def _timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _default_state(self) -> dict[str, Any]:
        return {
            "version": 1,
            "strategy": self.task_manager.strategy,
            "status": "not_started",
            "current_task_index": 0,
            "current_task_name": self.tasks[0].name if self.tasks else None,
            "current_checkpoint": None,
            "completed_tasks": [],
            "updated_at": self._timestamp(),
        }

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return self._default_state()
        with open(self.state_path, "r") as f:
            state = json.load(f)
        if state.get("strategy") != self.task_manager.strategy:
            raise ValueError(
                f"Resume state strategy mismatch at {self.state_path}: "
                f"expected={self.task_manager.strategy}, found={state.get('strategy')}"
            )
        return state

    def _save_state(self, state: dict[str, Any]) -> None:
        state["updated_at"] = self._timestamp()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w") as f:
            json.dump(state, f, indent=2)

    def _resolve_resume_plan(self, state: dict[str, Any]) -> tuple[int, Path | None]:
        if not self.tasks:
            return 0, None

        if str(state.get("status", "")) == "completed":
            return len(self.tasks), None

        completed = {int(i) for i in state.get("completed_tasks", [])}
        first_incomplete = 0
        while first_incomplete < len(self.tasks) and first_incomplete in completed:
            first_incomplete += 1

        current_task_index = int(state.get("current_task_index", first_incomplete))
        start_index = max(first_incomplete, min(current_task_index, len(self.tasks) - 1))

        checkpoint_raw = state.get("current_checkpoint")
        checkpoint_path = Path(checkpoint_raw) if checkpoint_raw else None
        if checkpoint_path is not None and not checkpoint_path.exists():
            self.logger.warning("Resume checkpoint not found at %s. Continuing without intra-task resume.", checkpoint_path)
            checkpoint_path = None

        # If there is no in-progress checkpoint for the start task, begin from task start.
        if start_index != current_task_index:
            checkpoint_path = None

        return start_index, checkpoint_path

    def run(self) -> None:
        state = self._load_state()
        start_task_index, resume_checkpoint = self._resolve_resume_plan(state)

        if start_task_index >= len(self.tasks):
            self.logger.info("All continual tasks are already completed for run root: %s", self.dirs["root"])
            return

        self.logger.info(
            "Continual resume plan | start_task=%d/%d | checkpoint=%s",
            start_task_index + 1,
            len(self.tasks),
            str(resume_checkpoint) if resume_checkpoint else "<none>",
        )

        for task in self.tasks[start_task_index:]:
            self.engine = self.build_engine()
            self.configure_engine()

            task_dir = self.dirs["root"] / "tasks" / self.task_manager.task_dir_name(task)
            task_dir.mkdir(parents=True, exist_ok=True)

            task_training_cfg = merge_dicts(task.training_cfg, {"output_dir": str(task_dir)})
            save_task_snapshot(self.cfg, self.task_manager, task, task_training_cfg, task_dir, self.base_model_cfg)

            self.logger.info("Starting task %s (%s)", task.index + 1, task.name)
            self.logger.info("  Dataset: %s", task.dataset_cfg.get("name", "unknown"))
            self.logger.info("  Output: %s", task_dir)

            state["status"] = "in_progress"
            state["current_task_index"] = task.index
            state["current_task_name"] = task.name
            if task.index != start_task_index:
                state["current_checkpoint"] = None
            self._save_state(state)

            datamodule = VoxTellDataModule(task.dataset_cfg)
            trainer = Trainer(self.engine, task_training_cfg, hooks=self, task=task)
            self.logger.info("Training task %d/%d: %s", task.index + 1, len(self.tasks), task.name)
            task_resume_checkpoint = resume_checkpoint if task.index == start_task_index else None
            task_metrics = trainer.fit(datamodule, resume_from=task_resume_checkpoint)
            self.logger.info("Task %s training completed", task.name)

            completed = {int(i) for i in state.get("completed_tasks", [])}
            completed.add(task.index)
            state["completed_tasks"] = sorted(completed)
            state["current_checkpoint"] = None
            self._save_state(state)

            self.after_task(task, task_dir, task_training_cfg, task_metrics=task_metrics)

        state["status"] = "completed"
        state["current_checkpoint"] = None
        self._save_state(state)

        self.logger.info(f"{self.strategy_name} strategy completed successfully")

    def build_engine(self) -> VoxTellEngine:
        return VoxTellEngine(self.base_model_cfg)

    def configure_engine(self) -> None:
        return None

    def on_train_batch_end(
        self,
        task: ContinualTask,
        batch_idx: int,
        epoch: int,
        batch_loss: float,
    ) -> None:
        return None

    def on_recovery_checkpoint(
        self,
        *,
        task: ContinualTask,
        checkpoint_path: Path,
        epoch: int,
        batch_idx: int,
        stage: str,
    ) -> None:
        try:
            state = self._load_state()
        except Exception:
            state = self._default_state()

        state["status"] = "in_progress"
        state["current_task_index"] = task.index
        state["current_task_name"] = task.name
        state["current_checkpoint"] = str(checkpoint_path)
        state["resume_position"] = {
            "epoch": int(epoch),
            "batch_idx": int(batch_idx),
            "stage": str(stage),
        }
        self._save_state(state)

    def require_engine(self) -> VoxTellEngine:
        if self.engine is None:
            raise RuntimeError("Strategy engine has not been initialized")
        return self.engine

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
        return EvaluationSpec(
            model_cfg={
                **trained_model_cfg,
                "checkpoint_path": str(task_dir / "checkpoints" / checkpoint_name),
            },
        )

    def after_task(
        self,
        task: ContinualTask,
        task_dir: Path,
        task_training_cfg: dict[str, Any],
        task_metrics: dict[str, Any] | None = None,
    ) -> None:
        self.base_model_cfg={
            **self.base_model_cfg,
            "checkpoint_path": str(task_dir / "checkpoints" / "best_model.pt"),
        }
