from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def merge_dicts(base: dict[str, Any] | None, override: dict[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = deepcopy(base or {})
    for key, value in (override or {}).items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


@dataclass(frozen=True)
class ContinualTask:
    index: int
    name: str
    dataset_cfg: dict[str, Any]
    training_cfg: dict[str, Any]
    model_cfg: dict[str, Any]
    is_retention: bool = False


class ContinualTaskManager:
    def __init__(self, cfg: dict[str, Any] | None = None):
        self.cfg = cfg or {}
        self.base_dataset_cfg = self.cfg.get("dataset", {})
        self.base_training_cfg = self.cfg.get("training", {})
        self.base_model_cfg = self.cfg.get("model", {})
        self.continual_cfg = self.cfg.get("continual", {})

    @property
    def strategy(self) -> str:
        return str(self.continual_cfg.get("strategy", self.continual_cfg.get("baseline", "naive_sequential_finetuning")))

    @property
    def output_dir(self) -> str:
        return str(self.continual_cfg.get("output_dir", self.base_training_cfg.get("output_dir", "./experiments/continual")))

    @property
    def from_scratch(self) -> bool:
        return bool(self.continual_cfg.get("from_scratch", True))

    @property
    def lora_cfg(self) -> dict[str, Any]:
        return dict(self.continual_cfg.get("lora", {}))

    @property
    def cpe_clip_cfg(self) -> dict[str, Any]:
        return dict(self.continual_cfg.get("cpe_clip", {}))

    def _build_tasks(self, raw_tasks: list[dict[str, Any]], *, is_retention: bool = False) -> list[ContinualTask]:
        tasks: list[ContinualTask] = []
        for index, task_cfg in enumerate(raw_tasks):
            default_name = "retention_task_%02d" % (index + 1) if is_retention else "task_%02d" % (index + 1)
            task_name = str(task_cfg.get("name", default_name))
            task_dataset_cfg = merge_dicts(self.base_dataset_cfg, task_cfg.get("dataset", {}))
            task_training_cfg = merge_dicts(self.base_training_cfg, task_cfg.get("training", {}))
            task_model_cfg = merge_dicts(self.base_model_cfg, task_cfg.get("model", {}))
            tasks.append(
                ContinualTask(
                    index=index,
                    name=task_name,
                    dataset_cfg=task_dataset_cfg,
                    training_cfg=task_training_cfg,
                    model_cfg=task_model_cfg,
                    is_retention=is_retention,
                )
            )
        return tasks

    def retention_tasks(self) -> list[ContinualTask]:
        raw_tasks = list(self.continual_cfg.get("retention_tasks", []))
        return self._build_tasks(raw_tasks, is_retention=True)

    def tasks(self, include_retention: bool = False) -> list[ContinualTask]:
        raw_tasks = list(self.continual_cfg.get("tasks", []))
        if not raw_tasks:
            raise ValueError("continual.tasks must contain at least one task")

        training_tasks = self._build_tasks(raw_tasks)
        if include_retention:
            return self.retention_tasks() + training_tasks
        return training_tasks

    @staticmethod
    def task_dir_name(task: ContinualTask, task_number: int | None = None) -> str:
        slug = task.name.lower().replace(" ", "_").replace("/", "_")
        number = task.index + 1 if task_number is None else task_number
        return f"task_{number:02d}_{slug}"
