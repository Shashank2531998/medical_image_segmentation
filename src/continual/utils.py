from __future__ import annotations

from pathlib import Path
from typing import Any

from src.continual.task_manager import ContinualTask, ContinualTaskManager, merge_dicts
from src.utils.config import save_config_snapshot


def build_base_model_cfg(task_manager: ContinualTaskManager, tasks: list[ContinualTask]) -> dict[str, Any]:
    base_model_cfg = merge_dicts(task_manager.base_model_cfg, tasks[0].model_cfg)
    if task_manager.from_scratch:
        base_model_cfg["reinit_weights"] = True
    return base_model_cfg


def save_task_snapshot(
    cfg: dict[str, Any],
    task_manager: ContinualTaskManager,
    task: ContinualTask,
    task_training_cfg: dict[str, Any],
    task_dir: Path,
    base_model_cfg: dict[str, Any],
) -> None:
    task_cfg_snapshot = merge_dicts(
        cfg,
        {
            "dataset": task.dataset_cfg,
            "training": task_training_cfg,
            "model": base_model_cfg,
            "continual": {
                **task_manager.continual_cfg,
                "active_task": task.name,
            },
        },
    )
    save_config_snapshot(task_cfg_snapshot, task_dir)