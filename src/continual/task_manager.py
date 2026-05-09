"""Task manager for continual learning task sequencing and evaluation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from src.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class ContinualTask:
    """Single task in a continual learning sequence."""

    task_id: str
    """Unique identifier for this task."""

    dataset_name: str
    """Dataset adapter name (e.g., 'aeropath', 'veela')."""

    dataset_root: str
    """Path to dataset root directory."""

    train_root: Optional[str] = None
    """Optional separate train directory (if None, uses dataset_root with split)."""

    val_root: Optional[str] = None
    """Optional separate validation directory."""

    val_fraction: float = 0.2
    """Fraction of data to use for validation if no val_root."""

    batch_size: int = 2
    """Training batch size for this task."""

    num_epochs: int = 10
    """Number of epochs to train on this task."""

    learning_rate: float = 1e-4
    """Learning rate for this task."""

    target_structures: list[str] = field(default_factory=list)
    """Anatomical structures to segment (e.g., ['trachea', 'lung'])."""

    seed: int = 42
    """Random seed for reproducibility."""

    def to_dict(self) -> dict:
        """Export task to dictionary."""
        return {
            "task_id": self.task_id,
            "dataset_name": self.dataset_name,
            "dataset_root": self.dataset_root,
            "train_root": self.train_root,
            "val_root": self.val_root,
            "val_fraction": self.val_fraction,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "target_structures": self.target_structures,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ContinualTask:
        """Create task from dictionary."""
        return cls(**data)


class TaskManager:
    """Manages task sequences for continual learning.
    
    Responsibilities:
    - Load and sequence multiple tasks
    - Expose task-aware prompts and splits
    - Track evaluation order
    - Maintain task metadata during training
    """

    def __init__(self):
        """Initialize task manager."""
        self.tasks: list[ContinualTask] = []
        self.current_task_idx: int = 0
        self.completed_tasks: list[str] = []

    def add_task(self, task: ContinualTask) -> None:
        """Add a task to the sequence."""
        self.tasks.append(task)
        logger.info("Added task %s (total: %d tasks)", task.task_id, len(self.tasks))

    def load_from_config(self, config_path: str | Path) -> None:
        """Load task sequence from YAML configuration.
        
        Config format:
        ```yaml
        strategy: "replay"  # or finetune, freeze, adapter
        tasks:
          - task_id: "task_0_airway"
            dataset_name: "aeropath"
            dataset_root: "./data/AeroPath/"
            batch_size: 2
            num_epochs: 10
            target_structures: ["trachea"]
          - task_id: "task_1_lung"
            ...
        ```
        """
        config_path = Path(config_path)
        with open(config_path) as f:
            config = yaml.safe_load(f)

        tasks_config = config.get("tasks", [])
        if not tasks_config:
            raise ValueError(f"No tasks found in {config_path}")

        logger.info("Loading %d tasks from %s", len(tasks_config), config_path)

        for task_config in tasks_config:
            task = ContinualTask.from_dict(task_config)
            self.add_task(task)

    def get_task(self, idx: Optional[int] = None) -> Optional[ContinualTask]:
        """Get task by index (None = current task)."""
        if idx is None:
            idx = self.current_task_idx
        if 0 <= idx < len(self.tasks):
            return self.tasks[idx]
        return None

    def current_task(self) -> Optional[ContinualTask]:
        """Get current task."""
        return self.get_task(self.current_task_idx)

    def next_task(self) -> bool:
        """Advance to next task. Return True if there is a next task."""
        if self.current_task_idx < len(self.tasks) - 1:
            task = self.current_task()
            if task:
                self.completed_tasks.append(task.task_id)
            self.current_task_idx += 1
            return True
        return False

    def __len__(self) -> int:
        """Total number of tasks."""
        return len(self.tasks)

    def __iter__(self):
        """Iterate over tasks in order."""
        return iter(self.tasks)

    def task_index(self) -> tuple[int, int]:
        """Get (current_idx, total_tasks)."""
        return self.current_task_idx, len(self.tasks)

    def get_datamodule_config(self) -> dict:
        """Get DataModule config for current task.
        
        Returns config dict compatible with VoxTellDataModule.
        """
        task = self.current_task()
        if not task:
            return {}

        return {
            "name": task.dataset_name,
            "train_root": task.train_root or task.dataset_root,
            "val_root": task.val_root,
            "val_fraction": task.val_fraction,
            "seed": task.seed,
            "batch_size": task.batch_size,
        }

    def get_trainer_config(self) -> dict:
        """Get Trainer config for current task.
        
        Returns config dict for optimizer, learning rate, etc.
        """
        task = self.current_task()
        if not task:
            return {}

        return {
            "epochs": task.num_epochs,
            "optimizer": {
                "lr": task.learning_rate,
                "weight_decay": 3e-5,
                "momentum": 0.99,
                "poly_power": 1.0,
            },
        }

    def get_task_structures(self) -> list[str]:
        """Get anatomical structures for current task."""
        task = self.current_task()
        return task.target_structures if task else []

    def get_evaluation_order(self) -> list[str]:
        """Get order of tasks for final evaluation (all completed + current)."""
        eval_order = self.completed_tasks.copy()
        task = self.current_task()
        if task and task.task_id not in eval_order:
            eval_order.append(task.task_id)
        return eval_order

    def to_dict(self) -> dict:
        """Export task sequence to dictionary."""
        return {
            "tasks": [task.to_dict() for task in self.tasks],
            "current_task_idx": self.current_task_idx,
            "completed_tasks": self.completed_tasks,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TaskManager:
        """Create TaskManager from dictionary."""
        manager = cls()
        for task_data in data.get("tasks", []):
            manager.add_task(ContinualTask.from_dict(task_data))
        manager.current_task_idx = data.get("current_task_idx", 0)
        manager.completed_tasks = data.get("completed_tasks", [])
        return manager
