"""Continual learning strategies for VoxTell.

Strategies define how the trainer handles multiple sequential tasks:
- Finetune: Basic sequential learning (baseline, no mitigation)
- Freeze: Lock encoder, train only decoder/text fusion
- Replay: Mix exemplars from previous tasks with current task data
- Adapter: Add task-specific adapter layers while keeping backbone frozen
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.utils.logging import get_logger


logger = get_logger(__name__)


class ContinualStrategy(ABC):
    """Base class for continual learning strategies."""

    @abstractmethod
    def on_task_start(self, task_id: str, task_idx: int, total_tasks: int) -> dict:
        """Called when starting a new task. Return config updates."""
        pass

    @abstractmethod
    def on_task_end(self, task_id: str, metrics: dict) -> None:
        """Called when a task completes. Store exemplars, update adapter, etc."""
        pass

    @abstractmethod
    def adapt_batch(
        self, batch: dict, task_id: str, task_idx: int
    ) -> dict:
        """Optionally modify batch (e.g., add replay exemplars)."""
        return batch

    @abstractmethod
    def get_model_modifications(self) -> dict:
        """Return model modifications (frozen layers, adapters, etc.)."""
        pass


@dataclass
class FinetuneStrategy(ContinualStrategy):
    """Basic sequential finetuning without catastrophic forgetting mitigation."""

    name: str = "finetune"
    description: str = "Sequential finetuning with no explicit anti-forgetting measures"
    task_history: list[str] = None

    def __post_init__(self):
        if self.task_history is None:
            self.task_history = []

    def on_task_start(self, task_id: str, task_idx: int, total_tasks: int) -> dict:
        """No modifications needed for baseline finetuning."""
        logger.info(
            "FinetuneStrategy: Starting task %d/%d (%s)", task_idx + 1, total_tasks, task_id
        )
        return {}

    def on_task_end(self, task_id: str, metrics: dict) -> None:
        """Track completed tasks."""
        self.task_history.append(task_id)
        logger.info(
            "FinetuneStrategy: Completed task %s. History: %s",
            task_id,
            self.task_history,
        )

    def adapt_batch(self, batch: dict, task_id: str, task_idx: int) -> dict:
        """No batch adaptation needed."""
        return batch

    def get_model_modifications(self) -> dict:
        """No model modifications."""
        return {}


@dataclass
class FreezeStrategy(ContinualStrategy):
    """Freeze encoder, train only decoder for new tasks."""

    name: str = "freeze"
    description: str = "Freeze encoder, adapt decoder and text fusion to new tasks"
    task_history: list[str] = None
    frozen_layers: list[str] = None

    def __post_init__(self):
        if self.task_history is None:
            self.task_history = []
        if self.frozen_layers is None:
            self.frozen_layers = ["encoder"]

    def on_task_start(self, task_id: str, task_idx: int, total_tasks: int) -> dict:
        """Freeze encoder for tasks after the first."""
        logger.info(
            "FreezeStrategy: Starting task %d/%d (%s)", task_idx + 1, total_tasks, task_id
        )
        if task_idx > 0:
            logger.info("FreezeStrategy: Freezing encoder for task %s", task_id)
            return {"freeze_encoder": True}
        return {}

    def on_task_end(self, task_id: str, metrics: dict) -> None:
        """Track completed tasks."""
        self.task_history.append(task_id)

    def adapt_batch(self, batch: dict, task_id: str, task_idx: int) -> dict:
        """No batch adaptation needed."""
        return batch

    def get_model_modifications(self) -> dict:
        """Return layers to freeze."""
        return {"freeze_encoder": True} if len(self.task_history) > 0 else {}


@dataclass
class ReplayStrategy(ContinualStrategy):
    """Mix exemplars from previous tasks with current task data.
    
    Requires exemplar storage from completed tasks.
    """

    name: str = "replay"
    description: str = "Mix exemplars from previous tasks to prevent catastrophic forgetting"
    exemplars_per_task: int = 100
    replay_weight: float = 0.2  # 20% replay, 80% new data
    task_history: list[str] = None
    task_exemplars: dict[str, list[tuple]] = None

    def __post_init__(self):
        if self.task_history is None:
            self.task_history = []
        if self.task_exemplars is None:
            self.task_exemplars = {}

    def on_task_start(self, task_id: str, task_idx: int, total_tasks: int) -> dict:
        """Prepare replay exemplar loading."""
        logger.info(
            "ReplayStrategy: Starting task %d/%d (%s)", task_idx + 1, total_tasks, task_id
        )
        if task_idx > 0:
            n_replay_tasks = len(self.task_history)
            logger.info(
                "ReplayStrategy: Will mix exemplars from %d previous task(s)",
                n_replay_tasks,
            )
            return {"use_replay": True, "replay_weight": self.replay_weight}
        return {}

    def on_task_end(self, task_id: str, metrics: dict) -> None:
        """Store exemplars from completed task."""
        self.task_history.append(task_id)
        # Exemplars would be populated by DataModule
        if task_id not in self.task_exemplars:
            self.task_exemplars[task_id] = []
        logger.info(
            "ReplayStrategy: Stored exemplars for task %s (total: %d)",
            task_id,
            len(self.task_exemplars.get(task_id, [])),
        )

    def adapt_batch(self, batch: dict, task_id: str, task_idx: int) -> dict:
        """Batch is already mixed by DataModule if use_replay=True."""
        return batch

    def get_model_modifications(self) -> dict:
        """No model modifications needed."""
        return {}


@dataclass
class AdapterStrategy(ContinualStrategy):
    """Add task-specific adapter layers while keeping backbone frozen."""

    name: str = "adapter"
    description: str = "Task-specific adapters with frozen backbone"
    adapter_dim: int = 64
    task_history: list[str] = None
    task_adapters: dict[str, dict] = None

    def __post_init__(self):
        if self.task_history is None:
            self.task_history = []
        if self.task_adapters is None:
            self.task_adapters = {}

    def on_task_start(self, task_id: str, task_idx: int, total_tasks: int) -> dict:
        """Create/activate task-specific adapter."""
        logger.info(
            "AdapterStrategy: Starting task %d/%d (%s)", task_idx + 1, total_tasks, task_id
        )
        if task_id not in self.task_adapters:
            self.task_adapters[task_id] = {
                "adapter_dim": self.adapter_dim,
                "is_new": True,
            }
        return {
            "use_adapters": True,
            "task_id": task_id,
            "adapter_config": self.task_adapters[task_id],
        }

    def on_task_end(self, task_id: str, metrics: dict) -> None:
        """Mark adapter as trained."""
        self.task_history.append(task_id)
        if task_id in self.task_adapters:
            self.task_adapters[task_id]["is_new"] = False
        logger.info("AdapterStrategy: Adapter for task %s marked as trained", task_id)

    def adapt_batch(self, batch: dict, task_id: str, task_idx: int) -> dict:
        """No batch adaptation needed."""
        return batch

    def get_model_modifications(self) -> dict:
        """Return adapter configuration for model."""
        return {
            "use_adapters": True,
            "adapters": self.task_adapters,
        }


def create_strategy(strategy_name: str, **kwargs) -> ContinualStrategy:
    """Factory function to create strategy instances.
    
    Args:
        strategy_name: "finetune", "freeze", "replay", or "adapter"
        **kwargs: Strategy-specific configuration
        
    Returns:
        ContinualStrategy instance
    """
    strategies = {
        "finetune": FinetuneStrategy,
        "freeze": FreezeStrategy,
        "replay": ReplayStrategy,
        "adapter": AdapterStrategy,
    }

    if strategy_name not in strategies:
        raise ValueError(
            f"Unknown strategy: {strategy_name}. Choose from {list(strategies.keys())}"
        )

    strategy_class = strategies[strategy_name]
    return strategy_class(**kwargs)
