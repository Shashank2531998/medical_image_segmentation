"""Continual learning module for VoxTell.

Provides strategy-based adaptation mechanisms that integrate with the existing
trainer/datamodule/builder stack without forking separate training logic.
"""

from src.continual.strategies import (
    ContinualStrategy,
    FinetuneStrategy,
    FreezeStrategy,
    ReplayStrategy,
    AdapterStrategy,
    create_strategy,
)
from src.continual.task_manager import TaskManager, ContinualTask

__all__ = [
    "ContinualStrategy",
    "FinetuneStrategy",
    "FreezeStrategy",
    "ReplayStrategy",
    "AdapterStrategy",
    "create_strategy",
    "TaskManager",
    "ContinualTask",
]
