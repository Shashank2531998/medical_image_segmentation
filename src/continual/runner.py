from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

from src.continual.task_manager import ContinualTask, ContinualTaskManager
from src.utils.logging import get_logger

from . import strategies as _strategy_registry  # noqa: F401
from .strategies.registry import create_strategy


def _load_builtin_strategies() -> None:
    strategies_module = import_module("src.continual.strategies")
    load_builtin_strategies = getattr(strategies_module, "load_builtin_strategies")
    load_builtin_strategies()


logger = get_logger(__name__)


def run_continual_strategy(
    cfg: dict[str, Any],
    task_manager: ContinualTaskManager,
    tasks: list[ContinualTask],
    dirs: dict[str, Path],
    logger
) -> None:
    _load_builtin_strategies()
    strategy = create_strategy(
        task_manager.strategy,
        cfg=cfg,
        task_manager=task_manager,
        tasks=tasks,
        dirs=dirs,
        logger=logger,
    )
    strategy.run()