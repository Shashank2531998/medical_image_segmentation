from __future__ import annotations

from pathlib import Path
from typing import Any

from src.continual.task_manager import ContinualTask, ContinualTaskManager
from src.utils.logging import get_logger

from .strategies.registry import create_strategy


logger = get_logger(__name__)


def run_continual_strategy(
    cfg: dict[str, Any],
    task_manager: ContinualTaskManager,
    tasks: list[ContinualTask],
    dirs: dict[str, Path],
) -> None:
    strategy = create_strategy(
        task_manager.strategy,
        cfg=cfg,
        task_manager=task_manager,
        tasks=tasks,
        dirs=dirs,
        logger=logger,
    )
    strategy.run()