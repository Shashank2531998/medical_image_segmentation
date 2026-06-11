from __future__ import annotations

from pathlib import Path
from typing import Any

from src.continual.task_manager import ContinualTask, ContinualTaskManager

from src.continual.strategies.base import BaseContinualStrategy
from src.continual.strategies.registry import register_strategy


@register_strategy
class NaiveSequentialFinetuningStrategy(BaseContinualStrategy):
    strategy_name = "naive_sequential_finetuning"


def run_naive_sequential_finetuning(
    *,
    cfg: dict[str, Any],
    task_manager: ContinualTaskManager,
    tasks: list[ContinualTask],
    dirs: dict[str, Path],
    logger,
) -> None:
    NaiveSequentialFinetuningStrategy(
        cfg=cfg,
        task_manager=task_manager,
        tasks=tasks,
        dirs=dirs,
        logger=logger,
    ).run()