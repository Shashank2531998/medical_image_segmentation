from __future__ import annotations

from pathlib import Path

import numpy as np

from src.continual.evaluation import compute_all_metrics
from src.continual.task_manager import ContinualTaskManager, merge_dicts
from src.data.datamodule import VoxTellDataModule
from src.evaluation.test import Evaluator
from src.utils.config import load_config
from src.utils.logging import get_logger


logger = get_logger(__name__)


def _load_experiment_config(experiment_root: Path) -> dict:
    config_path = experiment_root / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Expected continual config snapshot at {config_path}")
    return load_config(config_path)


def evaluate_continual_experiment(
    experiment_root: str | Path,
    *,
    checkpoint_name: str = "final_checkpoint.pt",
) -> dict:
    experiment_root = Path(experiment_root)
    cfg = _load_experiment_config(experiment_root)

    task_manager = ContinualTaskManager(cfg)
    tasks = task_manager.tasks()
    if not tasks:
        raise ValueError("continual.tasks must contain at least one task")

    base_model_cfg = merge_dicts(task_manager.base_model_cfg, tasks[0].model_cfg)
    eval_matrix = np.full((len(tasks), len(tasks)), np.nan, dtype=float)

    logger.info("Evaluating continual experiment at %s", experiment_root)
    logger.info("Tasks=%d | strategy=%s", len(tasks), task_manager.strategy)

    for task in tasks:
        task_dir = experiment_root / "tasks" / task_manager.task_dir_name(task)
        if task_manager.strategy == "lora":
            adapter_path = task_dir / "lora_adapter.pt"
            if not adapter_path.exists():
                raise FileNotFoundError(f"Expected LoRA adapter for task {task.name} at {adapter_path}")

            model_cfg_for_eval = {
                **base_model_cfg,
                "lora_cfg": dict(task_manager.lora_cfg),
                "lora_adapter_path": str(adapter_path),
            }
            logger.info("Loading LoRA adapter for task %s from %s", task.name, adapter_path)
            evaluator = Evaluator(
                model_cfg=model_cfg_for_eval,
                eval_cfg={"output_dir": str(task_dir / "eval_out")},
            )
        else:
            ckpt_src = task_dir / "checkpoints" / checkpoint_name
            if not ckpt_src.exists():
                raise FileNotFoundError(f"Expected checkpoint for task {task.name} at {ckpt_src}")

            model_cfg_for_eval = {
                **base_model_cfg,
                "checkpoint_path": str(ckpt_src),
            }
            evaluator = Evaluator(
                model_cfg=model_cfg_for_eval,
                eval_cfg={"output_dir": str(task_dir / "eval_out")},
            )

        completed_tasks = tasks[: task.index + 1]
        for eval_idx, eval_task in enumerate(completed_tasks):
            metrics = evaluator.evaluate(VoxTellDataModule(eval_task.dataset_cfg))
            eval_matrix[task.index, eval_idx] = float(metrics.get("Average Dice", np.nan))

        np.save(experiment_root / "eval_matrix.npy", eval_matrix)

    metrics = compute_all_metrics(eval_matrix)
    A = metrics["A"]
    F = metrics["F"]

    np.savetxt(experiment_root / "eval_matrix.csv", eval_matrix, delimiter=",")
    np.savetxt(experiment_root / "A_values.csv", A, delimiter=",")
    np.savetxt(experiment_root / "F_values.csv", F, delimiter=",")

    for t in range(len(tasks)):
        logger.info("Task %d | A=%.4f | F=%.4f", t + 1, float(A[t]), float(F[t]))

    logger.info("Continual experiment evaluation completed successfully")
    return {"eval_matrix": eval_matrix, "A": A, "F": F}