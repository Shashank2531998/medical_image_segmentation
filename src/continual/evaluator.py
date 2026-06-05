from __future__ import annotations

from pathlib import Path
import csv
import json

import numpy as np
import pandas as pd

from src.continual.metrics import compute_all_metrics
from src.continual.task_manager import ContinualTaskManager, merge_dicts
from src.continual.strategies import get_strategy_class
from src.data.datamodule import VoxTellDataModule
from src.evaluation.test import Evaluator
from src.utils.config import load_config
from src.utils.logging import get_logger


logger = get_logger(__name__)


def _json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _save_evaluation_state(experiment_root: Path, raw_metrics, prompt_set: set[str], prompt_to_task: dict[str, int]) -> None:
    state_path = experiment_root / "evaluation_state.json"
    state = {
        "raw_metrics": _json_safe(raw_metrics),
        "prompt_set": sorted(prompt_set),
        "prompt_to_task": prompt_to_task,
    }
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)


def _load_evaluation_state(experiment_root: Path) -> tuple[list[list[dict | None]], set[str], dict[str, int]]:
    state_path = experiment_root / "evaluation_state.json"
    if not state_path.exists():
        return [[None for _ in range(0)] for _ in range(0)], set(), {}

    with open(state_path, "r") as f:
        state = json.load(f)

    raw_metrics = state.get("raw_metrics", [])
    prompt_set = set(state.get("prompt_set", []))
    prompt_to_task = dict(state.get("prompt_to_task", {}))
    return raw_metrics, prompt_set, prompt_to_task


def _load_experiment_config(experiment_root: Path) -> dict:
    config_path = experiment_root / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Expected continual config snapshot at {config_path}")
    return load_config(config_path)


class ContinualExperimentEvaluator:
    def __init__(self, experiment_root: str | Path, checkpoint_name: str = "best_model.pt") -> None:
        self.experiment_root = Path(experiment_root)
        self.checkpoint_name = checkpoint_name
        self.cfg = _load_experiment_config(self.experiment_root)

        self.task_manager = ContinualTaskManager(self.cfg)
        self.training_tasks = self.task_manager.tasks()
        if not self.training_tasks:
            raise ValueError("continual.tasks must contain at least one task")

        self.retention_tasks = self.task_manager.retention_tasks()
        self.evaluation_tasks = self.training_tasks + self.retention_tasks
        self.model_tasks = self.training_tasks

        self.pretrained_model_cfg = dict(self.task_manager.base_model_cfg)
        self.trained_model_cfgs = [merge_dicts(self.task_manager.base_model_cfg, task.model_cfg) for task in self.training_tasks]
        self.strategy_class = get_strategy_class(self.task_manager.strategy)

        self.row_labels = [f"{task.name} (CL)" for task in self.training_tasks]
        self.row_labels.extend([f"{task.name} (Retention)" for task in self.retention_tasks])
        self.column_labels = ["Pretrained"]
        self.column_labels.extend([f"After {task.name}" for task in self.training_tasks])

        logger.info("Evaluating continual experiment at %s", self.experiment_root)
        logger.info(
            "Tasks=%d | retention=%d | strategy=%s",
            len(self.training_tasks),
            len(self.retention_tasks),
            self.task_manager.strategy,
        )

    def build_model_evaluators(self) -> list[Evaluator]:
        evaluators: list[Evaluator] = []
        evaluators.append(
            Evaluator(
                model_cfg=self.pretrained_model_cfg,
                eval_cfg={"output_dir": str(self.experiment_root / "tasks" / "pretrained" / "test")},
            )
        )

        for task_idx, task in enumerate(self.training_tasks):
            task_dir = self.experiment_root / "tasks" / self.task_manager.task_dir_name(task)
            evaluation_spec = self.strategy_class.build_evaluation_spec(
                task_manager=self.task_manager,
                task=task,
                task_dir=task_dir,
                trained_model_cfg=self.trained_model_cfgs[task_idx],
                checkpoint_name=self.checkpoint_name,
            )
            evaluators.append(
                Evaluator(
                    model_cfg=evaluation_spec.model_cfg,
                    eval_cfg={"output_dir": str(task_dir / "test"), **evaluation_spec.eval_cfg},
                )
            )
        return evaluators

    def evaluate_models_on_tasks(self, evaluators: list[Evaluator]):
        rows = len(self.evaluation_tasks)
        cols = len(evaluators)

        saved_raw_metrics, saved_prompt_set, saved_prompt_to_task = _load_evaluation_state(self.experiment_root)
        raw_metrics = [[None for _ in range(cols)] for _ in range(rows)]

        if saved_raw_metrics:
            for r in range(min(rows, len(saved_raw_metrics))):
                for c in range(min(cols, len(saved_raw_metrics[r]))):
                    raw_metrics[r][c] = saved_raw_metrics[r][c]

        prompt_set: set[str] = set(saved_prompt_set)
        self.prompt_to_task: dict[str, int] = dict(saved_prompt_to_task)

        # ---------------------------
        # Collect metrics
        # ---------------------------
        for col_idx, evaluator in enumerate(evaluators):
            for row_idx, eval_task in enumerate(self.evaluation_tasks):
                if raw_metrics[row_idx][col_idx] is not None:
                    continue

                metrics = evaluator.evaluate(
                    VoxTellDataModule(eval_task.dataset_cfg)
                )

                raw_metrics[row_idx][col_idx] = metrics

                for k in metrics.keys():
                    if k not in ("Average Dice", "Average Loss"):
                        prompt_set.add(k)

                        # Remember which task this prompt belongs to
                        if k not in self.prompt_to_task:
                            self.prompt_to_task[k] = row_idx

                _save_evaluation_state(self.experiment_root, raw_metrics, prompt_set, self.prompt_to_task)

        self.prompt_list = sorted(prompt_set)

        # ---------------------------
        # Task-level matrix
        # ---------------------------
        self.task_matrix = np.full((rows, cols), np.nan, dtype=float)

        for r in range(rows):
            for c in range(cols):
                cell = raw_metrics[r][c]

                if cell is not None:
                    self.task_matrix[r, c] = float(cell["Average Dice"])

        # ---------------------------
        # Prompt-level matrices
        # ---------------------------
        self.per_prompt_matrices: dict[str, np.ndarray] = {}

        for prompt in self.prompt_list:
            mat = np.full((rows, cols), np.nan, dtype=float)

            for r in range(rows):
                for c in range(cols):
                    cell_metrics = raw_metrics[r][c] or {}

                    pm = cell_metrics.get(prompt)

                    if pm is not None:
                        mat[r, c] = float(pm["mean_dice"])

            self.per_prompt_matrices[prompt] = mat

        return self.task_matrix

    def compute_and_save_metrics(self):
        df = pd.DataFrame(
            self.task_matrix,
            index=self.row_labels,
            columns=self.column_labels,
        )

        df.to_csv(
            self.experiment_root / "eval_matrix.csv"
        )

        # ---------------------------
        # Continual Learning metrics (A, F)
        # ---------------------------

        metrics = compute_all_metrics(self.task_matrix, len(self.training_tasks))

        final_forgetting = (
            float(metrics["F"][-1])
            if len(metrics["F"])
            else float("nan")
        )

        valid_zs = metrics["ZS"][~np.isnan(metrics["ZS"])]

        avg_zs = (
            float(np.mean(valid_zs))
            if len(valid_zs)
            else float("nan")
        )

        avg_retention_drop = (
            float(np.mean(metrics["retention_drop"]))
            if len(metrics["retention_drop"])
            else float("nan")
        )

        logger.info("=" * 80)
        logger.info("Continual Learning Metrics")
        logger.info("=" * 80)
        logger.info(
            "Final Average Performance (A_final): %.4f",
            metrics["A_final"],
        )
        logger.info(
            "Final Forgetting: %.4f",
            final_forgetting,
        )
        logger.info(
            "Average Zero-Shot Transfer: %.4f",
            avg_zs,
        )
        logger.info(
            "Average Retention Drop: %.4f",
            avg_retention_drop,
        )

        if len(metrics["retention_drop"]):
            logger.info(
                "Retention Drops: %s",
                np.round(metrics["retention_drop"], 4).tolist(),
            )

        logger.info("=" * 80)


        metrics_to_save = {
            "Average Performance": metrics["A"].tolist(),
            "Forgetting measure": metrics["F"].tolist(),
            "Zero shot transfer": metrics["ZS"].tolist(),
            "A_final": float(metrics["A_final"]),
            "retention_drop": metrics["retention_drop"].tolist(),
            "Final Forgetting": final_forgetting,
            "Average Zero Shot Transfer": avg_zs,
            "Average Retention drop": avg_retention_drop
        }

        with open(
            self.experiment_root / "metrics.json",
            "w",
        ) as f:
            json.dump(
                metrics_to_save,
                f,
                indent=4,
            )

        # ==================================================
        # Save prompt-level matrices (diagnostics only)
        # ==================================================
        per_prompt_dir = self.experiment_root / "per_prompt"
        per_prompt_dir.mkdir(parents=True, exist_ok=True)

        for prompt, mat in self.per_prompt_matrices.items():
            safe_name = "".join(
                c if c.isalnum() or c in ("-", "_") else "_"
                for c in prompt
            ).strip("_")

            with open(
                per_prompt_dir / f"eval_matrix_{safe_name}.csv",
                "w",
                newline="",
            ) as f:
                writer = csv.writer(f)

                writer.writerow([""] + self.column_labels)

                for label, row in zip(self.row_labels, mat):
                    writer.writerow([label] + row.tolist())

        # Save metadata
        with open(per_prompt_dir / "prompts.json", "w") as f:
            json.dump(self.prompt_list, f, indent=2)

        with open(per_prompt_dir / "prompt_to_task.json", "w") as f:
            json.dump(self.prompt_to_task, f, indent=2)

    def run(self):
        evaluators = self.build_model_evaluators()
        self.evaluate_models_on_tasks(evaluators)
        self.compute_and_save_metrics()

        logger.info("Continual experiment evaluation completed successfully")


def evaluate_continual_experiment(
    experiment_root: str | Path,
    *,
    checkpoint_name: str = "best_model.pt",
) -> dict:
    evaluator = ContinualExperimentEvaluator(experiment_root, checkpoint_name)
    return evaluator.run()