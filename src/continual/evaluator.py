from __future__ import annotations

from pathlib import Path
import csv
import json

import numpy as np

from src.continual.metrics import compute_all_metrics
from src.continual.task_manager import ContinualTaskManager, merge_dicts
from src.continual.strategies import get_strategy_class
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

        self.row_labels = [f"Training task {i + 1}" for i in range(len(self.training_tasks))]
        self.row_labels.extend([f"Retention task {i + 1}" for i in range(len(self.retention_tasks))])
        self.column_labels = ["Pretrained model"]
        self.column_labels.extend([f"Model trained on Task {i + 1}" for i in range(len(self.training_tasks))])

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
                eval_cfg={"output_dir": str(self.experiment_root / "tasks" / "pretrained" / "eval_out")},
            )
        )

        for task_idx, task in enumerate(self.training_tasks):
            task_dir = self.experiment_root / "tasks" / self.task_manager.task_dir_name(task)
            model_cfg_for_eval = self.strategy_class.build_model_cfg_for_evaluation(
                task_manager=self.task_manager,
                task=task,
                task_dir=task_dir,
                trained_model_cfg=self.trained_model_cfgs[task_idx],
                checkpoint_name=self.checkpoint_name,
            )
            evaluators.append(
                Evaluator(
                    model_cfg=model_cfg_for_eval,
                    eval_cfg={"output_dir": str(task_dir / "test")},
                )
            )
        return evaluators

    def evaluate_models_on_tasks(self, evaluators: list[Evaluator]) -> np.ndarray:
        # store raw metrics per cell so we can build per-prompt matrices afterwards
        rows = len(self.evaluation_tasks)
        cols = len(evaluators)
        raw_metrics = [[None for _ in range(cols)] for _ in range(rows)]
        prompt_set: set[str] = set()

        for col_idx, evaluator in enumerate(evaluators):
            for row_idx, eval_task in enumerate(self.evaluation_tasks):
                metrics = evaluator.evaluate(VoxTellDataModule(eval_task.dataset_cfg))
                raw_metrics[row_idx][col_idx] = metrics
                # collect prompt keys (exclude overall summary keys)
                for k in metrics.keys():
                    if k not in ("Average Dice", "Average Loss"):
                        prompt_set.add(k)

        self.prompt_list = sorted(prompt_set)
        # build per-prompt matrices: dict prompt -> np.ndarray(rows, cols)
        self.per_prompt_matrices: dict[str, np.ndarray] = {}
        for p in self.prompt_list:
            mat = np.full((rows, cols), np.nan, dtype=float)
            for r in range(rows):
                for c in range(cols):
                    cell_metrics = raw_metrics[r][c] or {}
                    val = np.nan
                    pm = cell_metrics.get(p)
                    val = float(pm["mean_dice"])
                    mat[r, c] = val
            self.per_prompt_matrices[p] = mat

        return self.per_prompt_matrices

    def compute_and_save_metrics(self):
        # save per-prompt matrices and metadata
        per_prompt_dir = self.experiment_root / "per_prompt"
        per_prompt_dir.mkdir(parents=True, exist_ok=True)
        for p, mat in self.per_prompt_matrices.items():
            safe_name = "".join(c if c.isalnum() or c in ("-","_") else "_" for c in p).strip("_")
            np.save(per_prompt_dir / f"eval_matrix_{safe_name}.npy", mat)
            with open(per_prompt_dir / f"eval_matrix_{safe_name}.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([""] + self.column_labels)
                for label, row in zip(self.row_labels, mat):
                    writer.writerow([label] + row.tolist())

        # save prompt list
        with open(per_prompt_dir / "prompts.json", "w") as f:
            json.dump(self.prompt_list, f, indent=2)

        # compute and save per-prompt metrics
        per_prompt_results: dict[str, dict[str, np.ndarray]] = {}
        for p, mat in self.per_prompt_matrices.items():
            train_matrix = mat[: len(self.training_tasks), 1:]
            metrics = compute_all_metrics(train_matrix)
            A = metrics["A"]
            F = metrics["F"]
            per_prompt_results[p] = {"A": A, "F": F}
            # save per-prompt A/F
            safe_name = "".join(c if c.isalnum() or c in ("-","_") else "_" for c in p).strip("_")
            np.savetxt(self.experiment_root / "per_prompt" / f"A_values_{safe_name}.csv", A, delimiter=",")
            np.savetxt(self.experiment_root / "per_prompt" / f"F_values_{safe_name}.csv", F, delimiter=",")

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