#!/usr/bin/env python3
import argparse
from pathlib import Path

from src.cli.common import set_offline_mode

set_offline_mode()

from src.continual import ContinualTaskManager, merge_dicts
from src.continual import apply_loralib_lora, save_lora_adapter
from src.continual.experiment_evaluator import evaluate_continual_experiment
from src.data.datamodule import VoxTellDataModule
from src.engine.model_engine import VoxTellEngine
from src.training.trainer import Trainer
from src.utils.config import load_config, save_config_snapshot
from src.utils.io import make_experiment_dir
from src.utils.logging import get_logger


logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    eval_group = parser.add_mutually_exclusive_group()
    eval_group.add_argument(
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="Run continual evaluation after training (default)",
    )
    eval_group.add_argument(
        "--no_evaluate",
        dest="evaluate",
        action="store_false",
        help="Skip the continual evaluation pass after training",
    )
    parser.set_defaults(evaluate=True)
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("Loading continual config from %s", args.config)
    cfg = load_config(Path(args.config))

    task_manager = ContinualTaskManager(cfg)
    tasks = task_manager.tasks()

    run_root = task_manager.output_dir
    dirs = make_experiment_dir(run_root, subdirs=["logs", "tasks"])
    save_config_snapshot(cfg, dirs["root"])
    logger.info("Continual run root: %s", dirs["root"])
    logger.info(
        "Strategy=%s | from_scratch=%s | tasks=%d",
        task_manager.strategy,
        task_manager.from_scratch,
        len(tasks),
    )

    base_model_cfg = merge_dicts(task_manager.base_model_cfg, tasks[0].model_cfg)
    if task_manager.from_scratch:
        base_model_cfg["reinit_weights"] = True

    engine = VoxTellEngine(base_model_cfg)

    base_model = engine.model

    if task_manager.strategy == "lora":
        logger.info("Applying LoRA adaptation to base model...")
        base_model = apply_loralib_lora(base_model, task_manager.lora_cfg)
        total_params = sum(p.numel() for p in base_model.parameters())
        trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
        logger.info(
            "Model parameters - Total: %d | Trainable: %d | Frozen: %d",
            total_params,
            trainable_params,
            total_params - trainable_params,
        )
        engine.model = base_model
        logger.info("LoRA adaptation complete. Model ready for continual learning.")

    for task in tasks:
        task_dir = dirs["root"] / "tasks" / task_manager.task_dir_name(task)
        task_dir.mkdir(parents=True, exist_ok=True)

        task_training_cfg = merge_dicts(task.training_cfg, {"output_dir": str(task_dir)})
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

        logger.info("Starting task %s (%s)", task.index + 1, task.name)
        logger.info("  Dataset: %s", task.dataset_cfg.get("name", "unknown"))
        logger.info("  Output: %s", task_dir)

        datamodule = VoxTellDataModule(task.dataset_cfg)
        trainer = Trainer(engine, task_training_cfg)
        logger.info("Training task %d/%d: %s", task.index + 1, len(tasks), task.name)
        trainer.fit(datamodule)
        logger.info("Task %s training completed", task.name)

        if task_manager.strategy == "lora":
            lora_bias = str(task_manager.lora_cfg.get("bias", "none"))
            save_lora_adapter(engine.model, task_dir / "lora_adapter.pt", bias=lora_bias)
            logger.info("Saved LoRA adapter for task %s", task.name)

    logger.info("Sequential fine-tuning baseline completed successfully")

    if args.evaluate:
        evaluate_continual_experiment(dirs["root"])
    else:
        logger.info("Skipping continual evaluation by request")


if __name__ == "__main__":
    main()
