#!/usr/bin/env python3
"""CLI for continual learning training.

Wire continual learning strategies into the existing trainer/datamodule/builder
stack without forking separate training logic.

Usage:
    python -m src.cli.continual \
      --config configs/experiments/continual_example.yaml \
      --strategy replay \
      --model ./models/voxtell_v1.1/ \
      --output experiments/continual_run
"""

import argparse
import logging
from pathlib import Path

import yaml
import os

# Set Hugging Face Hub environment variables for ma
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from src.continual.strategies import create_strategy
from src.continual.task_manager import TaskManager
from src.data.datamodule import VoxTellDataModule
from src.training.trainer import Trainer
from src.utils.config import load_config, save_config
from src.utils.logging import get_logger


logger = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Continual Learning Training for VoxTell",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sequential finetuning (baseline)
  python -m src.cli.continual \\
    --config configs/experiments/continual_example.yaml \\
    --strategy finetune \\
    --model ./models/voxtell_v1.1/

  # Replay-based continual learning
  python -m src.cli.continual \\
    --config configs/experiments/continual_example.yaml \\
    --strategy replay \\
    --model ./models/voxtell_v1.1/ \\
    --output experiments/continual_replay
        """,
    )
    p.add_argument(
        "--config",
        required=True,
        help="Path to continual learning task sequence config (YAML)",
    )
    p.add_argument(
        "--strategy",
        choices=["finetune", "freeze", "replay", "adapter"],
        default="finetune",
        help="Continual learning strategy",
    )
    p.add_argument(
        "--model",
        required=True,
        help="Path to model directory containing plans.json",
    )
    p.add_argument(
        "--output",
        default="experiments/continual",
        help="Output directory for experiment runs",
    )
    p.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Compute device",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return p.parse_args()


def main():
    """Main training loop for continual learning."""
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_dir)

    # Load task sequence
    logger.info("Loading task sequence from %s", args.config)
    task_manager = TaskManager()
    task_manager.load_from_config(args.config)
    logger.info("Loaded %d tasks", len(task_manager))

    # Create continual learning strategy
    logger.info("Using strategy: %s", args.strategy)
    strategy = create_strategy(args.strategy)

    # Load base trainer config
    base_trainer_config = {
        "device": args.device,
        "seed": args.seed,
        "epochs": 1,  # Will be overridden per task
        "use_deterministic_algorithms": False,
        "optimizer": {
            "lr": 1e-4,
            "weight_decay": 3e-5,
            "momentum": 0.99,
            "poly_power": 1.0,
        },
    }

    # Train each task sequentially
    task_results = {}

    for task_idx, task in enumerate(task_manager.tasks):
        task_id = task.task_id
        task_output = output_dir / f"task_{task_idx:02d}_{task_id}"
        task_output.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 80)
        logger.info("TASK %d/%d: %s", task_idx + 1, len(task_manager), task_id)
        logger.info("Structures: %s", task.target_structures)
        logger.info("Epochs: %d, Batch Size: %d, LR: %g", 
                   task.num_epochs, task.batch_size, task.learning_rate)
        logger.info("=" * 80)

        # Invoke strategy hook for task start
        strategy_mods = strategy.on_task_start(task_id, task_idx, len(task_manager))

        # Build trainer config for this task
        trainer_config = base_trainer_config.copy()
        trainer_config["epochs"] = task.num_epochs
        trainer_config["output_dir"] = str(task_output)
        if "optimizer" in strategy_mods and "lr" in strategy_mods["optimizer"]:
            trainer_config["optimizer"]["lr"] = strategy_mods["optimizer"]["lr"]
        else:
            trainer_config["optimizer"]["lr"] = task.learning_rate

        # Build model config
        model_config = {
            "dir": args.model,
        }
        model_config.update(strategy.get_model_modifications())

        # Create trainer (using existing base trainer, not forking)
        trainer = Trainer(trainer_config, model_config)

        # Build datamodule for this task (with strategy adaptations)
        dm_config = task_manager.get_datamodule_config()
        dm_config.update({
            "patch_size": [192, 192, 192],
            "num_workers": 4,
        })
        datamodule = VoxTellDataModule(dm_config)

        # Invoke strategy hook for batch adaptation (e.g., add replay data)
        # This would require modifying DataModule to accept strategy
        # For now, document the integration point

        # Save task config snapshot
        task_config = {
            "task": task.to_dict(),
            "strategy": args.strategy,
            "strategy_mods": strategy_mods,
            "trainer": trainer_config,
            "model": model_config,
        }
        save_config(task_config, task_output / "task_config.yaml")

        try:
            # Train this task using existing trainer (no separate trainer!)
            logger.info("Training task %s...", task_id)
            trainer.fit(datamodule, model_dir=args.model)

            task_results[task_id] = {
                "status": "completed",
                "output_dir": str(task_output),
            }

            # Invoke strategy hook for task end
            strategy.on_task_end(task_id, {})

            logger.info("Task %s completed successfully", task_id)

        except Exception as e:
            logger.exception("Task %s failed: %s", task_id, e)
            task_results[task_id] = {
                "status": "failed",
                "error": str(e),
            }

    logger.info("=" * 80)
    logger.info("Continual Learning Training Complete")
    logger.info("=" * 80)
    logger.info("Task Results:")
    for task_id, result in task_results.items():
        logger.info("  %s: %s", task_id, result["status"])

    # Save experiment summary
    summary = {
        "strategy": args.strategy,
        "num_tasks": len(task_manager),
        "task_history": strategy.task_history if hasattr(strategy, "task_history") else [],
        "results": task_results,
    }
    save_config(summary, output_dir / "experiment_summary.yaml")
    logger.info("Experiment summary saved to %s/experiment_summary.yaml", output_dir)


if __name__ == "__main__":
    main()
