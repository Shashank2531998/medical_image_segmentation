#!/usr/bin/env python3
import argparse
from pathlib import Path

from src.cli.common import set_offline_mode

set_offline_mode()

from src.continual.task_manager import ContinualTaskManager
from src.continual.evaluator import evaluate_continual_experiment
from src.continual.runner import run_continual_strategy
from src.utils.config import load_config, save_config_snapshot
from src.utils.io import make_experiment_dir
from src.utils.logging import get_logger


logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--resume_experiment",
        default=None,
        help="Path to an existing continual experiment directory to resume in-place",
    )
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
    resume_experiment = args.resume_experiment or task_manager.continual_cfg.get("resume_experiment")

    if resume_experiment:
        resume_root = Path(resume_experiment)
        resume_root.mkdir(parents=True, exist_ok=True)
        dirs = {
            "root": resume_root,
            "logs": resume_root / "logs",
            "tasks": resume_root / "tasks",
        }
        dirs["logs"].mkdir(parents=True, exist_ok=True)
        dirs["tasks"].mkdir(parents=True, exist_ok=True)
        logger.info("Resuming continual experiment in existing run root: %s", dirs["root"])
    else:
        dirs = make_experiment_dir(run_root, subdirs=["logs", "tasks"])

    save_config_snapshot(cfg, dirs["root"])
    logger.info("Continual run root: %s", dirs["root"])
    logger.info(
        "Strategy=%s | from_scratch=%s | tasks=%d",
        task_manager.strategy,
        task_manager.from_scratch,
        len(tasks),
    )
    run_continual_strategy(cfg, task_manager, tasks, dirs, logger=logger)

    if args.evaluate:
        evaluate_continual_experiment(dirs["root"])
    else:
        logger.info("Skipping continual evaluation by request")


if __name__ == "__main__":
    main()
