#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.cli.common import load_run_config, set_offline_mode

set_offline_mode()

from src.data.datamodule import VoxTellDataModule
from src.engine.model_engine import VoxTellEngine
from src.evaluation.test import Evaluator
from src.training.trainer import Trainer
from src.utils.logging import get_logger


logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="voxtell")
    parser.add_argument("--config", required=True, help="Path to the YAML config file")
    parser.add_argument("--train_only", action="store_true", help="Run training only")
    parser.add_argument("--test_only", action="store_true", help="Run testing only")
    return parser


def run_train(config_path: str | Path) -> None:
    logger.info("Loading training config from %s", config_path)
    _, data_cfg, train_cfg, model_cfg, dirs = load_run_config(config_path, "training")
    logger.info("Experiment root: %s", dirs["root"])

    datamodule = VoxTellDataModule(data_cfg)
    model_engine = VoxTellEngine(model_cfg)
    trainer = Trainer(model_engine, train_cfg)
    trainer.fit(datamodule)


def run_test(config_path: str | Path) -> None:
    logger.info("Loading evaluation config from %s", config_path)
    _, data_cfg, eval_cfg, model_cfg, dirs = load_run_config(config_path, "evaluation", subdirs=["logs"])
    logger.info("Experiment root: %s", dirs["root"])

    datamodule = VoxTellDataModule(data_cfg)
    evaluator = Evaluator(model_cfg, eval_cfg)
    evaluator.evaluate(datamodule)


def run_train_then_test(config_path: str | Path) -> None:
    logger.info("Loading training config from %s", config_path)
    _, data_cfg, train_cfg, model_cfg, dirs = load_run_config(
        config_path,
        "training",
        subdirs=["checkpoints", "logs"],
    )
    logger.info("Experiment root: %s", dirs["root"])

    datamodule = VoxTellDataModule(data_cfg)
    model_engine = VoxTellEngine(model_cfg)
    trainer = Trainer(model_engine, train_cfg)
    trainer.fit(datamodule)

    final_checkpoint = dirs["root"] / "checkpoints" / "final_checkpoint.pt"
    if not final_checkpoint.exists():
        raise FileNotFoundError(f"Expected final checkpoint at {final_checkpoint}")

    test_dir = dirs["root"] / "test"
    test_dir.mkdir(parents=True, exist_ok=True)

    eval_model_cfg = {
        **model_cfg,
        "checkpoint_path": str(final_checkpoint),
    }
    eval_cfg = {
        "output_dir": str(test_dir),
    }

    evaluator = Evaluator(eval_model_cfg, eval_cfg)
    evaluator.evaluate(datamodule)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.train_only and args.test_only:
        parser.error("--train_only and --test_only cannot be used together")

    if args.train_only:
        run_train(args.config)
    elif args.test_only:
        run_test(args.config)
    else:
        run_train_then_test(args.config)


if __name__ == "__main__":
    main()