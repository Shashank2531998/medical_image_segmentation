#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.cli.common import set_offline_mode

set_offline_mode()

from src.continual.evaluator import evaluate_continual_experiment
from src.utils.logging import get_logger


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="voxtell-continual-eval")
    parser.add_argument(
        "--experiment_root",
        required=True,
        help="Path to a saved continual learning experiment root containing config.yaml and task folders",
    )
    parser.add_argument(
        "--checkpoint_name",
        default="best_model.pt",
        help="Checkpoint filename to evaluate inside each task checkpoint directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("Running continual experiment evaluation from %s", args.experiment_root)
    evaluate_continual_experiment(Path(args.experiment_root), checkpoint_name=args.checkpoint_name)


if __name__ == "__main__":
    main()