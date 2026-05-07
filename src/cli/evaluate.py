#!/usr/bin/env python3
import argparse

from src.evaluation.eval import run_eval_from_config
from src.utils.logging import get_logger


logger = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument(
        "--split",
        default="evaluation",
        choices=("evaluation", "validation", "testing"),
        help="Config section to evaluate: evaluation, validation, or testing",
    )
    return p.parse_args()


def main():
    args = parse_args()
    metrics_name_map = {
        "evaluation": "metrics.json",
        "validation": "validation_metrics.json",
        "testing": "test_metrics.json",
    }
    out_file = run_eval_from_config(
        args.config,
        split_key=args.split,
        metrics_name=metrics_name_map[args.split],
    )
    logger.info("Saved %s results to %s", args.split, out_file)


if __name__ == '__main__':
    main()
