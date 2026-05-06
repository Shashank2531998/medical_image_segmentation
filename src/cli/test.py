#!/usr/bin/env python3
import argparse

from src.cli.eval_common import run_eval_from_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to YAML config")
    return p.parse_args()


def main():
    args = parse_args()
    out_file = run_eval_from_config(args.config, split_key="testing", metrics_name="test_metrics.json")
    print(f"Saved test results to {out_file}")


if __name__ == '__main__':
    main()
