#!/usr/bin/env python3
import argparse
from pathlib import Path
import yaml
import torch

from src.evaluation.eval import evaluate_dataset
from src.utils.io import make_experiment_dir, write_metrics


def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(Path(args.config))

    dataset_name = cfg["dataset"]["name"]
    dataset_root = Path(cfg["dataset"]["root"]) 
    max_cases = cfg["dataset"].get("max_cases", None)

    model_dir = Path(cfg["model"]["dir"]) 
    device = torch.device(cfg["model"].get("device", "cuda"))

    prompts = cfg["evaluation"]["prompts"]

    results = evaluate_dataset(
        dataset_name=dataset_name,
        dataset_root=dataset_root,
        prompts=prompts,
        model_dir=model_dir,
        device=device,
        max_cases=max_cases,
    )

    # persist results into an experiment folder
    out_root = cfg.get("output", {}).get("experiment_root", "experiments")
    dirs = make_experiment_dir(out_root)
    write_metrics(results, dirs["root"], name="metrics.json")
    print(f"Saved results to {dirs['root'] / 'metrics.json'}")


if __name__ == '__main__':
    main()
