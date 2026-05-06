#!/usr/bin/env python3
import argparse
from pathlib import Path
import yaml
import torch

from src.evaluation.eval import evaluate_dataset


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

    import json
    out = cfg.get("output", {}).get("json_path", None)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {out}")
    else:
        print(results)


if __name__ == '__main__':
    main()
