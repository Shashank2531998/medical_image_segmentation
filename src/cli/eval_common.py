#!/usr/bin/env python3
from pathlib import Path

import torch

from src.evaluation.eval import evaluate_dataset
from src.utils.config import load_config, save_config_snapshot
from src.utils.io import make_experiment_dir, write_metrics


def run_eval_from_config(config_path: str | Path, split_key: str, metrics_name: str) -> Path:
    cfg = load_config(Path(config_path))

    dataset_cfg = cfg.get("dataset", {})
    model_cfg = cfg.get("model", {})
    split_cfg = cfg.get(split_key, cfg.get("evaluation", {}))

    dataset_name = split_cfg.get("dataset_name", dataset_cfg.get("name"))
    dataset_root = Path(split_cfg.get("dataset_root", dataset_cfg.get("root")))
    max_cases = split_cfg.get("max_cases", dataset_cfg.get("max_cases", None))

    model_dir = Path(model_cfg["dir"])
    device = torch.device(model_cfg.get("device", "cuda"))
    prompts = split_cfg.get("prompts", cfg.get("evaluation", {}).get("prompts", []))

    if not prompts:
        raise ValueError(f"No prompts provided in '{split_key}' or 'evaluation' section")

    results = evaluate_dataset(
        dataset_name=dataset_name,
        dataset_root=dataset_root,
        prompts=prompts,
        model_dir=model_dir,
        device=device,
        max_cases=max_cases,
    )

    out_root = cfg.get("output", {}).get("experiment_root", "experiments")
    dirs = make_experiment_dir(out_root)
    save_config_snapshot(cfg, dirs["root"], name="config.yaml")
    out_file = write_metrics(results, dirs["root"], name=metrics_name)
    return out_file
