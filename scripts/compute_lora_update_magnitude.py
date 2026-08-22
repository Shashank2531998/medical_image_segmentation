#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute LoRA parameter update magnitudes for a continual LoRA experiment."
    )
    parser.add_argument(
        "--experiment_root",
        required=True,
        help="Path to the continual experiment root containing a tasks/ directory",
    )
    parser.add_argument(
        "--output_json",
        default=None,
        help="Optional output JSON path. Defaults to <experiment_root>/lora_update_magnitudes.json",
    )
    parser.add_argument(
        "--output_csv",
        default=None,
        help="Optional output CSV path. Defaults to <experiment_root>/lora_update_magnitudes.csv",
    )
    return parser.parse_args()


def _task_sort_key(path: Path) -> tuple[int, str]:
    name = path.name
    if name.startswith("task_"):
        try:
            return (int(name.split("_", 1)[1].split("_", 1)[0]), name)
        except ValueError:
            return (10**9, name)
    return (10**9, name)


def _collect_task_dirs(experiment_root: Path) -> list[Path]:
    tasks_root = experiment_root / "tasks"
    if not tasks_root.exists():
        raise FileNotFoundError(f"Expected task directory at {tasks_root}")

    task_dirs = [
        path
        for path in tasks_root.iterdir()
        if path.is_dir() and path.name != "pretrained" and (path / "lora_adapter.pt").exists()
    ]
    task_dirs.sort(key=_task_sort_key)
    if not task_dirs:
        raise FileNotFoundError(
            f"No task directories with lora_adapter.pt were found under {tasks_root}"
        )
    return task_dirs


def _load_adapter_state(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError(f"Expected a state dictionary in {path}, got {type(state)!r}")
    return {key: value.detach().float() for key, value in state.items() if isinstance(value, torch.Tensor)}


def _empty_state_like(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: torch.zeros_like(value) for key, value in state.items()}


def _compute_delta_norm(prev_state: dict[str, torch.Tensor], curr_state: dict[str, torch.Tensor]) -> float:
    all_keys = sorted(set(prev_state) | set(curr_state))
    total_sq = 0.0
    for key in all_keys:
        prev_tensor = prev_state.get(key)
        curr_tensor = curr_state.get(key)
        if prev_tensor is None:
            prev_tensor = torch.zeros_like(curr_tensor)
        if curr_tensor is None:
            curr_tensor = torch.zeros_like(prev_tensor)
        if prev_tensor.shape != curr_tensor.shape:
            raise ValueError(
                f"Shape mismatch for LoRA parameter {key}: {tuple(prev_tensor.shape)} vs {tuple(curr_tensor.shape)}"
            )
        diff = curr_tensor - prev_tensor
        total_sq += float(diff.pow(2).sum().item())
    return math.sqrt(total_sq)


def _compute_parameter_norms(state: dict[str, torch.Tensor]) -> dict[str, float]:
    return {key: math.sqrt(float(tensor.pow(2).sum().item())) for key, tensor in state.items()}


def _compute_relative_update(prev_state: dict[str, torch.Tensor], curr_state: dict[str, torch.Tensor]) -> float:
    prev_norm = _compute_delta_norm(_empty_state_like(prev_state), prev_state)
    if prev_norm == 0.0:
        return float("inf") if _compute_delta_norm(prev_state, curr_state) > 0 else 0.0
    return _compute_delta_norm(prev_state, curr_state) / prev_norm


def compute_magnitudes(experiment_root: Path) -> dict[str, Any]:
    task_dirs = _collect_task_dirs(experiment_root)
    previous_state: dict[str, torch.Tensor] | None = None
    previous_task_name = "initial"
    transitions: list[dict[str, Any]] = []

    for task_dir in task_dirs:
        adapter_path = task_dir / "lora_adapter.pt"
        current_state = _load_adapter_state(adapter_path)
        if previous_state is None:
            prev_state_for_delta = _empty_state_like(current_state)
        else:
            prev_state_for_delta = previous_state

        delta_norm = _compute_delta_norm(prev_state_for_delta, current_state)
        relative_update = _compute_relative_update(prev_state_for_delta, current_state)
        layer_norms = {
            key: math.sqrt(float((current_state[key] - prev_state_for_delta.get(key, torch.zeros_like(current_state[key]))).pow(2).sum().item()))
            for key in sorted(set(current_state) | set(prev_state_for_delta))
            if key in current_state
        }
        transitions.append(
            {
                "from_task": previous_task_name,
                "to_task": task_dir.name,
                "adapter_path": str(adapter_path),
                "delta_norm": delta_norm,
                "relative_update": relative_update,
                "layer_norms": layer_norms,
            }
        )
        previous_state = current_state
        previous_task_name = task_dir.name

    return {
        "experiment_root": str(experiment_root.resolve()),
        "task_directories": [task_dir.name for task_dir in task_dirs],
        "transitions": transitions,
    }


def write_outputs(results: dict[str, Any], experiment_root: Path, output_json: Path | None, output_csv: Path | None) -> None:
    if output_json is None:
        output_json = experiment_root / "lora_update_magnitudes.json"
    if output_csv is None:
        output_csv = experiment_root / "lora_update_magnitudes.csv"

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["from_task", "to_task", "adapter_path", "delta_norm", "relative_update"])
        for transition in results["transitions"]:
            writer.writerow(
                [
                    transition["from_task"],
                    transition["to_task"],
                    transition["adapter_path"],
                    transition["delta_norm"],
                    transition["relative_update"],
                ]
            )

        for transition in results["transitions"]:
            for layer_name, layer_norm in transition.get("layer_norms", {}).items():
                writer.writerow([
                    transition["from_task"],
                    transition["to_task"],
                    f"layer:{layer_name}",
                    layer_norm,
                    "",
                ])


def main() -> None:
    args = parse_args()
    experiment_root = Path(args.experiment_root).expanduser().resolve()
    results = compute_magnitudes(experiment_root)
    write_outputs(
        results,
        experiment_root,
        Path(args.output_json).expanduser().resolve() if args.output_json else None,
        Path(args.output_csv).expanduser().resolve() if args.output_csv else None,
    )
    print(f"Wrote LoRA update magnitudes to {experiment_root / 'lora_update_magnitudes.json'}")
    print(f"Wrote LoRA update magnitudes to {experiment_root / 'lora_update_magnitudes.csv'}")


if __name__ == "__main__":
    main()
