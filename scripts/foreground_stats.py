#!/usr/bin/env python3
"""Compute foreground voxel/volume statistics per task defined in a continual config.

Usage examples:
  python scripts/foreground_stats.py --config configs/continual/shared_lora.yaml
  python scripts/foreground_stats.py --config configs/continual/shared_lora.yaml --out stats.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics
from typing import Iterable

import nibabel as nib
import numpy as np
import yaml
from tqdm import tqdm

from src.data.adapters import get_dataset_adapter


def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def iter_annotation_files(dataset_root: Path) -> Iterable[Path]:
    # Prefer explicit annotations/ directory used by adapters
    ann_dir = dataset_root / "annotations"
    if ann_dir.exists() and ann_dir.is_dir():
        for p in sorted(ann_dir.iterdir()):
            if p.is_file() and p.name.endswith((".nii", ".nii.gz")):
                yield p
        return

    # Fallback: search for nifti files under root, but skip an images/ folder if present
    images_dir = dataset_root / "images"
    for p in sorted(dataset_root.rglob("*.nii*")):
        try:
            if images_dir.exists() and images_dir in p.parents:
                continue
        except Exception:
            pass
        yield p


def case_id_from_path(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    return path.stem


def voxel_volume_mm3(img: nib.Nifti1Image) -> float:
    zooms = img.header.get_zooms()
    if not zooms:
        return 1.0
    prod = 1.0
    for z in zooms:
        prod *= float(z)
    return prod


def compute_stats_for_mask(path: Path) -> dict:
    img = nib.load(str(path))
    data = np.asarray(img.dataobj)
    fg_voxels = int(np.count_nonzero(data != 0))
    tot_voxels = int(data.size)
    vv = voxel_volume_mm3(img)
    fg_volume = fg_voxels * vv
    return {
        "case_id": case_id_from_path(path),
        "file": str(path),
        "foreground_voxels": fg_voxels,
        "total_voxels": tot_voxels,
        "voxel_volume_mm3": vv,
        "foreground_volume_mm3": fg_volume,
    }


def summarize_task(rows: list[dict]) -> dict:
    fgs = [r["foreground_voxels"] for r in rows]
    vols = [r["foreground_volume_mm3"] for r in rows]
    if not rows:
        return {
            "cases": 0,
            "sum_voxels": 0,
            "mean_voxels": 0,
            "median_voxels": 0,
            "std_voxels": 0,
            "sum_volume_mm3": 0.0,
        }
    return {
        "cases": len(rows),
        "sum_voxels": sum(fgs),
        "mean_voxels": statistics.mean(fgs),
        "median_voxels": statistics.median(fgs),
        "std_voxels": statistics.pstdev(fgs) if len(fgs) > 1 else 0.0,
        "sum_volume_mm3": sum(vols),
        "mean_volume_mm3": statistics.mean(vols),
    }


def tasks_from_config(cfg: dict) -> list[dict]:
    # Expect path like continual.tasks
    cont = cfg.get("continual") or cfg.get("continual", {})
    tasks = cont.get("tasks") if cont else None
    if tasks:
        return tasks
    # Fall back to top-level 'tasks'
    return cfg.get("tasks", [])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", "-c", type=Path, default=Path("configs/continual/shared_lora.yaml"))
    p.add_argument("--out", "-o", type=Path, default=None, help="CSV output file")
    p.add_argument("--task", "-t", type=str, default=None, help="Only process a single task name")
    args = p.parse_args()

    cfg = load_config(args.config)
    tasks = tasks_from_config(cfg)
    results = []

    for task in tasks:
        name = task.get("name") if isinstance(task, dict) else str(task)
        if args.task and name != args.task:
            continue
        dataset = task.get("dataset") if isinstance(task, dict) else None
        root = None
        if dataset and isinstance(dataset, dict):
            root = Path(dataset.get("root", ""))
        else:
            root = Path()

        print(f"Processing task {name} -> {root}")
        rows = []
        if not root.exists():
            print(f"  dataset root does not exist: {root}")
        else:
            try:
                adapter = get_dataset_adapter(name, root)
            except Exception as e:
                print(f"  failed to get adapter for {name}: {e}")
                adapter = None

            if adapter is None:
                print(f"  no adapter available for task {name}")
            else:
                cases = adapter.cases()
                samples = adapter.build_training_samples(cases)
                for s in tqdm(samples, desc=f"{name}"):
                    # s is a TrainingSample; iterate mask paths + labels
                    for idx, mask_path in enumerate(s.mask_paths):
                        if mask_path is None:
                            # empty mask
                            fg_voxels = 0
                            tot_voxels = 0
                            vv = 0.0
                            fg_volume = 0.0
                            file_path = None
                        else:
                            try:
                                img = nib.load(str(mask_path))
                                data = np.asarray(img.dataobj)
                                label = None
                                if idx < len(s.mask_labels):
                                    label = s.mask_labels[idx]

                                if label is None:
                                    mask = data > 0
                                else:
                                    mask = data == float(label)

                                fg_voxels = int(np.count_nonzero(mask))
                                tot_voxels = int(data.size)
                                vv = voxel_volume_mm3(img)
                                fg_volume = fg_voxels * vv
                                file_path = str(mask_path)
                            except Exception as e:
                                print(f"  failed to load mask {mask_path}: {e}")
                                continue

                        rows.append({
                            "case_id": case_id_from_path(s.image_path),
                            "file": file_path or "",
                            "foreground_voxels": fg_voxels,
                            "total_voxels": tot_voxels,
                            "voxel_volume_mm3": vv,
                            "foreground_volume_mm3": fg_volume,
                            "prompt": (s.prompts[idx] if idx < len(s.prompts) else ""),
                            "label": (s.mask_labels[idx] if idx < len(s.mask_labels) else None),
                        })

        summary = summarize_task(rows)
        print(f"  Cases: {summary['cases']}")
        print(f"  Sum foreground voxels: {summary['sum_voxels']}")
        print(f"  Mean per-case voxels: {summary.get('mean_voxels', 0):.1f}")
        print(f"  Median per-case voxels: {summary.get('median_voxels', 0):.1f}")
        print(f"  Std dev voxels: {summary.get('std_voxels', 0):.1f}")
        print(f"  Sum foreground volume (mm^3): {summary.get('sum_volume_mm3', 0):.1f}")

        for r in rows:
            results.append({"task": name, **r})

        # Print per-case details for this task to console
        if rows:
            print("\nPer-case details:")
            for r in rows:
                print(
                    f"Task: {name} | Prompt: {r.get('prompt','')} | Label: {r.get('label')} | Case: {r['case_id']} | "
                    f"FG voxels: {r['foreground_voxels']} | Total voxels: {r['total_voxels']} | "
                    f"Voxel mm^3: {r['voxel_volume_mm3']:.6g} | FG volume mm^3: {r['foreground_volume_mm3']:.2f}"
                )
        else:
            print("No results to display for this task.")


if __name__ == "__main__":
    main()
