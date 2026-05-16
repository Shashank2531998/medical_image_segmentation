from __future__ import annotations

from datetime import datetime
from pathlib import Path
import secrets
from typing import Dict
from typing import List


def make_experiment_dir(
    root: str | Path = "experiments", 
    name: str | None = None, 
    subdirs: List[str] = ["checkpoints", "logs"]
) -> Dict[str, Path]:
    
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rand = secrets.token_hex(4)
    exp_name = name or f"exp_{ts}_{rand}"
    exp_dir = root / exp_name
    exp_dir.mkdir(parents=True, exist_ok=False)

    paths: dict[str, Path] = {"root": exp_dir}

    for subdir in subdirs:
        path = exp_dir / subdir
        path.mkdir(parents=True, exist_ok=True)
        paths[subdir] = path

    return paths


def write_metrics(metrics: dict, exp_dir: Path, name: str = "metrics.json") -> Path:
    import json
    out = Path(exp_dir) / name
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    return out
