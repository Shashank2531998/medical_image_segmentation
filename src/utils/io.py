from __future__ import annotations

from datetime import datetime
from pathlib import Path
import secrets
from typing import Dict


def make_experiment_dir(root: str | Path = "experiments", name: str | None = None) -> Dict[str, Path]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    rand = secrets.token_hex(4)
    exp_name = name or f"exp_{ts}_{rand}"
    exp_dir = root / exp_name
    exp_dir.mkdir(parents=True, exist_ok=False)

    checkpoints = exp_dir / "checkpoints"
    logs = exp_dir / "logs"
    predictions = exp_dir / "predictions"

    checkpoints.mkdir()
    logs.mkdir()
    predictions.mkdir()

    return {
        "root": exp_dir,
        "checkpoints": checkpoints,
        "logs": logs,
        "predictions": predictions,
    }


def write_metrics(metrics: dict, exp_dir: Path, name: str = "metrics.json") -> Path:
    import json
    out = Path(exp_dir) / name
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    return out
