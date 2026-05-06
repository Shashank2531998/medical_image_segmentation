from __future__ import annotations

from pathlib import Path
import yaml


def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_config_snapshot(cfg: dict, out_dir: Path, name: str = "config.yaml") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / name
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f)
    return out_path
