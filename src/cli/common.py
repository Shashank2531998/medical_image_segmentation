from __future__ import annotations

import os
from pathlib import Path

from src.utils.config import load_config, save_config_snapshot
from src.utils.io import make_experiment_dir


def set_offline_mode() -> None:
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"


def load_run_config(
    config_path: str | Path,
    section: str,
    subdirs: list[str] | None = None,
) -> tuple[dict, dict, dict, dict, dict[str, Path]]:
    cfg = load_config(Path(config_path))

    data_cfg = cfg.get("dataset", {})
    run_cfg = dict(cfg.get(section, {}))
    model_cfg = dict(cfg.get("model", {}))

    out_root = run_cfg.get("output_dir", "experiments")
    dirs = make_experiment_dir(out_root, subdirs=subdirs or ["checkpoints", "logs"])
    save_config_snapshot(cfg, dirs["root"])
    run_cfg["output_dir"] = str(dirs["root"])

    return cfg, data_cfg, run_cfg, model_cfg, dirs