#!/usr/bin/env python3
import argparse
from pathlib import Path
import yaml

from src.training.trainer import Trainer
from src.data.datamodule import VoxTellDataModule
from src.utils.config import load_config
from src.utils.io import make_experiment_dir


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

    data_cfg = cfg.get("dataset", {})
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})

    # create experiment folder and snapshot config
    out_root = train_cfg.get("output_dir", "experiments")
    dirs = make_experiment_dir(out_root)
    # save resolved config
    from src.utils.config import save_config_snapshot
    save_config_snapshot(cfg, dirs["root"]) 

    # make trainer write artifacts into this experiment root
    train_cfg["output_dir"] = str(dirs["root"])

    datamodule = VoxTellDataModule(data_cfg)
    trainer = Trainer(train_cfg, model_cfg)

    trainer.fit(datamodule)


if __name__ == '__main__':
    main()
