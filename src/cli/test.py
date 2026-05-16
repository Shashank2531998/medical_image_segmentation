#!/usr/bin/env python3
import argparse
from pathlib import Path
import os

# Set Hugging Face Hub environment variables for ma
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from src.evaluation.test import Evaluator
from src.data.datamodule import VoxTellTestDataModule
from src.utils.config import load_config
from src.utils.io import make_experiment_dir
from src.utils.logging import get_logger


logger = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    logger.info("Loading training config from %s", args.config)
    cfg = load_config(Path(args.config))

    data_cfg = cfg.get("dataset", {})
    train_cfg = cfg.get("evaluation", {})
    model_cfg = cfg.get("model", {})

    # create experiment folder and snapshot config
    out_root = train_cfg.get("output_dir", "experiments")
    dirs = make_experiment_dir(out_root, subdirs=["logs"])
    # save resolved config
    from src.utils.config import save_config_snapshot
    save_config_snapshot(cfg, dirs["root"])
    logger.info("Experiment root: %s", dirs["root"])

    # make trainer write artifacts into this experiment root
    train_cfg["output_dir"] = str(dirs["root"])

    datamodule = VoxTellTestDataModule(data_cfg)
    evaluator = Evaluator(model_cfg, train_cfg)

    evaluator.evaluate(datamodule)


if __name__ == '__main__':
    main()
