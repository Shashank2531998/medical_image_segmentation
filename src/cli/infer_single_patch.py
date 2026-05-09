#!/usr/bin/env python3
from pathlib import Path
import argparse
import torch
import os

from src.data.datamodule import VoxTellDataModule
from src.inference.postprocessing import logits_to_segmentation
from src.utils.config import load_config

# Set Hugging Face Hub environment variables for ma
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from src.inference.predictor import predict_image, get_predictor, save_all_segmentations
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
    model_cfg = cfg.get("model", {})

    device = torch.device('cuda')
    model_dir = model_cfg.get("dir", None)
    predictor = get_predictor(model_dir, device)
    
    datamodule = VoxTellDataModule(data_cfg)
    train_loader = datamodule.train_dataloader()
    first_batch = next(iter(train_loader))

    imgs = first_batch["image"].to('cuda')
    prompts = first_batch["prompts"]

    # Embed text prompts
    embeddings = predictor.embed_text_prompts(prompts)

    # Predict segmentation logits
    prediction = predictor.predict_sliding_window_return_logits(imgs[0], embeddings).to('cpu')
    print(prediction.max())


if __name__ == '__main__':
    main()
