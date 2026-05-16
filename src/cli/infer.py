#!/usr/bin/env python3
from pathlib import Path
import argparse
import torch
import os

# Set Hugging Face Hub environment variables for ma
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from src.inference.predictor import predict_image, get_predictor, save_all_segmentations
from src.utils.io import make_experiment_dir
from src.utils.logging import get_logger


logger = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--prompts", nargs="+", required=True)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    return p.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    device = torch.device(args.device)

    predictor = get_predictor(args.model, device)

    # create experiment folder and store predictions there
    dirs = make_experiment_dir("experiments/inference", subdirs=[])
    predictions_out = dirs["root"]

    segmentations = predict_image(
        predictor,
        input_path,
        args.prompts,
        verbose=True,
    )

    save_all_segmentations(segmentations, predictions_out, input_path, args.prompts, save_combined=False, verbose=True)
    logger.info("Saved predictions to %s", predictions_out)


if __name__ == '__main__':
    main()
