#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
import os 
import torch 

# Set Hugging Face Hub environment variables for ma 
os.environ["TRANSFORMERS_OFFLINE"] = "1" 
os.environ["HF_HUB_OFFLINE"] = "1"

from src.inference.predict_core import (
    predict_image,
    save_all_segmentations,
    get_predictor,
)


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-p", "--prompts", nargs="+", required=True)

    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument("--save-combined", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


# ============================================================================
# MAIN
# ============================================================================

def main():
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    # device setup
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    if args.verbose:
        print(f"Using device: {device}")

    predictor = get_predictor(args.model, device)

    segmentations = predict_image(
        predictor,
        input_path,
        args.prompts,
        verbose=args.verbose,
    )

    output_folder = Path(args.output)

    save_all_segmentations(
        segmentations,
        output_folder,
        input_path,
        args.prompts,
        save_combined=args.save_combined,
        verbose=args.verbose,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())