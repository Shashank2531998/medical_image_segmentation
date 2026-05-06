#!/usr/bin/env python3
from pathlib import Path
import argparse
import torch

from src.inference.predict_core import predict_image, get_predictor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--prompts", nargs="+", required=True)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    predictor = get_predictor(args.model, device)

    segmentations = predict_image(
        predictor,
        input_path,
        args.prompts,
        verbose=True,
    )

    from src.inference.predict_core import save_all_segmentations

    save_all_segmentations(segmentations, output_path, input_path, args.prompts, save_combined=False, verbose=True)


if __name__ == '__main__':
    main()
