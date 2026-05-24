#!/usr/bin/env python3
import os
# Set Hugging Face Hub environment variables for ma
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"


from pathlib import Path
import argparse
import torch
import nibabel as nib
import numpy as np
import monai


from src.inference.predictor import predict_image, get_predictor, save_all_segmentations
from src.utils.io import make_experiment_dir
from src.utils.logging import get_logger


logger = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--prompts", nargs="+", required=True)
    p.add_argument("--mask", nargs="+", required=False,
                   help="Optional ground-truth mask file(s). Provide one per prompt or a single mask for all prompts.")
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

    # If masks provided, compute Dice metric(s)
    if args.mask:
        mask_paths = [Path(p) for p in args.mask]

        def compute_dice(pred_arr, gt_arr):
            # 1. Convert to binary PyTorch tensors
            pred_bin = torch.tensor(pred_arr > 0, dtype=torch.uint8)
            gt_bin = torch.tensor(gt_arr > 0, dtype=torch.uint8)
            
            # 2. Add required Batch and Channel dimensions [B, C, H, W, D]
            pred_bin = pred_bin.unsqueeze(0).unsqueeze(0)
            gt_bin = gt_bin.unsqueeze(0).unsqueeze(0)
            
            # 3. Compute and aggregate
            metric_fn = monai.metrics.DiceMetric(include_background=True, reduction="none", ignore_empty=False)
            metric_fn(pred_bin, gt_bin)
            return metric_fn.aggregate().mean().item()

        gt_masks = []
        for mp in mask_paths:
            if not mp.exists():
                logger.warning("Mask file does not exist: %s", mp)
                gt_masks.append(None)
                continue
            gt = nib.load(str(mp)).get_fdata()
            gt_masks.append(gt)

        per_prompt_dice = []
        # segmentations are nibabel images (reoriented to original space)
        for i, seg_nib in enumerate(segmentations):
            seg_arr = seg_nib.get_fdata()

            # choose ground truth mask
            if len(gt_masks) == len(segmentations):
                gt = gt_masks[i]
            elif len(gt_masks) == 1:
                gt = gt_masks[0]
            else:
                logger.warning("Number of provided masks (%d) doesn't match number of predictions (%d). Skipping dice for prompt %d.", len(gt_masks), len(segmentations), i)
                per_prompt_dice.append(None)
                continue

            if gt is None:
                per_prompt_dice.append(None)
                continue

            if seg_arr.shape != gt.shape:
                logger.warning("Shape mismatch between prediction (%s) and ground truth (%s) for prompt %d. Skipping.", seg_arr.shape, gt.shape, i)
                per_prompt_dice.append(None)
                continue

            dice_val = compute_dice(seg_arr, gt)
            per_prompt_dice.append(dice_val)
            logger.info("Prompt %s | Dice: %.6f", args.prompts[i], dice_val)

        # report mean over available values
        valid = [d for d in per_prompt_dice if d is not None]
        if valid:
            mean_dice = float(np.mean(valid))
            logger.info("Mean Dice: %.6f", mean_dice)
        else:
            logger.warning("No valid Dice values computed.")


if __name__ == '__main__':
    main()
