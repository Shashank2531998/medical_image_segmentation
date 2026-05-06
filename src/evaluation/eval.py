from collections import defaultdict
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import torch

from src.adapters.aeropath import AeroPathAdapter
from src.adapters.base import DatasetAdapter
from src.evaluation.metrics import dice_coefficient
from src.inference.predict_core import predict_image, get_predictor


# ============================================================================  
# DATASET REGISTRY  
# ============================================================================  

DATASET_ADAPTERS = {
    "aeropath": AeroPathAdapter,
}


def get_adapter(dataset_name: str, dataset_root: Path) -> DatasetAdapter:
    if dataset_name not in DATASET_ADAPTERS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(DATASET_ADAPTERS.keys())}"
        )
    return DATASET_ADAPTERS[dataset_name](dataset_root)


# ============================================================================  
# CORE EVALUATION LOGIC  
# ============================================================================  

def _load_binary_mask(path: Path) -> np.ndarray:
    data = nib.load(str(path)).get_fdata()
    return (data > 0).astype(np.uint8)


def evaluate_case(predictor, case, prompts: list[str]) -> list[dict]:
    print(f"\n[CASE] {case.case_id}")

    segmentations = predict_image(
        predictor,
        case.image_path,
        prompts,
        verbose=True
    )

    if len(segmentations) != len(prompts):
        raise ValueError("Prompt/output mismatch")

    results = []

    for target_name, target_path in case.target_paths.items():
        target = _load_binary_mask(target_path)

        for prompt, pred in zip(prompts, segmentations):

            if target_name not in prompt.lower():
                print(
                    f"Skipping Dice calculation for prompt {prompt} "
                    f"in case {case.case_id}"
                )
                continue

            if pred.shape != target.shape:
                raise ValueError(f"Shape mismatch {case.case_id}/{target_name}")

            if isinstance(pred, nib.Nifti1Image):
                pred = pred.get_fdata()
                pred = (pred > 0).astype(np.uint8)

            score = dice_coefficient(pred, target)

            results.append({
                "prompt": prompt,
                "dice": float(score),
            })

            print(f"  {prompt} → {target_name}: {score:.4f}")

    return results


def evaluate_dataset(
    dataset_name: str,
    dataset_root: Path,
    prompts: list[str],
    model_dir: Path,
    device: torch.device,
    max_cases: Optional[int] = None,
) -> dict:

    adapter = get_adapter(dataset_name, dataset_root)
    cases = adapter.cases()

    if max_cases:
        cases = cases[:max_cases]

    print(f"Evaluating {len(cases)} cases")
    print(f"Prompts: {prompts}")

    predictor = get_predictor(model_dir, device)

    prompt_scores = defaultdict(list)

    for i, case in enumerate(cases):
        print(f"\n[{i+1}/{len(cases)}] {case.case_id}")

        case_results = evaluate_case(predictor, case, prompts)

        for r in case_results:
            prompt_scores[r["prompt"]].append(r["dice"])

    summary = {}

    for prompt, scores in prompt_scores.items():
        summary[prompt] = {
            "mean_dice": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "count": len(scores),
        }

    return summary
