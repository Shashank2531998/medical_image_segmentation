from collections import defaultdict
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import torch

from src.data.adapters import get_dataset_adapter
from src.data.adapters.base import DatasetAdapter
from src.evaluation.metrics import dice_coefficient
from src.inference.predictor import predict_image, get_predictor
from src.utils.logging import get_logger
from src.utils.config import load_config, save_config_snapshot
from src.utils.io import make_experiment_dir, write_metrics


logger = get_logger(__name__)


# ============================================================================  
# DATASET REGISTRY  
# ============================================================================  

def get_adapter(dataset_name: str, dataset_root: Path) -> DatasetAdapter:
    return get_dataset_adapter(dataset_name, dataset_root)


# ============================================================================  
# CORE EVALUATION LOGIC  
# ============================================================================  

def _load_binary_mask(path: Path) -> np.ndarray:
    data = nib.load(str(path)).get_fdata()
    return (data > 0).astype(np.uint8)


def evaluate_case(predictor, case, prompts: list[str]) -> list[dict]:
    logger.info("[CASE] %s", case.case_id)

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
                logger.info(
                    "Skipping Dice calculation for prompt %s in case %s",
                    prompt,
                    case.case_id,
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

            logger.info("  %s -> %s: %.4f", prompt, target_name, score)

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

    logger.info("Evaluating %d cases", len(cases))
    logger.info("Prompts: %s", prompts)

    predictor = get_predictor(model_dir, device)

    prompt_scores = defaultdict(list)

    for i, case in enumerate(cases):
        logger.info("[%d/%d] %s", i + 1, len(cases), case.case_id)

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


def run_eval_from_config(config_path: str | Path, split_key: str, metrics_name: str) -> Path:
    cfg = load_config(Path(config_path))

    dataset_cfg = cfg.get("dataset", {})
    model_cfg = cfg.get("model", {})
    split_cfg = cfg.get(split_key, cfg.get("evaluation", {}))

    dataset_name = split_cfg.get("dataset_name", dataset_cfg.get("name"))
    dataset_root = Path(split_cfg.get("dataset_root", dataset_cfg.get("root")))
    max_cases = split_cfg.get("max_cases", dataset_cfg.get("max_cases", None))

    model_dir = Path(model_cfg["dir"])
    device = torch.device(model_cfg.get("device", "cuda"))
    adapter = get_adapter(dataset_name, dataset_root)
    prompts = split_cfg.get("prompts", cfg.get("evaluation", {}).get("prompts", []))
    if not prompts:
        prompts = adapter.default_prompts()

    if not prompts:
        raise ValueError(f"No prompts provided in '{split_key}' or derived from dataset adapter")

    results = evaluate_dataset(
        dataset_name=dataset_name,
        dataset_root=dataset_root,
        prompts=prompts,
        model_dir=model_dir,
        device=device,
        max_cases=max_cases,
    )

    out_root = cfg.get("output", {}).get("experiment_root", "experiments")
    dirs = make_experiment_dir(out_root)
    save_config_snapshot(cfg, dirs["root"], name="config.yaml")
    out_file = write_metrics(results, dirs["root"], name=metrics_name)
    return out_file
