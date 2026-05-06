import argparse
from pathlib import Path
import yaml
import json
import os 
import torch 

# Set Hugging Face Hub environment variables for ma 
os.environ["TRANSFORMERS_OFFLINE"] = "1" 
os.environ["HF_HUB_OFFLINE"] = "1"

from src.evaluation.eval import evaluate_dataset


# ============================================================================  
# CONFIG LOADING  
# ============================================================================  

def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ============================================================================  
# MAIN  
# ============================================================================  

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    cfg = load_config(config_path)

    # -------------------------
    # Resolve config values
    # -------------------------
    dataset_name = cfg["dataset"]["name"]
    dataset_root = Path(cfg["dataset"]["root"])
    max_cases = cfg["dataset"].get("max_cases", None)

    model_dir = Path(cfg["model"]["dir"])
    device = torch.device(cfg["model"].get("device", "cuda"))

    prompts = cfg["evaluation"]["prompts"]

    output_json = cfg.get("output", {}).get("json_path", None)

    # -------------------------
    # Validate paths
    # -------------------------
    if not dataset_root.exists():
        raise FileNotFoundError(dataset_root)

    if not model_dir.exists():
        raise FileNotFoundError(model_dir)

    print(f"Using device: {device}")
    print(f"Dataset: {dataset_name}")
    print(f"Prompts: {prompts}")

    # -------------------------
    # Run evaluation
    # -------------------------
    results = evaluate_dataset(
        dataset_name=dataset_name,
        dataset_root=dataset_root,
        prompts=prompts,
        model_dir=model_dir,
        device=device,
        max_cases=max_cases,
    )

    # -------------------------
    # Print summary
    # -------------------------
    for prompt, r in results.items():
        print(f"\nPrompt: {prompt}")
        print(f"  Mean Dice: {r['mean_dice']:.4f}")
        print(f"  Std Dice : {r['std']:.4f}")
        print(f"  Samples  : {r['count']}")

    # -------------------------
    # Optional JSON export
    # -------------------------
    if output_json:
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nSaved results to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())