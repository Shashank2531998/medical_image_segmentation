VoxTell — project layout and experiment conventions

Experiment layout

- experiments/exp_<id>/
    - config.yaml          # snapshot of resolved config for the run
    - checkpoints/         # saved model checkpoints
    - logs/                # run logs (run.log)
    - metrics.json         # evaluation metrics (for evaluate runs)
    - predictions/         # outputs from inference runs

Quick CLI

- Train: python src/cli/train.py --config configs/experiments/train_debug.yaml
- Continual baseline: python src/cli/continual.py --config configs/continual/naive_sequential_finetuning.yaml
- Continual LoRA: python src/cli/continual.py --config configs/continual/lora_sequential_finetuning.yaml
- Evaluate: python src/cli/evaluate.py --config configs/experiments/aeropath_eval.yaml --split evaluation
- Validate: python src/cli/evaluate.py --config configs/experiments/aeropath_eval.yaml --split validation
- Test: python src/cli/evaluate.py --config configs/experiments/aeropath_eval.yaml --split testing
- Infer: python src/cli/infer.py --input <image> --output <outdir> --model <modeldir> --prompts "liver"

Notes

- Config files live under `configs/`.
- CLIs create an `experiments/` run directory and snapshot the resolved config.
- Ensure required dependencies are installed from `requirements.txt` before running.
