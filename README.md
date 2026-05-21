VoxTell — project layout and experiment conventions

Experiment layout

- experiments/exp_<id>/
    - config.yaml          # snapshot of resolved config for the run
    - checkpoints/         # saved model checkpoints
    - logs/                # run logs (run.log)
    - metrics.json         # evaluation metrics (for evaluate runs)
    - predictions/         # outputs from inference runs

Quick CLI

- Train: python src/cli/voxtell.py --mode train --config configs/aeropath_train.yaml
- Train only: python src/cli/main.py --train_only --config configs/aeropath_train.yaml
- Test only: python src/cli/main.py --test_only --config configs/aeropath_eval.yaml
- Train then test: python src/cli/main.py --config configs/aeropath_train.yaml
- Continual baseline: python src/cli/continual.py --config configs/continual/naive_sequential_finetuning.yaml
- Continual LoRA: python src/cli/continual.py --config configs/continual/lora_sequential_finetuning.yaml
- Infer: python src/cli/infer.py --input <image> --output <outdir> --model <modeldir> --prompts "liver"

Notes

- Config files live under `configs/`.
- CLIs create an `experiments/` run directory and snapshot the resolved config.
- Ensure required dependencies are installed from `requirements.txt` before running.
