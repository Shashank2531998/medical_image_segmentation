#!/bin/bash
#SBATCH --job-name=voxtell_batch                   # Job name
#SBATCH --output=outputs/logs/voxtell_%j.out               # Standard output (%j = JobID)
#SBATCH --error=outputs/logs/voxtell_%j.err                # Standard error
#SBATCH --gres=gpu:a100:1                          # Request 1 GPU
#SBATCH --time=1-00:00:00                          # Max runtime
#SBATCH --partition=a100                           # Partition name

PYTHONPATH=. python src/cli/train.py --config configs/aeropath_train.yaml
