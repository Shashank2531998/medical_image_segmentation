#!/bin/bash
#SBATCH --job-name=voxtell_batch                   # Job name
#SBATCH --output=outputs/logs/voxtell_%j.out               # Standard output (%j = JobID)
#SBATCH --error=outputs/logs/voxtell_%j.err                # Standard error
#SBATCH --gres=gpu:a100:1                          # Request 1 GPU
#SBATCH --time=1-00:00:00                          # Max runtime
#SBATCH --partition=a100                           # Partition name

# Activate environment
conda activate voxtell

# -------------------------------
# Config handling (default + override)
# -------------------------------
CONFIG=${1:-configs/continual/naive_finetuning.yaml}

echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Using config: $CONFIG"
echo "======================================"

# Ensure logs directory exists (safe guard)
mkdir -p outputs/logs

# Run training
PYTHONPATH=. python src/cli/continual.py --config "$CONFIG"
