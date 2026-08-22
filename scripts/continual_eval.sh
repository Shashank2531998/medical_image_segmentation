#!/bin/bash
#SBATCH --job-name=voxtell_batch                   # Job name
#SBATCH --output=/home/vault/iwi5/iwi5326h/projects/VoxTell/outputs/logs/voxtell_%j.out               # Standard output (%j = JobID)
#SBATCH --error=/home/vault/iwi5/iwi5326h/projects/VoxTell/outputs/logs/voxtell_%j.err                # Standard error
#SBATCH --gres=gpu:a100:1                          # Request 1 GPU
#SBATCH --time=1-00:00:00                          # Max runtime
#SBATCH --partition=a100                           # Partition name

# Activate environment
conda activate voxtell

# -------------------------------
# Config handling (default + override)
# -------------------------------
EXPERIMENT_ROOT=${1:-/home/vault/iwi5/iwi5326h/projects/VoxTell/experiments/continual/lora/task_specific/exp_20260627_103019_9b49fc9d}

echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Using Experiment Root: $EXPERIMENT_ROOT"
echo "======================================"

# Ensure logs directory exists (safe guard)
mkdir -p outputs/logs

# Run training
PYTHONPATH=. python src/cli/continual_eval.py --experiment_root "$EXPERIMENT_ROOT"
