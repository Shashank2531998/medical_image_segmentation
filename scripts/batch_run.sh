#!/bin/bash
#SBATCH --job-name=voxtell_batch                   # Job name
#SBATCH --output=logs/voxtell_%j.out               # Standard output (%j = JobID)
#SBATCH --error=logs/voxtell_%j.err                # Standard error
#SBATCH --gres=gpu:a100:1                          # Request 1 GPU
#SBATCH --time=1-00:00:00                          # Max runtime
#SBATCH --partition=a100                           # Partition name

# Activate conda environment
conda activate voxtell
