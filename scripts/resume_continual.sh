#!/bin/bash
#SBATCH --job-name=voxtell_batch                   # Job name
#SBATCH --output=/home/vault/iwi5/iwi5326h/projects/VoxTell/outputs/logs/voxtell_%j.out               # Standard output (%j = JobID)
#SBATCH --error=/home/vault/iwi5/iwi5326h/projects/VoxTell/outputs/logs/voxtell_%j.err                # Standard error
#SBATCH --gres=gpu:a100:1                          # Request 1 GPU
#SBATCH --time=1-00:00:00                          # Max runtime
#SBATCH --partition=a100                           # Partition name

if [[ $# -ne 1 ]]; then
	echo "Usage: $0 <resume_experiment>" >&2
	exit 1
fi

resume_experiment="$1"
config_path="$resume_experiment/config.yaml"

if [[ ! -f "$config_path" ]]; then
	echo "Config not found: $config_path" >&2
	exit 1
fi

# Activate environment
conda activate voxtell

# -------------------------------
# Config handling (default + override)
# -------------------------------
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "======================================"

# Ensure logs directory exists (safe guard)
mkdir -p outputs/logs

# Run training
PYTHONPATH=. python src/cli/continual.py --config "$config_path" --resume_experiment "$resume_experiment"
