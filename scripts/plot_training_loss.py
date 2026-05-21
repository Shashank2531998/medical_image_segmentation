#!/usr/bin/env python
"""
Script to visualize training and validation loss from training logs.

This script parses training log files and creates plots showing:
1. Training loss vs epoch
2. Validation loss vs epoch
3. Combined training and validation loss

Usage:
    python scripts/plot_training_loss.py [--log-dir DIRECTORY] [--output-dir DIRECTORY]

Example:
    # Plot loss from a single experiment
    python scripts/plot_training_loss.py --log-dir experiments/fine_tuning/exp_20260514_141005_1ba4100a/logs

    # Plot and save to custom output directory
    python scripts/plot_training_loss.py \
        --log-dir experiments/fine_tuning/exp_20260514_141005_1ba4100a/logs \
        --output-dir plots

    # Plot loss from multiple experiments
    python scripts/plot_training_loss.py --log-dir experiments/fine_tuning/ --multi
"""

import argparse
import re
from pathlib import Path
from typing import Tuple, List, Dict
import json

import matplotlib.pyplot as plt
import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


def extract_loss_from_log(log_file: Path) -> Tuple[List[float], List[float], Dict]:
    """
    Extract training and validation loss from log file.
    
    Args:
        log_file: Path to the log file
        
    Returns:
        Tuple of (train_losses, val_losses, metadata)
    """
    train_losses = []
    val_losses = []
    metadata = {
        "total_epochs": 0,
        "log_file": str(log_file),
    }
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        # Pattern to match the epoch loss line
        # "Epoch 1/10 completed | Training Loss=0.3033, | Validation Loss=0.3120"
        epoch_pattern = r'Epoch (\d+)/(\d+) completed \| Training Loss=([\d.]+), \| Validation Loss=([\d.]+)'
        
        for line in lines:
            match = re.search(epoch_pattern, line)
            if match:
                epoch_num = int(match.group(1))
                total_epochs = int(match.group(2))
                train_loss = float(match.group(3))
                val_loss = float(match.group(4))
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                metadata["total_epochs"] = total_epochs
        
        if not train_losses:
            logger.warning(f"No loss entries found in {log_file}")
            return [], [], metadata
        
        logger.info(f"Extracted {len(train_losses)} epochs from {log_file}")
        return train_losses, val_losses, metadata
        
    except Exception as e:
        logger.error(f"Error reading log file {log_file}: {e}")
        return [], [], metadata


def plot_single_experiment(log_file: Path, output_dir: Path):
    """
    Plot loss curves for a single experiment.
    
    Args:
        log_file: Path to the log file
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_losses, val_losses, metadata = extract_loss_from_log(log_file)
    
    if not train_losses:
        logger.warning(f"Could not extract losses from {log_file}")
        return
    
    epochs = np.arange(1, len(train_losses) + 1)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Training Loss Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Training Loss
    ax1 = axes[0]
    ax1.plot(epochs, train_losses, marker='o', linestyle='-', linewidth=2, 
             markersize=6, color='#2E86AB', label='Training Loss')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Training Loss', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add min/max annotations
    min_train_idx = np.argmin(train_losses)
    ax1.scatter([epochs[min_train_idx]], [train_losses[min_train_idx]], 
               color='red', s=100, zorder=5)
    ax1.annotate(f'Min: {train_losses[min_train_idx]:.4f}',
                xy=(epochs[min_train_idx], train_losses[min_train_idx]),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Plot 2: Validation Loss
    ax2 = axes[1]
    ax2.plot(epochs, val_losses, marker='s', linestyle='-', linewidth=2, 
             markersize=6, color='#A23B72', label='Validation Loss')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Loss', fontsize=11)
    ax2.set_title('Validation Loss', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add min annotation
    min_val_idx = np.argmin(val_losses)
    ax2.scatter([epochs[min_val_idx]], [val_losses[min_val_idx]], 
               color='red', s=100, zorder=5)
    ax2.annotate(f'Best: {val_losses[min_val_idx]:.4f}\n(Epoch {epochs[min_val_idx]})',
                xy=(epochs[min_val_idx], val_losses[min_val_idx]),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Plot 3: Combined Training and Validation Loss
    ax3 = axes[2]
    ax3.plot(epochs, train_losses, marker='o', linestyle='-', linewidth=2, 
             markersize=6, color='#2E86AB', label='Training Loss')
    ax3.plot(epochs, val_losses, marker='s', linestyle='-', linewidth=2, 
             markersize=6, color='#A23B72', label='Validation Loss')
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Loss', fontsize=11)
    ax3.set_title('Training vs Validation Loss', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Add shaded region between curves if they diverge significantly
    divergence = np.abs(np.array(train_losses) - np.array(val_losses))
    if np.max(divergence) > 0.05:
        ax3.fill_between(epochs, train_losses, val_losses, alpha=0.2, color='gray', 
                         label='Train-Val Gap')
    
    plt.tight_layout()
    
    # Save figure
    plot_path = output_dir / 'training_loss.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved plot to {plot_path}")
    
    # Save metrics as JSON
    metrics = {
        "epochs": len(train_losses),
        "train_loss": {
            "min": float(np.min(train_losses)),
            "max": float(np.max(train_losses)),
            "mean": float(np.mean(train_losses)),
            "final": float(train_losses[-1]),
        },
        "val_loss": {
            "min": float(np.min(val_losses)),
            "max": float(np.max(val_losses)),
            "mean": float(np.mean(val_losses)),
            "final": float(val_losses[-1]),
            "best_epoch": int(min_val_idx + 1),
        },
        "stability": {
            "train_loss_std": float(np.std(train_losses)),
            "val_loss_std": float(np.std(val_losses)),
            "train_loss_improvement": float(train_losses[0] - train_losses[-1]),
            "val_loss_improvement": float(val_losses[0] - val_losses[-1]),
        }
    }
    
    metrics_path = output_dir / 'loss_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("TRAINING LOSS ANALYSIS SUMMARY")
    logger.info("="*60)
    logger.info(f"Total Epochs: {metrics['epochs']}")
    logger.info(f"\nTraining Loss:")
    logger.info(f"  Min:  {metrics['train_loss']['min']:.6f}")
    logger.info(f"  Max:  {metrics['train_loss']['max']:.6f}")
    logger.info(f"  Mean: {metrics['train_loss']['mean']:.6f}")
    logger.info(f"  Final: {metrics['train_loss']['final']:.6f}")
    logger.info(f"  Improvement: {metrics['stability']['train_loss_improvement']:.6f}")
    logger.info(f"  Std Dev: {metrics['stability']['train_loss_std']:.6f}")
    
    logger.info(f"\nValidation Loss:")
    logger.info(f"  Min:  {metrics['val_loss']['min']:.6f}")
    logger.info(f"  Max:  {metrics['val_loss']['max']:.6f}")
    logger.info(f"  Mean: {metrics['val_loss']['mean']:.6f}")
    logger.info(f"  Final: {metrics['val_loss']['final']:.6f}")
    logger.info(f"  Best Epoch: {metrics['val_loss']['best_epoch']}")
    logger.info(f"  Improvement: {metrics['stability']['val_loss_improvement']:.6f}")
    logger.info(f"  Std Dev: {metrics['stability']['val_loss_std']:.6f}")
    
    # Training Stability Assessment
    logger.info(f"\nTraining Stability Assessment:")
    train_improvement_ratio = (metrics['stability']['train_loss_improvement'] / 
                               metrics['train_loss']['min']) if metrics['train_loss']['min'] > 0 else 0
    
    if metrics['stability']['train_loss_std'] < 0.05:
        stability = "STABLE ✓"
    elif metrics['stability']['train_loss_std'] < 0.15:
        stability = "MODERATELY STABLE"
    else:
        stability = "UNSTABLE (high variance)"
    
    logger.info(f"  Training Loss Std Dev: {metrics['stability']['train_loss_std']:.6f} ({stability})")
    logger.info(f"  Overfitting Gap (final val - train): {(val_losses[-1] - train_losses[-1]):.6f}")
    logger.info("="*60)
    
    plt.close()


def plot_multiple_experiments(experiments_dir: Path, output_dir: Path):
    """
    Plot loss curves for multiple experiments.
    
    Args:
        experiments_dir: Directory containing multiple experiment folders
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all run.log files
    log_files = list(experiments_dir.rglob('run.log'))
    
    if not log_files:
        logger.warning(f"No log files found in {experiments_dir}")
        return
    
    logger.info(f"Found {len(log_files)} log files")
    
    # Create comparison plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(log_files)))
    
    for idx, log_file in enumerate(log_files):
        train_losses, val_losses, metadata = extract_loss_from_log(log_file)
        
        if not train_losses:
            continue
        
        epochs = np.arange(1, len(train_losses) + 1)
        exp_name = log_file.parent.parent.name  # Get experiment folder name
        
        ax.plot(epochs, val_losses, marker='o', linestyle='-', linewidth=2, 
               label=exp_name, color=colors[idx])
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title('Validation Loss Comparison - Multiple Experiments', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')
    
    plt.tight_layout()
    
    plot_path = output_dir / 'validation_loss_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved comparison plot to {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot training and validation loss from experiment logs"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        required=True,
        help="Path to the log directory or experiments directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Directory to save plots (default: plots)"
    )
    parser.add_argument(
        "--multi",
        action="store_true",
        help="Compare multiple experiments"
    )
    
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir)
    
    if not log_dir.exists():
        logger.error(f"Log directory does not exist: {log_dir}")
        return
    
    # Check if it's a single log file or directory structure
    if log_dir.is_file() and log_dir.name == 'run.log':
        # Single log file
        logger.info(f"Plotting single experiment from {log_dir}")
        plot_single_experiment(log_dir, output_dir)
    elif (log_dir / 'run.log').exists():
        # Directory with run.log
        logger.info(f"Plotting single experiment from {log_dir}")
        plot_single_experiment(log_dir / 'run.log', output_dir)
    elif args.multi:
        # Multiple experiments
        logger.info(f"Plotting multiple experiments from {log_dir}")
        plot_multiple_experiments(log_dir, output_dir)
    else:
        logger.info(f"Attempting to find run.log files in {log_dir}")
        run_log_files = list(log_dir.rglob('run.log'))
        if run_log_files:
            for log_file in run_log_files:
                exp_name = log_file.parent.parent.name
                exp_output_dir = output_dir / exp_name
                logger.info(f"Plotting {exp_name}...")
                plot_single_experiment(log_file, exp_output_dir)
        else:
            logger.error(f"Could not find run.log files in {log_dir}")


if __name__ == "__main__":
    main()
