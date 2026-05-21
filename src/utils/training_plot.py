import json

import numpy as np
import matplotlib.pyplot as plt
from src.utils.logging import get_logger

logger = get_logger(__name__)


def plot_training_loss(train_losses, val_losses, plots_dir):
    """
    Create plots of training and validation loss.
    Saves plots as PNG and metrics as JSON in the output directory.
    """
    
    epochs = np.arange(1, len(train_losses) + 1)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Training Loss Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Training Loss
    ax1 = axes[0]
    ax1.plot(epochs, train_losses, marker='o', linestyle='-', linewidth=2, 
                markersize=6, color='#2E86AB', label='Training Loss')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Training Loss', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add min annotation
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
        ax3.fill_between(epochs, train_losses, val_losses, alpha=0.2, color='gray')
    
    plt.tight_layout()
    
    # Save plot
    
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plots_dir / 'training_loss.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved training loss plot to {plot_path}")
    plt.close()
    
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
    
    metrics_path = plots_dir / 'loss_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved loss metrics to {metrics_path}")
    
    # Training Stability Assessment
    logger.info(f"\nTraining Stability Assessment:")
    if metrics['stability']['train_loss_std'] < 0.05:
        stability = "STABLE ✓"
    elif metrics['stability']['train_loss_std'] < 0.15:
        stability = "MODERATELY STABLE"
    else:
        stability = "UNSTABLE (high variance)"
    
    logger.info(f"  Training Loss Std Dev: {metrics['stability']['train_loss_std']:.6f} ({stability})")
    
    final_overfitting_gap = val_losses[-1] - train_losses[-1]
    logger.info(f"  Overfitting Gap (final val - train): {final_overfitting_gap:.6f}")
    
    if final_overfitting_gap > 0.1:
        logger.warning(f"  ⚠ High overfitting detected (gap={final_overfitting_gap:.6f})")
    elif final_overfitting_gap > 0.05:
        logger.info(f"  ⚠ Moderate overfitting (gap={final_overfitting_gap:.6f})")
    else:
        logger.info(f"  ✓ Good generalization (gap={final_overfitting_gap:.6f})")
    
    logger.info("="*70 + "\n")

