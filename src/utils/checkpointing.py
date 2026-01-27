"""
Checkpointing Utilities

This module provides functions for saving and loading model checkpoints.

Checkpoints include:
- Model state dictionary
- Optimizer state dictionary
- Training epoch number
- Loss history
- Configuration
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    filepath: str,
    config: Optional[Dict] = None
):
    """
    Save model checkpoint to disk.

    Args:
        model (nn.Module): Model to save
        optimizer (optim.Optimizer): Optimizer state to save
        epoch (int): Current training epoch
        metrics (Dict[str, float]): Dictionary of metrics (losses, scores)
        filepath (str): Path to save checkpoint file
        config (Optional[Dict]): Configuration dictionary. Default: None

    Example:
        >>> save_checkpoint(model, optimizer, epoch=10,
        ...                 metrics={"train_loss": 0.5, "val_loss": 0.6},
        ...                 filepath="checkpoints/model_epoch10.pth")
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics
    }

    if config is not None:
        checkpoint["config"] = config

    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> Tuple[int, Dict[str, float]]:
    """
    Load model checkpoint from disk.

    Args:
        filepath (str): Path to checkpoint file
        model (nn.Module): Model to load state into
        optimizer (Optional[optim.Optimizer]): Optimizer to load state into.
            If None, optimizer state is not loaded.
        device (Optional[torch.device]): Device to map checkpoint to.
            If None, uses current device.

    Returns:
        Tuple[int, Dict[str, float]]: (epoch, metrics)
            - epoch: Training epoch when checkpoint was saved
            - metrics: Dictionary of metrics from checkpoint

    Example:
        >>> model = UNet()
        >>> optimizer = torch.optim.Adam(model.parameters())
        >>> epoch, metrics = load_checkpoint("checkpoints/best_model.pth",
        ...                                  model, optimizer)
        >>> print(f"Loaded model from epoch {epoch}")
    """
    if device is None:
        checkpoint = torch.load(filepath, weights_only=False)
    else:
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    metrics = checkpoint.get("metrics", {})

    return epoch, metrics


def load_model_for_inference(
    filepath: str,
    model: nn.Module,
    device: Optional[torch.device] = None
) -> nn.Module:
    """
    Load model for inference (no optimizer, inference mode).

    Args:
        filepath (str): Path to checkpoint file
        model (nn.Module): Model to load state into
        device (Optional[torch.device]): Device to move model to

    Returns:
        nn.Module: Model loaded in inference mode

    Example:
        >>> model = UNet(num_classes=6)
        >>> model = load_model_for_inference("checkpoints/best_model.pth", model)
        >>> predictions = model(input_images)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model


def get_checkpoint_info(filepath: str) -> Dict:
    """
    Get information about a checkpoint without loading the full model.

    Args:
        filepath (str): Path to checkpoint file

    Returns:
        Dict: Dictionary containing checkpoint metadata:
            - epoch: Training epoch
            - metrics: Performance metrics
            - config: Configuration (if saved)

    Example:
        >>> info = get_checkpoint_info("checkpoints/best_model.pth")
        >>> print(f"Checkpoint epoch: {info['epoch']}")
        >>> print(f"Val loss: {info['metrics']['val_loss']:.4f}")
    """
    checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)

    info = {
        "epoch": checkpoint.get("epoch", None),
        "metrics": checkpoint.get("metrics", {}),
        "config": checkpoint.get("config", None)
    }

    return info


def find_best_checkpoint(checkpoint_dir: str, metric: str = "val_loss", mode: str = "min") -> Optional[str]:
    """
    Find the best checkpoint in a directory based on a metric.

    Args:
        checkpoint_dir (str): Directory containing checkpoint files
        metric (str): Metric to compare (e.g., "val_loss", "f1_macro")
        mode (str): "min" for loss, "max" for accuracy/F1

    Returns:
        Optional[str]: Path to best checkpoint, or None if no checkpoints found

    Example:
        >>> best_path = find_best_checkpoint("checkpoints", metric="val_loss", mode="min")
        >>> if best_path:
        >>>     model = load_model_for_inference(best_path, model)
    """
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        return None

    checkpoint_files = list(checkpoint_dir.glob("*.pth"))

    if len(checkpoint_files) == 0:
        return None

    best_checkpoint = None
    best_value = float('inf') if mode == "min" else float('-inf')

    for checkpoint_file in checkpoint_files:
        try:
            info = get_checkpoint_info(str(checkpoint_file))
            metrics = info.get("metrics", {})

            if metric in metrics:
                value = metrics[metric]

                if mode == "min" and value < best_value:
                    best_value = value
                    best_checkpoint = str(checkpoint_file)
                elif mode == "max" and value > best_value:
                    best_value = value
                    best_checkpoint = str(checkpoint_file)

        except Exception as e:
            print(f"Warning: Could not load {checkpoint_file}: {e}")
            continue

    return best_checkpoint
