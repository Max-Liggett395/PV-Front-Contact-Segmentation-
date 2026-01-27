#!/usr/bin/env python3
"""
Main Training Script for U-Net SEM Segmentation

Usage:
    python scripts/train.py --config config/train_config.yaml

Optional arguments:
    --resume PATH       Resume training from checkpoint
    --device DEVICE     Force device (cuda, mps, cpu)
"""

import argparse
import sys
from pathlib import Path
import yaml
import random

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.unet import UNet
from src.models.deeplabv3 import DeepLabV3, DeepLabV3Plus
from src.data.dataset import SEMDataset, get_image_filenames_from_dir
from src.data.transforms import get_transforms_from_config
from src.training.losses import get_loss_from_config
from src.training.trainer import Trainer
from src.evaluation.metrics import compute_f1_score


def create_model(model_config: dict):
    """
    Factory function to create model based on configuration.

    Args:
        model_config: Model configuration dictionary with 'name' key

    Returns:
        Model instance
    """
    model_name = model_config.get("name", "unet").lower()

    if model_name == "unet":
        return UNet(
            in_channels=model_config["in_channels"],
            num_classes=model_config["num_classes"],
            dropout=model_config.get("dropout", 0.3)
        )
    elif model_name == "deeplabv3":
        return DeepLabV3(
            in_channels=model_config["in_channels"],
            num_classes=model_config["num_classes"],
            backbone=model_config.get("backbone", "resnet50"),
            pretrained=model_config.get("pretrained", True),
            dropout=model_config.get("dropout", 0.1)
        )
    elif model_name in ["deeplabv3+", "deeplabv3plus"]:
        return DeepLabV3Plus(
            in_channels=model_config["in_channels"],
            num_classes=model_config["num_classes"],
            backbone=model_config.get("backbone", "resnet50"),
            pretrained=model_config.get("pretrained", True),
            dropout=model_config.get("dropout", 0.1),
            low_level_channels=model_config.get("low_level_channels", 48)
        )
    else:
        raise ValueError(f"Unknown model: {model_name}. Supported: unet, deeplabv3, deeplabv3+")


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(force_device: str = None) -> torch.device:
    """Auto-detect or force device selection."""
    if force_device:
        return torch.device(force_device)

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def create_data_splits(img_dir: str, config: dict) -> tuple:
    """
    Create train/val/test splits from dataset.

    Returns:
        tuple: (train_filenames, val_filenames, test_filenames)
    """
    # Get all image filenames
    all_filenames = get_image_filenames_from_dir(img_dir)

    # Extract split ratios
    train_split = config["dataset"]["train_split"]
    val_split = config["dataset"]["val_split"]
    test_split = config["dataset"]["test_split"]
    seed = config["dataset"]["seed"]

    # Verify splits sum to 1.0
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Splits must sum to 1.0"

    # First split: train vs (val + test)
    train_filenames, temp_filenames = train_test_split(
        all_filenames,
        train_size=train_split,
        random_state=seed,
        shuffle=True
    )

    # Second split: val vs test from remaining
    val_ratio = val_split / (val_split + test_split)
    val_filenames, test_filenames = train_test_split(
        temp_filenames,
        train_size=val_ratio,
        random_state=seed,
        shuffle=True
    )

    return train_filenames, val_filenames, test_filenames


def main():
    parser = argparse.ArgumentParser(description="Train U-Net for SEM segmentation")
    parser.add_argument("--config", type=str, default="config/train_config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device (cuda, mps, cpu)")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Set random seed for reproducibility
    seed = config.get("seed", 42)
    set_seed(seed)
    print(f"Random seed set to: {seed}")

    # Device selection
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Create data splits
    print("\nCreating data splits...")
    img_dir = config["dataset"]["images_dir"]
    label_dir = config["dataset"]["labels_dir"]

    train_filenames, val_filenames, test_filenames = create_data_splits(img_dir, config)

    print(f"Train set: {len(train_filenames)} images")
    print(f"Val set: {len(val_filenames)} images")
    print(f"Test set: {len(test_filenames)} images")

    # Create transforms
    train_transform = get_transforms_from_config(config, mode="train")
    val_transform = get_transforms_from_config(config, mode="val")

    # Create datasets
    train_dataset = SEMDataset(img_dir, label_dir, train_filenames, transform=train_transform)
    val_dataset = SEMDataset(img_dir, label_dir, val_filenames, transform=val_transform)

    # Create data loaders
    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    print(f"\nDataLoaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # Initialize model
    print("\nInitializing model...")
    model_config = config["model"]
    model = create_model(model_config)

    # Move model to device
    model = model.to(device)

    # Initialize LazyModules with a dummy forward pass
    model.training = False
    with torch.no_grad():
        dummy_input = torch.zeros(1, model_config["in_channels"], 768, 1024, device=device)
        _ = model(dummy_input)
    model.training = True

    model_name = model_config.get("name", "unet")
    print(f"\nModel '{model_name}' initialized successfully!")

    # Count parameters (filter out uninitialized lazy parameters)
    total_params = sum(
        p.numel() for p in model.parameters()
        if not isinstance(p, torch.nn.parameter.UninitializedParameter)
    )
    trainable_params = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad and not isinstance(p, torch.nn.parameter.UninitializedParameter)
    )
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Initialize optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 0.0)
    )

    # Initialize loss function
    loss_fn = get_loss_from_config(config)

    # Initialize metrics function (for validation)
    def metrics_fn(predictions, targets):
        return compute_f1_score(predictions, targets, num_classes=model_config["num_classes"])

    # Create checkpoint and log directories
    checkpoint_dir = config["checkpointing"]["save_dir"]
    log_dir = config["logging"]["log_dir"] if config["logging"].get("tensorboard", False) else None

    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        config=config,
        metrics_fn=metrics_fn,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir
    )

    # Run training
    num_epochs = config["training"]["num_epochs"]
    trainer.fit(num_epochs=num_epochs, resume_from=args.resume)

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
