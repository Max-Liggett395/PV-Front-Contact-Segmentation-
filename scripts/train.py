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
import json
import sys
from collections import Counter
from pathlib import Path
import yaml
import random

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.unet import UNet
from src.models.deeplabv3 import DeepLabV3, DeepLabV3Plus
from src.data.dataset import SEMDataset, get_image_filenames_from_dir, compute_dataset_statistics, compute_class_weights
from src.data.transforms import get_transforms_from_config
from src.training.losses import get_loss_from_config
from src.training.trainer import Trainer
from src.evaluation.metrics import compute_f1_score


def create_model(model_config: dict, num_magnifications: int = None, film_embedding_dim: int = 32):
    """
    Factory function to create model based on configuration.

    Args:
        model_config: Model configuration dictionary with 'name' key
        num_magnifications: Number of magnification categories for FiLM conditioning
        film_embedding_dim: Dimension of magnification embedding

    Returns:
        Model instance
    """
    model_name = model_config.get("name", "unet").lower()

    if model_name == "unet":
        return UNet(
            in_channels=model_config["in_channels"],
            num_classes=model_config["num_classes"],
            dropout=model_config.get("dropout", 0.3),
            num_magnifications=num_magnifications,
            film_embedding_dim=film_embedding_dim,
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


def bin_magnifications(magnification_map: dict, bin_edges: list) -> dict:
    """
    Bin continuous magnification values into discrete groups.

    Each magnification value is assigned to a bin based on which interval it
    falls into. Bins are defined by edges: [e0, e1, e2, ...] creates bins
    [0, e0), [e0, e1), [e1, e2), ..., [eN, inf).

    Args:
        magnification_map: dict mapping filename stem to magnification (KX)
        bin_edges: sorted list of bin boundaries (e.g., [1, 3, 5, 8])

    Returns:
        dict mapping filename stem to bin index (float, for compat with dataset)
    """
    import bisect
    binned_map = {}
    for stem, mag in magnification_map.items():
        bin_idx = bisect.bisect_right(bin_edges, mag)
        binned_map[stem] = float(bin_idx)
    return binned_map


def create_data_splits(img_dir: str, config: dict) -> tuple:
    """
    Create train/val/test splits from dataset.

    Returns:
        tuple: (train_filenames, val_filenames, test_filenames)
    """
    # Get all image filenames
    all_filenames = get_image_filenames_from_dir(img_dir)

    # Exclude images specified in config (e.g. low-magnification images)
    exclude_stems = set(config["dataset"].get("exclude", []))
    if exclude_stems:
        before = len(all_filenames)
        all_filenames = [f for f in all_filenames if Path(f).stem not in exclude_stems]
        print(f"Excluded {before - len(all_filenames)} images ({len(all_filenames)} remaining)")

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

    # Auto-compute dataset statistics from training set
    print("\nComputing dataset statistics from training set...")
    dataset_stats = compute_dataset_statistics(img_dir, train_filenames)
    print(f"  Mean: {dataset_stats['mean']}")
    print(f"  Std:  {dataset_stats['std']}")

    # Inject computed mean/std into config normalize sections
    for mode in ("train", "val"):
        if mode in config.get("augmentation", {}):
            if "normalize" in config["augmentation"][mode]:
                config["augmentation"][mode]["normalize"]["mean"] = dataset_stats["mean"]
                config["augmentation"][mode]["normalize"]["std"] = dataset_stats["std"]

    # Auto-compute class weights from training set
    print("\nComputing class weights from training set...")
    num_classes = config["model"]["num_classes"]
    class_weights = compute_class_weights(img_dir, label_dir, train_filenames, num_classes)
    config["loss"]["class_weights"] = class_weights
    class_names = config.get("class_names", {})
    for i, w in enumerate(class_weights):
        name = class_names.get(i, f"Class {i}")
        print(f"  {name}: {w:.4f}")

    # Load magnification map if FiLM is enabled
    film_config = config.get("film", {})
    magnification_map = None
    num_magnifications = None
    film_embedding_dim = film_config.get("embedding_dim", 32)

    if film_config.get("enabled", False):
        mag_map_path = film_config.get("magnification_map", "data/magnification_map.json")
        print(f"\nLoading magnification map from {mag_map_path}...")
        with open(mag_map_path, 'r') as f:
            magnification_map = json.load(f)
        print(f"  Loaded {len(magnification_map)} entries")
        print(f"  Unique magnifications: {len(set(magnification_map.values()))}")

        # Apply binning if configured
        bin_edges = film_config.get("bin_edges")
        if bin_edges:
            print(f"  Binning magnifications with edges: {bin_edges}")
            magnification_map = bin_magnifications(magnification_map, bin_edges)
            num_bins = len(set(magnification_map.values()))
            print(f"  After binning: {num_bins} bins")

    # Create transforms
    train_transform = get_transforms_from_config(config, mode="train")
    val_transform = get_transforms_from_config(config, mode="val")

    # Create datasets
    train_dataset = SEMDataset(img_dir, label_dir, train_filenames,
                               transform=train_transform, magnification_map=magnification_map)
    val_dataset = SEMDataset(img_dir, label_dir, val_filenames,
                             transform=val_transform, magnification_map=magnification_map)

    if magnification_map is not None:
        mag_categories = train_dataset.get_magnification_categories()
        num_magnifications = len(mag_categories)
        print(f"  Magnification categories: {num_magnifications}")
        print(f"  Values: {mag_categories}")

    # Create weighted sampler if configured
    batch_size = config["training"]["batch_size"]
    num_workers = config["training"].get("num_workers", 4)
    train_sampler = None
    train_shuffle = True

    sampling_config = config.get("sampling", {})
    if sampling_config.get("weighted_by_magnification", False) and magnification_map is not None:
        print("\nCreating WeightedRandomSampler by magnification...")
        mag_ids = train_dataset.get_sample_mag_ids()
        mag_counts = Counter(mag_ids)
        # Weight = 1 / count for each sample's magnification category
        sample_weights = [1.0 / mag_counts[mid] for mid in mag_ids]
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        train_shuffle = False  # Sampler and shuffle are mutually exclusive
        print(f"  Magnification distribution in training set:")
        train_mag_cats = train_dataset.get_magnification_categories()
        if train_mag_cats:
            for mid, count in sorted(mag_counts.items()):
                mag_val = train_mag_cats[mid] if mid < len(train_mag_cats) else "?"
                print(f"    {mag_val} KX: {count} images (weight={1.0/count:.4f})")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda")
    )

    print(f"\nDataLoaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # Initialize model
    print("\nInitializing model...")
    model_config = config["model"]
    model = create_model(model_config, num_magnifications=num_magnifications,
                         film_embedding_dim=film_embedding_dim)

    # Move model to device
    model = model.to(device)

    # Initialize LazyModules with a dummy forward pass
    with torch.no_grad():
        dummy_input = torch.zeros(1, model_config["in_channels"], 768, 1024, device=device)
        if num_magnifications:
            _ = model(dummy_input, mag_id=torch.zeros(1, dtype=torch.long, device=device))
        else:
            _ = model(dummy_input)
    model.train()

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

    # Initialize LR scheduler
    scheduler = None
    sched_config = config.get("training", {}).get("scheduler", {})
    if sched_config.get("enabled", False):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=sched_config.get("mode", "min"),
            factor=sched_config.get("factor", 0.5),
            patience=sched_config.get("patience", 10),
            min_lr=sched_config.get("min_lr", 1e-7),
        )
        print(f"\nLR Scheduler: ReduceLROnPlateau (factor={sched_config.get('factor', 0.5)}, "
              f"patience={sched_config.get('patience', 10)}, min_lr={sched_config.get('min_lr', 1e-7)})")

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
        log_dir=log_dir,
        scheduler=scheduler
    )

    # Run training
    num_epochs = config["training"]["num_epochs"]
    trainer.fit(num_epochs=num_epochs, resume_from=args.resume)

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
