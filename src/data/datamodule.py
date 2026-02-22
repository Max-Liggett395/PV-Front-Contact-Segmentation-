"""Data module for managing train/val splits and dataloaders."""

import os
import glob

import torch
from torch.utils.data import DataLoader, random_split

from .dataset import SEMDataset, get_train_transform, get_val_transform


class SEMDataModule:
    """Manages dataset splits and dataloaders."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.image_dir = cfg["image_dir"]
        self.mask_dir = cfg["mask_dir"]
        self.batch_size = cfg.get("batch_size", 1)
        self.num_workers = cfg.get("num_workers", 2)
        self.seed = cfg.get("seed", 42)
        self.train_split = cfg.get("train_split", 0.85)
        self.in_channels = cfg.get("in_channels", 1)

        self.train_dataset = None
        self.val_dataset = None

    def setup(self):
        """Create train/val splits and datasets."""
        # Find all images that have a corresponding mask
        image_paths = []
        for ext in ("*.png", "*.PNG", "*.jpg", "*.JPG"):
            image_paths.extend(glob.glob(os.path.join(self.image_dir, ext)))

        image_paths = [
            p for p in image_paths
            if os.path.exists(os.path.join(
                self.mask_dir,
                os.path.splitext(os.path.basename(p))[0] + ".npy"
            ))
        ]
        image_paths.sort()

        if not image_paths:
            raise RuntimeError(
                f"No matching image/mask pairs found.\n"
                f"  image_dir: {self.image_dir} (exists={os.path.isdir(self.image_dir)})\n"
                f"  mask_dir:  {self.mask_dir} (exists={os.path.isdir(self.mask_dir)})"
            )

        # Build full dataset (no transform yet — applied via wrapper)
        full_dataset = SEMDataset(
            image_paths, self.mask_dir, transform=None, in_channels=self.in_channels,
        )

        # Split
        train_size = int(self.train_split * len(full_dataset))
        val_size = len(full_dataset) - train_size

        train_subset, val_subset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(self.seed),
        )

        # Wrap subsets with transforms
        self.train_dataset = _TransformSubset(
            train_subset, get_train_transform(self.in_channels), self.in_channels,
        )
        self.val_dataset = _TransformSubset(
            val_subset, get_val_transform(self.in_channels), self.in_channels,
        )

        print(f"Split: {train_size} train / {val_size} val (from {len(full_dataset)} total)")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class _TransformSubset:
    """Wraps a Subset to apply a specific transform at __getitem__ time."""

    def __init__(self, subset, transform, in_channels=1):
        self.subset = subset
        self.transform = transform
        self.in_channels = in_channels

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, mask = self.subset[idx]
        # image/mask are numpy arrays since the base dataset has transform=None
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        if not torch.is_tensor(image):
            image = torch.tensor(image, dtype=torch.float32)
        if image.ndim == 2:
            image = image.unsqueeze(0)
        # For 3-channel: ToTensorV2 may output (H, W, 3) — need (3, H, W)
        if image.ndim == 3 and self.in_channels == 3 and image.shape[0] != 3:
            image = image.permute(2, 0, 1)
        if not torch.is_tensor(mask):
            mask = torch.tensor(mask, dtype=torch.long)

        return image, mask
