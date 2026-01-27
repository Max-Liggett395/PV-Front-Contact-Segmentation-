"""
SEM Dataset Module

This module provides the SEMDataset class for loading and preprocessing
cross-sectional SEM images and their corresponding segmentation masks.

Dataset structure:
- Images: PNG/JPG files in data/images/ (grayscale SEM images)
- Labels: .npy files in data/masks/npy/ (768x1024 uint8 arrays, values 0-5)
- JSON annotations: data/labels/ (VIA format, used to generate masks)

Classes:
    0: Background
    1: Silver (Ag)
    2: Glass
    3: Silicon (Si)
    4: Void
    5: Interfacial Void
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List, Callable

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class SEMDataset(Dataset):
    """
    Dataset class for loading SEM images and segmentation masks.

    Args:
        img_dir (str): Directory containing image files (.png, .jpg, .PNG)
        label_dir (str): Directory containing label files (.npy)
        img_filenames (Optional[List[str]]): List of image filenames to use.
            If None, all images in img_dir are used.
        transform (Optional[Callable]): Albumentations transform to apply
            to both image and mask. Should accept image and mask as keywords.

    Attributes:
        img_dir (Path): Path to image directory
        label_dir (Path): Path to label directory
        image_list (List[str]): List of image file paths
        transform (Optional[Callable]): Transform function
    """

    def __init__(
        self,
        img_dir: str,
        label_dir: str,
        img_filenames: Optional[List[str]] = None,
        transform: Optional[Callable] = None
    ):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform

        # Build image list
        if img_filenames is not None:
            # Use provided filenames
            self.image_list = [
                str(self.img_dir / fname) for fname in img_filenames
            ]
        else:
            # Use all images in directory
            self.image_list = [
                str(p) for p in self.img_dir.iterdir()
                if p.suffix.lower() in ['.png', '.jpg', '.jpeg']
            ]

        self.image_list.sort()  # Ensure consistent ordering

        # Verify at least one image exists
        if len(self.image_list) == 0:
            raise ValueError(f"No images found in {img_dir}")

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.image_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and return a single sample (image, label).

        Args:
            idx (int): Index of the sample to load

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - image: Tensor of shape [1, H, W] (grayscale)
                - label: Tensor of shape [H, W] with integer class labels (0-5)

        Note:
            Images are resized to 1024x768 using bilinear interpolation.
            Labels are resized using nearest-neighbor to preserve discrete values.
        """
        # Construct paths
        img_path = self.image_list[idx]
        img_basename = Path(img_path).stem
        label_path = self.label_dir / f"{img_basename}.npy"

        # Load image as grayscale and resize
        image = Image.open(img_path).convert("L")
        image = image.resize((1024, 768), Image.BILINEAR)

        # Load label and resize (use NEAREST to preserve discrete labels)
        label = np.load(label_path)
        label = Image.fromarray(label.astype(np.uint8))
        label = label.resize((1024, 768), Image.NEAREST)

        # Convert to numpy arrays
        image = np.array(image)
        label = np.array(label)

        # Apply transformations (Albumentations)
        if self.transform:
            augmented = self.transform(image=image, mask=label)
            image = augmented["image"]
            label = augmented["mask"]

        # Convert to tensors if not already
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32)

        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)

        # Ensure image has channel dimension [C, H, W]
        if image.ndim == 2:
            image = image.unsqueeze(0)

        return image, label

    def get_class_distribution(self) -> dict:
        """
        Compute the class distribution across the entire dataset.

        Returns:
            dict: Dictionary mapping class index to pixel count

        Note:
            This loads all labels and may be slow for large datasets.
        """
        class_counts = {i: 0 for i in range(6)}

        for idx in range(len(self)):
            _, label = self[idx]
            for class_idx in range(6):
                class_counts[class_idx] += (label == class_idx).sum().item()

        return class_counts

    def get_image_filename(self, idx: int) -> str:
        """
        Get the filename of the image at the given index.

        Args:
            idx (int): Index of the sample

        Returns:
            str: Image filename (without path)
        """
        return Path(self.image_list[idx]).name


def get_image_filenames_from_dir(img_dir: str) -> List[str]:
    """
    Get all image filenames from a directory.

    Args:
        img_dir (str): Path to image directory

    Returns:
        List[str]: Sorted list of image filenames (basenames only)
    """
    img_dir = Path(img_dir)
    filenames = [
        p.name for p in img_dir.iterdir()
        if p.suffix.lower() in ['.png', '.jpg', '.jpeg']
    ]
    return sorted(filenames)


def verify_dataset_integrity(img_dir: str, label_dir: str) -> Tuple[int, List[str]]:
    """
    Verify that all images have corresponding labels.

    Args:
        img_dir (str): Path to image directory
        label_dir (str): Path to label directory

    Returns:
        Tuple[int, List[str]]:
            - Number of valid image-label pairs
            - List of images without corresponding labels

    Raises:
        ValueError: If no valid image-label pairs are found
    """
    img_dir = Path(img_dir)
    label_dir = Path(label_dir)

    image_files = [
        p for p in img_dir.iterdir()
        if p.suffix.lower() in ['.png', '.jpg', '.jpeg']
    ]

    missing_labels = []
    valid_pairs = 0

    for img_path in image_files:
        label_path = label_dir / f"{img_path.stem}.npy"
        if label_path.exists():
            valid_pairs += 1
        else:
            missing_labels.append(img_path.name)

    if valid_pairs == 0:
        raise ValueError("No valid image-label pairs found")

    return valid_pairs, missing_labels
