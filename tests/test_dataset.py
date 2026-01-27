"""
Tests for SEMDataset class

Run with: pytest tests/test_dataset.py -v
"""

import sys
from pathlib import Path

import pytest
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import SEMDataset, get_image_filenames_from_dir, verify_dataset_integrity


def test_dataset_length():
    """Test that dataset contains 141 samples."""
    img_dir = "data/images"
    label_dir = "data/labels"

    dataset = SEMDataset(img_dir, label_dir)

    assert len(dataset) == 141, f"Expected 141 samples, got {len(dataset)}"


def test_dataset_item_shapes():
    """Test that dataset returns correct tensor shapes."""
    img_dir = "data/images"
    label_dir = "data/labels"

    dataset = SEMDataset(img_dir, label_dir)

    # Get first sample
    image, label = dataset[0]

    # Check shapes
    assert isinstance(image, torch.Tensor), "Image should be a tensor"
    assert isinstance(label, torch.Tensor), "Label should be a tensor"

    # Image should be [1, H, W] (grayscale with channel dimension)
    assert image.ndim == 3, f"Image should be 3D, got {image.ndim}D"
    assert image.shape[0] == 1, f"Image should have 1 channel, got {image.shape[0]}"
    assert image.shape[1] == 768, f"Image height should be 768, got {image.shape[1]}"
    assert image.shape[2] == 1024, f"Image width should be 1024, got {image.shape[2]}"

    # Label should be [H, W] (2D mask)
    assert label.ndim == 2, f"Label should be 2D, got {label.ndim}D"
    assert label.shape[0] == 768, f"Label height should be 768, got {label.shape[0]}"
    assert label.shape[1] == 1024, f"Label width should be 1024, got {label.shape[1]}"


def test_label_values():
    """Test that labels contain only valid class indices (0-5)."""
    img_dir = "data/images"
    label_dir = "data/labels"

    dataset = SEMDataset(img_dir, label_dir)

    # Check first 5 samples
    for i in range(min(5, len(dataset))):
        _, label = dataset[i]

        unique_values = torch.unique(label).numpy()

        # All values should be in range [0, 5]
        assert all(0 <= v <= 5 for v in unique_values), \
            f"Sample {i}: Invalid label values found: {unique_values}"


def test_image_label_correspondence():
    """Test that image and label filenames correspond correctly."""
    img_dir = "data/images"
    label_dir = "data/labels"

    num_pairs, missing = verify_dataset_integrity(img_dir, label_dir)

    assert num_pairs == 141, f"Expected 141 valid pairs, got {num_pairs}"
    assert len(missing) == 0, f"Missing labels for images: {missing}"


def test_dataset_with_transform():
    """Test dataset with augmentation transforms."""
    from src.data.transforms import get_train_transforms

    img_dir = "data/images"
    label_dir = "data/labels"

    transform = get_train_transforms()
    dataset = SEMDataset(img_dir, label_dir, transform=transform)

    # Get sample with transforms applied
    image, label = dataset[0]

    # Shapes should still be correct
    assert image.shape == (1, 768, 1024), f"Unexpected image shape: {image.shape}"
    assert label.shape == (768, 1024), f"Unexpected label shape: {label.shape}"

    # Image should be normalized (values likely in different range)
    # Just check it's a valid tensor
    assert torch.isfinite(image).all(), "Image contains non-finite values"
    assert torch.isfinite(label.float()).all(), "Label contains non-finite values"


def test_dataset_subset():
    """Test dataset with specific filenames (subset)."""
    img_dir = "data/images"
    label_dir = "data/labels"

    # Get all filenames and select first 10
    all_filenames = get_image_filenames_from_dir(img_dir)
    subset_filenames = all_filenames[:10]

    dataset = SEMDataset(img_dir, label_dir, subset_filenames)

    assert len(dataset) == 10, f"Expected 10 samples, got {len(dataset)}"


def test_get_image_filenames():
    """Test helper function to get image filenames."""
    img_dir = "data/images"

    filenames = get_image_filenames_from_dir(img_dir)

    # Should return list of strings
    assert isinstance(filenames, list), "Should return a list"
    assert all(isinstance(f, str) for f in filenames), "All items should be strings"

    # Should have correct number of files
    assert len(filenames) > 0, "Should find at least some image files"

    # Should be sorted
    assert filenames == sorted(filenames), "Filenames should be sorted"


def test_dataset_dtypes():
    """Test that tensors have correct data types."""
    img_dir = "data/images"
    label_dir = "data/labels"

    dataset = SEMDataset(img_dir, label_dir)

    image, label = dataset[0]

    # Image should be float32
    assert image.dtype == torch.float32, f"Image should be float32, got {image.dtype}"

    # Label should be long (int64) for classification
    assert label.dtype == torch.long, f"Label should be long, got {label.dtype}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
