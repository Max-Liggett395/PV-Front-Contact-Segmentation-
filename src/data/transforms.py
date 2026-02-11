"""
Data Augmentation Transforms

This module provides Albumentations-based augmentation pipelines for training
and validation datasets.

Per the paper findings:
- Morphological augmentations (GaussianBlur, noise) are most effective for SEM
- Geometric augmentations (flips, rotations) are less effective but included
- Validation uses normalization only

Note: All transforms work with both image and mask to maintain correspondence.
"""

import inspect
from typing import Dict, Any

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Albumentations v2+ renamed var_limit to std_range in GaussNoise
_GAUSS_NOISE_USE_STD_RANGE = "std_range" in inspect.signature(A.GaussNoise).parameters


def _make_gauss_noise(std_range, p):
    """Create GaussNoise compatible with both old (var_limit) and new (std_range) API."""
    if _GAUSS_NOISE_USE_STD_RANGE:
        return A.GaussNoise(std_range=std_range, p=p)
    # Convert std_range (fraction of 255) back to var_limit (pixel-scale variance)
    var_low = (std_range[0] * 255) ** 2
    var_high = (std_range[1] * 255) ** 2
    return A.GaussNoise(var_limit=(var_low, var_high), p=p)


def get_train_transforms(config: Dict[str, Any] = None) -> A.Compose:
    """
    Create training augmentation pipeline.

    Args:
        config (Dict[str, Any], optional): Configuration dictionary with
            augmentation parameters. If None, uses default values.

    Returns:
        A.Compose: Albumentations composition for training

    Default augmentations (per paper):
        - GaussianBlur: σ ~ U(0.5, 1.5), p=0.2
        - GaussNoise: var ~ U(10, 50), p=0.2
        - HorizontalFlip: p=0.5
        - VerticalFlip: p=0.5
        - Rotate: ±5°, p=0.3
        - Normalize: mean=0.0, std=1.0
    """
    if config is None:
        # Default configuration matching train_config.yaml
        config = {
            "gaussian_blur": {"enabled": True, "blur_limit": [3, 7], "sigma_limit": [0.5, 1.5], "p": 0.2},
            "gauss_noise": {"enabled": True, "std_range": [0.012, 0.028], "p": 0.2},
            "horizontal_flip": {"enabled": True, "p": 0.5},
            "vertical_flip": {"enabled": True, "p": 0.5},
            "rotate": {"enabled": True, "limit": 5, "p": 0.3},
            "normalize": {"mean": [0.0], "std": [1.0], "max_pixel_value": 255.0}
        }

    transforms_list = []

    # Morphological augmentations (most effective for SEM per paper)
    if config.get("gaussian_blur", {}).get("enabled", False):
        params = config["gaussian_blur"]
        transforms_list.append(
            A.GaussianBlur(
                blur_limit=tuple(params["blur_limit"]),
                sigma_limit=tuple(params["sigma_limit"]),
                p=params["p"]
            )
        )

    if config.get("gauss_noise", {}).get("enabled", False):
        params = config["gauss_noise"]
        transforms_list.append(
            _make_gauss_noise(
                std_range=tuple(params["std_range"]),
                p=params["p"]
            )
        )

    # Limited geometric augmentations (less effective but included)
    if config.get("horizontal_flip", {}).get("enabled", False):
        transforms_list.append(
            A.HorizontalFlip(p=config["horizontal_flip"]["p"])
        )

    if config.get("vertical_flip", {}).get("enabled", False):
        transforms_list.append(
            A.VerticalFlip(p=config["vertical_flip"]["p"])
        )

    if config.get("rotate", {}).get("enabled", False):
        params = config["rotate"]
        transforms_list.append(
            A.Rotate(
                limit=params["limit"],
                border_mode=0,  # BORDER_CONSTANT
                p=params["p"]
            )
        )

    # Normalization (always applied)
    norm_params = config.get("normalize", {"mean": [0.0], "std": [1.0], "max_pixel_value": 255.0})
    transforms_list.append(
        A.Normalize(
            mean=norm_params["mean"],
            std=norm_params["std"],
            max_pixel_value=norm_params["max_pixel_value"]
        )
    )

    # Convert to PyTorch tensors
    transforms_list.append(ToTensorV2())

    return A.Compose(transforms_list)


def get_val_transforms(config: Dict[str, Any] = None) -> A.Compose:
    """
    Create validation/test augmentation pipeline.

    Validation uses only normalization (no augmentation).

    Args:
        config (Dict[str, Any], optional): Configuration dictionary with
            normalization parameters. If None, uses default values.

    Returns:
        A.Compose: Albumentations composition for validation/test
    """
    if config is None:
        config = {
            "normalize": {"mean": [0.0], "std": [1.0], "max_pixel_value": 255.0}
        }

    norm_params = config.get("normalize", {"mean": [0.0], "std": [1.0], "max_pixel_value": 255.0})

    return A.Compose([
        A.Normalize(
            mean=norm_params["mean"],
            std=norm_params["std"],
            max_pixel_value=norm_params["max_pixel_value"]
        ),
        ToTensorV2()
    ])


def get_transforms_from_config(config_dict: Dict[str, Any], mode: str = "train") -> A.Compose:
    """
    Create transforms from a configuration dictionary.

    Args:
        config_dict (Dict[str, Any]): Full configuration dictionary
        mode (str): Either "train" or "val"

    Returns:
        A.Compose: Appropriate transform pipeline

    Example:
        >>> import yaml
        >>> with open("config/train_config.yaml") as f:
        >>>     config = yaml.safe_load(f)
        >>> train_transforms = get_transforms_from_config(config, mode="train")
        >>> val_transforms = get_transforms_from_config(config, mode="val")
    """
    if mode == "train":
        aug_config = config_dict.get("augmentation", {}).get("train", {})
        return get_train_transforms(aug_config)
    elif mode in ["val", "test"]:
        aug_config = config_dict.get("augmentation", {}).get("val", {})
        return get_val_transforms(aug_config)
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'train', 'val', or 'test'.")


# Preset configurations for quick experimentation
MINIMAL_AUGMENTATION = A.Compose([
    A.Normalize(mean=[0.0], std=[1.0], max_pixel_value=255.0),
    ToTensorV2()
])

MORPHOLOGICAL_ONLY = A.Compose([
    A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.5, 1.5), p=0.3),
    _make_gauss_noise(std_range=(0.012, 0.028), p=0.3),
    A.Normalize(mean=[0.0], std=[1.0], max_pixel_value=255.0),
    ToTensorV2()
])

GEOMETRIC_ONLY = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=5, border_mode=0, p=0.5),
    A.Normalize(mean=[0.0], std=[1.0], max_pixel_value=255.0),
    ToTensorV2()
])

HEAVY_AUGMENTATION = A.Compose([
    A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.5, 1.5), p=0.5),
    _make_gauss_noise(std_range=(0.012, 0.028), p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=10, border_mode=0, p=0.5),
    A.Normalize(mean=[0.0], std=[1.0], max_pixel_value=255.0),
    ToTensorV2()
])
