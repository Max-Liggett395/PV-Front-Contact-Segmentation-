"""Visualization utilities for SEM segmentation."""

import numpy as np
from PIL import Image

from src.data.dataset import CLASS_COLORS, CLASS_NAMES


def mask_to_rgb(mask):
    """Convert integer mask (H, W) to RGB image (H, W, 3)."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in enumerate(CLASS_COLORS):
        rgb[mask == cls_id] = color
    return rgb


def overlay_mask(image, mask, alpha=0.5):
    """Overlay colored mask on grayscale image.

    Args:
        image: (H, W) grayscale numpy array [0-255]
        mask: (H, W) integer class mask
        alpha: blend factor

    Returns:
        (H, W, 3) RGB numpy array
    """
    if image.ndim == 2:
        image_rgb = np.stack([image] * 3, axis=-1)
    else:
        image_rgb = image

    mask_rgb = mask_to_rgb(mask)
    blended = (alpha * mask_rgb + (1 - alpha) * image_rgb).astype(np.uint8)
    return blended
