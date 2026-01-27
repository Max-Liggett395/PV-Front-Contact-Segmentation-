"""
Visualization Utilities for Segmentation Results

This module provides functions for visualizing segmentation predictions.

Features:
- Colored overlays of predictions on original images
- Side-by-side comparison grids
- Class-specific color mapping per paper specifications

Class colors (RGB):
    0: Background - Red (255, 0, 0)
    1: Silver (Ag) - Green (0, 255, 0)
    2: Glass - Blue (0, 0, 255)
    3: Silicon (Si) - Yellow (255, 255, 0)
    4: Void - Magenta (255, 0, 255)
    5: Interfacial Void - Cyan (0, 255, 255)
"""

from typing import Dict, Optional, Tuple
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch


# Default class colors from config (RGB)
DEFAULT_CLASS_COLORS = {
    0: (255, 0, 0),      # Background - Red
    1: (0, 255, 0),      # Silver (Ag) - Green
    2: (0, 0, 255),      # Glass - Blue
    3: (255, 255, 0),    # Silicon (Si) - Yellow
    4: (255, 0, 255),    # Void - Magenta
    5: (0, 255, 255)     # Interfacial Void - Cyan
}

DEFAULT_CLASS_NAMES = {
    0: "Background",
    1: "Silver (Ag)",
    2: "Glass",
    3: "Silicon (Si)",
    4: "Void",
    5: "Interfacial Void"
}


def mask_to_rgb(
    mask: np.ndarray,
    class_colors: Optional[Dict[int, Tuple[int, int, int]]] = None
) -> np.ndarray:
    """
    Convert integer mask to RGB image using class colors.

    Args:
        mask (np.ndarray): Integer mask of shape [H, W] with class indices
        class_colors (Optional[Dict]): Mapping from class index to RGB tuple.
            If None, uses DEFAULT_CLASS_COLORS.

    Returns:
        np.ndarray: RGB image of shape [H, W, 3] with uint8 values

    Example:
        >>> mask = np.array([[0, 1], [2, 3]])  # 2x2 mask
        >>> rgb_mask = mask_to_rgb(mask)
        >>> rgb_mask.shape
        (2, 2, 3)
    """
    if class_colors is None:
        class_colors = DEFAULT_CLASS_COLORS

    h, w = mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_idx, color in class_colors.items():
        rgb_mask[mask == class_idx] = color

    return rgb_mask


def create_colored_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    class_colors: Optional[Dict[int, Tuple[int, int, int]]] = None
) -> np.ndarray:
    """
    Create colored overlay of segmentation mask on grayscale image.

    Args:
        image (np.ndarray): Grayscale image [H, W] or [H, W, 1], values 0-255
        mask (np.ndarray): Integer mask [H, W] with class indices
        alpha (float): Transparency of overlay (0=transparent, 1=opaque). Default: 0.5
        class_colors (Optional[Dict]): Class-to-color mapping. Default: None

    Returns:
        np.ndarray: Blended RGB image [H, W, 3] with uint8 values

    Example:
        >>> image = np.random.randint(0, 255, (768, 1024), dtype=np.uint8)
        >>> mask = np.random.randint(0, 6, (768, 1024))
        >>> overlay = create_colored_overlay(image, mask, alpha=0.5)
    """
    if class_colors is None:
        class_colors = DEFAULT_CLASS_COLORS

    # Ensure image is 2D
    if image.ndim == 3:
        image = image.squeeze()

    # Convert grayscale image to RGB
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)

    image_rgb = np.stack([image, image, image], axis=2)

    # Convert mask to RGB
    mask_rgb = mask_to_rgb(mask, class_colors)

    # Blend image and mask
    overlay = (alpha * mask_rgb + (1 - alpha) * image_rgb).astype(np.uint8)

    return overlay


def save_comparison_grid(
    image: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    save_path: str,
    class_colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
    class_names: Optional[Dict[int, str]] = None,
    dpi: int = 150
):
    """
    Create and save a side-by-side comparison grid.

    Grid layout: [Original Image | Ground Truth | Prediction | Overlay]

    Args:
        image (np.ndarray): Grayscale image [H, W] or [H, W, 1]
        ground_truth (np.ndarray): Ground truth mask [H, W]
        prediction (np.ndarray): Predicted mask [H, W]
        save_path (str): Path to save the comparison image
        class_colors (Optional[Dict]): Class-to-color mapping
        class_names (Optional[Dict]): Class index to name mapping
        dpi (int): Resolution for saved image. Default: 150

    Example:
        >>> save_comparison_grid(image, gt_mask, pred_mask,
        ...                      "results/comparison.png")
    """
    if class_colors is None:
        class_colors = DEFAULT_CLASS_COLORS
    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES

    # Ensure image is 2D
    if image.ndim == 3:
        image = image.squeeze()

    # Normalize image to 0-255
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)

    # Create RGB versions
    gt_rgb = mask_to_rgb(ground_truth, class_colors)
    pred_rgb = mask_to_rgb(prediction, class_colors)
    overlay = create_colored_overlay(image, prediction, alpha=0.5, class_colors=class_colors)

    # Create figure with 4 subplots
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title("Original Image", fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Ground truth
    axes[1].imshow(gt_rgb)
    axes[1].set_title("Ground Truth", fontsize=12, fontweight='bold')
    axes[1].axis('off')

    # Prediction
    axes[2].imshow(pred_rgb)
    axes[2].set_title("Prediction", fontsize=12, fontweight='bold')
    axes[2].axis('off')

    # Overlay
    axes[3].imshow(overlay)
    axes[3].set_title("Overlay (Image + Prediction)", fontsize=12, fontweight='bold')
    axes[3].axis('off')

    # Add legend with class colors
    patches = [
        mpatches.Patch(color=np.array(class_colors[i])/255.0, label=class_names[i])
        for i in sorted(class_colors.keys())
    ]
    fig.legend(handles=patches, loc='lower center', ncol=6, fontsize=10, frameon=True)

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    # Save figure
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def save_single_prediction(
    image: np.ndarray,
    prediction: np.ndarray,
    save_path: str,
    class_colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
    overlay: bool = True,
    alpha: float = 0.5,
    dpi: int = 150
):
    """
    Save a single prediction visualization (overlay or mask only).

    Args:
        image (np.ndarray): Grayscale image [H, W] or [H, W, 1]
        prediction (np.ndarray): Predicted mask [H, W]
        save_path (str): Path to save the visualization
        class_colors (Optional[Dict]): Class-to-color mapping
        overlay (bool): If True, save as overlay on image. If False, save mask only.
        alpha (float): Transparency for overlay. Default: 0.5
        dpi (int): Resolution for saved image. Default: 150

    Example:
        >>> save_single_prediction(image, pred_mask, "results/pred_overlay.png",
        ...                        overlay=True, alpha=0.6)
    """
    if class_colors is None:
        class_colors = DEFAULT_CLASS_COLORS

    # Ensure image is 2D
    if image.ndim == 3:
        image = image.squeeze()

    # Normalize image to 0-255
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)

    # Create visualization
    if overlay:
        vis = create_colored_overlay(image, prediction, alpha=alpha, class_colors=class_colors)
    else:
        vis = mask_to_rgb(prediction, class_colors)

    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    plt.imshow(vis)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert PyTorch tensor to NumPy array for visualization.

    Args:
        tensor (torch.Tensor): Input tensor (image or mask)

    Returns:
        np.ndarray: NumPy array

    Example:
        >>> image_tensor = torch.rand(1, 768, 1024)  # [C, H, W]
        >>> image_np = tensor_to_numpy(image_tensor)
        >>> image_np.shape
        (768, 1024)
    """
    # Detach from computation graph and move to CPU
    array = tensor.detach().cpu().numpy()

    # Remove batch dimension if present
    if array.ndim == 4:  # [B, C, H, W]
        array = array[0]

    # Remove channel dimension for single channel
    if array.ndim == 3 and array.shape[0] == 1:  # [1, H, W]
        array = array[0]

    return array


def visualize_batch(
    images: torch.Tensor,
    ground_truths: torch.Tensor,
    predictions: torch.Tensor,
    save_dir: str,
    prefix: str = "batch",
    max_samples: int = 4,
    class_colors: Optional[Dict[int, Tuple[int, int, int]]] = None
):
    """
    Visualize multiple samples from a batch.

    Args:
        images (torch.Tensor): Batch of images [B, C, H, W]
        ground_truths (torch.Tensor): Batch of ground truth masks [B, H, W]
        predictions (torch.Tensor): Batch of predictions [B, C, H, W] or [B, H, W]
        save_dir (str): Directory to save visualizations
        prefix (str): Filename prefix. Default: "batch"
        max_samples (int): Maximum number of samples to visualize. Default: 4
        class_colors (Optional[Dict]): Class-to-color mapping

    Example:
        >>> visualize_batch(image_batch, gt_batch, pred_batch,
        ...                 save_dir="results/batch_viz", max_samples=8)
    """
    if class_colors is None:
        class_colors = DEFAULT_CLASS_COLORS

    batch_size = images.shape[0]
    num_samples = min(batch_size, max_samples)

    # Convert predictions to class indices if needed
    if predictions.ndim == 4:  # [B, C, H, W]
        predictions = torch.argmax(predictions, dim=1)  # [B, H, W]

    for i in range(num_samples):
        # Extract and convert to numpy
        image_np = tensor_to_numpy(images[i])
        gt_np = tensor_to_numpy(ground_truths[i])
        pred_np = tensor_to_numpy(predictions[i])

        # Save comparison grid
        save_path = Path(save_dir) / f"{prefix}_sample_{i}.png"
        save_comparison_grid(image_np, gt_np, pred_np, str(save_path), class_colors=class_colors)
