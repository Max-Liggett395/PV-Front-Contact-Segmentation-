"""
Evaluation Metrics for Semantic Segmentation

This module provides metric functions for evaluating segmentation performance.

Supported metrics:
- F1 Score (per-class and macro-average)
- Intersection over Union (IoU / Jaccard Index)
- Dice Coefficient (per-class and macro-average)
- Pixel Accuracy
- Confusion Matrix

Per the paper:
- F1 scores >0.95 achieved across all 6 classes
- Per-class metrics crucial for identifying weak classes
"""

from typing import Dict, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix


def compute_f1_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 6,
    average: str = "macro",
    ignore_background: bool = False
) -> Dict[str, float]:
    """
    Compute F1 score for semantic segmentation.

    F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        predictions (torch.Tensor): Predicted logits [B, C, H, W] or
            predicted class indices [B, H, W]
        targets (torch.Tensor): Ground truth labels [B, H, W]
        num_classes (int): Number of classes. Default: 6
        average (str): Averaging strategy. Options: "macro", "none". Default: "macro"
        ignore_background (bool): Whether to exclude class 0 from averaging. Default: False

    Returns:
        Dict[str, float]: Dictionary with keys:
            - "f1_macro": Macro-averaged F1 score (if average="macro")
            - "f1_class_0", "f1_class_1", ...: Per-class F1 scores

    Example:
        >>> metrics = compute_f1_score(predictions, targets, num_classes=6)
        >>> print(f"Macro F1: {metrics['f1_macro']:.4f}")
        >>> print(f"Silver F1: {metrics['f1_class_1']:.4f}")
    """
    # Convert logits to class predictions if needed
    if predictions.ndim == 4:  # [B, C, H, W]
        predictions = torch.argmax(predictions, dim=1)  # [B, H, W]

    # Flatten tensors
    predictions_flat = predictions.flatten().cpu().numpy()
    targets_flat = targets.flatten().cpu().numpy()

    # Compute per-class F1 scores
    f1_scores = {}

    for class_idx in range(num_classes):
        # Binary masks for this class
        pred_mask = (predictions_flat == class_idx)
        target_mask = (targets_flat == class_idx)

        # True positives, false positives, false negatives
        tp = np.sum(pred_mask & target_mask)
        fp = np.sum(pred_mask & ~target_mask)
        fn = np.sum(~pred_mask & target_mask)

        # Compute precision and recall
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

        # Compute F1
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        f1_scores[f"f1_class_{class_idx}"] = float(f1)

    # Compute macro average
    if average == "macro":
        if ignore_background:
            class_indices = range(1, num_classes)
        else:
            class_indices = range(num_classes)

        f1_values = [f1_scores[f"f1_class_{i}"] for i in class_indices]
        f1_scores["f1_macro"] = float(np.mean(f1_values))

    return f1_scores


def compute_iou(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 6,
    average: str = "macro",
    ignore_background: bool = False
) -> Dict[str, float]:
    """
    Compute Intersection over Union (IoU / Jaccard Index).

    IoU = Intersection / Union = TP / (TP + FP + FN)

    Args:
        predictions (torch.Tensor): Predicted logits [B, C, H, W] or
            predicted class indices [B, H, W]
        targets (torch.Tensor): Ground truth labels [B, H, W]
        num_classes (int): Number of classes. Default: 6
        average (str): Averaging strategy. Options: "macro", "none". Default: "macro"
        ignore_background (bool): Whether to exclude class 0 from averaging. Default: False

    Returns:
        Dict[str, float]: Dictionary with keys:
            - "iou_macro": Macro-averaged IoU (if average="macro")
            - "iou_class_0", "iou_class_1", ...: Per-class IoU scores

    Example:
        >>> metrics = compute_iou(predictions, targets, num_classes=6)
        >>> print(f"Mean IoU: {metrics['iou_macro']:.4f}")
    """
    # Convert logits to class predictions if needed
    if predictions.ndim == 4:  # [B, C, H, W]
        predictions = torch.argmax(predictions, dim=1)  # [B, H, W]

    # Flatten tensors
    predictions_flat = predictions.flatten().cpu().numpy()
    targets_flat = targets.flatten().cpu().numpy()

    # Compute per-class IoU scores
    iou_scores = {}

    for class_idx in range(num_classes):
        # Binary masks for this class
        pred_mask = (predictions_flat == class_idx)
        target_mask = (targets_flat == class_idx)

        # Intersection and union
        intersection = np.sum(pred_mask & target_mask)
        union = np.sum(pred_mask | target_mask)

        # Compute IoU
        iou = intersection / (union + 1e-8)

        iou_scores[f"iou_class_{class_idx}"] = float(iou)

    # Compute macro average
    if average == "macro":
        if ignore_background:
            class_indices = range(1, num_classes)
        else:
            class_indices = range(num_classes)

        iou_values = [iou_scores[f"iou_class_{i}"] for i in class_indices]
        iou_scores["iou_macro"] = float(np.mean(iou_values))

    return iou_scores


def compute_dice_coefficient(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 6,
    average: str = "macro",
    ignore_background: bool = False
) -> Dict[str, float]:
    """
    Compute Dice coefficient (F1 equivalent for overlap).

    Dice = 2 * Intersection / (|Pred| + |Target|) = 2*TP / (2*TP + FP + FN)

    Args:
        predictions (torch.Tensor): Predicted logits [B, C, H, W] or
            predicted class indices [B, H, W]
        targets (torch.Tensor): Ground truth labels [B, H, W]
        num_classes (int): Number of classes. Default: 6
        average (str): Averaging strategy. Options: "macro", "none". Default: "macro"
        ignore_background (bool): Whether to exclude class 0 from averaging. Default: False

    Returns:
        Dict[str, float]: Dictionary with keys:
            - "dice_macro": Macro-averaged Dice coefficient (if average="macro")
            - "dice_class_0", "dice_class_1", ...: Per-class Dice scores

    Note:
        Dice coefficient is mathematically equivalent to F1 score.
    """
    # Convert logits to class predictions if needed
    if predictions.ndim == 4:  # [B, C, H, W]
        predictions = torch.argmax(predictions, dim=1)  # [B, H, W]

    # Flatten tensors
    predictions_flat = predictions.flatten().cpu().numpy()
    targets_flat = targets.flatten().cpu().numpy()

    # Compute per-class Dice scores
    dice_scores = {}

    for class_idx in range(num_classes):
        # Binary masks for this class
        pred_mask = (predictions_flat == class_idx)
        target_mask = (targets_flat == class_idx)

        # Intersection and cardinalities
        intersection = np.sum(pred_mask & target_mask)
        pred_cardinality = np.sum(pred_mask)
        target_cardinality = np.sum(target_mask)

        # Compute Dice
        dice = (2.0 * intersection) / (pred_cardinality + target_cardinality + 1e-8)

        dice_scores[f"dice_class_{class_idx}"] = float(dice)

    # Compute macro average
    if average == "macro":
        if ignore_background:
            class_indices = range(1, num_classes)
        else:
            class_indices = range(num_classes)

        dice_values = [dice_scores[f"dice_class_{i}"] for i in class_indices]
        dice_scores["dice_macro"] = float(np.mean(dice_values))

    return dice_scores


def compute_pixel_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> float:
    """
    Compute pixel-wise classification accuracy.

    Accuracy = (TP + TN) / Total Pixels

    Args:
        predictions (torch.Tensor): Predicted logits [B, C, H, W] or
            predicted class indices [B, H, W]
        targets (torch.Tensor): Ground truth labels [B, H, W]

    Returns:
        float: Pixel accuracy in range [0, 1]
    """
    # Convert logits to class predictions if needed
    if predictions.ndim == 4:  # [B, C, H, W]
        predictions = torch.argmax(predictions, dim=1)  # [B, H, W]

    # Compute accuracy
    correct = (predictions == targets).sum().item()
    total = targets.numel()

    accuracy = correct / total

    return float(accuracy)


def compute_confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 6
) -> np.ndarray:
    """
    Compute confusion matrix for segmentation.

    Args:
        predictions (torch.Tensor): Predicted logits [B, C, H, W] or
            predicted class indices [B, H, W]
        targets (torch.Tensor): Ground truth labels [B, H, W]
        num_classes (int): Number of classes. Default: 6

    Returns:
        np.ndarray: Confusion matrix of shape [num_classes, num_classes]
            - Rows: Ground truth classes
            - Columns: Predicted classes
            - Entry [i, j]: Number of pixels with true class i predicted as class j

    Example:
        >>> cm = compute_confusion_matrix(predictions, targets, num_classes=6)
        >>> print(f"Background misclassified as Silver: {cm[0, 1]}")
    """
    # Convert logits to class predictions if needed
    if predictions.ndim == 4:  # [B, C, H, W]
        predictions = torch.argmax(predictions, dim=1)  # [B, H, W]

    # Flatten tensors
    predictions_flat = predictions.flatten().cpu().numpy()
    targets_flat = targets.flatten().cpu().numpy()

    # Compute confusion matrix
    cm = sklearn_confusion_matrix(
        targets_flat,
        predictions_flat,
        labels=list(range(num_classes))
    )

    return cm


def compute_all_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 6
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.

    Args:
        predictions (torch.Tensor): Predicted logits [B, C, H, W] or
            predicted class indices [B, H, W]
        targets (torch.Tensor): Ground truth labels [B, H, W]
        num_classes (int): Number of classes. Default: 6

    Returns:
        Dict[str, float]: Dictionary containing all metrics:
            - F1 scores (macro + per-class)
            - IoU scores (macro + per-class)
            - Dice coefficients (macro + per-class)
            - Pixel accuracy

    Example:
        >>> metrics = compute_all_metrics(predictions, targets)
        >>> print(f"Macro F1: {metrics['f1_macro']:.4f}")
        >>> print(f"Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
    """
    metrics = {}

    # F1 scores
    f1_metrics = compute_f1_score(predictions, targets, num_classes=num_classes)
    metrics.update(f1_metrics)

    # IoU scores
    iou_metrics = compute_iou(predictions, targets, num_classes=num_classes)
    metrics.update(iou_metrics)

    # Dice coefficients
    dice_metrics = compute_dice_coefficient(predictions, targets, num_classes=num_classes)
    metrics.update(dice_metrics)

    # Pixel accuracy
    metrics["pixel_accuracy"] = compute_pixel_accuracy(predictions, targets)

    return metrics
