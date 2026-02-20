"""Evaluation metrics for semantic segmentation."""

import torch
import numpy as np


def compute_metrics(preds, targets, num_classes):
    """Compute segmentation metrics.

    Args:
        preds: (N, H, W) integer tensor of predicted classes
        targets: (N, H, W) integer tensor of ground truth classes
        num_classes: number of classes

    Returns:
        dict with miou, per_class_iou, f1_macro, pixel_accuracy
    """
    iou_per_class = []
    f1_per_class = []

    for c in range(num_classes):
        pred_c = (preds == c)
        target_c = (targets == c)

        intersection = (pred_c & target_c).sum().float()
        union = (pred_c | target_c).sum().float()
        tp = intersection
        fp = (pred_c & ~target_c).sum().float()
        fn = (~pred_c & target_c).sum().float()

        iou = (intersection / (union + 1e-8)).item() if union > 0 else float("nan")
        iou_per_class.append(iou)

        precision = (tp / (tp + fp + 1e-8)).item() if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn + 1e-8)).item() if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall + 1e-8) if (precision + recall) > 0 else 0.0
        f1_per_class.append(f1)

    valid_ious = [v for v in iou_per_class if not np.isnan(v)]
    miou = np.mean(valid_ious) if valid_ious else 0.0
    f1_macro = np.mean(f1_per_class)

    correct = (preds == targets).sum().float()
    total = targets.numel()
    pixel_acc = (correct / total).item()

    return {
        "miou": miou,
        "f1_macro": f1_macro,
        "pixel_accuracy": pixel_acc,
        "per_class_iou": iou_per_class,
    }
