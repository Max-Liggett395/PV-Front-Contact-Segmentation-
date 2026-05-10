"""Evaluation metrics for semantic segmentation."""

import torch
import numpy as np


def _compute_single(preds, targets, num_classes):
    """Compute IoU, F1, and pixel accuracy for a single (H, W) pair or pooled batch."""
    iou_per_class = []
    f1_per_class = []
    tp_total = 0.0
    fp_total = 0.0
    fn_total = 0.0

    for c in range(num_classes):
        pred_c = (preds == c)
        target_c = (targets == c)

        intersection = (pred_c & target_c).sum().float()
        union = (pred_c | target_c).sum().float()
        tp = intersection
        fp = (pred_c & ~target_c).sum().float()
        fn = (~pred_c & target_c).sum().float()
        tp_total += tp.item()
        fp_total += fp.item()
        fn_total += fn.item()

        iou = (intersection / (union + 1e-8)).item() if union > 0 else float("nan")
        iou_per_class.append(iou)

        precision = (tp / (tp + fp + 1e-8)).item() if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn + 1e-8)).item() if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall + 1e-8) if (precision + recall) > 0 else 0.0
        f1_per_class.append(f1)

    valid_ious = [v for v in iou_per_class if not np.isnan(v)]
    miou = np.mean(valid_ious) if valid_ious else 0.0
    f1_macro = np.mean(f1_per_class)

    # Micro F1: pooled across classes. For multiclass single-label segmentation
    # this equals pixel accuracy (each pixel contributes exactly one TP or one FP+FN).
    p_micro = tp_total / (tp_total + fp_total + 1e-8)
    r_micro = tp_total / (tp_total + fn_total + 1e-8)
    f1_micro = 2 * p_micro * r_micro / (p_micro + r_micro + 1e-8) if (p_micro + r_micro) > 0 else 0.0

    correct = (preds == targets).sum().float()
    total = targets.numel()
    pixel_acc = (correct / total).item()

    return miou, f1_macro, f1_micro, pixel_acc, iou_per_class


def compute_metrics(preds, targets, num_classes):
    """Compute segmentation metrics (global and per-image).

    Args:
        preds: (N, H, W) integer tensor of predicted classes
        targets: (N, H, W) integer tensor of ground truth classes
        num_classes: number of classes

    Returns:
        dict with global metrics (miou, f1_macro, pixel_accuracy, per_class_iou)
        and per-image averaged metrics (img_miou, img_f1_macro, img_pixel_accuracy)
    """
    # Global metrics (pooled across all images)
    miou, f1_macro, f1_micro, pixel_acc, iou_per_class = _compute_single(
        preds, targets, num_classes
    )

    # Per-image metrics (averaged across images, like sklearn per-image)
    img_mious = []
    img_f1s = []
    img_accs = []
    for i in range(preds.shape[0]):
        m, f, _fmi, a, _ = _compute_single(preds[i], targets[i], num_classes)
        img_mious.append(m)
        img_f1s.append(f)
        img_accs.append(a)

    return {
        "miou": miou,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "pixel_accuracy": pixel_acc,
        "per_class_iou": iou_per_class,
        "img_miou": np.mean(img_mious),
        "img_f1_macro": np.mean(img_f1s),
        "img_pixel_accuracy": np.mean(img_accs),
    }
