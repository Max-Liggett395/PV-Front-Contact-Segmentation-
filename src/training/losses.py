"""
Loss Functions for Semantic Segmentation

This module provides loss functions for training the U-Net segmentation model.

Supported losses:
- Weighted Cross-Entropy (primary choice per paper)
- Combined BCE + Dice Loss (alternative option)

Per the paper:
- Cross-entropy with class weights is the primary loss function
- Class weights: [1.0, 1.0, 1.5, 1.0, 1.5, 1.5]
  - Higher weights for Glass (2), Void (4), Interfacial Void (5)
  - Compensates for class imbalance
"""

from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted Cross-Entropy Loss for semantic segmentation.

    This is the primary loss function used in the paper, with class weights
    to handle imbalanced classes.

    Args:
        class_weights (Optional[List[float]]): Weight for each class.
            If None, uses uniform weights. Default from paper:
            [1.0, 1.0, 1.5, 1.0, 1.5, 1.5]
        ignore_index (int): Class index to ignore in loss computation.
            Default: -100 (no classes ignored)

    Shape:
        - Input: [B, C, H, W] (logits, raw model outputs)
        - Target: [B, H, W] (class indices, long tensor)
        - Output: scalar loss value
    """

    def __init__(
        self,
        class_weights: Optional[List[float]] = None,
        ignore_index: int = -100
    ):
        super().__init__()

        if class_weights is None:
            # Default weights from paper
            class_weights = [1.0, 1.0, 1.5, 1.0, 1.5, 1.5]

        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.ignore_index = ignore_index

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted cross-entropy loss.

        Args:
            predictions (torch.Tensor): Raw logits [B, C, H, W]
            targets (torch.Tensor): Ground truth labels [B, H, W]

        Returns:
            torch.Tensor: Scalar loss value
        """
        # Move class weights to same device as predictions
        if self.class_weights.device != predictions.device:
            self.class_weights = self.class_weights.to(predictions.device)

        targets = targets.long()
        loss = F.cross_entropy(
            predictions,
            targets,
            weight=self.class_weights,
            ignore_index=self.ignore_index,
            reduction='mean'
        )

        return loss


class DiceLoss(nn.Module):
    """
    Dice Loss for semantic segmentation.

    Dice loss measures overlap between predicted and ground truth segmentations.
    Ranges from 0 (perfect overlap) to 1 (no overlap).

    Args:
        smooth (float): Smoothing factor to avoid division by zero. Default: 1.0
        ignore_background (bool): Whether to exclude background class. Default: True
        ignore_index (int): Pixel-level ignore index. Pixels with this target
            value are excluded from the Dice computation. Default: -100

    Shape:
        - Input: [B, C, H, W] (probabilities after softmax)
        - Target: [B, H, W] (class indices, long tensor)
        - Output: scalar loss value
    """

    def __init__(self, smooth: float = 1.0, ignore_background: bool = True,
                 ignore_index: int = -100):
        super().__init__()
        self.smooth = smooth
        self.ignore_background = ignore_background
        self.ignore_index = ignore_index

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice loss.

        Args:
            predictions (torch.Tensor): Predicted logits [B, C, H, W]
            targets (torch.Tensor): Ground truth labels [B, H, W]

        Returns:
            torch.Tensor: Scalar loss value
        """
        # Apply softmax to get probabilities
        predictions = F.softmax(predictions, dim=1)

        # Build pixel mask for ignored indices
        valid_mask = None
        if self.ignore_index >= 0:
            valid_mask = (targets != self.ignore_index).float()

        # Convert targets to one-hot encoding (clamp ignore_index to valid range)
        num_classes = predictions.shape[1]
        safe_targets = targets.clone().long()
        if self.ignore_index >= 0:
            safe_targets[targets == self.ignore_index] = 0
        targets_one_hot = F.one_hot(safe_targets, num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        # Determine which classes to include
        if self.ignore_background:
            start_idx = 1
        else:
            start_idx = 0

        # Compute Dice coefficient for each class
        dice_scores = []
        for c in range(start_idx, num_classes):
            pred_c = predictions[:, c, :, :]
            target_c = targets_one_hot[:, c, :, :]

            if valid_mask is not None:
                pred_c = pred_c * valid_mask
                target_c = target_c * valid_mask

            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()

            dice_c = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice_c)

        # Average Dice score across classes
        dice_score = torch.stack(dice_scores).mean()

        # Return loss (1 - Dice)
        return 1.0 - dice_score


class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross-Entropy and Dice Loss.

    This loss combines BCE (pixel-wise classification) with Dice (global overlap).
    Useful for handling class imbalance and encouraging spatial consistency.

    Args:
        bce_weight (float): Weight for BCE component. Default: 0.5
        dice_weight (float): Weight for Dice component. Default: 0.5
        smooth (float): Smoothing factor for Dice loss. Default: 1.0
        class_weights (Optional[List[float]]): Weight for each class in CE component.
        ignore_index (int): Class index to ignore in loss computation. Default: -100

    Shape:
        - Input: [B, C, H, W] (logits, raw model outputs)
        - Target: [B, H, W] (class indices, long tensor)
        - Output: scalar loss value
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1.0,
        class_weights: Optional[List[float]] = None,
        ignore_index: int = -100
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index
        self.class_weights = None
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        ignore_bg = (ignore_index == 0)
        self.dice_loss = DiceLoss(smooth=smooth, ignore_background=ignore_bg,
                                  ignore_index=ignore_index)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute combined BCE + Dice loss.

        Args:
            predictions (torch.Tensor): Raw logits [B, C, H, W]
            targets (torch.Tensor): Ground truth labels [B, H, W]

        Returns:
            torch.Tensor: Scalar loss value
        """
        # Move class weights to correct device
        weights = self.class_weights
        if weights is not None and weights.device != predictions.device:
            self.class_weights = self.class_weights.to(predictions.device)
            weights = self.class_weights

        # BCE component (using cross-entropy for multi-class)
        targets = targets.long()
        bce_loss = F.cross_entropy(predictions, targets, weight=weights,
                                   ignore_index=self.ignore_index, reduction='mean')

        # Dice component
        dice_loss = self.dice_loss(predictions, targets)

        # Weighted combination
        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss

        return total_loss


def get_loss_function(
    loss_type: str,
    class_weights: Optional[List[float]] = None,
    **kwargs
) -> nn.Module:
    """
    Factory function to create loss function by name.

    Args:
        loss_type (str): Type of loss function. Options:
            - "cross_entropy": Weighted Cross-Entropy
            - "bce_dice": Combined BCE + Dice
            - "dice": Dice Loss only
        class_weights (Optional[List[float]]): Class weights for cross-entropy
        **kwargs: Additional arguments for specific loss functions

    Returns:
        nn.Module: Loss function module

    Example:
        >>> loss_fn = get_loss_function("cross_entropy", class_weights=[1.0, 1.0, 1.5, 1.0, 1.5, 1.5])
        >>> loss = loss_fn(predictions, targets)
    """
    if loss_type == "cross_entropy":
        return WeightedCrossEntropyLoss(class_weights=class_weights, **kwargs)
    elif loss_type == "bce_dice":
        return BCEDiceLoss(class_weights=class_weights, **kwargs)
    elif loss_type == "dice":
        return DiceLoss(**kwargs)
    else:
        raise ValueError(
            f"Unknown loss type: {loss_type}. "
            f"Available options: 'cross_entropy', 'bce_dice', 'dice'"
        )


def get_loss_from_config(config_dict: dict) -> nn.Module:
    """
    Create loss function from configuration dictionary.

    Args:
        config_dict (dict): Configuration dictionary with 'loss' section

    Returns:
        nn.Module: Loss function module

    Example:
        >>> import yaml
        >>> with open("config/train_config.yaml") as f:
        >>>     config = yaml.safe_load(f)
        >>> loss_fn = get_loss_from_config(config)
    """
    loss_config = config_dict.get("loss", {})
    loss_type = loss_config.get("type", "cross_entropy")
    class_weights = loss_config.get("class_weights", None)
    ignore_index = loss_config.get("ignore_index", -100)

    return get_loss_function(loss_type, class_weights=class_weights,
                             ignore_index=ignore_index)
