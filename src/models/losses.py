"""Loss functions for semantic segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice loss for multi-class segmentation."""

    def __init__(self, smooth=1.0, ignore_index=-1):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        # One-hot encode targets
        valid_mask = targets != self.ignore_index
        targets_clean = targets.clone()
        targets_clean[~valid_mask] = 0
        one_hot = F.one_hot(targets_clean, num_classes).permute(0, 3, 1, 2).float()

        # Mask out ignored pixels
        valid_mask = valid_mask.unsqueeze(1).float()
        probs = probs * valid_mask
        one_hot = one_hot * valid_mask

        # Compute per-class dice
        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + one_hot.sum(dim=dims)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class DiceCELoss(nn.Module):
    """Combined Dice + Cross-Entropy loss."""

    def __init__(self, class_weights=None, dice_weight=0.5, ce_weight=0.5, ignore_index=-1):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = DiceLoss(ignore_index=ignore_index)

        weight = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, logits, targets):
        dice = self.dice_loss(logits, targets)
        ce = self.ce_loss(logits, targets)
        return self.dice_weight * dice + self.ce_weight * ce


def create_loss(cfg):
    """Create loss function from config."""
    loss_type = cfg.get("type", "dice_ce").lower()
    class_weights = cfg.get("class_weights", None)
    ignore_index = cfg.get("ignore_index", -1)

    if loss_type == "ce":
        weight = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        return nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
    elif loss_type == "dice":
        return DiceLoss(ignore_index=ignore_index)
    elif loss_type == "dice_ce":
        return DiceCELoss(
            class_weights=class_weights,
            dice_weight=cfg.get("dice_weight", 0.5),
            ce_weight=cfg.get("ce_weight", 0.5),
            ignore_index=ignore_index,
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
