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


class FocalLoss(nn.Module):
    """Focal loss for multi-class segmentation.

    Reduces the relative loss for well-classified examples so that training
    focuses on hard, misclassified pixels.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Class balancing weight.  Can be a single float (applied to all
            classes) or a list of per-class floats.  Defaults to 1.0 (no
            balancing).
        gamma: Focusing exponent.  Higher values down-weight easy examples
            more aggressively.  Defaults to 2.0.
        class_weights: Optional per-class tensor passed to the underlying
            cross-entropy computation (separate from alpha).
        ignore_index: Target value that is excluded from loss computation.
    """

    def __init__(
        self,
        alpha: float | list[float] = 1.0,
        gamma: float = 2.0,
        class_weights: list[float] | None = None,
        ignore_index: int = -1,
    ):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

        if isinstance(alpha, (int, float)):
            self.alpha = float(alpha)
        else:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))

        weight = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        # Reduction='none' so we can apply the focal modulation per pixel.
        self.ce_loss = nn.CrossEntropyLoss(
            weight=weight, ignore_index=ignore_index, reduction="none"
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Float tensor of shape (N, C, H, W).
            targets: Long tensor of shape (N, H, W).

        Returns:
            Scalar loss value.
        """
        # Per-pixel cross-entropy, shape (N, H, W).
        ce = self.ce_loss(logits, targets)

        # Softmax probabilities for the true class, shape (N, H, W).
        # We need p_t = softmax(logits)[class], clamped to avoid log(0).
        probs = F.softmax(logits, dim=1)  # (N, C, H, W)

        # Gather the probability for the ground-truth class at each pixel.
        # Replace ignored positions with a dummy index (0) so gather doesn't error;
        # those pixels will be zeroed out by the valid_mask below.
        valid_mask = targets != self.ignore_index  # (N, H, W)
        safe_targets = targets.clone()
        safe_targets[~valid_mask] = 0
        p_t = probs.gather(dim=1, index=safe_targets.unsqueeze(1)).squeeze(1)  # (N, H, W)

        # Focal modulation factor.
        focal_factor = (1.0 - p_t) ** self.gamma  # (N, H, W)

        # Apply per-class alpha if provided as a tensor.
        if isinstance(self.alpha, torch.Tensor):
            alpha_t = self.alpha[safe_targets]  # (N, H, W)
        else:
            alpha_t = self.alpha

        loss = alpha_t * focal_factor * ce  # (N, H, W)

        # Average over valid pixels only.
        n_valid = valid_mask.sum().clamp(min=1)
        return loss[valid_mask].sum() / n_valid


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


class FocalDiceLoss(nn.Module):
    """Combined Focal + Dice loss.

    Args:
        alpha: Passed directly to FocalLoss (scalar or per-class list).
        gamma: Focusing exponent for FocalLoss.
        class_weights: Optional per-class weights for FocalLoss CE component.
        focal_weight: Scalar weight applied to the focal loss term.
        dice_weight: Scalar weight applied to the dice loss term.
        ignore_index: Target value excluded from both loss components.
    """

    def __init__(
        self,
        alpha: float | list[float] = 1.0,
        gamma: float = 2.0,
        class_weights: list[float] | None = None,
        focal_weight: float = 0.5,
        dice_weight: float = 0.5,
        ignore_index: int = -1,
    ):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.focal_loss = FocalLoss(
            alpha=alpha,
            gamma=gamma,
            class_weights=class_weights,
            ignore_index=ignore_index,
        )
        self.dice_loss = DiceLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        focal = self.focal_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.focal_weight * focal + self.dice_weight * dice


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
    elif loss_type == "focal":
        return FocalLoss(
            alpha=cfg.get("alpha", 1.0),
            gamma=cfg.get("gamma", 2.0),
            class_weights=class_weights,
            ignore_index=ignore_index,
        )
    elif loss_type == "focal_dice":
        return FocalDiceLoss(
            alpha=cfg.get("alpha", 1.0),
            gamma=cfg.get("gamma", 2.0),
            class_weights=class_weights,
            focal_weight=cfg.get("focal_weight", 0.5),
            dice_weight=cfg.get("dice_weight", 0.5),
            ignore_index=ignore_index,
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
