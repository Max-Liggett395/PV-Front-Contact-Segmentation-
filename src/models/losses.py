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


class LovaszSoftmaxLoss(nn.Module):
    """Lovász-Softmax loss that directly optimizes the IoU/Jaccard metric.

    Well-suited for imbalanced multi-class segmentation.  Implements the
    multi-class extension of the Lovász hinge described in:
        "The Lovász-Softmax loss: A tractable surrogate for the optimization
        of the intersection-over-union measure in neural networks"
        (Berman et al., CVPR 2018).

    Args:
        classes: Which classes to include in the loss.  ``'present'`` skips
            classes absent from the target batch (default); ``'all'`` includes
            every class.
        ignore_index: Target value excluded from loss computation.
    """

    def __init__(self, classes: str = "present", ignore_index: int = -1):
        super().__init__()
        if classes not in ("present", "all"):
            raise ValueError(f"classes must be 'present' or 'all', got '{classes}'")
        self.classes = classes
        self.ignore_index = ignore_index

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
        """Subgradient of the Lovász extension w.r.t. sorted errors."""
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1.0 - gt_sorted).float().cumsum(0)
        jaccard = 1.0 - intersection / union
        if p > 1:
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard

    def _lovasz_softmax_flat(
        self, probas: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Multi-class Lovász-Softmax on flattened (P, C) / (P,) tensors."""
        if probas.numel() == 0:
            return probas * 0.0

        C = probas.size(1)
        losses = []
        for c in range(C):
            fg = (labels == c).float()  # foreground indicator for class c
            if self.classes == "present" and fg.sum() == 0:
                continue
            if C == 1:
                fg_errors = (fg - (1.0 - probas[:, 0])).abs()
            else:
                fg_errors = (fg - probas[:, c]).abs()
            errors_sorted, perm = torch.sort(fg_errors, dim=0, descending=True)
            fg_sorted = fg[perm.data]
            losses.append(torch.dot(errors_sorted, self._lovasz_grad(fg_sorted)))

        if not losses:
            return torch.tensor(0.0, device=probas.device, requires_grad=True)
        return torch.stack(losses).mean()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Lovász-Softmax loss.

        Args:
            logits: Float tensor of shape (N, C, H, W).
            targets: Long tensor of shape (N, H, W).

        Returns:
            Scalar loss value.
        """
        # Mask out ignored pixels before flattening.
        valid_mask = targets != self.ignore_index  # (N, H, W)

        # Flatten spatial dims: (N*H*W, C) and (N*H*W,)
        N, C, H, W = logits.shape
        probas = F.softmax(logits, dim=1)  # (N, C, H, W)
        probas_flat = probas.permute(0, 2, 3, 1).reshape(-1, C)  # (N*H*W, C)
        targets_flat = targets.reshape(-1)  # (N*H*W,)
        valid_flat = valid_mask.reshape(-1)  # (N*H*W,)

        probas_valid = probas_flat[valid_flat]
        targets_valid = targets_flat[valid_flat]

        return self._lovasz_softmax_flat(probas_valid, targets_valid)


class LabelSmoothingCELoss(nn.Module):
    """Cross-entropy loss with label smoothing.

    Converts hard one-hot targets to soft targets:
        soft_target = (1 - smoothing) * one_hot + smoothing / num_classes

    and computes the cross-entropy against those soft targets.

    Args:
        smoothing: Label smoothing factor in [0, 1).  0.0 recovers standard CE.
        class_weights: Optional per-class weight tensor for loss scaling.
        ignore_index: Target value excluded from loss computation.
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        class_weights: list[float] | None = None,
        ignore_index: int = -1,
    ):
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError(f"smoothing must be in [0, 1), got {smoothing}")
        self.smoothing = smoothing
        self.ignore_index = ignore_index

        weight = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        self.register_buffer("class_weights", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute label-smoothed cross-entropy.

        Args:
            logits: Float tensor of shape (N, C, H, W).
            targets: Long tensor of shape (N, H, W).

        Returns:
            Scalar loss value.
        """
        N, C, H, W = logits.shape
        valid_mask = targets != self.ignore_index  # (N, H, W)

        # Build soft targets over valid pixels only.
        safe_targets = targets.clone()
        safe_targets[~valid_mask] = 0  # dummy index so one_hot doesn't error

        # one_hot: (N, H, W, C) -> (N, C, H, W)
        one_hot = F.one_hot(safe_targets, C).permute(0, 3, 1, 2).float()
        smooth_val = self.smoothing / C
        soft_targets = (1.0 - self.smoothing) * one_hot + smooth_val  # (N, C, H, W)

        # Log-softmax for numerical stability.
        log_probs = F.log_softmax(logits, dim=1)  # (N, C, H, W)

        # Per-pixel CE: -sum_c soft_target_c * log_prob_c
        if self.class_weights is not None:
            # Broadcast class_weights: (C,) -> (1, C, 1, 1)
            w = self.class_weights.view(1, C, 1, 1)
            per_pixel = -(soft_targets * log_probs * w).sum(dim=1)  # (N, H, W)
        else:
            per_pixel = -(soft_targets * log_probs).sum(dim=1)  # (N, H, W)

        # Average over valid pixels only.
        n_valid = valid_mask.sum().clamp(min=1)
        return per_pixel[valid_mask].sum() / n_valid


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
    elif loss_type == "lovasz":
        return LovaszSoftmaxLoss(
            classes=cfg.get("classes", "present"),
            ignore_index=ignore_index,
        )
    elif loss_type == "label_smoothing_ce":
        return LabelSmoothingCELoss(
            smoothing=cfg.get("smoothing", 0.1),
            class_weights=class_weights,
            ignore_index=ignore_index,
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
