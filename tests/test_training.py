"""
Tests for training components (losses, checkpointing, trainer)

Run with: pytest tests/test_training.py -v
"""

import sys
from pathlib import Path
import tempfile
import shutil

import pytest
import torch
import torch.optim as optim

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.unet import UNet
from src.training.losses import WeightedCrossEntropyLoss, DiceLoss, BCEDiceLoss, get_loss_function
from src.utils.checkpointing import save_checkpoint, load_checkpoint


def test_weighted_cross_entropy_loss():
    """Test weighted cross-entropy loss computation."""
    loss_fn = WeightedCrossEntropyLoss(class_weights=[1.0, 1.0, 1.5, 1.0, 1.5, 1.5])

    # Create dummy predictions and targets
    predictions = torch.randn(2, 6, 768, 1024)  # [B, C, H, W]
    targets = torch.randint(0, 6, (2, 768, 1024))  # [B, H, W]

    # Compute loss
    loss = loss_fn(predictions, targets)

    # Loss should be a scalar
    assert loss.ndim == 0, "Loss should be a scalar"

    # Loss should be positive
    assert loss.item() >= 0, "Loss should be non-negative"

    # Loss should be finite
    assert torch.isfinite(loss), "Loss should be finite"


def test_dice_loss():
    """Test Dice loss computation."""
    loss_fn = DiceLoss(smooth=1.0, ignore_background=True)

    predictions = torch.randn(2, 6, 768, 1024)
    targets = torch.randint(0, 6, (2, 768, 1024))

    loss = loss_fn(predictions, targets)

    assert loss.ndim == 0, "Loss should be a scalar"
    assert 0 <= loss.item() <= 1, "Dice loss should be in [0, 1]"
    assert torch.isfinite(loss), "Loss should be finite"


def test_bce_dice_loss():
    """Test combined BCE + Dice loss."""
    loss_fn = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)

    predictions = torch.randn(2, 6, 768, 1024)
    targets = torch.randint(0, 6, (2, 768, 1024))

    loss = loss_fn(predictions, targets)

    assert loss.ndim == 0, "Loss should be a scalar"
    assert loss.item() >= 0, "Loss should be non-negative"
    assert torch.isfinite(loss), "Loss should be finite"


def test_get_loss_function_factory():
    """Test loss function factory."""
    # Test cross-entropy
    loss_ce = get_loss_function("cross_entropy", class_weights=[1.0, 1.0, 1.5, 1.0, 1.5, 1.5])
    assert isinstance(loss_ce, WeightedCrossEntropyLoss)

    # Test Dice
    loss_dice = get_loss_function("dice")
    assert isinstance(loss_dice, DiceLoss)

    # Test BCE+Dice
    loss_bce_dice = get_loss_function("bce_dice")
    assert isinstance(loss_bce_dice, BCEDiceLoss)


def test_checkpoint_save_load():
    """Test saving and loading checkpoints."""
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "test_checkpoint.pth"

        # Create model and optimizer
        model = UNet(in_channels=1, num_classes=6, dropout=0.3)
        dummy_input = torch.randn(1, 1, 768, 1024)
        _ = model(dummy_input)

        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        # Save checkpoint
        metrics = {"train_loss": 0.5, "val_loss": 0.6}
        save_checkpoint(model, optimizer, epoch=10, metrics=metrics, filepath=str(checkpoint_path))

        # Verify file exists
        assert checkpoint_path.exists(), "Checkpoint file should be created"

        # Create new model and optimizer
        model_new = UNet(in_channels=1, num_classes=6, dropout=0.3)
        _ = model_new(dummy_input)
        optimizer_new = optim.Adam(model_new.parameters(), lr=1e-4)

        # Load checkpoint
        epoch, loaded_metrics = load_checkpoint(str(checkpoint_path), model_new, optimizer_new)

        # Verify loaded values
        assert epoch == 10, f"Expected epoch 10, got {epoch}"
        assert loaded_metrics["train_loss"] == 0.5
        assert loaded_metrics["val_loss"] == 0.6


def test_overfitting_single_batch():
    """Test that model can overfit on a single batch (sanity check)."""
    # Create small model for faster testing
    model = UNet(in_channels=1, num_classes=6, dropout=0.0)  # No dropout for overfitting

    # Create single batch
    images = torch.randn(2, 1, 768, 1024)
    labels = torch.randint(0, 6, (2, 768, 1024))

    # Initialize optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()

    # Train for several iterations
    initial_loss = None
    final_loss = None

    for iteration in range(50):
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        if iteration == 0:
            initial_loss = loss.item()
        if iteration == 49:
            final_loss = loss.item()

    # Loss should decrease significantly
    assert final_loss < initial_loss * 0.5, \
        f"Model should overfit: initial loss {initial_loss:.4f}, final loss {final_loss:.4f}"


def test_loss_backward_pass():
    """Test that loss allows gradient computation."""
    loss_fn = WeightedCrossEntropyLoss()

    predictions = torch.randn(1, 6, 768, 1024, requires_grad=True)
    targets = torch.randint(0, 6, (1, 768, 1024))

    loss = loss_fn(predictions, targets)

    # Backward pass
    loss.backward()

    # Check that predictions have gradients
    assert predictions.grad is not None, "Predictions should have gradients"
    assert torch.any(predictions.grad != 0), "Gradients should be non-zero"


def test_checkpoint_without_optimizer():
    """Test loading checkpoint without optimizer (inference mode)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "test_checkpoint.pth"

        model = UNet(in_channels=1, num_classes=6, dropout=0.3)
        dummy_input = torch.randn(1, 1, 768, 1024)
        _ = model(dummy_input)

        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        save_checkpoint(model, optimizer, epoch=5, metrics={}, filepath=str(checkpoint_path))

        model_new = UNet(in_channels=1, num_classes=6, dropout=0.3)
        _ = model_new(dummy_input)

        # Load without optimizer
        epoch, _ = load_checkpoint(str(checkpoint_path), model_new, optimizer=None)

        assert epoch == 5, f"Expected epoch 5, got {epoch}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
