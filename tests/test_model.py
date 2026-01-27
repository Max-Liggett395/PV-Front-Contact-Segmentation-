"""
Tests for U-Net model architecture

Run with: pytest tests/test_model.py -v
"""

import sys
from pathlib import Path

import pytest
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.unet import UNet, create_unet


def test_model_initialization():
    """Test that U-Net model can be initialized."""
    model = UNet(in_channels=1, num_classes=6, dropout=0.3)

    assert model is not None, "Model should be initialized"
    assert isinstance(model, torch.nn.Module), "Model should be a PyTorch module"


def test_model_forward_pass():
    """Test that model can perform forward pass and output has correct shape."""
    model = UNet(in_channels=1, num_classes=6, dropout=0.3)

    batch_size = 2
    input_tensor = torch.randn(batch_size, 1, 768, 1024)

    output = model(input_tensor)

    expected_shape = (batch_size, 6, 768, 1024)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"


def test_model_output_range():
    """Test that model outputs logits (not necessarily bounded)."""
    model = UNet(in_channels=1, num_classes=6, dropout=0.3)

    input_tensor = torch.randn(1, 1, 768, 1024)

    output = model(input_tensor)

    assert torch.isfinite(output).all(), "Output contains non-finite values"


def test_model_parameter_count():
    """Test that model has approximately 7.8M parameters (plus or minus 5 percent)."""
    model = UNet(in_channels=1, num_classes=6, dropout=0.3)

    dummy_input = torch.randn(1, 1, 768, 1024)
    _ = model(dummy_input)

    num_params = model.count_parameters()

    expected = 7.8e6
    tolerance = 0.05

    lower_bound = expected * (1 - tolerance)
    upper_bound = expected * (1 + tolerance)

    assert lower_bound <= num_params <= upper_bound, \
        f"Expected ~{expected/1e6:.1f}M parameters (±5%), got {num_params/1e6:.2f}M"


def test_model_gradient_flow():
    """Test that gradients flow through the model during backpropagation."""
    model = UNet(in_channels=1, num_classes=6, dropout=0.3)

    input_tensor = torch.randn(1, 1, 768, 1024, requires_grad=True)
    target = torch.randint(0, 6, (1, 768, 1024))

    output = model(input_tensor)

    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(output, target)

    loss.backward()

    has_gradients = False
    for param in model.parameters():
        if param.grad is not None and torch.any(param.grad != 0):
            has_gradients = True
            break

    assert has_gradients, "Model parameters should have gradients after backward pass"


def test_model_different_batch_sizes():
    """Test that model works with different batch sizes."""
    model = UNet(in_channels=1, num_classes=6, dropout=0.3)

    _ = model(torch.randn(1, 1, 768, 1024))

    batch_sizes = [1, 2, 4, 8]

    for batch_size in batch_sizes:
        input_tensor = torch.randn(batch_size, 1, 768, 1024)
        output = model(input_tensor)

        expected_shape = (batch_size, 6, 768, 1024)
        assert output.shape == expected_shape, \
            f"Batch size {batch_size}: expected shape {expected_shape}, got {output.shape}"


def test_create_unet_factory():
    """Test factory function for creating U-Net."""
    model = create_unet(in_channels=1, num_classes=6, dropout=0.3)

    assert model is not None, "Factory function should return a model"
    assert isinstance(model, UNet), "Should return UNet instance"


def test_model_training_mode():
    """Test that model can switch between train and inference modes."""
    model = UNet(in_channels=1, num_classes=6, dropout=0.3)

    assert model.training, "Model should start in training mode"

    model.eval()
    assert not model.training, "Model should be in inference mode"

    model.train()
    assert model.training, "Model should be in training mode"


def test_model_device_movement():
    """Test that model can be moved to different devices."""
    model = UNet(in_channels=1, num_classes=6, dropout=0.3)

    model = model.to("cpu")
    assert next(model.parameters()).device.type == "cpu", "Model should be on CPU"

    if torch.cuda.is_available():
        model = model.to("cuda")
        assert next(model.parameters()).device.type == "cuda", "Model should be on CUDA"


def test_model_reproducibility():
    """Test that model produces same output with same seed."""
    torch.manual_seed(42)
    model1 = UNet(in_channels=1, num_classes=6, dropout=0.3)

    torch.manual_seed(42)
    model2 = UNet(in_channels=1, num_classes=6, dropout=0.3)

    torch.manual_seed(42)
    input_tensor = torch.randn(1, 1, 768, 1024)

    model1.eval()
    model2.eval()

    output1 = model1(input_tensor)
    output2 = model2(input_tensor)

    assert torch.allclose(output1, output2, atol=1e-6), "Outputs should be identical with same seed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
