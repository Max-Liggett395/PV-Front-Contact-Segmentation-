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

from src.models.unet import UNet, FiLMLayer, create_unet


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


def test_film_layer():
    """Test FiLMLayer produces correct output shape and near-identity init."""
    film = FiLMLayer(condition_dim=32, num_features=64)

    x = torch.randn(2, 64, 16, 16)
    cond = torch.randn(2, 32)

    out = film(x, cond)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    assert torch.isfinite(out).all(), "FiLM output contains non-finite values"

    # With zero conditioning, output should be close to input (gamma~1, beta~0)
    cond_zero = torch.zeros(2, 32)
    out_zero = film(x, cond_zero)
    assert torch.allclose(out_zero, x, atol=0.1), "FiLM with zero input should be near-identity"


def test_film_unet_initialization():
    """Test that FiLM-conditioned U-Net can be initialized."""
    model = UNet(in_channels=1, num_classes=6, dropout=0.3,
                 num_magnifications=10, film_embedding_dim=32)

    assert model.film_enabled, "FiLM should be enabled"
    assert hasattr(model, 'mag_embedding'), "Should have magnification embedding"
    assert hasattr(model, 'film1'), "Should have FiLM layer 1"
    assert hasattr(model, 'film5'), "Should have FiLM layer 5 (bottleneck)"


def test_film_unet_forward_with_mag_id():
    """Test FiLM-conditioned U-Net forward pass with magnification IDs."""
    model = UNet(in_channels=1, num_classes=6, dropout=0.3,
                 num_magnifications=10, film_embedding_dim=32)

    batch_size = 2
    x = torch.randn(batch_size, 1, 64, 64)
    mag_ids = torch.tensor([3, 7])

    output = model(x, mag_id=mag_ids)

    expected_shape = (batch_size, 6, 64, 64)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    assert torch.isfinite(output).all(), "Output contains non-finite values"


def test_film_unet_forward_without_mag_id():
    """Test FiLM-conditioned U-Net works without mag_id (backward compat)."""
    model = UNet(in_channels=1, num_classes=6, dropout=0.3,
                 num_magnifications=10, film_embedding_dim=32)

    x = torch.randn(1, 1, 64, 64)
    output = model(x)  # No mag_id

    assert output.shape == (1, 6, 64, 64), f"Wrong shape: {output.shape}"


def test_film_unet_different_mag_ids_produce_different_outputs():
    """Test that different magnification IDs produce different outputs."""
    model = UNet(in_channels=1, num_classes=6, dropout=0.3,
                 num_magnifications=10, film_embedding_dim=32)

    # Initialize lazy modules
    x = torch.randn(1, 1, 64, 64)
    _ = model(x, mag_id=torch.tensor([0]))

    # Same input, different mag_ids
    torch.manual_seed(42)
    x = torch.randn(1, 1, 64, 64)

    model.training = False  # Disable dropout for deterministic comparison
    out1 = model(x, mag_id=torch.tensor([0]))
    out2 = model(x, mag_id=torch.tensor([5]))

    assert not torch.allclose(out1, out2, atol=1e-6), \
        "Different mag_ids should produce different outputs"


def test_create_unet_factory_with_film():
    """Test factory function with FiLM parameters."""
    model = create_unet(num_magnifications=5, film_embedding_dim=16)

    assert model.film_enabled, "FiLM should be enabled"
    assert isinstance(model, UNet)

    x = torch.randn(1, 1, 64, 64)
    mag_ids = torch.tensor([2])
    output = model(x, mag_id=mag_ids)
    assert output.shape == (1, 6, 64, 64)


def test_film_unet_gradient_flow():
    """Test that gradients flow through FiLM layers."""
    model = UNet(in_channels=1, num_classes=6, dropout=0.3,
                 num_magnifications=10, film_embedding_dim=32)

    x = torch.randn(1, 1, 64, 64, requires_grad=True)
    mag_ids = torch.tensor([3])
    target = torch.randint(0, 6, (1, 64, 64))

    output = model(x, mag_id=mag_ids)
    loss = torch.nn.CrossEntropyLoss()(output, target)
    loss.backward()

    # Check FiLM layers have gradients
    assert model.film1.gamma_fc.weight.grad is not None, "FiLM gamma should have gradients"
    assert model.film1.beta_fc.weight.grad is not None, "FiLM beta should have gradients"
    assert model.mag_embedding.weight.grad is not None, "Mag embedding should have gradients"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
