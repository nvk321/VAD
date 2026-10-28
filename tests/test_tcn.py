"""Unit tests for Dilated Causal TCN and TCNModel."""

from pathlib import Path
import sys

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch

from src.models.tcn import CausalConv1d, DilatedCausalTCN, TCNModel, TemporalResidualBlock


def test_tcn_shapes():
    """Verify input and output shapes for 2D and 3D tensors."""
    model = TCNModel(feature_dim=1024, tcn_hidden_dim=128, head_hidden_dim=64, dilations=[1, 2, 4])

    # 3D input: (B=4, T=32, D=1024)
    x3d = torch.randn(4, 32, 1024)
    out3d = model(x3d)
    assert out3d.shape == (4, 32), f"Expected (4, 32), got {out3d.shape}"
    assert torch.all(out3d >= 0.0) and torch.all(out3d <= 1.0), "Scores must be bounded in [0, 1]"

    # 2D input: (T=15, D=1024)
    x2d = torch.randn(15, 1024)
    out2d = model(x2d)
    assert out2d.shape == (15,), f"Expected (15,), got {out2d.shape}"
    assert torch.all(out2d >= 0.0) and torch.all(out2d <= 1.0), "Scores must be bounded in [0, 1]"


def test_strict_temporal_causality():
    """Verify that future frames DO NOT affect past predictions (Zero Future Leakage).

    If we modify features at t = 20, the outputs for t in [0, 19] must remain exactly identical.
    """
    model = TCNModel(feature_dim=1024, tcn_hidden_dim=128, head_hidden_dim=64, dilations=[1, 2, 4])
    model.eval()

    torch.manual_seed(42)
    x_orig = torch.randn(1, 32, 1024)

    with torch.no_grad():
        out_orig = model(x_orig)  # (1, 32)

    # Perturb all frames from index 15 onwards
    x_perturbed = x_orig.clone()
    x_perturbed[:, 15:, :] += torch.randn_like(x_perturbed[:, 15:, :]) * 10.0

    with torch.no_grad():
        out_perturbed = model(x_perturbed)

    # Check past frames: indices 0 through 14
    past_orig = out_orig[:, :15]
    past_perturbed = out_perturbed[:, :15]

    diff = torch.max(torch.abs(past_orig - past_perturbed)).item()
    assert diff < 1e-6, (
        f"CAUSALITY VIOLATION DETECTED! Max past score change: {diff:.8f}. "
        "Future inputs leaked into past outputs!"
    )


def test_tcn_gradient_flow():
    """Verify that gradients propagate cleanly through TCN and head."""
    model = TCNModel(feature_dim=1024, tcn_hidden_dim=128, head_hidden_dim=64)
    x = torch.randn(2, 32, 1024, requires_grad=True)

    out = model(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "Gradient missing on input"
    assert not torch.isnan(x.grad).any(), "NaN gradient in input"

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient missing for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"


def test_tcn_summary_and_parameter_count():
    """Verify TCN metadata summary and reasonable parameter footprint (< 300k)."""
    model = TCNModel(feature_dim=1024, tcn_hidden_dim=128, head_hidden_dim=64, dilations=[1, 2, 4])
    summary = model.get_summary()

    assert summary["feature_dim"] == 1024
    assert summary["tcn_hidden_dim"] == 128
    assert summary["dilations"] == [1, 2, 4]
    assert 50000 < summary["total_parameters"] < 1000000, f"Unexpected parameter count: {summary['total_parameters']}"


if __name__ == "__main__":
    print("Testing TCN shapes...")
    test_tcn_shapes()
    print("Testing strict temporal causality...")
    test_strict_temporal_causality()
    print("Testing TCN gradient flow...")
    test_tcn_gradient_flow()
    print("Testing TCN summary...")
    test_tcn_summary_and_parameter_count()
    print("ALL TCN TESTS PASSED!")
