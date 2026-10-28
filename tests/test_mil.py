"""Unit tests for MILHead and MILBaselineModel."""

from pathlib import Path
import sys

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.models.mil_head import MILBaselineModel, MILHead


def test_mil_head_shapes():
    """Verify input and output shapes for 2D and 3D tensors."""
    head = MILHead(input_dim=1024, hidden_dim=64, dropout=0.2)

    # Test 3D batch input: (B=4, T=32, D=1024)
    x3d = torch.randn(4, 32, 1024)
    out3d = head(x3d)
    assert out3d.shape == (4, 32), f"Expected shape (4, 32), got {out3d.shape}"
    assert torch.all(out3d >= 0.0) and torch.all(out3d <= 1.0), "Scores must be bounded in [0, 1]"

    # Test 2D single video input: (T=15, D=1024)
    x2d = torch.randn(15, 1024)
    out2d = head(x2d)
    assert out2d.shape == (15,), f"Expected shape (15,), got {out2d.shape}"
    assert torch.all(out2d >= 0.0) and torch.all(out2d <= 1.0), "Scores must be bounded in [0, 1]"


def test_mil_baseline_model():
    """Verify MILBaselineModel wrapper, parameters, and metadata."""
    model = MILBaselineModel(feature_dim=1024, hidden_dim=64, dropout=0.2)
    x = torch.randn(2, 32, 1024)
    scores = model(x)
    assert scores.shape == (2, 32)
    assert not torch.isnan(scores).any()

    summary = model.get_summary()
    assert summary["feature_dim"] == 1024
    assert summary["total_parameters"] > 0
    assert summary["trainable_parameters"] == summary["total_parameters"]


def test_gradients_exist():
    """Verify that gradients propagate cleanly through MILHead."""
    head = MILHead(input_dim=1024, hidden_dim=64)
    x = torch.randn(2, 32, 1024, requires_grad=True)
    out = head(x)
    loss = out.sum()
    loss.backward()

    for name, param in head.named_parameters():
        assert param.grad is not None, f"Gradient missing for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"


if __name__ == "__main__":
    print("Running test_mil_head_shapes()...")
    test_mil_head_shapes()
    print("Running test_mil_baseline_model()...")
    test_mil_baseline_model()
    print("Running test_gradients_exist()...")
    test_gradients_exist()
    print("ALL MIL MODEL TESTS PASSED SUCCESSFULLY!")
