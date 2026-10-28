"""Unit tests for MILRankingLoss."""

from pathlib import Path
import sys

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.losses.ranking_loss import MILRankingLoss


def test_ranking_loss_scalar_and_finite():
    """Verify that ranking loss returns a valid scalar tensor and finite components."""
    criterion = MILRankingLoss(margin=1.0, lambda_smooth=8e-5, lambda_sparse=8e-5)

    # Abnormal scores higher than normal: hinge should be small or 0
    abnormal = torch.tensor([[0.9, 0.8, 0.7, 0.6]])
    normal = torch.tensor([[0.1, 0.2, 0.05, 0.1]])

    loss, components = criterion(abnormal, normal)

    assert isinstance(loss, torch.Tensor), "Loss must be a torch.Tensor"
    assert loss.dim() == 0, "Loss must be scalar"
    assert not torch.isnan(loss), "Loss must not be NaN"
    assert not torch.isinf(loss), "Loss must not be Inf"
    assert loss.item() >= 0.0, "Loss must be non-negative"

    assert "total_loss" in components
    assert "ranking_loss" in components
    assert "smoothness_loss" in components
    assert "sparsity_loss" in components


def test_ranking_loss_backprop():
    """Verify that gradients flow back into input scores."""
    criterion = MILRankingLoss(margin=1.0, lambda_smooth=0.1, lambda_sparse=0.1)

    abnormal = torch.tensor([[0.3, 0.4, 0.2]], requires_grad=True)
    normal = torch.tensor([[0.7, 0.6, 0.5]], requires_grad=True)

    loss, _ = criterion(abnormal, normal)
    loss.backward()

    assert abnormal.grad is not None, "Abnormal scores should have gradients"
    assert normal.grad is not None, "Normal scores should have gradients"
    assert not torch.isnan(abnormal.grad).any()
    assert not torch.isnan(normal.grad).any()


def test_margin_ranking_logic():
    """Verify that max(abnormal) - max(normal) > margin yields zero ranking loss."""
    criterion = MILRankingLoss(margin=0.5, lambda_smooth=0.0, lambda_sparse=0.0)

    # max_abnormal = 0.9, max_normal = 0.1, diff = 0.8 > margin(0.5) => hinge = max(0, 0.5 - 0.9 + 0.1) = 0
    abn = torch.tensor([[0.1, 0.9, 0.2]])
    norm = torch.tensor([[0.05, 0.1, 0.0]])

    loss, components = criterion(abn, norm)
    assert abs(components["ranking_loss"]) < 1e-6, "Ranking loss should be 0 when margin is satisfied"


if __name__ == "__main__":
    print("Running test_ranking_loss_scalar_and_finite()...")
    test_ranking_loss_scalar_and_finite()
    print("Running test_ranking_loss_backprop()...")
    test_ranking_loss_backprop()
    print("Running test_margin_ranking_logic()...")
    test_margin_ranking_logic()
    print("ALL RANKING LOSS TESTS PASSED SUCCESSFULLY!")
