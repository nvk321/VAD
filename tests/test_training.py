"""Unit and integration tests for training logic, checkpoints, and resume."""

from pathlib import Path
import shutil
import sys

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.optim import Adam
from src.evaluation.metrics import compute_video_level_metrics
from src.losses.ranking_loss import MILRankingLoss
from src.models.mil_head import MILBaselineModel


def test_checkpoint_save_and_resume():
    """Verify that model and optimizer states save and resume cleanly."""
    tmp_ckpt_dir = Path("experiments/_tmp_ckpt")
    tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = tmp_ckpt_dir / "last.pt"

    # Setup initial model & optimizer
    model1 = MILBaselineModel(feature_dim=1024, hidden_dim=64)
    opt1 = Adam(model1.parameters(), lr=1e-3)

    # Perform a dummy update
    x_abn = torch.randn(2, 32, 1024)
    x_norm = torch.randn(2, 32, 1024)
    criterion = MILRankingLoss()

    loss, _ = criterion(model1(x_abn), model1(x_norm))
    loss.backward()
    opt1.step()

    # Save checkpoint
    saved_state = {
        "epoch": 5,
        "model_state_dict": model1.state_dict(),
        "optimizer_state_dict": opt1.state_dict(),
        "best_val_roc_auc": 0.825,
        "best_val_pr_auc": 0.791,
        "best_epoch": 4,
    }
    torch.save(saved_state, ckpt_path)
    assert ckpt_path.is_file()

    # Load into fresh model
    model2 = MILBaselineModel(feature_dim=1024, hidden_dim=64)
    opt2 = Adam(model2.parameters(), lr=1e-3)

    loaded = torch.load(ckpt_path, weights_only=False)
    model2.load_state_dict(loaded["model_state_dict"])
    opt2.load_state_dict(loaded["optimizer_state_dict"])

    assert loaded["epoch"] == 5
    assert loaded["best_val_roc_auc"] == 0.825

    # Check that model weights match exactly
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.equal(p1, p2), "Model parameters should match after loading checkpoint"

    # Cleanup
    shutil.rmtree(tmp_ckpt_dir)


def test_validation_metrics_calculation():
    """Verify compute_video_level_metrics on controlled ground-truth and predictions."""
    # Perfect predictions: abnormal higher than normal
    y_true = [1, 1, 0, 0]
    y_pred = [0.95, 0.85, 0.10, 0.05]
    metrics = compute_video_level_metrics(y_true, y_pred)

    assert metrics["roc_auc"] == 1.0, f"Expected ROC-AUC 1.0, got {metrics['roc_auc']}"
    assert metrics["pr_auc"] == 1.0, f"Expected PR-AUC 1.0, got {metrics['pr_auc']}"

    # Inverted predictions: normal higher than abnormal
    y_pred_inverted = [0.05, 0.10, 0.85, 0.95]
    metrics_inv = compute_video_level_metrics(y_true, y_pred_inverted)
    assert metrics_inv["roc_auc"] == 0.0, f"Expected ROC-AUC 0.0, got {metrics_inv['roc_auc']}"


if __name__ == "__main__":
    print("Running test_checkpoint_save_and_resume()...")
    test_checkpoint_save_and_resume()
    print("Running test_validation_metrics_calculation()...")
    test_validation_metrics_calculation()
    print("ALL TRAINING & CHECKPOINT TESTS PASSED SUCCESSFULLY!")
