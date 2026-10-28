"""Unit tests for ExperimentTracker, artifact serialization, and plotting."""

import csv
import json
from pathlib import Path
import shutil
import sys

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.experiment_tracker import ExperimentTracker


def test_experiment_tracker_lifecycle():
    """Verify tracker initialization, logging, plot generation, and registry updates."""
    test_exp_id = "test_exp_tracker_unit"
    test_root = Path("experiments/_test_tmp")

    if test_root.exists():
        shutil.rmtree(test_root)

    tracker = ExperimentTracker(
        experiment_id=test_exp_id,
        experiments_root=test_root,
        config_snapshot={"test_param": 123},
    )

    # 1. Verify directory creation
    assert tracker.root_dir.is_dir()
    assert tracker.checkpoints_dir.is_dir()
    assert tracker.plots_dir.is_dir()
    assert tracker.metrics_dir.is_dir()
    assert (tracker.root_dir / "config_snapshot.yaml").is_file()

    # 2. Log mock epochs
    for ep in range(1, 4):
        tracker.log_epoch({
            "epoch": ep,
            "train_total_loss": 0.5 / ep,
            "train_ranking_loss": 0.4 / ep,
            "train_smoothness_loss": 0.05 / ep,
            "train_sparsity_loss": 0.05 / ep,
            "val_roc_auc": 0.6 + (ep * 0.05),
            "val_pr_auc": 0.5 + (ep * 0.05),
            "learning_rate": 0.0001,
            "epoch_time_sec": 1.2,
            "peak_vram_mb": 100.0,
        })

    # Verify CSV and JSON metrics files
    csv_file = tracker.metrics_dir / "epoch_metrics.csv"
    json_file = tracker.metrics_dir / "epoch_metrics.json"
    assert csv_file.is_file()
    assert json_file.is_file()

    with open(csv_file, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        assert len(rows) == 3
        assert rows[-1]["epoch"] == "3"

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert len(data) == 3

    # 3. Generate plots
    plots = tracker.generate_plots()
    assert len(plots) >= 2
    for p in plots:
        assert p.is_file()
        assert p.stat().st_size > 0

    # 4. Save summary
    summary_path = tracker.save_summary(
        model_name="TestModel",
        feature_dim=1024,
        num_parameters=5000,
        best_epoch=3,
        best_val_roc_auc=0.75,
        best_val_pr_auc=0.65,
        peak_vram_mb=120.0,
    )
    assert summary_path.is_file()
    with open(summary_path, "r", encoding="utf-8") as f:
        sdata = json.load(f)
        assert sdata["status"] == "MEASURED"
        assert sdata["best_epoch"] == 3

    # 5. Update central registry
    reg_path = test_root / "experiment_registry.csv"
    tracker.update_registry(
        backbone="S3D",
        temporal_model="None",
        persistence="None",
        risk="None",
        val_auc=0.75,
        val_ap=0.65,
        registry_path=reg_path,
    )
    assert reg_path.is_file()
    with open(reg_path, "r", encoding="utf-8") as f:
        r_rows = list(csv.DictReader(f))
        assert len(r_rows) == 1
        assert r_rows[0]["Experiment"] == test_exp_id
        assert r_rows[0]["Val_AUC"] == "0.7500"

    # Cleanup temporary test directory
    shutil.rmtree(test_root)


if __name__ == "__main__":
    print("Running test_experiment_tracker_lifecycle()...")
    test_experiment_tracker_lifecycle()
    print("ALL EXPERIMENT TRACKER TESTS PASSED SUCCESSFULLY!")
