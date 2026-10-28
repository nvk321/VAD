"""Experiment tracking, plotting, checkpoint management, and paper-ready artifact serialization.

Ensures automatic generation of:
- summary.json & run_metadata.json
- epoch_metrics.csv
- training & validation curve plots (PNG)
- central experiments/experiment_registry.csv updates
"""

import csv
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import torch


class ExperimentTracker:
    """Manages experiment directory structure, metrics logging, artifact plots, and registry."""

    def __init__(
        self,
        experiment_id: str,
        experiments_root: Union[str, Path] = "experiments",
        config_snapshot: Optional[Dict[str, Any]] = None,
    ):
        self.experiment_id = experiment_id
        self.root_dir = Path(experiments_root) / experiment_id
        self.checkpoints_dir = self.root_dir / "checkpoints"
        self.logs_dir = self.root_dir / "logs"
        self.tb_dir = self.logs_dir / "tensorboard"
        self.metrics_dir = self.root_dir / "metrics"
        self.plots_dir = self.root_dir / "plots"
        self.predictions_dir = self.root_dir / "predictions"

        for d in [
            self.checkpoints_dir,
            self.logs_dir,
            self.tb_dir,
            self.metrics_dir,
            self.plots_dir,
            self.predictions_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

        self.epoch_records: List[Dict[str, Any]] = []
        self.start_time = time.perf_counter()

        if config_snapshot:
            snapshot_path = self.root_dir / "config_snapshot.yaml"
            import yaml
            with open(snapshot_path, "w", encoding="utf-8") as f:
                yaml.dump(config_snapshot, f, default_flow_style=False)

    def log_epoch(self, epoch_data: Dict[str, Any]) -> None:
        """Record an epoch's metrics to in-memory history and append to CSV."""
        self.epoch_records.append(epoch_data)

        csv_path = self.metrics_dir / "epoch_metrics.csv"
        file_exists = csv_path.is_file()

        fieldnames = list(epoch_data.keys())
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(epoch_data)

        # Update JSON dump of all epochs
        json_path = self.metrics_dir / "epoch_metrics.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.epoch_records, f, indent=2)

    def generate_plots(self) -> List[Path]:
        """Generate high-resolution paper-ready plots of training curves."""
        if not self.epoch_records:
            return []

        epochs = [r["epoch"] for r in self.epoch_records]
        generated = []

        # Style configurations for publication-quality figures
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

        # 1. Training Total Loss
        plt.figure(figsize=(7, 4.5), dpi=300)
        train_loss = [r.get("train_total_loss", r.get("train_loss", 0.0)) for r in self.epoch_records]
        plt.plot(epochs, train_loss, color="#1f77b4", linewidth=2.0, marker="o", markersize=3, label="Train Loss")
        plt.title(f"{self.experiment_id} — Training Loss Curve", fontsize=12, fontweight="bold")
        plt.xlabel("Epoch", fontsize=10)
        plt.ylabel("Total Loss", fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(frameon=True)
        plt.tight_layout()
        p1 = self.plots_dir / "training_loss.png"
        plt.savefig(p1)
        plt.close()
        generated.append(p1)

        # 2. Validation ROC-AUC & PR-AUC Curves
        val_aucs = [r.get("val_roc_auc") for r in self.epoch_records if "val_roc_auc" in r]
        val_epochs = [r["epoch"] for r in self.epoch_records if "val_roc_auc" in r]
        val_pr_aucs = [r.get("val_pr_auc") for r in self.epoch_records if "val_pr_auc" in r]

        if val_aucs:
            plt.figure(figsize=(7, 4.5), dpi=300)
            plt.plot(val_epochs, val_aucs, color="#2ca02c", linewidth=2.0, marker="s", markersize=3, label="Val ROC-AUC")
            if val_pr_aucs:
                plt.plot(val_epochs, val_pr_aucs, color="#ff7f0e", linewidth=2.0, marker="^", markersize=3, label="Val PR-AUC")
            plt.title(f"{self.experiment_id} — Validation Metrics vs Epoch", fontsize=12, fontweight="bold")
            plt.xlabel("Epoch", fontsize=10)
            plt.ylabel("Score", fontsize=10)
            plt.ylim(0.0, 1.05)
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend(frameon=True)
            plt.tight_layout()
            p2 = self.plots_dir / "validation_auc.png"
            plt.savefig(p2)
            plt.close()
            generated.append(p2)

        # 3. Loss Components Breakdown
        has_components = all(k in self.epoch_records[0] for k in ["train_ranking_loss", "train_smoothness_loss", "train_sparsity_loss"])
        if has_components:
            plt.figure(figsize=(7, 4.5), dpi=300)
            plt.plot(epochs, [r["train_ranking_loss"] for r in self.epoch_records], label="Ranking Loss", linewidth=1.8)
            plt.plot(epochs, [r["train_smoothness_loss"] for r in self.epoch_records], label="Smoothness Loss", linewidth=1.5, linestyle="--")
            plt.plot(epochs, [r["train_sparsity_loss"] for r in self.epoch_records], label="Sparsity Loss", linewidth=1.5, linestyle=":")
            plt.title(f"{self.experiment_id} — Loss Components Decomposition", fontsize=12, fontweight="bold")
            plt.xlabel("Epoch", fontsize=10)
            plt.ylabel("Component Loss Value", fontsize=10)
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend(frameon=True)
            plt.tight_layout()
            p3 = self.plots_dir / "loss_components.png"
            plt.savefig(p3)
            plt.close()
            generated.append(p3)

        return generated

    def save_summary(
        self,
        model_name: str,
        feature_dim: int,
        num_parameters: int,
        best_epoch: int,
        best_val_roc_auc: float,
        best_val_pr_auc: float,
        peak_vram_mb: float,
        dataset_name: str = "UCF-Crime",
        seed: int = 42,
    ) -> Path:
        """Write machine-readable summary.json for research paper compilation."""
        total_time = time.perf_counter() - self.start_time

        summary = {
            "experiment": self.experiment_id,
            "dataset": dataset_name,
            "seed": seed,
            "model": model_name,
            "feature_dim": feature_dim,
            "num_parameters": num_parameters,
            "best_epoch": best_epoch,
            "best_val_roc_auc": round(best_val_roc_auc, 4),
            "best_val_pr_auc": round(best_val_pr_auc, 4),
            "training_time_seconds": round(total_time, 2),
            "peak_vram_mb": round(peak_vram_mb, 2),
            "status": "MEASURED",
        }

        summary_path = self.metrics_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary_path

    def update_registry(
        self,
        backbone: str,
        temporal_model: str,
        persistence: str,
        risk: str,
        val_auc: float,
        val_ap: float,
        test_auc: str = "TBD",
        fps: str = "TBD",
        vram_mb: Optional[float] = None,
        status: str = "MEASURED",
        registry_path: Union[str, Path] = "experiments/experiment_registry.csv",
    ) -> Path:
        """Append or update this experiment's row in the central paper registry."""
        registry_path = Path(registry_path)
        registry_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "Experiment",
            "Backbone",
            "Temporal_Model",
            "Persistence",
            "Risk",
            "Val_AUC",
            "Val_AP",
            "Test_AUC",
            "FPS",
            "VRAM_MB",
            "Status",
        ]

        rows = []
        found = False
        if registry_path.is_file():
            with open(registry_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    if r.get("Experiment") == self.experiment_id:
                        found = True
                        r["Backbone"] = backbone
                        r["Temporal_Model"] = temporal_model
                        r["Persistence"] = persistence
                        r["Risk"] = risk
                        r["Val_AUC"] = f"{val_auc:.4f}"
                        r["Val_AP"] = f"{val_ap:.4f}"
                        r["Test_AUC"] = test_auc
                        r["FPS"] = fps
                        r["VRAM_MB"] = f"{vram_mb:.1f}" if vram_mb is not None else "TBD"
                        r["Status"] = status
                    rows.append(r)

        if not found:
            rows.append({
                "Experiment": self.experiment_id,
                "Backbone": backbone,
                "Temporal_Model": temporal_model,
                "Persistence": persistence,
                "Risk": risk,
                "Val_AUC": f"{val_auc:.4f}",
                "Val_AP": f"{val_ap:.4f}",
                "Test_AUC": test_auc,
                "FPS": fps,
                "VRAM_MB": f"{vram_mb:.1f}" if vram_mb is not None else "TBD",
                "Status": status,
            })

        with open(registry_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        return registry_path
