"""Training Pipeline for Weakly Supervised Video Anomaly Detection (Experiment A: Baseline MIL).

Supports:
- Canonical MIL ranking loss with temporal smoothness and sparsity regularization
- Live Rich dashboard with training progress, loss breakdown, and validation metrics
- TensorBoard logging and telemetry
- Resumable training and automatic checkpointing (last.pt, best_auc.pt, best_pr_auc.pt)
- Validation-based early stopping (ROC-AUC)
- Automatic generation of paper-ready summary.json, plots, and experiment registry updates
"""

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, Optional, Tuple

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from src.data.feature_dataset import CachedFeatureDataset, MILPairDataset
from src.evaluation.metrics import compute_video_level_metrics
from src.losses.ranking_loss import MILRankingLoss
from src.models.mil_head import MILBaselineModel
from src.utils.experiment_tracker import ExperimentTracker
from src.utils.seed import seed_everything


console = Console()


def evaluate_validation(
    model: torch.nn.Module,
    val_dataset: CachedFeatureDataset,
    device: str,
) -> Tuple[Dict[str, float], list]:
    """Run validation pass and compute video-level ROC-AUC and PR-AUC."""
    model.eval()
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    y_true = []
    y_pred_scores = []
    predictions_log = []

    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)  # (1, T, D)
            label = int(batch["label"].item())
            video_id = batch["video_id"][0]
            category = batch["category"][0]

            scores = model(features)  # (1, T) or (T,)
            if scores.dim() > 1:
                scores = scores.squeeze(0)

            # In weakly supervised MIL, video-level score is the peak snippet anomaly score
            video_score = float(torch.max(scores).item())

            y_true.append(label)
            y_pred_scores.append(video_score)
            predictions_log.append({
                "video_id": video_id,
                "category": category,
                "label": label,
                "predicted_anomaly_score": round(video_score, 4),
            })

    metrics = compute_video_level_metrics(y_true, y_pred_scores)
    return metrics, predictions_log


def run_training(
    experiment_id: str = "exp_A_mil_baseline",
    features_dir: str = "data/features/smoke_test",
    train_manifest: str = "data/manifests/ucf_crime_train.csv",
    val_manifest: str = "data/manifests/ucf_crime_val.csv",
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 5e-4,
    margin: float = 1.0,
    lambda_smooth: float = 8e-5,
    lambda_sparse: float = 8e-5,
    patience: int = 15,
    feature_dim: int = 1024,
    hidden_dim: int = 64,
    dropout: float = 0.2,
    device_name: str = "auto",
    seed: int = 42,
    resume_path: Optional[str] = None,
    max_train_videos: Optional[int] = None,
    max_val_videos: Optional[int] = None,
):
    """Main training routine with live Rich telemetry and checkpoint management."""
    seed_everything(seed)

    # 1. Device selection
    if device_name == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_name

    # 2. Initialize Experiment Tracker
    config_dict = {
        "experiment_id": experiment_id,
        "features_dir": str(features_dir),
        "train_manifest": str(train_manifest),
        "val_manifest": str(val_manifest),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "margin": margin,
        "lambda_smooth": lambda_smooth,
        "lambda_sparse": lambda_sparse,
        "patience": patience,
        "feature_dim": feature_dim,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "device": device,
        "seed": seed,
    }
    tracker = ExperimentTracker(experiment_id=experiment_id, config_snapshot=config_dict)
    tb_writer = SummaryWriter(log_dir=str(tracker.tb_dir))

    console.print(Panel.fit(
        f"[bold cyan]Weakly Supervised MIL Baseline Training (Exp A)[/bold cyan]\n"
        f"Experiment ID : [green]{experiment_id}[/green]\n"
        f"Device        : [magenta]{device}[/magenta] ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})\n"
        f"Features Dir  : [yellow]{features_dir}[/yellow]\n"
        f"Batch Size    : {batch_size} pairs | Epochs: {epochs} | LR: {lr}\n"
        f"TensorBoard   : [cyan]{tracker.tb_dir}[/cyan]",
        title="Training Configuration"
    ))

    # 3. Datasets & Loaders
    console.print("Initializing cached feature datasets...")
    train_dataset = MILPairDataset(
        features_dir=features_dir,
        manifest_path=train_manifest,
        num_segments=32,
        seed=seed,
    )
    val_dataset = CachedFeatureDataset(
        features_dir=features_dir,
        manifest_path=val_manifest,
        num_segments=32,
    )

    actual_batch_size = min(batch_size, len(train_dataset))
    train_loader = DataLoader(
        train_dataset,
        batch_size=actual_batch_size,
        shuffle=True,
        drop_last=False,
    )

    console.print(
        f"Train Bags: [green]{len(train_dataset.abnormal_dataset)} abnormal / "
        f"{len(train_dataset.normal_dataset)} normal[/green] | "
        f"Val Bags: [green]{len(val_dataset)}[/green] | "
        f"Steps/Epoch: {len(train_loader)}"
    )

    # 4. Model, Loss, Optimizer, Scheduler
    model = MILBaselineModel(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)

    criterion = MILRankingLoss(
        margin=margin,
        lambda_smooth=lambda_smooth,
        lambda_sparse=lambda_sparse,
    )

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # 5. Checkpoint Resume
    start_epoch = 1
    best_val_roc_auc = 0.0
    best_val_pr_auc = 0.0
    best_epoch = 1
    epochs_no_improve = 0

    if resume_path:
        resume_file = Path(resume_path)
        if resume_file.is_file():
            console.print(f"Resuming training from checkpoint: [yellow]{resume_file}[/yellow]")
            ckpt = torch.load(resume_file, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_val_roc_auc = ckpt.get("best_val_roc_auc", 0.0)
            best_val_pr_auc = ckpt.get("best_val_pr_auc", 0.0)
            best_epoch = ckpt.get("best_epoch", 1)
            console.print(f"Resumed at epoch {start_epoch} (Best AUC so far: {best_val_roc_auc:.4f})")

    # 6. Training Loop
    total_start_time = time.perf_counter()

    for epoch in range(start_epoch, epochs + 1):
        epoch_start_time = time.perf_counter()
        model.train()

        running_total_loss = 0.0
        running_ranking_loss = 0.0
        running_smoothness_loss = 0.0
        running_sparsity_loss = 0.0
        num_batches = len(train_loader)

        with Progress(
            SpinnerColumn(),
            TextColumn(f"[bold cyan]Epoch {epoch:02d}/{epochs:02d}[/bold cyan]"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
            transient=True,
        ) as pbar:
            step_task = pbar.add_task("Training...", total=num_batches)

            for abn_feats, norm_feats in train_loader:
                abn_feats = abn_feats.to(device)    # (B, 32, D)
                norm_feats = norm_feats.to(device)  # (B, 32, D)

                optimizer.zero_grad()

                scores_abn = model(abn_feats)       # (B, 32)
                scores_norm = model(norm_feats)     # (B, 32)

                loss, loss_dict = criterion(scores_abn, scores_norm)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                running_total_loss += loss_dict["total_loss"]
                running_ranking_loss += loss_dict["ranking_loss"]
                running_smoothness_loss += loss_dict["smoothness_loss"]
                running_sparsity_loss += loss_dict["sparsity_loss"]

                pbar.advance(step_task)

        scheduler.step()
        epoch_time = time.perf_counter() - epoch_start_time

        avg_total_loss = running_total_loss / max(1, num_batches)
        avg_ranking_loss = running_ranking_loss / max(1, num_batches)
        avg_smoothness_loss = running_smoothness_loss / max(1, num_batches)
        avg_sparsity_loss = running_sparsity_loss / max(1, num_batches)

        # 7. Validation Evaluation
        val_metrics, val_preds = evaluate_validation(model, val_dataset, device)
        val_roc_auc = val_metrics["roc_auc"]
        val_pr_auc = val_metrics["pr_auc"]

        # 8. Memory measurement
        vram_mb = torch.cuda.max_memory_allocated() / (1024**2) if device == "cuda" else 0.0
        current_lr = scheduler.get_last_lr()[0]

        # 9. Telemetry & TensorBoard Logging
        tb_writer.add_scalar("train/total_loss", avg_total_loss, epoch)
        tb_writer.add_scalar("train/ranking_loss", avg_ranking_loss, epoch)
        tb_writer.add_scalar("train/smoothness_loss", avg_smoothness_loss, epoch)
        tb_writer.add_scalar("train/sparsity_loss", avg_sparsity_loss, epoch)
        tb_writer.add_scalar("val/roc_auc", val_roc_auc, epoch)
        tb_writer.add_scalar("val/pr_auc", val_pr_auc, epoch)
        tb_writer.add_scalar("system/learning_rate", current_lr, epoch)
        tb_writer.add_scalar("system/epoch_time_sec", epoch_time, epoch)
        if device == "cuda":
            tb_writer.add_scalar("system/gpu_memory_mb", vram_mb, epoch)

        epoch_record = {
            "epoch": epoch,
            "train_total_loss": round(avg_total_loss, 5),
            "train_ranking_loss": round(avg_ranking_loss, 5),
            "train_smoothness_loss": round(avg_smoothness_loss, 5),
            "train_sparsity_loss": round(avg_sparsity_loss, 5),
            "val_roc_auc": round(val_roc_auc, 4),
            "val_pr_auc": round(val_pr_auc, 4),
            "learning_rate": round(current_lr, 7),
            "epoch_time_sec": round(epoch_time, 2),
            "peak_vram_mb": round(vram_mb, 1),
        }
        tracker.log_epoch(epoch_record)

        # 10. Checkpoint Management
        is_best_auc = val_roc_auc > best_val_roc_auc
        is_best_pr = val_pr_auc > best_val_pr_auc

        if is_best_auc:
            best_val_roc_auc = val_roc_auc
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_roc_auc": val_roc_auc,
                "val_pr_auc": val_pr_auc,
                "best_epoch": best_epoch,
                "config": config_dict,
                "experiment_id": experiment_id,
            }, tracker.checkpoints_dir / "best_auc.pt")
        else:
            epochs_no_improve += 1

        if is_best_pr:
            best_val_pr_auc = val_pr_auc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_roc_auc": val_roc_auc,
                "val_pr_auc": val_pr_auc,
                "best_epoch": best_epoch,
                "config": config_dict,
                "experiment_id": experiment_id,
            }, tracker.checkpoints_dir / "best_pr_auc.pt")

        # Always save last.pt for uninterrupted resumability
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_roc_auc": val_roc_auc,
            "val_pr_auc": val_pr_auc,
            "best_val_roc_auc": best_val_roc_auc,
            "best_val_pr_auc": best_val_pr_auc,
            "best_epoch": best_epoch,
            "config": config_dict,
            "experiment_id": experiment_id,
        }, tracker.checkpoints_dir / "last.pt")

        # Save validation predictions snapshot
        val_pred_csv = tracker.predictions_dir / "validation_predictions.csv"
        with open(val_pred_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["video_id", "category", "label", "predicted_anomaly_score"])
            writer.writeheader()
            writer.writerows(val_preds)

        # 11. Live Progress Display Table (Section 8 requirement)
        table = Table(title=f"Epoch {epoch:02d}/{epochs:02d} Summary", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="dim", width=22)
        table.add_column("Value", style="bold green", width=18)
        table.add_column("Context / Status", style="cyan", width=24)

        table.add_row("Train Total Loss", f"{avg_total_loss:.4f}", f"Rank: {avg_ranking_loss:.4f}")
        table.add_row("Smoothness / Sparsity", f"{avg_smoothness_loss:.4f} / {avg_sparsity_loss:.4f}", "Regularization")
        table.add_row("Val ROC-AUC (MEASURED)", f"{val_roc_auc:.4f}", f"Best: {best_val_roc_auc:.4f} (Ep {best_epoch})")
        table.add_row("Val PR-AUC (MEASURED)", f"{val_pr_auc:.4f}", f"Best: {best_val_pr_auc:.4f}")
        table.add_row("GPU Peak VRAM", f"{vram_mb:.1f} MB", f"Cap: 4.0 GB ({vram_mb/1024:.2f} GB)")
        table.add_row("Epoch Time / LR", f"{epoch_time:.2f} s", f"LR: {current_lr:.6f}")

        console.print(table)

        # 12. Early Stopping Check
        if epochs_no_improve >= patience:
            console.print(f"[bold yellow]Early stopping triggered at epoch {epoch} (Patience: {patience} epochs without ROC-AUC improvement)[/bold yellow]")
            break

    total_training_time = time.perf_counter() - total_start_time
    tb_writer.close()

    # 13. Paper-Ready Artifact Generation
    console.print("\nGenerating paper-ready publication figures...")
    generated_plots = tracker.generate_plots()
    for p in generated_plots:
        console.print(f"  Saved plot: [green]{p}[/green]")

    model_summary = model.get_summary()
    tracker.save_summary(
        model_name="MILBaseline_MLP",
        feature_dim=feature_dim,
        num_parameters=model_summary["total_parameters"],
        best_epoch=best_epoch,
        best_val_roc_auc=best_val_roc_auc,
        best_val_pr_auc=best_val_pr_auc,
        peak_vram_mb=vram_mb,
        dataset_name="UCF-Crime",
        seed=seed,
    )

    tracker.update_registry(
        backbone="S3D (1024-d)",
        temporal_model="None (MIL Baseline)",
        persistence="None",
        risk="None",
        val_auc=best_val_roc_auc,
        val_ap=best_val_pr_auc,
        test_auc="TBD",
        fps="TBD",
        vram_mb=vram_mb,
        status="MEASURED",
    )

    console.print("\n" + "=" * 65)
    console.print("[bold green]EXPERIMENT A TRAINING COMPLETED[/bold green]")
    console.print(f"  Experiment ID     : [bold cyan]{experiment_id}[/bold cyan]")
    console.print(f"  Best Val ROC-AUC  : [bold green]{best_val_roc_auc:.4f}[/bold green] (Epoch {best_epoch}) [MEASURED]")
    console.print(f"  Best Val PR-AUC   : [bold green]{best_val_pr_auc:.4f}[/bold green] [MEASURED]")
    console.print(f"  Total Duration    : {total_training_time:.1f} s")
    console.print(f"  Peak VRAM         : {vram_mb:.1f} MB")
    console.print(f"  Checkpoints       : {tracker.checkpoints_dir.resolve()}")
    console.print(f"  Metrics & Summary : {tracker.metrics_dir.resolve()}")
    console.print(f"  Plots             : {tracker.plots_dir.resolve()}")
    console.print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Weakly Supervised MIL Baseline Model")
    parser.add_argument("--config", type=str, default="configs/training.yaml")
    parser.add_argument("--experiment", type=str, default="exp_A_mil_baseline")
    parser.add_argument("--features_dir", type=str, default="data/features/smoke_test")
    parser.add_argument("--train_manifest", type=str, default="data/manifests/ucf_crime_train.csv")
    parser.add_argument("--val_manifest", type=str, default="data/manifests/ucf_crime_val.csv")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--max_train_videos", type=int, default=None)
    parser.add_argument("--max_val_videos", type=int, default=None)

    args = parser.parse_args()

    run_training(
        experiment_id=args.experiment,
        features_dir=args.features_dir,
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        margin=args.margin,
        patience=args.patience,
        device_name=args.device,
        seed=args.seed,
        resume_path=args.resume,
        max_train_videos=args.max_train_videos,
        max_val_videos=args.max_val_videos,
    )
