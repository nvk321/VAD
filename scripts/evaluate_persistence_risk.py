"""Evaluation and Calibration Script for Temporal Persistence (Exp C) and Risk Layer (Exp D).

Answers:
1. Experiment C: Does temporal persistence reduce alert fragmentation and chattering
   compared to raw snippet-level thresholding?
2. Experiment D: Can transparent heuristic risk prioritization reliably stratify
   detected incidents into operational tiers (Low / Medium / High) without ground-truth risk labels?

Strictly respects test set quarantine: evaluated solely on the 241 validation videos.
"""

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Dict, List

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import torch
from torch.utils.data import DataLoader

from src.data.feature_dataset import CachedFeatureDataset
from src.models.tcn import TCNModel
from src.postprocessing.event_formation import EventFormationEngine, evaluate_persistence_ablation
from src.postprocessing.smoothing import exponential_moving_average
from src.risk.decision_layer import RiskDecisionLayer


console = Console()


def run_persistence_and_risk_evaluation(
    checkpoint_path: str = "experiments/exp_B_tcn/checkpoints/best_auc.pt",
    features_dir: str = "data/features/ucf_crime_s3d",
    val_manifest: str = "data/manifests/ucf_crime_val.csv",
    device_name: str = "auto",
):
    """Run end-to-end evaluation of Stage 4 (Persistence) and Stage 5 (Risk)."""
    if device_name == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_name

    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

    console.print(Panel.fit(
        f"[bold cyan]Running Persistence (Exp C) & Risk (Exp D) Evaluation[/bold cyan]\n"
        f"Checkpoint : [yellow]{ckpt_file}[/yellow]\n"
        f"Manifest   : [green]{val_manifest}[/green]\n"
        f"Features   : [yellow]{features_dir}[/yellow]\n"
        f"Device     : [magenta]{device}[/magenta]",
        title="Evaluation Suite"
    ))

    # 1. Load trained TCN Model
    ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    feature_dim = config.get("feature_dim", 1024)
    tcn_hidden_dim = config.get("tcn_hidden_dim", 128)
    head_hidden_dim = config.get("head_hidden_dim", 64)
    dilations = config.get("dilations", [1, 2, 4])

    model = TCNModel(
        feature_dim=feature_dim,
        tcn_hidden_dim=tcn_hidden_dim,
        head_hidden_dim=head_hidden_dim,
        dilations=dilations,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 2. Engines
    engine = EventFormationEngine(
        threshold_high=0.60,
        threshold_low=0.40,
        min_persistence_sec=1.5,
        max_gap_sec=1.2,
        snippet_duration_sec=0.533,
    )

    risk_layer = RiskDecisionLayer(
        weight_magnitude=0.65,
        weight_duration=0.35,
        duration_saturation_sec=8.0,
        theta_high=0.70,
        theta_low=0.45,
        theta_base=0.25,
    )

    # 3. Load Validation Dataset
    val_dataset = CachedFeatureDataset(
        features_dir=features_dir,
        manifest_path=val_manifest,
        num_segments=None,  # Use raw sequential snippets for temporal persistence
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    exp_c_dir = Path("experiments/exp_C_persistence")
    exp_d_dir = Path("experiments/exp_D_risk")
    exp_c_dir.mkdir(parents=True, exist_ok=True)
    exp_d_dir.mkdir(parents=True, exist_ok=True)
    (exp_c_dir / "plots").mkdir(exist_ok=True)
    (exp_d_dir / "plots").mkdir(exist_ok=True)

    persistence_records = []
    risk_records = []

    total_raw_blocks = 0
    total_raw_chatter = 0
    total_formed_events = 0
    total_formed_duration = 0.0

    risk_tier_counts = {"Normal": 0, "Low Risk": 0, "Medium Risk": 0, "High Risk": 0}
    category_risk_counts: Dict[str, Dict[str, int]] = {}

    sample_normal_video = None
    sample_anomaly_video = None

    console.print(f"Evaluating {len(val_loader)} validation videos...")

    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)  # (1, T, D)
            label = int(batch["label"].item())
            video_id = batch["video_id"][0]
            category = batch["category"][0]

            scores_tensor = model(features).squeeze(0)  # (T,)
            raw_scores = scores_tensor.cpu().numpy()
            smoothed_scores = exponential_moving_average(raw_scores, alpha=0.25)

            # --- Experiment C: Persistence & Event Formation ---
            ablation = evaluate_persistence_ablation(
                raw_scores=raw_scores,
                smoothed_scores=smoothed_scores,
                threshold=0.50,
                engine=engine,
            )
            formed_events = engine.form_events(smoothed_scores)

            total_raw_blocks += ablation["raw_alert_blocks"]
            total_raw_chatter += ablation["raw_chatter_toggles"]
            total_formed_events += ablation["formed_events_count"]
            total_formed_duration += ablation["total_event_duration_sec"]

            persistence_records.append({
                "video_id": video_id,
                "category": category,
                "label": label,
                "raw_alert_blocks": ablation["raw_alert_blocks"],
                "raw_chatter_toggles": ablation["raw_chatter_toggles"],
                "formed_events_count": ablation["formed_events_count"],
                "total_event_duration_sec": ablation["total_event_duration_sec"],
                "fragmentation_reduction_pct": ablation["fragmentation_reduction_pct"],
            })

            # --- Experiment D: Risk Decision Support ---
            peak_val = float(np.max(smoothed_scores)) if len(smoothed_scores) > 0 else 0.0
            video_risk = risk_layer.assess_video(formed_events, overall_peak_score=peak_val)

            risk_tier_counts[video_risk.risk_level] += 1
            if category not in category_risk_counts:
                category_risk_counts[category] = {"Normal": 0, "Low Risk": 0, "Medium Risk": 0, "High Risk": 0}
            category_risk_counts[category][video_risk.risk_level] += 1

            risk_records.append({
                "video_id": video_id,
                "category": category,
                "label": label,
                "risk_level": video_risk.risk_level,
                "risk_score": video_risk.risk_score,
                "peak_score": video_risk.peak_score,
                "duration_sec": video_risk.duration_sec,
                "persistence_ratio": video_risk.persistence_ratio,
                "magnitude_contribution": video_risk.magnitude_contribution,
                "duration_contribution": video_risk.duration_contribution,
                "explanation": video_risk.explanation,
            })

            # Cache representative samples for paper plots
            if label == 0 and sample_normal_video is None:
                sample_normal_video = (video_id, category, raw_scores, smoothed_scores, formed_events, video_risk)
            if label == 1 and sample_anomaly_video is None and len(formed_events) > 0:
                sample_anomaly_video = (video_id, category, raw_scores, smoothed_scores, formed_events, video_risk)

    # 4. Save Experiment C Persistence Results
    overall_fragmentation_red = max(
        0.0,
        (total_raw_blocks - total_formed_events) / max(1, total_raw_blocks) * 100.0
    )

    exp_c_summary = {
        "experiment": "exp_C_persistence",
        "dataset": "UCF-Crime Validation (241 videos)",
        "smoothing": "Exponential Moving Average (alpha=0.25)",
        "threshold_high": engine.threshold_high,
        "threshold_low": engine.threshold_low,
        "min_persistence_sec": engine.min_persistence_sec,
        "max_gap_sec": engine.max_gap_sec,
        "total_raw_alert_blocks": total_raw_blocks,
        "total_raw_chatter_toggles": total_raw_chatter,
        "total_formed_events": total_formed_events,
        "total_formed_duration_sec": round(total_formed_duration, 2),
        "overall_fragmentation_reduction_pct": round(overall_fragmentation_red, 1),
        "status": "MEASURED",
    }

    with open(exp_c_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(exp_c_summary, f, indent=2)

    with open(exp_c_dir / "persistence_ablation.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(persistence_records[0].keys()))
        writer.writeheader()
        writer.writerows(persistence_records)

    # 5. Save Experiment D Risk Results
    exp_d_summary = {
        "experiment": "exp_D_risk",
        "dataset": "UCF-Crime Validation (241 videos)",
        "weight_magnitude": risk_layer.w_mag,
        "weight_duration": risk_layer.w_dur,
        "duration_saturation_sec": risk_layer.duration_sat,
        "theta_high": risk_layer.theta_high,
        "theta_low": risk_layer.theta_low,
        "theta_base": risk_layer.theta_base,
        "tier_distribution": risk_tier_counts,
        "status": "MEASURED",
    }

    with open(exp_d_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(exp_d_summary, f, indent=2)

    with open(exp_d_dir / "risk_assessment.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(risk_records[0].keys()))
        writer.writeheader()
        writer.writerows(risk_records)

    # 6. Generate Publication Figures
    console.print("\nGenerating paper-ready publication figures...")

    # Figure: Persistence Ablation Bar Chart
    fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
    methods = ["Raw Thresholding\n(Instance s >= 0.50)", "Persistence-Based\n(Hysteresis + Tau_min)"]
    counts = [total_raw_blocks, total_formed_events]
    bars = ax.bar(methods, counts, color=["#d9534f", "#2b8a3e"], width=0.45)
    ax.set_ylabel("Total Number of Generated Alerts", fontsize=11, fontweight="bold")
    ax.set_title("Experiment C: Alert Fragmentation Reduction\n(241 UCF-Crime Validation Videos)", fontsize=12, fontweight="bold")
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 10, f"{int(yval):,}", ha="center", va="bottom", fontweight="bold")
    ax.set_ylim(0, max(counts) * 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig_pers_path = exp_c_dir / "plots" / "persistence_ablation.png"
    plt.savefig(fig_pers_path)
    plt.close()
    console.print(f"  Saved plot: [green]{fig_pers_path}[/green]")

    # Figure: Risk Tier Distribution
    fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
    tiers = list(risk_tier_counts.keys())
    tier_vals = [risk_tier_counts[t] for t in tiers]
    colors = ["#6c757d", "#17a2b8", "#ffc107", "#dc3545"]
    bars = ax.bar(tiers, tier_vals, color=colors, width=0.5)
    ax.set_ylabel("Number of Videos", fontsize=11, fontweight="bold")
    ax.set_title("Experiment D: Operational Risk Tier Stratification\n(Validation Set Distribution)", fontsize=12, fontweight="bold")
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 3, f"{int(yval)}", ha="center", va="bottom", fontweight="bold")
    ax.set_ylim(0, max(tier_vals) * 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig_risk_path = exp_d_dir / "plots" / "risk_distribution.png"
    plt.savefig(fig_risk_path)
    plt.close()
    console.print(f"  Saved plot: [green]{fig_risk_path}[/green]")

    # Figure: Qualitative Timeline Examples (Normal vs Anomalous with formed events)
    if sample_anomaly_video is not None:
        vid_id, cat, raw, smooth, events, risk = sample_anomaly_video
        fig, ax = plt.subplots(figsize=(10, 4), dpi=300)
        time_axis = np.arange(len(raw)) * 0.533
        ax.plot(time_axis, raw, label="Raw Anomaly Score s(t)", color="#adb5bd", alpha=0.7, linestyle=":")
        ax.plot(time_axis, smooth, label="Smoothed Score s_EMA(t)", color="#0066cc", linewidth=2.0)
        ax.axhline(0.60, color="#d9534f", linestyle="--", alpha=0.7, label="Threshold High (0.60)")
        ax.axhline(0.40, color="#f0ad4e", linestyle="--", alpha=0.7, label="Threshold Low (0.40)")

        for e in events:
            ax.axvspan(e.start_sec, e.end_sec, color="#dc3545", alpha=0.25, label=f"Detected Event (R={risk.risk_score:.2f}, {risk.risk_level})")

        ax.set_xlabel("Time (seconds)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Anomaly Score", fontsize=11, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"Anomalous Incident Timeline: {vid_id} ({cat})\nOperational Alert: {risk.risk_level} (R={risk.risk_score:.3f})", fontsize=12, fontweight="bold")
        # Remove duplicate labels in legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper right", framealpha=0.9)
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        sample_plot_path = exp_c_dir / "plots" / "qualitative_anomaly_timeline.png"
        plt.savefig(sample_plot_path)
        plt.close()
        console.print(f"  Saved plot: [green]{sample_plot_path}[/green]")

    # 7. Update Registry
    registry_file = Path("experiments/experiment_registry.csv")
    with open(registry_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "exp_C_persistence",
            "S3D (1024-d)",
            "Dilated Causal TCN (d=[1,2,4])",
            f"EMA + Hysteresis ({overall_fragmentation_red:.1f}% red)",
            "None",
            "0.8750",
            "0.8705",
            "TBD",
            "TBD",
            "44.1",
            "MEASURED",
        ])
        writer.writerow([
            "exp_D_risk",
            "S3D (1024-d)",
            "Dilated Causal TCN (d=[1,2,4])",
            "EMA + Hysteresis",
            f"Deterministic ({risk_tier_counts['High Risk']} High, {risk_tier_counts['Medium Risk']} Med, {risk_tier_counts['Low Risk']} Low)",
            "0.8750",
            "0.8705",
            "TBD",
            "TBD",
            "44.1",
            "MEASURED",
        ])

    # 8. Rich Display Tables
    table_c = Table(title="Experiment C: Persistence Ablation Summary", show_header=True, header_style="bold green")
    table_c.add_column("Metric", style="dim", width=32)
    table_c.add_column("Value", style="bold cyan", width=22)
    table_c.add_row("Raw Alert Blocks (Fragmented)", f"{total_raw_blocks:,}")
    table_c.add_row("Raw Threshold Toggles (Chatter)", f"{total_raw_chatter:,}")
    table_c.add_row("Formed Incident Events", f"{total_formed_events:,}")
    table_c.add_row("Alert Fragmentation Reduction", f"{overall_fragmentation_red:.1f}% [MEASURED]")
    table_c.add_row("Total Formed Event Duration", f"{total_formed_duration:.1f} s")
    console.print(table_c)

    table_d = Table(title="Experiment D: Operational Risk Distribution", show_header=True, header_style="bold magenta")
    table_d.add_column("Operational Tier", style="dim", width=24)
    table_d.add_column("Count (out of 241)", style="bold green", width=20)
    table_d.add_column("Percentage", style="cyan", width=20)
    for tier, count in risk_tier_counts.items():
        pct = count / len(val_dataset) * 100.0
        table_d.add_row(tier, f"{count}", f"{pct:.1f}%")
    console.print(table_d)

    console.print("\n[bold green]EXPERIMENTS C & D COMPLETED SUCCESSFULLY![/bold green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Persistence (Exp C) and Risk (Exp D)")
    parser.add_argument("--checkpoint", type=str, default="experiments/exp_B_tcn/checkpoints/best_auc.pt")
    parser.add_argument("--features_dir", type=str, default="data/features/ucf_crime_s3d")
    parser.add_argument("--val_manifest", type=str, default="data/manifests/ucf_crime_val.csv")
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()
    run_persistence_and_risk_evaluation(
        checkpoint_path=args.checkpoint,
        features_dir=args.features_dir,
        val_manifest=args.val_manifest,
        device_name=args.device,
    )
