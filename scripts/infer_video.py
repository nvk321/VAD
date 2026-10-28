"""Inference and Incident Timeline Demonstration for Video Anomaly Detection.

Demonstrates the complete end-to-end inference stack:
1. Model Inference (Dilated Causal TCN or Baseline MIL Head)
2. Temporal Smoothing (EMA)
3. Event Formation (Dual-Threshold Hysteresis + Persistence Filter)
4. Transparent Risk Assessment (Low / Medium / High Operational Triage)
5. Publication-Ready Timeline Visualization & Rich Incident Report
"""

import argparse
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

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

from src.models.mil_head import MILBaselineModel
from src.models.tcn import TCNModel
from src.postprocessing.event_formation import EventFormationEngine
from src.postprocessing.smoothing import exponential_moving_average
from src.risk.decision_layer import RiskDecisionLayer


console = Console()


def plot_anomaly_timeline(
    video_id: str,
    snippet_scores: np.ndarray,
    smoothed_scores: np.ndarray,
    events: List[Any],
    threshold_high: float = 0.60,
    threshold_low: float = 0.40,
    output_path: Optional[Path] = None,
) -> Path:
    """Generate publication-ready anomaly score timeline visualization."""
    T = len(snippet_scores)
    time_sec = np.arange(T) * 0.533  # ~0.533s per 16-frame snippet at 30 fps (10-frame sample step)

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    ax.plot(time_sec, snippet_scores, label="Raw Snippet Anomaly Score $s_t$", color="#adb5bd", linestyle=":", alpha=0.7, linewidth=1.2)
    ax.plot(time_sec, smoothed_scores, label="Smoothed Score $\\tilde{s}_t$ (EMA $\\alpha=0.25$)", color="#1f77b4", linewidth=2.2)

    ax.axhline(y=threshold_high, color="#d62728", linestyle="--", alpha=0.8, linewidth=1.2, label=f"Event High Trigger $T_{{high}}={threshold_high:.2f}$")
    ax.axhline(y=threshold_low, color="#ff7f0e", linestyle="--", alpha=0.8, linewidth=1.2, label=f"Event Low Sustain $T_{{low}}={threshold_low:.2f}$")

    # Shade formed event intervals
    for event in events:
        ax.axvspan(event.start_sec, event.end_sec, color="#d62728", alpha=0.25)
        ax.text(
            (event.start_sec + event.end_sec) / 2.0,
            min(1.0, event.peak_score + 0.05),
            f"Event {event.event_id}\n({event.duration_sec:.1f}s)",
            color="#800000",
            fontsize=8,
            fontweight="bold",
            ha="center",
        )

    ax.set_title(f"Video Anomaly Score Timeline & Formed Incidents — {video_id}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (seconds)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Continuous Anomaly Score $s_t \\in [0, 1]$", fontsize=10, fontweight="bold")
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlim(0, max(1.0, time_sec[-1]))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8.5)

    plt.tight_layout()

    if output_path is None:
        out_dir = Path("outputs/demo")
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"timeline_{video_id}.png"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path)
    plt.close()
    return output_path


def load_model_from_checkpoint(checkpoint_path: Path, device: str = "cpu") -> Tuple[torch.nn.Module, str]:
    """Load either a TCNModel or MILBaselineModel from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    state_dict = ckpt["model_state_dict"]

    # Detect if TCN or MIL
    is_tcn = any("tcn" in k for k in state_dict.keys())

    if is_tcn:
        model = TCNModel(
            feature_dim=config.get("feature_dim", 1024),
            tcn_hidden_dim=config.get("tcn_hidden_dim", 128),
            head_hidden_dim=config.get("head_hidden_dim", 64),
            dilations=config.get("dilations", [1, 2, 4]),
        )
        model_name = "S3D + Dilated Causal TCN"
    else:
        model = MILBaselineModel(
            feature_dim=config.get("feature_dim", 1024),
            hidden_dim=config.get("head_hidden_dim", 512),
        )
        model_name = "S3D + Baseline MIL Head"

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, model_name


def run_inference(
    checkpoint_path: str,
    feature_file: str,
    threshold_high: float = 0.60,
    threshold_low: float = 0.40,
    min_persistence_sec: float = 1.5,
    max_gap_sec: float = 1.2,
    device: str = "cpu",
    output_dir: str = "outputs/demo",
) -> Dict[str, Any]:
    """Run full inference, persistence, and risk assessment pipeline."""
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    feat_path = Path(feature_file)
    if not feat_path.is_file():
        raise FileNotFoundError(f"Feature file not found: {feat_path}")

    # 1. Load model and feature tensor
    model, model_name = load_model_from_checkpoint(ckpt_path, device=device)

    feat_data = torch.load(feat_path, map_location="cpu", weights_only=False)
    features = feat_data["features"]
    if not isinstance(features, torch.Tensor):
        features = torch.tensor(features, dtype=torch.float32)

    video_id = feat_data.get("video_id", feat_path.stem)
    category = feat_data.get("category", "Unknown")
    ground_truth_label = int(feat_data.get("label", -1))
    gt_str = "Anomalous (1)" if ground_truth_label == 1 else "Normal (0)" if ground_truth_label == 0 else "Unknown"

    # 2. Run inference
    start_time = time.perf_counter()
    with torch.no_grad():
        if isinstance(model, TCNModel):
            inp = features.unsqueeze(0).to(device)  # (1, T, D)
            scores = model(inp).squeeze(0).cpu().numpy()
        else:
            scores = model(features.to(device)).cpu().numpy()
    inference_time_ms = (time.perf_counter() - start_time) * 1000

    # 3. Postprocessing: Temporal Smoothing
    scores_smooth = exponential_moving_average(scores, alpha=0.25)

    # 4. Postprocessing: Event Formation
    engine = EventFormationEngine(
        threshold_high=threshold_high,
        threshold_low=threshold_low,
        min_persistence_sec=min_persistence_sec,
        max_gap_sec=max_gap_sec,
        snippet_duration_sec=0.533,
    )
    events = engine.form_events(scores_smooth)

    # 5. Risk Assessment
    peak_score = float(np.max(scores_smooth)) if len(scores_smooth) > 0 else 0.0
    risk_layer = RiskDecisionLayer(
        weight_magnitude=0.65,
        weight_duration=0.35,
        duration_saturation_sec=8.0,
        theta_high=0.70,
        theta_low=0.45,
        theta_base=0.25,
    )
    video_risk = risk_layer.assess_video(events, overall_peak_score=peak_score)

    # 6. Save Timeline Visualization
    plot_path = plot_anomaly_timeline(
        video_id=video_id,
        snippet_scores=scores,
        smoothed_scores=scores_smooth,
        events=events,
        threshold_high=threshold_high,
        threshold_low=threshold_low,
        output_path=Path(output_dir) / f"timeline_{video_id}.png",
    )

    # 7. Print Rich Incident Analysis Table
    table = Table(title=f"Incident Analysis Report: {video_id}", show_header=True, header_style="bold cyan")
    table.add_column("Property", style="dim", width=24)
    table.add_column("Value", style="bold green", width=36)

    table.add_row("Video Identifier", video_id)
    table.add_row("Category", category)
    table.add_row("Ground-Truth Label", gt_str)
    table.add_row("Model Architecture", model_name)
    table.add_row("Total Snippets", f"{len(scores)} (~{len(scores)*0.533:.1f} s video)")
    table.add_row("Peak Anomaly Score", f"{peak_score:.4f}")
    table.add_row("Formed Events", f"{len(events)} incident(s)")
    table.add_row("Operational Risk Tier", f"[{video_risk.risk_level.upper()}] (Score: {video_risk.risk_score:.3f})")
    table.add_row("Risk Explainability", video_risk.explanation)
    table.add_row("Inference Latency", f"{inference_time_ms:.2f} ms")
    table.add_row("Timeline Plot Saved", str(plot_path))

    console.print(table)

    return {
        "video_id": video_id,
        "category": category,
        "ground_truth_label": ground_truth_label,
        "model_name": model_name,
        "peak_score": peak_score,
        "events_count": len(events),
        "risk_level": video_risk.risk_level,
        "risk_score": video_risk.risk_score,
        "explanation": video_risk.explanation,
        "latency_ms": inference_time_ms,
        "plot_path": str(plot_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Video Anomaly Detection Inference & Risk Triage")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint (.pt)")
    parser.add_argument("--feature_file", type=str, required=True, help="Path to cached feature (.pt)")
    parser.add_argument("--threshold_high", type=float, default=0.60, help="Event trigger threshold")
    parser.add_argument("--threshold_low", type=float, default=0.40, help="Event sustain threshold")
    parser.add_argument("--min_persistence", type=float, default=1.5, help="Minimum persistence duration (sec)")
    parser.add_argument("--max_gap", type=float, default=1.2, help="Bridging gap threshold (sec)")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--output_dir", type=str, default="outputs/demo", help="Output directory for plots")

    args = parser.parse_args()

    run_inference(
        checkpoint_path=args.checkpoint,
        feature_file=args.feature_file,
        threshold_high=args.threshold_high,
        threshold_low=args.threshold_low,
        min_persistence_sec=args.min_persistence,
        max_gap_sec=args.max_gap,
        device=args.device,
        output_dir=args.output_dir,
    )
