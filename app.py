"""Real-Time Video Anomaly Detection & Operational Risk Assessment Application.

Comprehensive Multi-Tab Surveillance and Research System:
    Tab 1: 🎥 Live Anomaly Detection (Raw Video Upload, Sequential Processing,
           Progressive Timeline, Real-Time HUD Overlay, Downloads)
    Tab 2: 🧪 UCF-Crime Benchmark Examples (Instant evaluation across 13 crime classes)
    Tab 3: 📊 Research & Evaluation Dashboard (Ablations, Metrics, Paper Artifacts)
    Tab 4: ⚙️ System Architecture & Scientific Methodology (Honest positioning)

Preserves Frozen Architecture & Research Invariants:
    Frozen S3D Backbone -> Dilated Causal TCN -> Online Causal EMA
    -> Dual-Threshold Hysteresis -> Persistence-Based Event Formation
    -> Transparent Deterministic Risk Decision Layer.
"""

from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Dict, Generator, List, Optional, Tuple, Union

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.inference.annotated_video import create_annotated_video
from src.inference.video_stream import StreamUpdate, VideoStreamInference
from src.models.tcn import TCNModel
from src.postprocessing.event_formation import AnomalyEvent, EventFormationEngine
from src.postprocessing.smoothing import exponential_moving_average
from src.risk.decision_layer import RiskAssessment, RiskDecisionLayer
from src.utils.logger import get_logger

logger = get_logger("app")

# Directories
OUTPUTS_DIR = Path("outputs/demo")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
DEMO_SAMPLES_DIR = Path("data/demo_samples")
FEATURES_DIR = Path("data/features/ucf_crime_s3d")
PAPER_ARTIFACTS_DIR = Path("experiments/paper_artifacts")

# Benchmark video options for Tab 2
BENCHMARK_SAMPLES = {
    "[High Risk] Robbery037 (Armed Robbery)": "Robbery037_x264",
    "[High Risk] RoadAccidents147 (Highway Collision)": "RoadAccidents147_x264",
    "[High Risk] Fighting034 (Violent Physical Altercation)": "Fighting034_x264",
    "[High Risk] Burglary023 (Forced Commercial Entry)": "Burglary023_x264",
    "[High Risk] Arson015 (Structure Fire / Arson)": "Arson015_x264",
    "[High Risk] Stealing084 (Property Theft Incident)": "Stealing084_x264",
    "[High Risk] Abuse015 (Physical Abuse Incident)": "Abuse015_x264",
    "[Medium Risk] Arrest016 (Law Enforcement Action)": "Arrest016_x264",
    "[Medium Risk] Shoplifting030 (Concealed Theft)": "Shoplifting030_x264",
    "[Normal Video] Normal_Videos008 (Calm Street Surveillance)": "Normal_Videos008_x264",
    "[Normal Video] Normal_Videos012 (Quiet Corridor CCTV)": "Normal_Videos012_x264",
    "[Normal Video] Normal_Videos028 (Standard Pedestrian Movement)": "Normal_Videos028_x264",
    "[Normal Video] Normal_Videos035 (Indoor Activity)": "Normal_Videos035_x264",
    "[Normal Video] Normal_Videos089 (Normal Retail Store Footage)": "Normal_Videos089_x264",
}

# Cached Models for Tab 2
_TAB2_MODEL = None
_TAB2_DEVICE = None


def get_tab2_model(device: str = "auto") -> Tuple[torch.nn.Module, str]:
    """Retrieve or load cached TCN model for Tab 2 rapid benchmark evaluation."""
    global _TAB2_MODEL, _TAB2_DEVICE
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    if _TAB2_MODEL is not None and _TAB2_DEVICE == dev:
        return _TAB2_MODEL, dev

    ckpt_path = Path("experiments/exp_B_tcn/checkpoints/best_auc.pt")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    cfg = ckpt.get("config", {})
    model = TCNModel(
        feature_dim=cfg.get("feature_dim", 1024),
        tcn_hidden_dim=cfg.get("tcn_hidden_dim", 128),
        head_hidden_dim=cfg.get("head_hidden_dim", 64),
        dilations=cfg.get("dilations", [1, 2, 4]),
    ).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _TAB2_MODEL = model
    _TAB2_DEVICE = dev
    return model, dev


# ==============================================================================
# TAB 1: Progressive Raw Video Inference Logic
# ==============================================================================

def render_live_plot(
    timestamps: List[float],
    raw_scores: List[float],
    smooth_scores: List[float],
    events: List[AnomalyEvent],
    threshold_high: float = 0.60,
    threshold_low: float = 0.40,
    current_time_sec: float = 0.0,
    total_duration_sec: float = 1.0,
):
    """Draw dynamic timeline plot that grows progressively as video is processed."""
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=180)

    t_arr = np.array(timestamps) if timestamps else np.array([0.0])
    raw_arr = np.array(raw_scores) if raw_scores else np.array([0.0])
    smooth_arr = np.array(smooth_scores) if smooth_scores else np.array([0.0])

    # Plot anomaly curves
    ax.plot(t_arr, raw_arr, label="Raw Snippet Anomaly s(t)", color="#adb5bd", linestyle=":", alpha=0.7, linewidth=1.2)
    ax.plot(t_arr, smooth_arr, label="Smoothed Score s̃(t) (Causal EMA)", color="#1971c2", linewidth=2.2)

    # Threshold horizontal guides
    ax.axhline(threshold_high, color="#e03131", linestyle="--", alpha=0.85, linewidth=1.2, label=f"Event High Trigger ({threshold_high:.2f})")
    ax.axhline(threshold_low, color="#f08c00", linestyle="--", alpha=0.85, linewidth=1.2, label=f"Event Low Sustain ({threshold_low:.2f})")

    # Shade formed incident intervals
    for ev in events:
        ax.axvspan(ev.start_sec, min(ev.end_sec, current_time_sec), color="#e03131", alpha=0.22)
        mid_x = (ev.start_sec + min(ev.end_sec, current_time_sec)) / 2.0
        ax.text(
            mid_x,
            min(1.0, ev.peak_score + 0.05),
            f"Event {ev.event_id}\n({ev.duration_sec:.1f}s)",
            color="#960000",
            fontsize=8.5,
            fontweight="bold",
            ha="center",
        )

    # Cursor line showing current video progress
    if current_time_sec > 0:
        ax.axvline(current_time_sec, color="#2f9e44", linestyle="-", linewidth=1.8, label="Current Video Time")

    ax.set_xlabel("Video Time (seconds)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Continuous Anomaly Score", fontsize=10, fontweight="bold")
    ax.set_ylim(-0.05, 1.1)
    max_x = max(1.0, total_duration_sec, current_time_sec)
    ax.set_xlim(0, max_x)
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(loc="upper right", framealpha=0.92, fontsize=8.5)
    plt.tight_layout()
    return fig


def format_status_card(update: StreamUpdate) -> str:
    """Generate modern, rich HTML operational triage alert card."""
    risk_level = update.current_risk.risk_level if update.current_risk else "Normal"
    risk_score = update.current_risk.risk_score if update.current_risk else 0.0
    explanation = update.current_risk.explanation if update.current_risk else "Processing surveillance stream..."

    styles = {
        "High Risk": ("#e03131", "#fff5f5", "🔴 HIGH RISK INCIDENT"),
        "Medium Risk": ("#f08c00", "#fff9db", "🟠 MEDIUM RISK ANOMALY"),
        "Low Risk": ("#1971c2", "#e7f5ff", "🔵 LOW RISK NOTICE"),
        "Normal": ("#2b8a3e", "#ebfbee", "🟢 NORMAL SURVEILLANCE"),
    }
    color, bg, header_text = styles.get(risk_level, ("#495057", "#f8f9fa", "SURVEILLANCE ACTIVE"))

    return f"""
    <div style='background-color: {bg}; border: 2px solid {color}; border-radius: 12px; padding: 16px; margin-bottom: 8px;'>
        <div style='display: flex; justify-content: space-between; align-items: center;'>
            <div>
                <span style='font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: 1.2px; color: #555;'>Operational Triage Decision</span>
                <h2 style='margin: 3px 0 0 0; color: {color}; font-size: 24px; font-weight: 900;'>{header_text}</h2>
            </div>
            <div style='text-align: right;'>
                <span style='font-size: 11px; color: #666;'>Risk Metric R</span>
                <div style='font-size: 28px; font-weight: 900; color: #111;'>{risk_score:.3f}</div>
            </div>
        </div>
        <p style='margin: 10px 0 6px 0; font-size: 13.5px; line-height: 1.45; color: #333;'>
            <strong>Decision Breakdown:</strong> {explanation}
        </p>
        <div style='display: flex; flex-wrap: wrap; gap: 16px; font-size: 12.5px; color: #555; margin-top: 10px; padding-top: 8px; border-top: 1px solid #ddd;'>
            <span><strong>Current Score:</strong> {update.current_raw_score:.3f}</span>
            <span><strong>Smoothed EMA:</strong> {update.current_smooth_score:.3f}</span>
            <span><strong>Video Time:</strong> {update.video_timestamp_sec:.1f} / {update.total_duration_sec:.1f} s</span>
            <span><strong>Processing Speed:</strong> {update.processing_fps:.1f} FPS</span>
            <span><strong>Model Latency:</strong> {update.model_latency_ms:.1f} ms</span>
            <span><strong>Detected Incidents:</strong> {len(update.active_events)}</span>
        </div>
    </div>
    """


def run_streaming_detection(
    uploaded_video_file: Optional[str],
    preset_choice: str,
    overlay_toggle: bool,
    ema_alpha: float = 0.25,
    threshold_high: float = 0.60,
    threshold_low: float = 0.40,
    min_persistence: float = 1.5,
    max_gap: float = 1.2,
) -> Generator[Tuple, None, None]:
    """Progressive streaming inference pipeline with live UI yield."""
    # 1. Determine video path
    if uploaded_video_file is not None and Path(uploaded_video_file).is_file():
        video_path = Path(uploaded_video_file)
    elif preset_choice != "Custom Upload" and (DEMO_SAMPLES_DIR / f"{preset_choice}.mp4").is_file():
        video_path = DEMO_SAMPLES_DIR / f"{preset_choice}.mp4"
    else:
        # Default fallback to Robbery037 demo video
        video_path = DEMO_SAMPLES_DIR / "Robbery037.mp4"

    if not video_path.is_file():
        err_card = f"<div style='color: red; padding: 12px;'>Error: Video file not found: {video_path}</div>"
        yield (
            err_card,
            None,
            pd.DataFrame(),
            "Error: Video file missing",
            str(video_path),
            None, None, None, None,
        )
        return

    # 2. Instantiate Streaming Engine
    streamer = VideoStreamInference(
        checkpoint_path="experiments/exp_B_tcn/checkpoints/best_auc.pt",
        backbone_type="s3d",
        device="auto",
        snippet_length=16,
        snippet_stride=16,
        ema_alpha=ema_alpha,
        threshold_high=threshold_high,
        threshold_low=threshold_low,
        min_persistence_sec=min_persistence,
        max_gap_sec=max_gap,
    )

    last_update = None
    video_stem = video_path.stem

    for update in streamer.process_video_stream(video_path):
        last_update = update

        status_html = format_status_card(update)
        fig = render_live_plot(
            timestamps=update.timestamps_history,
            raw_scores=update.scores_history,
            smooth_scores=update.smoothed_history,
            events=update.active_events,
            threshold_high=threshold_high,
            threshold_low=threshold_low,
            current_time_sec=update.video_timestamp_sec,
            total_duration_sec=update.total_duration_sec,
        )

        # Form events table
        if update.active_events:
            table_rows = []
            for e in update.active_events:
                ev_risk = streamer.risk_layer.assess_event(e)
                table_rows.append({
                    "Event ID": f"Event {e.event_id}",
                    "Start (s)": f"{e.start_sec:.1f}s",
                    "End (s)": f"{e.end_sec:.1f}s",
                    "Duration": f"{e.duration_sec:.1f}s",
                    "Peak Anomaly": f"{e.peak_score:.3f}",
                    "Mean Anomaly": f"{e.mean_score:.3f}",
                    "Risk Score": f"{ev_risk.risk_score:.3f}",
                    "Operational Tier": ev_risk.risk_level,
                })
            events_df = pd.DataFrame(table_rows)
        else:
            events_df = pd.DataFrame([{
                "Event ID": "-",
                "Start (s)": "-",
                "End (s)": "-",
                "Duration": "0.0s",
                "Peak Anomaly": f"{update.current_smooth_score:.3f}",
                "Mean Anomaly": "-",
                "Risk Score": f"{update.current_risk.risk_score:.3f}" if update.current_risk else "0.000",
                "Operational Tier": "Normal (No Events)",
            }])

        telemetry_text = (
            f"Frame Processing: {update.frames_processed} / {update.total_frames} frames "
            f"({(update.frames_processed / max(1, update.total_frames)) * 100:.1f}%)\n"
            f"Video Time: {update.video_timestamp_sec:.2f} s / {update.total_duration_sec:.2f} s\n"
            f"Processing Throughput: {update.processing_fps:.1f} FPS | Model Latency: {update.model_latency_ms:.2f} ms\n"
            f"Hardware Device: {streamer.device.upper()} (NVIDIA RTX 3050 Laptop GPU)\n"
            f"Architecture: Frozen S3D Backbone (1024-d) + Dilated Causal TCN (Zero Future Leakage)"
        )

        # In live streaming, yield display without blocking for full video write
        yield (
            status_html,
            fig,
            events_df,
            telemetry_text,
            str(video_path),
            None, None, None, None,
        )

    # 3. Post-Processing & Downloadable Artifact Generation (Once inference finishes)
    if last_update is not None:
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = OUTPUTS_DIR / f"{video_stem}_{run_timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # 3A. Save Timeline Plot
        plot_path = run_dir / f"timeline_{video_stem}.png"
        fig.savefig(plot_path, dpi=250)
        plt.close(fig)

        # 3B. Save Anomaly Scores CSV
        scores_df = pd.DataFrame({
            "snippet_idx": np.arange(len(last_update.scores_history)),
            "timestamp_sec": last_update.timestamps_history,
            "raw_anomaly_score": last_update.scores_history,
            "smoothed_anomaly_score": last_update.smoothed_history,
        })
        scores_csv_path = run_dir / f"{video_stem}_scores.csv"
        scores_df.to_csv(scores_csv_path, index=False)

        # 3C. Save Incident Report CSV
        events_csv_path = run_dir / f"{video_stem}_incidents.csv"
        events_df.to_csv(events_csv_path, index=False)

        # 3D. Save Full JSON Summary
        summary_data = {
            "video_identifier": video_stem,
            "total_duration_sec": last_update.total_duration_sec,
            "total_frames": last_update.total_frames,
            "processing_fps": last_update.processing_fps,
            "peak_anomaly_score": max(last_update.smoothed_history) if last_update.smoothed_history else 0.0,
            "overall_risk_level": last_update.current_risk.risk_level if last_update.current_risk else "Normal",
            "overall_risk_score": last_update.current_risk.risk_score if last_update.current_risk else 0.0,
            "risk_explanation": last_update.current_risk.explanation if last_update.current_risk else "",
            "total_events_detected": len(last_update.active_events),
            "events": [
                {
                    "event_id": e.event_id,
                    "start_sec": e.start_sec,
                    "end_sec": e.end_sec,
                    "duration_sec": e.duration_sec,
                    "peak_score": e.peak_score,
                    "mean_score": e.mean_score,
                }
                for e in last_update.active_events
            ],
        }
        json_path = run_dir / f"{video_stem}_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

        # 3E. Render Downloadable Annotated Video if toggle enabled
        annotated_video_path = None
        display_video = str(video_path)
        if overlay_toggle:
            annotated_out_path = run_dir / f"{video_stem}_annotated.mp4"
            create_annotated_video(
                input_video_path=video_path,
                output_video_path=annotated_out_path,
                timestamps=last_update.timestamps_history,
                raw_scores=last_update.scores_history,
                smoothed_scores=last_update.smoothed_history,
                events=last_update.active_events,
                risk_layer=streamer.risk_layer,
            )
            annotated_video_path = str(annotated_out_path)
            display_video = str(annotated_out_path)

        # Final yield with download file paths populated
        yield (
            status_html,
            fig,
            events_df,
            telemetry_text + f"\n\nArtifacts saved to run directory: {run_dir}",
            display_video,
            str(annotated_video_path) if annotated_video_path else None,
            str(scores_csv_path),
            str(events_csv_path),
            str(json_path),
        )


# ==============================================================================
# TAB 2: UCF-Crime Benchmark Samples Logic (Instant Pre-cached Evaluation)
# ==============================================================================

def analyze_benchmark_sample(
    sample_key: str,
    ema_alpha: float = 0.25,
    threshold_high: float = 0.60,
    threshold_low: float = 0.40,
    min_persistence: float = 1.5,
    max_gap: float = 1.2,
):
    """Fast inspection of UCF-Crime benchmark samples from cached features."""
    video_id = BENCHMARK_SAMPLES.get(sample_key, "Robbery037_x264")
    feat_path = FEATURES_DIR / f"{video_id}.pt"

    if not feat_path.is_file():
        return (
            f"<div style='color: red;'>Cached feature file not found: {feat_path}</div>",
            None,
            pd.DataFrame(),
            "Error: Feature tensor missing",
        )

    data = torch.load(feat_path, map_location="cpu", weights_only=False)
    features = data["features"]
    if not isinstance(features, torch.Tensor):
        features = torch.tensor(features, dtype=torch.float32)

    category = data.get("category", "Unknown")
    ground_truth_label = int(data.get("label", -1))
    gt_str = "Anomalous" if ground_truth_label == 1 else "Normal" if ground_truth_label == 0 else "Unknown"

    model, device = get_tab2_model()
    t0 = time.perf_counter()
    with torch.no_grad():
        feat_gpu = features.unsqueeze(0).to(device)
        raw_scores = model(feat_gpu).squeeze(0).cpu().numpy()
    tcn_ms = (time.perf_counter() - t0) * 1000.0

    smooth_scores = exponential_moving_average(raw_scores, alpha=ema_alpha)

    engine = EventFormationEngine(
        threshold_high=threshold_high,
        threshold_low=threshold_low,
        min_persistence_sec=min_persistence,
        max_gap_sec=max_gap,
        snippet_duration_sec=0.533,
    )
    events = engine.form_events(smooth_scores)

    risk_layer = RiskDecisionLayer(
        weight_magnitude=0.65,
        weight_duration=0.35,
        duration_saturation_sec=8.0,
        theta_high=0.70,
        theta_low=0.45,
        theta_base=0.25,
    )
    overall_peak = float(np.max(smooth_scores)) if len(smooth_scores) > 0 else 0.0
    overall_risk = risk_layer.assess_video(events, overall_peak_score=overall_peak)

    # HTML Card
    color_map = {
        "High Risk": ("#e03131", "#fff5f5"),
        "Medium Risk": ("#f08c00", "#fff9db"),
        "Low Risk": ("#1971c2", "#e7f5ff"),
        "Normal": ("#2b8a3e", "#ebfbee"),
    }
    badge_color, bg_color = color_map.get(overall_risk.risk_level, ("#6c757d", "#f8f9fa"))

    status_html = f"""
    <div style='background-color: {bg_color}; border: 2px solid {badge_color}; border-radius: 12px; padding: 16px; margin-bottom: 8px;'>
        <div style='display: flex; justify-content: space-between; align-items: center;'>
            <div>
                <span style='font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: 1.2px; color: #555;'>Operational Triage Decision</span>
                <h2 style='margin: 3px 0 0 0; color: {badge_color}; font-size: 24px; font-weight: 900;'>{overall_risk.risk_level}</h2>
            </div>
            <div style='text-align: right;'>
                <span style='font-size: 11px; color: #666;'>Risk Metric R</span>
                <div style='font-size: 28px; font-weight: 900; color: #111;'>{overall_risk.risk_score:.3f}</div>
            </div>
        </div>
        <p style='margin: 10px 0 6px 0; font-size: 13.5px; line-height: 1.45; color: #333;'>
            <strong>Explainability Breakdown:</strong> {overall_risk.explanation}
        </p>
        <div style='display: flex; flex-wrap: wrap; gap: 16px; font-size: 12.5px; color: #555; margin-top: 10px; padding-top: 8px; border-top: 1px solid #ddd;'>
            <span><strong>Video:</strong> {video_id}</span>
            <span><strong>Category:</strong> {category}</span>
            <span><strong>Ground Truth:</strong> {gt_str}</span>
            <span><strong>Formed Events:</strong> {len(events)}</span>
            <span><strong>TCN Latency:</strong> {tcn_ms:.2f} ms</span>
        </div>
    </div>
    """

    # Timeline Plot
    timestamps = np.arange(len(raw_scores)) * 0.533
    fig = render_live_plot(
        timestamps=list(timestamps),
        raw_scores=list(raw_scores),
        smooth_scores=list(smooth_scores),
        events=events,
        threshold_high=threshold_high,
        threshold_low=threshold_low,
        current_time_sec=float(timestamps[-1]) if len(timestamps) > 0 else 1.0,
        total_duration_sec=float(timestamps[-1]) if len(timestamps) > 0 else 1.0,
    )

    # Table
    if events:
        table_rows = []
        for e in events:
            ev_risk = risk_layer.assess_event(e)
            table_rows.append({
                "Event ID": f"Event {e.event_id}",
                "Start (s)": f"{e.start_sec:.1f}s",
                "End (s)": f"{e.end_sec:.1f}s",
                "Duration": f"{e.duration_sec:.1f}s",
                "Peak Anomaly": f"{e.peak_score:.3f}",
                "Mean Anomaly": f"{e.mean_score:.3f}",
                "Risk Score": f"{ev_risk.risk_score:.3f}",
                "Operational Tier": ev_risk.risk_level,
            })
        events_df = pd.DataFrame(table_rows)
    else:
        events_df = pd.DataFrame([{
            "Event ID": "-",
            "Start (s)": "-",
            "End (s)": "-",
            "Duration": "0.0s",
            "Peak Anomaly": f"{overall_peak:.3f}",
            "Mean Anomaly": "-",
            "Risk Score": f"{overall_risk.risk_score:.3f}",
            "Operational Tier": "Normal (No Events)",
        }])

    diagnostics = (
        f"Sample: {video_id} | Category: {category} | Ground Truth: {gt_str}\n"
        f"Sequence Length: {len(raw_scores)} snippets ({len(raw_scores)*0.533:.1f} s video duration)\n"
        f"Inference Latency: {tcn_ms:.2f} ms | Evaluated on: {device.upper()}"
    )

    return status_html, fig, events_df, diagnostics


# ==============================================================================
# MAIN APPLICATION INTERFACE
# ==============================================================================

def create_full_application() -> gr.Blocks:
    """Constructs the comprehensive 4-tab Gradio Application."""
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
    )

    with gr.Blocks(title="Real-Time Video Anomaly Detection & Risk Assessment") as app:
        gr.Markdown(
            """
            # 🚨 Real-Time Video Anomaly Detection & Operational Risk Assessment
            ### Undergraduate Research Capstone System | Frozen S3D Backbone + Dilated Causal TCN + Temporal Persistence
            *Bridging the gap from continuous snippet-level anomaly scores to actionable, incident-level surveillance alerts.*
            """
        )

        with gr.Tabs() as tabs:

            # ------------------------------------------------------------------
            # TAB 1: Live Anomaly Detection (Primary Workflow)
            # ------------------------------------------------------------------
            with gr.Tab("🎥 Live Detection (Raw Video)", id="tab_live"):
                gr.Markdown(
                    """
                    Upload an arbitrary surveillance video (`.mp4`, `.avi`, `.mov`) or select a preloaded sample.
                    The system decodes frames sequentially, feeds 16-frame clips into the frozen S3D backbone,
                    evaluates the causal TCN, smooths scores with online EMA, forms persistent events, and assesses operational risk.
                    """
                )

                with gr.Row():
                    with gr.Column(scale=4):
                        video_input = gr.Video(
                            label="Upload Surveillance Video",
                            sources=["upload"],
                        )
                        preset_dropdown = gr.Dropdown(
                            label="Or Select a Demo Preset",
                            choices=["Custom Upload", "Robbery037", "Normal_Videos008", "Arson015"],
                            value="Robbery037",
                        )
                        overlay_toggle = gr.Checkbox(
                            label="Show Detection Overlay (Live HUD Banner)",
                            value=True,
                            info="When enabled, generates video with operational anomaly/risk HUD overlays.",
                        )
                        start_btn = gr.Button("▶ START ANOMALY DETECTION", variant="primary", size="lg")

                        with gr.Accordion("⚙️ Advanced Decision Parameters", open=False):
                            ema_slider = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Causal EMA Smoothing Factor (alpha)")
                            high_slider = gr.Slider(0.40, 0.90, value=0.60, step=0.05, label="Event Trigger High Threshold (T_high)")
                            low_slider = gr.Slider(0.10, 0.60, value=0.40, step=0.05, label="Event Sustain Low Threshold (T_low)")
                            min_dur_slider = gr.Slider(0.5, 5.0, value=1.5, step=0.5, label="Minimum Persistence Duration (sec)")
                            max_gap_slider = gr.Slider(0.5, 4.0, value=1.2, step=0.2, label="Gap Bridging Threshold (sec)")

                    with gr.Column(scale=6):
                        status_card = gr.HTML(
                            value="<div style='padding: 16px; border: 2px dashed #bbb; border-radius: 12px; text-align: center; color: #777;'>Upload a video and click 'START ANOMALY DETECTION' to begin live triage.</div>"
                        )
                        video_output_display = gr.Video(
                            label="Surveillance Playback",
                            interactive=False,
                        )

                with gr.Row():
                    timeline_plot = gr.Plot(label="Progressive Anomaly Score Timeline s(t)")

                with gr.Row():
                    events_table = gr.Dataframe(
                        label="Detected Persistent Incidents (Operational Events)",
                        interactive=False,
                    )

                with gr.Row():
                    telemetry_box = gr.Textbox(
                        label="System Telemetry & Hardware Profiling",
                        interactive=False,
                        lines=4,
                    )

                with gr.Row():
                    gr.Markdown("### 📥 Download Incident Artifacts")
                with gr.Row():
                    annotated_video_download = gr.File(label="Annotated Video (.mp4)")
                    scores_csv_download = gr.File(label="Continuous Scores (.csv)")
                    events_csv_download = gr.File(label="Incident Report (.csv)")
                    json_report_download = gr.File(label="Telemetry Summary (.json)")

                # Callback wiring for Tab 1
                start_btn.click(
                    fn=run_streaming_detection,
                    inputs=[
                        video_input,
                        preset_dropdown,
                        overlay_toggle,
                        ema_slider,
                        high_slider,
                        low_slider,
                        min_dur_slider,
                        max_gap_slider,
                    ],
                    outputs=[
                        status_card,
                        timeline_plot,
                        events_table,
                        telemetry_box,
                        video_output_display,
                        annotated_video_download,
                        scores_csv_download,
                        events_csv_download,
                        json_report_download,
                    ],
                )

            # ------------------------------------------------------------------
            # TAB 2: UCF-Crime Benchmark Examples (Instant Pre-cached Evaluation)
            # ------------------------------------------------------------------
            with gr.Tab("🧪 UCF-Crime Benchmark Examples", id="tab_benchmark"):
                gr.Markdown(
                    """
                    Inspect pre-cached benchmark samples across all 13 UCF-Crime anomaly categories plus normal videos.
                    This runs inference in under 50 ms directly on extracted features, ideal for quick viva demonstrations.
                    """
                )

                with gr.Row():
                    with gr.Column(scale=4):
                        benchmark_dropdown = gr.Dropdown(
                            label="Select Benchmark UCF-Crime Video",
                            choices=list(BENCHMARK_SAMPLES.keys()),
                            value=list(BENCHMARK_SAMPLES.keys())[0],
                        )
                        t2_analyze_btn = gr.Button("🔍 Run Benchmark Evaluation", variant="primary")

                    with gr.Column(scale=6):
                        t2_status_card = gr.HTML()

                with gr.Row():
                    t2_timeline_plot = gr.Plot(label="Benchmark Temporal Anomaly Timeline")

                with gr.Row():
                    t2_events_table = gr.Dataframe(label="Detected Incidents", interactive=False)

                with gr.Row():
                    t2_diag_box = gr.Textbox(label="Sample Diagnostics", interactive=False)

                t2_analyze_btn.click(
                    fn=analyze_benchmark_sample,
                    inputs=[benchmark_dropdown],
                    outputs=[t2_status_card, t2_timeline_plot, t2_events_table, t2_diag_box],
                )
                benchmark_dropdown.change(
                    fn=analyze_benchmark_sample,
                    inputs=[benchmark_dropdown],
                    outputs=[t2_status_card, t2_timeline_plot, t2_events_table, t2_diag_box],
                )

            # ------------------------------------------------------------------
            # TAB 3: Research & Evaluation Dashboard
            # ------------------------------------------------------------------
            with gr.Tab("📊 Research & Evaluation Dashboard", id="tab_research"):
                gr.Markdown(
                    """
                    ### Controlled Empirical Research Ablation Suite
                    All metrics below were **empirically measured on our local NVIDIA RTX 3050 Laptop GPU**
                    following the official UCF-Crime validation protocol.
                    """
                )

                with gr.Row():
                    gr.Markdown(
                        """
                        | Experiment | Architecture / Stage | Val ROC-AUC | Val PR-AUC | Alert Fragmentation Reduction | Snippet Latency | Effective Video FPS | Peak VRAM | Status |
                        | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
                        | **Exp A** | S3D (1024-d) + MIL Head | 0.8253 | 0.8236 | — | 0.011 ms | 93,023* | 43.8 MB | **MEASURED** |
                        | **Exp B** | S3D + Dilated Causal TCN + MIL | **0.8750** | **0.8705** | — | 0.092 ms | 10,914* | 44.1 MB | **MEASURED** |
                        | **Exp C** | S3D + TCN + Persistence | 0.8750 | 0.8705 | **-43.5% (74 consolidated events)** | 0.092 ms | 10,914* | 44.1 MB | **MEASURED** |
                        | **Exp D** | (C) + Deterministic Risk Layer | 0.8750 | 0.8705 | 53 High / 14 Med / 174 Normal | 0.093 ms | 10,800* | 44.1 MB | **MEASURED** |
                        | **Exp E** | **Full Decoupled Pipeline (E2E)** | **0.8750** | **0.8705** | **-43.5% Fragmentation** | **20.23 ms** | **791.1 FPS** | **1.08 GB** | **MEASURED** |

                        *\*Downstream scoring latency excluding 3D video decoding. E2E includes full S3D feature extraction.*
                        """
                    )

                with gr.Row():
                    with gr.Column(scale=6):
                        roc_pr_img = PAPER_ARTIFACTS_DIR / "plots/figure2_roc_pr_comparison.png"
                        if roc_pr_img.is_file():
                            gr.Image(value=str(roc_pr_img), label="Figure 2: ROC and Precision-Recall Curves (Exp A vs Exp B)")
                    with gr.Column(scale=6):
                        curves_img = PAPER_ARTIFACTS_DIR / "plots/figure1_training_curves_comparison.png"
                        if curves_img.is_file():
                            gr.Image(value=str(curves_img), label="Figure 1: Training & Validation Loss Progression")

                with gr.Row():
                    with gr.Column(scale=6):
                        persist_img = PAPER_ARTIFACTS_DIR / "plots/persistence_ablation.png"
                        if persist_img.is_file():
                            gr.Image(value=str(persist_img), label="Persistence Ablation: Raw Thresholds vs Hysteresis Events")
                    with gr.Column(scale=6):
                        risk_img = PAPER_ARTIFACTS_DIR / "plots/risk_distribution.png"
                        if risk_img.is_file():
                            gr.Image(value=str(risk_img), label="Validation Operational Risk Distribution")

            # ------------------------------------------------------------------
            # TAB 4: System Architecture & Scientific Methodology
            # ------------------------------------------------------------------
            with gr.Tab("⚙️ System Architecture & About", id="tab_about"):
                gr.Markdown(
                    """
                    ### Scientific Problem Statement & Three-Task Distinction
                    Traditional video anomaly detection systems produce continuous anomaly scores $s_t \\in [0, 1]$,
                    leaving security operators to manually interpret oscillating values. This research investigates whether
                    weakly supervised spatiotemporal representations can be augmented with **causal temporal modeling**,
                    **temporal persistence**, and a **transparent risk-oriented decision layer** to produce stable incident-level alerts:

                    ```text
                    [ Task 1: Anomaly Detection ]
                    Is the 16-frame snippet anomalous?
                    Output: Continuous anomaly score s_t in [0.0, 1.0]
                            ↓
                    [ Task 2: Incident Formation ]
                    Does the anomaly persist across time to constitute a meaningful event?
                    Output: Formed temporal event [t_start, t_end] with duration Δt
                            ↓
                    [ Task 3: Operational Risk Prioritization ]
                    How should the detected event be prioritized for security response?
                    Output: Continuous risk metric R and operational alert (Normal / Low / Medium / High Risk)
                    ```

                    ### Core Research Laws & Honest Academic Positioning
                    - **No False Novelty:** We did NOT invent Video Anomaly Detection (VAD), Multiple Instance Learning (MIL), I3D, or TCNs.
                    - **No Fake Ground Truth:** Ground-truth risk/severity labels DO NOT exist in UCF-Crime or XD-Violence. The risk layer is a **derived, heuristic, transparent decision support layer**, NOT a supervised classifier.
                    - **No Spatial Bounding Boxes:** Our system is a temporal sequence detector. It does not perform spatial object localization. Overlays represent frame-level triage alerts.
                    - **Zero Future Leakage:** The Causal TCN uses explicit left-padding $(K-1) \\cdot d$ and `ChannelLayerNorm`, mathematically guaranteeing that future video snippets do not contaminate past predictions.
                    - **Hardware Budget Compliance:** Entire pipeline executed on a local 4 GB VRAM budget (NVIDIA GeForce RTX 3050 Laptop GPU), peaking at 1.08 GB VRAM.
                    """
                )

    return app


if __name__ == "__main__":
    demo = create_full_application()
    theme = gr.themes.Soft(primary_hue="blue", secondary_hue="slate")
    demo.launch(theme=theme, server_name="127.0.0.1", server_port=7860, share=False)
