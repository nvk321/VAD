"""Interactive Demonstration Interface for Video Anomaly Detection & Risk Assessment.

Built with Gradio. Demonstrates:
1. Continuous Anomaly Score Timeline: s(t) across temporal snippets.
2. Temporal Smoothing & Event Formation: Suppresses noise and extracts stable incident intervals.
3. Transparent Deterministic Risk Decision Layer: Classifies events into Normal / Low / Medium / High Risk.
4. Explainable Decision Support: Full operational breakdown of risk drivers.

Uses trained S3D + Dilated Causal TCN model checkpoint (zero future leakage).
"""

from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.models.tcn import TCNModel
from src.postprocessing.event_formation import AnomalyEvent, EventFormationEngine
from src.postprocessing.smoothing import exponential_moving_average
from src.risk.decision_layer import RiskAssessment, RiskDecisionLayer


# 1. Available Pre-cached Demonstration Samples
SAMPLE_VIDEOS = {
    "[High Risk Anomaly] Robbery037 (Armed Robbery)": "Robbery037_x264",
    "[High Risk Anomaly] RoadAccidents147 (Highway Collision)": "RoadAccidents147_x264",
    "[High Risk Anomaly] Fighting034 (Violent Physical Altercation)": "Fighting034_x264",
    "[High Risk Anomaly] Burglary023 (Forced Commercial Entry)": "Burglary023_x264",
    "[High Risk Anomaly] Arson015 (Structure Fire / Arson)": "Arson015_x264",
    "[High Risk Anomaly] Stealing084 (Property Theft Incident)": "Stealing084_x264",
    "[High Risk Anomaly] Abuse015 (Physical Abuse Incident)": "Abuse015_x264",
    "[Medium Risk Anomaly] Arrest016 (Law Enforcement Action)": "Arrest016_x264",
    "[Medium Risk Anomaly] Shoplifting030 (Concealed Theft)": "Shoplifting030_x264",
    "[Normal Video] Normal_Videos008 (Calm Street Surveillance)": "Normal_Videos008_x264",
    "[Normal Video] Normal_Videos012 (Quiet Corridor CCTV)": "Normal_Videos012_x264",
    "[Normal Video] Normal_Videos028 (Standard Pedestrian Movement)": "Normal_Videos028_x264",
    "[Normal Video] Normal_Videos035 (Indoor Activity)": "Normal_Videos035_x264",
    "[Normal Video] Normal_Videos089 (Normal Retail Store Footage)": "Normal_Videos089_x264",
}

DEFAULT_CHECKPOINT = "experiments/exp_B_tcn/checkpoints/best_auc.pt"
FEATURES_DIR = Path("data/features/ucf_crime_s3d")


# 2. Global Model Loader (Cached)
_LOADED_MODEL = None
_LOADED_DEVICE = None


def get_model(device: str = "auto") -> Tuple[torch.nn.Module, str]:
    """Load or retrieve the cached TCN model."""
    global _LOADED_MODEL, _LOADED_DEVICE

    if device == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        dev = device

    if _LOADED_MODEL is not None and _LOADED_DEVICE == dev:
        return _LOADED_MODEL, dev

    ckpt_path = Path(DEFAULT_CHECKPOINT)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Trained checkpoint not found at: {ckpt_path}. Please train Exp B first.")

    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    config = ckpt.get("config", {})

    model = TCNModel(
        feature_dim=config.get("feature_dim", 1024),
        tcn_hidden_dim=config.get("tcn_hidden_dim", 128),
        head_hidden_dim=config.get("head_hidden_dim", 64),
        dilations=config.get("dilations", [1, 2, 4]),
    ).to(dev)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _LOADED_MODEL = model
    _LOADED_DEVICE = dev
    return model, dev


def analyze_video(
    selected_sample: str,
    custom_feature_file: Optional[str] = None,
    ema_alpha: float = 0.25,
    threshold_high: float = 0.60,
    threshold_low: float = 0.40,
    min_persistence_sec: float = 1.5,
    max_gap_sec: float = 1.2,
    weight_magnitude: float = 0.65,
    weight_duration: float = 0.35,
):
    """Main inference and analysis pipeline for the web demonstration interface."""
    start_time = time.perf_counter()

    # Determine feature file path
    if custom_feature_file is not None and Path(custom_feature_file).is_file():
        feat_path = Path(custom_feature_file)
        video_id = feat_path.stem
    else:
        video_id = SAMPLE_VIDEOS.get(selected_sample, "Robbery037_x264")
        feat_path = FEATURES_DIR / f"{video_id}.pt"

    if not feat_path.is_file():
        return (
            f"<div style='color: red;'>Feature file not found on disk: {feat_path}</div>",
            None,
            pd.DataFrame(),
            "Error: Feature file missing",
        )

    # 1. Load spatiotemporal features
    data = torch.load(feat_path, map_location="cpu", weights_only=False)
    features = data["features"]  # (T, 1024)
    if not isinstance(features, torch.Tensor):
        features = torch.tensor(features, dtype=torch.float32)

    category = data.get("category", "Unknown")
    ground_truth_label = int(data.get("label", -1))
    gt_str = "Anomalous" if ground_truth_label == 1 else "Normal" if ground_truth_label == 0 else "Unknown"

    # 2. Run Causal TCN Inference
    model, device = get_model()
    tcn_start = time.perf_counter()
    with torch.no_grad():
        feat_gpu = features.unsqueeze(0).to(device)  # (1, T, 1024)
        scores_raw = model(feat_gpu).squeeze(0).cpu().numpy()
    tcn_latency_ms = (time.perf_counter() - tcn_start) * 1000.0

    # 3. Apply Temporal Smoothing
    scores_smooth = exponential_moving_average(scores_raw, alpha=ema_alpha)

    # 4. Form Events with Dual-Threshold Hysteresis and Persistence
    engine = EventFormationEngine(
        threshold_high=threshold_high,
        threshold_low=threshold_low,
        min_persistence_sec=min_persistence_sec,
        max_gap_sec=max_gap_sec,
        snippet_duration_sec=0.533,
    )
    events = engine.form_events(scores_smooth)

    # 5. Assess Operational Risk
    overall_peak = float(np.max(scores_smooth)) if len(scores_smooth) > 0 else 0.0
    risk_layer = RiskDecisionLayer(
        weight_magnitude=weight_magnitude,
        weight_duration=weight_duration,
        duration_saturation_sec=8.0,
        theta_high=0.70,
        theta_low=0.45,
        theta_base=0.25,
    )
    overall_risk = risk_layer.assess_video(events, overall_peak_score=overall_peak)

    total_latency_ms = (time.perf_counter() - start_time) * 1000.0

    # 6. Format HTML Status Card
    color_map = {
        "High Risk": ("#d9534f", "#fdf2f2"),
        "Medium Risk": ("#f0ad4e", "#fdfcf0"),
        "Low Risk": ("#0275d8", "#f0f7fd"),
        "Normal": ("#2b8a3e", "#f0fdf4"),
    }
    badge_color, bg_color = color_map.get(overall_risk.risk_level, ("#6c757d", "#f8f9fa"))

    status_html = f"""
    <div style='background-color: {bg_color}; border: 2px solid {badge_color}; border-radius: 10px; padding: 18px; margin-bottom: 12px;'>
        <div style='display: flex; justify-content: space-between; align-items: center;'>
            <div>
                <span style='font-size: 13px; font-weight: bold; text-transform: uppercase; letter-spacing: 1px; color: #555;'>Operational Triage Decision</span>
                <h2 style='margin: 4px 0; color: {badge_color}; font-size: 26px; font-weight: 800;'>{overall_risk.risk_level}</h2>
            </div>
            <div style='text-align: right;'>
                <span style='font-size: 12px; color: #777;'>Continuous Risk Score</span>
                <div style='font-size: 28px; font-weight: 900; color: #222;'>{overall_risk.risk_score:.3f}</div>
            </div>
        </div>
        <p style='margin: 10px 0 6px 0; font-size: 14px; line-height: 1.5; color: #333;'>
            <strong>Explainability Breakdown:</strong> {overall_risk.explanation}
        </p>
        <div style='display: flex; gap: 20px; font-size: 13px; color: #666; margin-top: 10px; padding-top: 8px; border-top: 1px solid #ddd;'>
            <span><strong>Video:</strong> {video_id}</span>
            <span><strong>Category:</strong> {category}</span>
            <span><strong>Ground Truth:</strong> {gt_str}</span>
            <span><strong>Formed Events:</strong> {len(events)}</span>
            <span><strong>TCN Latency:</strong> {tcn_latency_ms:.2f} ms</span>
        </div>
    </div>
    """

    # 7. Render High-Resolution Plot
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=200)
    time_axis = np.arange(len(scores_raw)) * 0.533

    ax.plot(time_axis, scores_raw, label="Raw Snippet Score s(t)", color="#adb5bd", linestyle=":", alpha=0.6, linewidth=1.2)
    ax.plot(time_axis, scores_smooth, label=f"Smoothed s(t) (EMA alpha={ema_alpha})", color="#0066cc", linewidth=2.2)

    ax.axhline(threshold_high, color="#d9534f", linestyle="--", alpha=0.8, linewidth=1.2, label=f"Threshold High ({threshold_high:.2f})")
    ax.axhline(threshold_low, color="#f0ad4e", linestyle="--", alpha=0.8, linewidth=1.2, label=f"Threshold Low ({threshold_low:.2f})")

    # Shade formed event intervals
    for idx, e in enumerate(events):
        event_risk = risk_layer.assess_event(e)
        ev_color = badge_color
        ax.axvspan(e.start_sec, e.end_sec, color=ev_color, alpha=0.25)
        ax.text(
            (e.start_sec + e.end_sec) / 2.0,
            min(1.0, e.peak_score + 0.05),
            f"Event {e.event_id}\n({event_risk.risk_level})",
            color="#800000",
            fontsize=9,
            fontweight="bold",
            ha="center",
        )

    ax.set_xlabel("Time (seconds)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Anomaly Score", fontsize=11, fontweight="bold")
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlim(0, max(1.0, time_axis[-1]))
    ax.set_title(f"Temporal Anomaly Timeline & Incident Detection — {video_id} ({category})", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    plt.tight_layout()

    # 8. Events Table DataFrame
    if events:
        table_rows = []
        for e in events:
            ev_risk = risk_layer.assess_event(e)
            table_rows.append({
                "Event ID": e.event_id,
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
        f"Sequence Length: {len(scores_raw)} snippets ({len(scores_raw)*0.533:.1f} s video duration)\n"
        f"Hardware: {device.upper()} | Model: S3D + Dilated Causal TCN\n"
        f"Inference Latency: {tcn_latency_ms:.2f} ms | Pipeline E2E: {total_latency_ms:.1f} ms"
    )

    return status_html, fig, events_df, diagnostics


def create_demo_app():
    """Build the Gradio interface."""
    with gr.Blocks(title="Video Anomaly Detection & Risk Assessment") as demo:
        gr.Markdown(
            """
            # 🚨 Real-Time Video Anomaly Detection & Operational Risk Assessment
            ### Research Demonstration System | Frozen S3D Backbone + Dilated Causal TCN + Multiple Instance Learning
            **Mission:** Bridge the gap between raw temporal anomaly scores and actionable incident-level alerts
            through **causal temporal modeling**, **persistence filtering**, and a **transparent risk decision layer**.
            """
        )

        with gr.Row():
            with gr.Column(scale=4):
                sample_dropdown = gr.Dropdown(
                    label="Select Benchmark UCF-Crime Video",
                    choices=list(SAMPLE_VIDEOS.keys()),
                    value=list(SAMPLE_VIDEOS.keys())[0],
                )
                custom_file = gr.File(
                    label="Optional: Upload Custom .pt Feature Tensor",
                    file_types=[".pt"],
                    type="filepath",
                )

                with gr.Accordion("⚙️ Decision Engine Parameters", open=False):
                    ema_slider = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="EMA Smoothing Factor (alpha)")
                    high_slider = gr.Slider(0.40, 0.90, value=0.60, step=0.05, label="Event Trigger High Threshold (T_high)")
                    low_slider = gr.Slider(0.10, 0.60, value=0.40, step=0.05, label="Event Sustain Low Threshold (T_low)")
                    min_dur_slider = gr.Slider(0.5, 5.0, value=1.5, step=0.5, label="Minimum Persistence Duration (sec)")
                    max_gap_slider = gr.Slider(0.5, 4.0, value=1.2, step=0.2, label="Bridge Gap Threshold (sec)")

                analyze_btn = gr.Button("🔍 Run Anomaly Detection & Risk Assessment", variant="primary", size="lg")

            with gr.Column(scale=6):
                status_output = gr.HTML(label="Operational Alert Badge")
                plot_output = gr.Plot(label="Continuous Temporal Anomaly Timeline")

        with gr.Row():
            events_table = gr.Dataframe(label="Detected Persistent Events", interactive=False)

        with gr.Row():
            diag_output = gr.Textbox(label="System Telemetry & Performance", interactive=False)

        # Wire callbacks
        analyze_btn.click(
            fn=analyze_video,
            inputs=[
                sample_dropdown,
                custom_file,
                ema_slider,
                high_slider,
                low_slider,
                min_dur_slider,
                max_gap_slider,
            ],
            outputs=[status_output, plot_output, events_table, diag_output],
        )

        sample_dropdown.change(
            fn=analyze_video,
            inputs=[
                sample_dropdown,
                custom_file,
                ema_slider,
                high_slider,
                low_slider,
                min_dur_slider,
                max_gap_slider,
            ],
            outputs=[status_output, plot_output, events_table, diag_output],
        )

        # Trigger initial run on load
        demo.load(
            fn=analyze_video,
            inputs=[
                sample_dropdown,
                custom_file,
                ema_slider,
                high_slider,
                low_slider,
                min_dur_slider,
                max_gap_slider,
            ],
            outputs=[status_output, plot_output, events_table, diag_output],
        )

    return demo


if __name__ == "__main__":
    demo = create_demo_app()
    theme = gr.themes.Soft(primary_hue="blue", secondary_hue="slate")
    demo.launch(theme=theme, server_name="127.0.0.1", server_port=7860, share=False)
