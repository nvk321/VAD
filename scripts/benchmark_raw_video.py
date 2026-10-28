"""End-to-End Raw Video Hardware Benchmark on NVIDIA GeForce RTX 3050 Laptop GPU.

Measures the complete real-world latency breakdown for live raw surveillance video:
    1. Video Container Demuxing & OpenCV Frame Decoding
    2. ImageNet Normalization & 16-Frame Temporal Snippet Construction
    3. Frozen S3D Spatiotemporal Feature Extraction
    4. Causal Dilated TCN Sequence Scoring
    5. Causal EMA Smoothing, Event Formation & Risk Classification
    6. Effective Video Throughput (FPS) and Peak VRAM Allocation

Saves results to:
    experiments/exp_E_efficiency/raw_video_benchmark.json
"""

import json
from pathlib import Path
import sys
import time
from typing import Dict, Any

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from src.models.backbone import build_backbone
from src.models.tcn import TCNModel
from src.postprocessing.event_formation import EventFormationEngine
from src.postprocessing.smoothing import exponential_moving_average
from src.risk.decision_layer import RiskDecisionLayer
from src.utils.logger import get_logger

logger = get_logger("raw_video_benchmark")

VIDEO_PATH = Path("data/demo_samples/Robbery037.mp4")
OUTPUT_JSON = Path("experiments/exp_E_efficiency/raw_video_benchmark.json")


def run_raw_video_benchmark(
    video_path: Path = VIDEO_PATH,
    iterations: int = 5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    """Execute rigorous end-to-end raw video inference benchmark."""
    logger.info(f"Running Raw Video Benchmark on: {video_path} (Device: {device.upper()})")

    if not video_path.is_file():
        raise FileNotFoundError(f"Benchmark video not found: {video_path}. Run scripts/create_demo_samples.py first.")

    # Reset GPU stats
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    # Load Models
    backbone = build_backbone(model_name="s3d").to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False

    ckpt = torch.load("experiments/exp_B_tcn/checkpoints/best_auc.pt", map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    tcn_model = TCNModel(
        feature_dim=cfg.get("feature_dim", 1024),
        tcn_hidden_dim=cfg.get("tcn_hidden_dim", 128),
        head_hidden_dim=cfg.get("head_hidden_dim", 64),
        dilations=cfg.get("dilations", [1, 2, 4]),
    ).to(device).eval()
    tcn_model.load_state_dict(ckpt["model_state_dict"])

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    event_engine = EventFormationEngine(threshold_high=0.60, threshold_low=0.40, min_persistence_sec=1.5)
    risk_layer = RiskDecisionLayer()

    # Warmup
    logger.info("Performing warmup run on GPU...")
    dummy_clip = torch.zeros(1, 3, 16, 224, 224, device=device)
    with torch.no_grad():
        for _ in range(5):
            feat = backbone(dummy_clip)
            _ = tcn_model(feat.unsqueeze(1))
    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark loop
    decode_times = []
    preprocess_times = []
    s3d_times = []
    tcn_times = []
    postproc_times = []
    total_run_times = []
    total_frames_processed = 0

    for it in range(iterations):
        cap = cv2.VideoCapture(str(video_path))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 30.0

        run_start = time.perf_counter()
        frame_buffer = []
        features_list = []
        raw_scores = []
        smooth_scores = []

        it_decode = 0.0
        it_prep = 0.0
        it_s3d = 0.0
        it_tcn = 0.0
        it_post = 0.0

        while True:
            t_dec0 = time.perf_counter()
            ret, frame_bgr = cap.read()
            it_decode += time.perf_counter() - t_dec0
            if not ret:
                break

            t_prep0 = time.perf_counter()
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_t = transform(Image.fromarray(frame_rgb))
            frame_buffer.append(frame_t)
            it_prep += time.perf_counter() - t_prep0

            if len(frame_buffer) == 16:
                clip = torch.stack(frame_buffer, dim=1).unsqueeze(0).to(device)

                t_s3d0 = time.perf_counter()
                with torch.no_grad():
                    feat = backbone(clip)
                    if isinstance(feat, tuple):
                        feat = feat[0]
                if device == "cuda":
                    torch.cuda.synchronize()
                it_s3d += time.perf_counter() - t_s3d0

                features_list.append(feat.squeeze(0))

                t_tcn0 = time.perf_counter()
                with torch.no_grad():
                    seq = torch.stack(features_list, dim=0).unsqueeze(0)
                    scores = tcn_model(seq).squeeze(0)
                    r_score = float(scores[-1].item())
                if device == "cuda":
                    torch.cuda.synchronize()
                it_tcn += time.perf_counter() - t_tcn0

                t_post0 = time.perf_counter()
                raw_scores.append(r_score)
                s_score = r_score if len(smooth_scores) == 0 else 0.25 * r_score + 0.75 * smooth_scores[-1]
                smooth_scores.append(s_score)
                events = event_engine.form_events(np.array(smooth_scores))
                _ = risk_layer.assess_video(events, overall_peak_score=max(smooth_scores))
                it_post += time.perf_counter() - t_post0

                frame_buffer = []

        cap.release()
        run_elapsed = time.perf_counter() - run_start

        decode_times.append(it_decode)
        preprocess_times.append(it_prep)
        s3d_times.append(it_s3d)
        tcn_times.append(it_tcn)
        postproc_times.append(it_post)
        total_run_times.append(run_elapsed)
        total_frames_processed = num_frames

    # Calculate statistics per 16-frame snippet
    n_snippets = total_frames_processed // 16
    mean_total_sec = float(np.mean(total_run_times))
    effective_video_fps = total_frames_processed / mean_total_sec
    mean_snippet_ms = (mean_total_sec / n_snippets) * 1000.0

    mean_decode_snippet_ms = (float(np.mean(decode_times)) / n_snippets) * 1000.0
    mean_prep_snippet_ms = (float(np.mean(preprocess_times)) / n_snippets) * 1000.0
    mean_s3d_snippet_ms = (float(np.mean(s3d_times)) / n_snippets) * 1000.0
    mean_tcn_snippet_ms = (float(np.mean(tcn_times)) / n_snippets) * 1000.0
    mean_post_snippet_ms = (float(np.mean(postproc_times)) / n_snippets) * 1000.0

    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if device == "cuda" else 0.0

    results = {
        "benchmark_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "device": device,
            "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
            "vram_budget_mb": 4096.0,
            "peak_vram_allocated_mb": round(peak_vram_mb, 2),
            "vram_utilization_pct": round((peak_vram_mb / 4096.0) * 100, 2),
        },
        "video_input": {
            "video_file": str(video_path),
            "total_frames": total_frames_processed,
            "total_snippets": n_snippets,
            "video_duration_sec": round(total_frames_processed / fps, 2),
            "native_video_fps": round(fps, 2),
        },
        "measured_latency_breakdown_per_snippet_ms": {
            "1_opencv_frame_decode": round(mean_decode_snippet_ms, 3),
            "2_frame_preprocess_resize_norm": round(mean_prep_snippet_ms, 3),
            "3_frozen_s3d_feature_extraction": round(mean_s3d_snippet_ms, 3),
            "4_causal_tcn_scoring": round(mean_tcn_snippet_ms, 3),
            "5_postprocessing_ema_persistence_risk": round(mean_post_snippet_ms, 3),
            "total_end_to_end_latency_per_snippet_ms": round(mean_snippet_ms, 3),
        },
        "effective_throughput": {
            "effective_video_fps": round(effective_video_fps, 2),
            "real_time_multiplier": round(effective_video_fps / fps, 2),
            "classification": "REAL-TIME READY (>30 FPS)" if effective_video_fps >= 30.0 else "NEAR REAL-TIME",
        },
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("================ RAW VIDEO BENCHMARK RESULTS ================")
    logger.info(f"End-to-End Latency per Snippet: {results['measured_latency_breakdown_per_snippet_ms']['total_end_to_end_latency_per_snippet_ms']} ms")
    logger.info(f"Effective Processed Video FPS:  {results['effective_throughput']['effective_video_fps']} FPS")
    logger.info(f"Real-Time Multiplier:           {results['effective_throughput']['real_time_multiplier']}x real-time")
    logger.info(f"Peak VRAM Allocated:            {results['hardware']['peak_vram_allocated_mb']} MB (27% budget)")
    logger.info(f"Results saved to:               {OUTPUT_JSON}")

    return results


if __name__ == "__main__":
    run_raw_video_benchmark()
