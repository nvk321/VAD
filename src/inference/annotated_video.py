"""Video Frame Annotation & Downloadable Annotated Video Generation.

Provides frame-level operational HUD overlays for real-time surveillance triage:
    - Status Banner: NORMAL / LOW RISK / MEDIUM RISK / HIGH RISK
    - Continuous Anomaly Score & Smoothed Score
    - Deterministic Risk Score R in [0, 1]
    - Elapsed Video Timestamp
    - Active Persistent Event Identification

IMPORTANT SCIENTIFIC GOVERNANCE:
This module draws frame-level status HUD boxes only. It strictly DOES NOT
draw bounding boxes around spatial objects (persons, vehicles, weapons),
preserving honest academic positioning that our system performs temporal
video anomaly detection rather than spatial object localization.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from src.postprocessing.event_formation import AnomalyEvent
from src.risk.decision_layer import RiskAssessment, RiskDecisionLayer
from src.utils.logger import get_logger

logger = get_logger("annotated_video")


# Distinct BGR color scheme for surveillance overlay
COLOR_MAP_BGR = {
    "High Risk": (45, 45, 215),      # Vivid Crimson Red
    "Medium Risk": (30, 160, 240),   # Vibrant Amber / Orange
    "Low Risk": (210, 130, 10),      # Clear Cerulean Blue
    "Normal": (60, 160, 50),         # Stable Emerald Green
}


def annotate_frame(
    frame_bgr: np.ndarray,
    raw_score: float,
    smooth_score: float,
    risk_assessment: Optional[RiskAssessment],
    timestamp_sec: float,
    active_event: Optional[AnomalyEvent] = None,
) -> np.ndarray:
    """Renders a semi-transparent operational status HUD box on a video frame.

    Args:
        frame_bgr: Input video frame in BGR format.
        raw_score: Current continuous anomaly score s_t.
        smooth_score: Smoothed anomaly score s_tilde_t.
        risk_assessment: Current RiskAssessment object from decision layer.
        timestamp_sec: Current video timestamp in seconds.
        active_event: AnomalyEvent if currently inside an incident duration.

    Returns:
        Annotated frame with surveillance HUD overlay.
    """
    h, w = frame_bgr.shape[:2]
    annotated = frame_bgr.copy()

    risk_level = risk_assessment.risk_level if risk_assessment else "Normal"
    risk_score = risk_assessment.risk_score if risk_assessment else 0.0
    theme_color = COLOR_MAP_BGR.get(risk_level, (100, 100, 100))

    # Determine HUD box geometry
    box_w = min(w - 20, 360)
    box_h = 120 if active_event is None else 145
    x1, y1 = 15, 15
    x2, y2 = x1 + box_w, y1 + box_h

    # Create semi-transparent dark background
    overlay = annotated.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 24, 28), -1)
    # Blend with original frame (alpha=0.78 for readability against any CCTV lighting)
    cv2.addWeighted(overlay, 0.78, annotated, 0.22, 0, annotated)

    # Accent color bar on left edge of HUD box
    cv2.rectangle(annotated, (x1, y1), (x1 + 6, y2), theme_color, -1)
    # Thin outer border
    cv2.rectangle(annotated, (x1, y1), (x2, y2), theme_color, 1)

    # Title Banner
    if risk_level == "High Risk":
        title = "[!] HIGH RISK ANOMALY"
    elif risk_level == "Medium Risk":
        title = "[!] MEDIUM RISK ANOMALY"
    elif risk_level == "Low Risk":
        title = "[*] LOW RISK ANOMALY"
    else:
        title = "[+] NORMAL ACTIVITY"

    cv2.putText(
        annotated,
        title,
        (x1 + 16, y1 + 25),
        cv2.FONT_HERSHEY_DUPLEX,
        0.58,
        theme_color,
        1,
        cv2.LINE_AA,
    )

    # Line 1: Anomaly Score & Smoothed Score
    text_line1 = f"Anomaly: {raw_score:.3f} (EMA: {smooth_score:.3f})"
    cv2.putText(
        annotated,
        text_line1,
        (x1 + 16, y1 + 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )

    # Line 2: Risk Score & Level
    text_line2 = f"Risk Score: {risk_score:.3f} [{risk_level.upper()}]"
    cv2.putText(
        annotated,
        text_line2,
        (x1 + 16, y1 + 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )

    # Line 3: Timestamp
    text_line3 = f"Video Time: {timestamp_sec:.2f} s"
    cv2.putText(
        annotated,
        text_line3,
        (x1 + 16, y1 + 94),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    # Line 4 (Optional): Active Formed Event
    if active_event is not None:
        text_line4 = f"Incident: {active_event.start_sec:.1f}s - {active_event.end_sec:.1f}s ({active_event.duration_sec:.1f}s)"
        cv2.putText(
            annotated,
            text_line4,
            (x1 + 16, y1 + 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            theme_color,
            1,
            cv2.LINE_AA,
        )

    return annotated


import shutil
import subprocess
import imageio_ffmpeg


def validate_video_file(video_path: Union[str, Path]) -> Tuple[bool, float, str]:
    """Validates whether a video file exists, is non-empty, and has readable metadata.

    Returns:
        (is_valid, duration_sec, error_message)
    """
    v_path = Path(video_path)
    if not v_path.is_file():
        return False, 0.0, f"File not found: {v_path}"

    if v_path.stat().st_size < 1000:
        return False, 0.0, f"File too small ({v_path.stat().st_size} bytes): likely empty or corrupted"

    cap = cv2.VideoCapture(str(v_path))
    if not cap.isOpened():
        return False, 0.0, f"OpenCV unable to open container: {v_path.name}"

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if frames <= 0 or w <= 0 or h <= 0:
        return False, 0.0, f"Invalid video dimensions or frame count ({w}x{h}, {frames} frames)"

    fps = fps if (fps > 0 and not np.isnan(fps)) else 30.0
    duration_sec = frames / fps
    return True, duration_sec, ""


def create_clean_video(
    input_video_path: Union[str, Path],
    output_video_path: Union[str, Path],
) -> Path:
    """Creates a browser-compatible H.264 clean video without HUD overlay.

    If input is already browser-compatible H.264 with faststart, copies it;
    otherwise remuxes/transcodes via FFmpeg to guarantee HTML5 compatibility.
    """
    inp_path = Path(input_video_path)
    out_path = Path(output_video_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i", str(inp_path),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-movflags", "+faststart",
        "-an",
        str(out_path),
    ]

    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return out_path
    except Exception as e:
        logger.warning(f"Clean video FFmpeg encode failed, falling back to direct copy: {e}")
        shutil.copy2(inp_path, out_path)
        return out_path


def create_annotated_video(
    input_video_path: Union[str, Path],
    output_video_path: Union[str, Path],
    timestamps: List[float],
    raw_scores: List[float],
    smoothed_scores: List[float],
    events: List[AnomalyEvent],
    risk_layer: RiskDecisionLayer,
) -> Path:
    """Renders and encodes a full annotated video file with surveillance HUD overlay.

    Uses bundled FFmpeg with libx264 and +faststart to guarantee instant,
    error-free HTML5 video playback across all modern web browsers.

    Args:
        input_video_path: Path to the original input video.
        output_video_path: Path to save the annotated video (.mp4).
        timestamps: List of timestamps for each evaluated snippet.
        raw_scores: List of raw continuous anomaly scores s_t.
        smoothed_scores: List of smoothed anomaly scores s_tilde_t.
        events: Formed persistent events.
        risk_layer: RiskDecisionLayer for operational score calculation.

    Returns:
        Path to the generated annotated video file.
    """
    inp_path = Path(input_video_path)
    out_path = Path(output_video_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(inp_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video for annotation: {inp_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0 or np.isnan(fps):
        fps = 30.0

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{w}x{h}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "20",
        "-movflags", "+faststart",
        str(out_path),
    ]

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    total_snippets = len(timestamps)
    frame_idx = 0

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            current_time = frame_idx / fps

            # Find closest snippet index up to current time (strict causality)
            snippet_idx = 0
            if total_snippets > 0:
                for i, ts in enumerate(timestamps):
                    if ts <= current_time:
                        snippet_idx = i
                    else:
                        break

            r_score = raw_scores[snippet_idx] if snippet_idx < len(raw_scores) else 0.0
            s_score = smoothed_scores[snippet_idx] if snippet_idx < len(smoothed_scores) else 0.0

            # Check if current time falls within an active incident interval
            active_event = None
            for ev in events:
                if ev.start_sec <= current_time <= ev.end_sec:
                    active_event = ev
                    break

            # Assess risk for current state
            if active_event is not None:
                risk = risk_layer.assess_event(active_event)
            else:
                risk = risk_layer.assess_video([], overall_peak_score=s_score)

            # Render HUD on frame
            annotated_frame = annotate_frame(
                frame_bgr=frame_bgr,
                raw_score=r_score,
                smooth_score=s_score,
                risk_assessment=risk,
                timestamp_sec=current_time,
                active_event=active_event,
            )

            proc.stdin.write(annotated_frame.tobytes())
            frame_idx += 1

    finally:
        cap.release()
        if proc.stdin:
            proc.stdin.close()
        proc.wait()

    logger.info(f"H.264 Annotated video saved successfully to: {out_path} ({frame_idx} frames)")
    return out_path

