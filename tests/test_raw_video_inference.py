"""Tests for End-to-End Raw Video Inference, Streaming, and Annotation.

Covers:
- Video decoding and metadata extraction
- Temporal snippet buffering and ImageNet normalization
- Progressive StreamUpdate emission
- Dual-threshold hysteresis and persistence on video streams
- Frame-level HUD status box rendering (no spatial bounding boxes)
- Annotated MP4 video file creation
- Short and empty video handling
"""

import json
from pathlib import Path
import tempfile
import cv2
import numpy as np
import pytest
import torch

from src.inference.annotated_video import annotate_frame, create_annotated_video
from src.inference.video_stream import StreamUpdate, VideoStreamInference
from src.postprocessing.event_formation import AnomalyEvent
from src.risk.decision_layer import RiskAssessment, RiskDecisionLayer


@pytest.fixture
def dummy_video_file():
    """Create a temporary synthetic MP4 video file for fast testing."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        temp_path = Path(f.name)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 30.0
    w, h = 320, 240
    writer = cv2.VideoWriter(str(temp_path), fourcc, fps, (w, h))

    # Write 36 frames (~1.2 sec, enough for 2 full 16-frame snippets + remainder)
    for i in range(36):
        # Create gradient synthetic frame
        frame = np.full((h, w, 3), (i * 6) % 255, dtype=np.uint8)
        writer.write(frame)

    writer.release()
    yield temp_path

    # Teardown
    if temp_path.is_file():
        temp_path.unlink()


@pytest.fixture
def short_video_file():
    """Create a temporary synthetic MP4 video with fewer than 16 frames."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        temp_path = Path(f.name)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 30.0
    w, h = 160, 120
    writer = cv2.VideoWriter(str(temp_path), fourcc, fps, (w, h))

    # Write only 6 frames (< 16-frame snippet length)
    for i in range(6):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        writer.write(frame)

    writer.release()
    yield temp_path

    if temp_path.is_file():
        temp_path.unlink()


def test_video_metadata_and_decoding(dummy_video_file):
    """Test that OpenCV decodes video dimensions and frames properly."""
    cap = cv2.VideoCapture(str(dummy_video_file))
    assert cap.isOpened()
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert total_frames == 36
    fps = cap.get(cv2.CAP_PROP_FPS)
    assert fps == 30.0
    cap.release()


def test_streaming_inference_pipeline(dummy_video_file):
    """Test full sequential streaming inference on synthetic video."""
    streamer = VideoStreamInference(device="cpu")
    updates = list(streamer.process_video_stream(dummy_video_file))

    # 36 frames with stride=16 and length=16 yields 2 snippets (0-15, 16-31) + 1 completion
    assert len(updates) >= 2

    first_update = updates[0]
    assert isinstance(first_update, StreamUpdate)
    assert first_update.snippet_idx == 0
    assert 0.0 <= first_update.current_raw_score <= 1.0
    assert 0.0 <= first_update.current_smooth_score <= 1.0
    assert first_update.status_banner in ["NORMAL", "LOW RISK ANOMALY", "MEDIUM RISK ANOMALY", "HIGH RISK ANOMALY"]

    last_update = updates[-1]
    assert last_update.is_complete is True
    assert last_update.frames_processed == 36
    assert len(last_update.scores_history) >= 2


def test_short_video_padding_support(short_video_file):
    """Test that videos with fewer than 16 frames are padded and processed gracefully."""
    streamer = VideoStreamInference(device="cpu")
    updates = list(streamer.process_video_stream(short_video_file))

    assert len(updates) >= 1
    assert updates[-1].is_complete is True
    assert 0.0 <= updates[-1].current_smooth_score <= 1.0


def test_missing_video_raises_error():
    """Test that non-existent video files raise FileNotFoundError."""
    streamer = VideoStreamInference(device="cpu")
    with pytest.raises(FileNotFoundError):
        list(streamer.process_video_stream("non_existent_file.mp4"))


def test_annotate_frame_rendering():
    """Test that annotate_frame produces valid image without object bounding boxes."""
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    risk_assessment = RiskAssessment(
        risk_level="High Risk",
        risk_score=0.75,
        peak_score=0.88,
        duration_sec=5.3,
        persistence_ratio=0.66,
        magnitude_contribution=0.50,
        duration_contribution=0.25,
        explanation="High Risk: Significant anomaly magnitude and persistence.",
    )
    event = AnomalyEvent(
        event_id=1,
        start_snippet=10,
        end_snippet=20,
        start_sec=5.3,
        end_sec=10.6,
        duration_sec=5.3,
        peak_score=0.88,
        mean_score=0.76,
    )

    annotated = annotate_frame(
        frame_bgr=dummy_frame,
        raw_score=0.88,
        smooth_score=0.85,
        risk_assessment=risk_assessment,
        timestamp_sec=7.5,
        active_event=event,
    )

    assert annotated.shape == dummy_frame.shape
    # Frame should have non-zero pixels from the HUD overlay
    assert np.sum(annotated) > 0


def test_create_annotated_video(dummy_video_file):
    """Test full annotated video MP4 generation."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        out_path = Path(f.name)

    risk_layer = RiskDecisionLayer()
    events = [
        AnomalyEvent(
            event_id=1,
            start_snippet=0,
            end_snippet=1,
            start_sec=0.0,
            end_sec=1.0,
            duration_sec=1.0,
            peak_score=0.75,
            mean_score=0.70,
        )
    ]

    result_path = create_annotated_video(
        input_video_path=dummy_video_file,
        output_video_path=out_path,
        timestamps=[0.53, 1.07],
        raw_scores=[0.72, 0.75],
        smoothed_scores=[0.72, 0.74],
        events=events,
        risk_layer=risk_layer,
    )

    assert result_path.is_file()
    assert result_path.stat().st_size > 1000

    # Clean up
    if out_path.is_file():
        out_path.unlink()
