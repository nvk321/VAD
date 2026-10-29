"""Inference and streaming package for Real-Time Video Anomaly Detection & Risk Assessment."""

from src.inference.video_stream import (
    StreamUpdate,
    VideoStreamInference,
    get_streaming_inference_engine,
)
from src.inference.annotated_video import (
    annotate_frame,
    create_annotated_video,
    create_clean_video,
    validate_video_file,
)

__all__ = [
    "StreamUpdate",
    "VideoStreamInference",
    "get_streaming_inference_engine",
    "annotate_frame",
    "create_annotated_video",
    "create_clean_video",
    "validate_video_file",
]

