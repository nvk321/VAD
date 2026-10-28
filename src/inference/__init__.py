"""Inference and streaming package for Real-Time Video Anomaly Detection & Risk Assessment."""

from src.inference.video_stream import StreamUpdate, VideoStreamInference
from src.inference.annotated_video import annotate_frame, create_annotated_video

__all__ = [
    "StreamUpdate",
    "VideoStreamInference",
    "annotate_frame",
    "create_annotated_video",
]
