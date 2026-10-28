"""Streaming Raw Video Inference Engine for Video Anomaly Detection & Risk Assessment.

Processes raw video files (.mp4, .avi, .mov) directly from frames:
    Raw Video -> Frame Decoding -> 16-Frame Temporal Snippet Buffer
              -> Frozen S3D Backbone -> Causal TCN Anomaly Head
              -> Causal EMA Smoothing -> Hysteresis & Persistence Filter
              -> Deterministic Risk Decision Layer -> StreamUpdate

Emits progressive StreamUpdate objects containing live anomaly scores,
timeline history, formed incident intervals, and operational risk triage.
"""

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Generator, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from src.models.backbone import build_backbone
from src.models.tcn import TCNModel
from src.postprocessing.event_formation import AnomalyEvent, EventFormationEngine
from src.risk.decision_layer import RiskAssessment, RiskDecisionLayer
from src.utils.logger import get_logger

logger = get_logger("video_stream")


@dataclass
class StreamUpdate:
    """Telemetry and state emitted after each processed temporal snippet."""

    snippet_idx: int
    video_timestamp_sec: float
    total_duration_sec: float
    current_raw_score: float
    current_smooth_score: float
    scores_history: List[float] = field(default_factory=list)
    smoothed_history: List[float] = field(default_factory=list)
    timestamps_history: List[float] = field(default_factory=list)
    active_events: List[AnomalyEvent] = field(default_factory=list)
    current_risk: Optional[RiskAssessment] = None
    processing_fps: float = 0.0
    model_latency_ms: float = 0.0
    frames_processed: int = 0
    total_frames: int = 0
    is_complete: bool = False
    status_banner: str = "NORMAL"


class VideoStreamInference:
    """End-to-end streaming inference pipeline for raw video anomaly detection."""

    def __init__(
        self,
        checkpoint_path: str = "experiments/exp_B_tcn/checkpoints/best_auc.pt",
        backbone_type: str = "s3d",
        device: str = "auto",
        snippet_length: int = 16,
        snippet_stride: int = 16,
        ema_alpha: float = 0.25,
        threshold_high: float = 0.60,
        threshold_low: float = 0.40,
        min_persistence_sec: float = 1.5,
        max_gap_sec: float = 1.2,
        weight_magnitude: float = 0.65,
        weight_duration: float = 0.35,
        duration_saturation_sec: float = 8.0,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.backbone_type = backbone_type
        self.snippet_length = snippet_length
        self.snippet_stride = snippet_stride
        self.ema_alpha = ema_alpha

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Initializing VideoStreamInference on device: {self.device}")

        # 1. Load Frozen Backbone
        self.backbone = build_backbone(model_name=self.backbone_type)
        self.backbone.to(self.device)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        # 2. Load Causal TCN Anomaly Model
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found at: {self.checkpoint_path}")

        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        cfg = ckpt.get("config", {})
        self.model = TCNModel(
            feature_dim=cfg.get("feature_dim", 1024),
            tcn_hidden_dim=cfg.get("tcn_hidden_dim", 128),
            head_hidden_dim=cfg.get("head_hidden_dim", 64),
            dilations=cfg.get("dilations", [1, 2, 4]),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        # 3. Standard Kinetics / ImageNet Transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # 4. Postprocessing Engines
        self.threshold_high = threshold_high
        self.threshold_low = threshold_low
        self.min_persistence_sec = min_persistence_sec
        self.max_gap_sec = max_gap_sec

        self.risk_layer = RiskDecisionLayer(
            weight_magnitude=weight_magnitude,
            weight_duration=weight_duration,
            duration_saturation_sec=duration_saturation_sec,
            theta_high=0.70,
            theta_low=0.45,
            theta_base=0.25,
        )

    def process_video_stream(
        self,
        video_path: Union[str, Path],
    ) -> Generator[StreamUpdate, None, None]:
        """Streamingly processes a raw video file and yields progressive StreamUpdate objects.

        Args:
            video_path: Path to the input video (.mp4, .avi, .mov, etc.).

        Yields:
            StreamUpdate for each processed 16-frame temporal snippet.
        """
        video_file = Path(video_path)
        if not video_file.is_file():
            raise FileNotFoundError(f"Input video file does not exist: {video_file}")

        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file with OpenCV: {video_file}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0 or np.isnan(fps):
            fps = 30.0  # Robust default for CCTV with missing header FPS

        snippet_duration_sec = self.snippet_length / fps
        total_duration_sec = total_frames / fps if total_frames > 0 else 0.0

        event_engine = EventFormationEngine(
            threshold_high=self.threshold_high,
            threshold_low=self.threshold_low,
            min_persistence_sec=self.min_persistence_sec,
            max_gap_sec=self.max_gap_sec,
            snippet_duration_sec=snippet_duration_sec,
        )

        frame_buffer: List[torch.Tensor] = []
        features_list: List[torch.Tensor] = []
        raw_scores_history: List[float] = []
        smooth_scores_history: List[float] = []
        timestamps_history: List[float] = []

        frames_processed = 0
        snippet_idx = 0
        overall_start_time = time.perf_counter()

        logger.info(
            f"Starting streaming inference: {video_file.name} | "
            f"Frames: {total_frames} | FPS: {fps:.1f} | Duration: {total_duration_sec:.1f}s"
        )

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frames_processed += 1
            # Convert BGR (OpenCV) to RGB (PIL)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            frame_tensor = self.transform(pil_img)  # (3, 224, 224)
            frame_buffer.append(frame_tensor)

            # Once rolling buffer reaches snippet length (16 frames)
            if len(frame_buffer) == self.snippet_length:
                t0 = time.perf_counter()

                # 1. Format as (1, 3, 16, 224, 224)
                clip = torch.stack(frame_buffer, dim=1).unsqueeze(0).to(self.device)

                # 2. Extract S3D feature representation (1, 1024)
                with torch.no_grad():
                    feat = self.backbone(clip)
                    if isinstance(feat, tuple):
                        feat = feat[0]
                    features_list.append(feat.squeeze(0))  # (1024,)

                    # 3. Run Causal TCN on all accumulated features (1, T, 1024)
                    full_feat_seq = torch.stack(features_list, dim=0).unsqueeze(0)
                    scores_seq = self.model(full_feat_seq).squeeze(0)  # (T,)
                    raw_score = float(scores_seq[-1].item())

                model_latency_ms = (time.perf_counter() - t0) * 1000.0

                # 4. Online Causal EMA Smoothing
                if len(smooth_scores_history) == 0:
                    smooth_score = raw_score
                else:
                    smooth_score = self.ema_alpha * raw_score + (1.0 - self.ema_alpha) * smooth_scores_history[-1]

                current_timestamp = frames_processed / fps
                raw_scores_history.append(raw_score)
                smooth_scores_history.append(smooth_score)
                timestamps_history.append(current_timestamp)

                # 5. Form Events up to current snippet
                active_events = event_engine.form_events(np.array(smooth_scores_history))

                # 6. Assess Operational Risk
                current_peak = max(smooth_scores_history) if smooth_scores_history else 0.0
                current_risk = self.risk_layer.assess_video(active_events, overall_peak_score=current_peak)

                elapsed_total = time.perf_counter() - overall_start_time
                processing_fps = frames_processed / max(0.001, elapsed_total)

                # Status Banner
                if current_risk.risk_level == "High Risk":
                    status_banner = "HIGH RISK ANOMALY"
                elif current_risk.risk_level == "Medium Risk":
                    status_banner = "MEDIUM RISK ANOMALY"
                elif current_risk.risk_level == "Low Risk":
                    status_banner = "LOW RISK ANOMALY"
                else:
                    status_banner = "NORMAL"

                # 7. Yield progressive update
                yield StreamUpdate(
                    snippet_idx=snippet_idx,
                    video_timestamp_sec=current_timestamp,
                    total_duration_sec=total_duration_sec,
                    current_raw_score=raw_score,
                    current_smooth_score=smooth_score,
                    scores_history=list(raw_scores_history),
                    smoothed_history=list(smooth_scores_history),
                    timestamps_history=list(timestamps_history),
                    active_events=active_events,
                    current_risk=current_risk,
                    processing_fps=processing_fps,
                    model_latency_ms=model_latency_ms,
                    frames_processed=frames_processed,
                    total_frames=total_frames,
                    is_complete=False,
                    status_banner=status_banner,
                )

                snippet_idx += 1

                # Slide buffer by stride
                if self.snippet_stride < self.snippet_length:
                    frame_buffer = frame_buffer[self.snippet_stride:]
                else:
                    frame_buffer = []

        # Handle leftover frames (< 16) at end of video
        if len(frame_buffer) > 0 and snippet_idx == 0:
            # Video has fewer than 16 frames total: pad with last frame to enable inference
            while len(frame_buffer) < self.snippet_length:
                frame_buffer.append(frame_buffer[-1])

            t0 = time.perf_counter()
            clip = torch.stack(frame_buffer, dim=1).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.backbone(clip)
                if isinstance(feat, tuple):
                    feat = feat[0]
                features_list.append(feat.squeeze(0))
                full_feat_seq = torch.stack(features_list, dim=0).unsqueeze(0)
                scores_seq = self.model(full_feat_seq).squeeze(0)
                raw_score = float(scores_seq[-1].item())
            model_latency_ms = (time.perf_counter() - t0) * 1000.0

            smooth_score = raw_score
            current_timestamp = total_duration_sec
            raw_scores_history.append(raw_score)
            smooth_scores_history.append(smooth_score)
            timestamps_history.append(current_timestamp)

            active_events = event_engine.form_events(np.array(smooth_scores_history))
            current_risk = self.risk_layer.assess_video(active_events, overall_peak_score=smooth_score)
            elapsed_total = time.perf_counter() - overall_start_time
            processing_fps = frames_processed / max(0.001, elapsed_total)

            yield StreamUpdate(
                snippet_idx=snippet_idx,
                video_timestamp_sec=current_timestamp,
                total_duration_sec=total_duration_sec,
                current_raw_score=raw_score,
                current_smooth_score=smooth_score,
                scores_history=list(raw_scores_history),
                smoothed_history=list(smooth_scores_history),
                timestamps_history=list(timestamps_history),
                active_events=active_events,
                current_risk=current_risk,
                processing_fps=processing_fps,
                model_latency_ms=model_latency_ms,
                frames_processed=frames_processed,
                total_frames=total_frames,
                is_complete=True,
                status_banner="NORMAL" if current_risk.risk_level == "Normal" else f"{current_risk.risk_level.upper()} ANOMALY",
            )

        cap.release()

        # Emit final completed state
        if len(raw_scores_history) > 0:
            final_peak = max(smooth_scores_history) if smooth_scores_history else 0.0
            final_events = event_engine.form_events(np.array(smooth_scores_history))
            final_risk = self.risk_layer.assess_video(final_events, overall_peak_score=final_peak)
            elapsed_total = time.perf_counter() - overall_start_time
            processing_fps = frames_processed / max(0.001, elapsed_total)

            status_banner = "NORMAL" if final_risk.risk_level == "Normal" else f"{final_risk.risk_level.upper()} ANOMALY"

            yield StreamUpdate(
                snippet_idx=len(raw_scores_history) - 1,
                video_timestamp_sec=total_duration_sec,
                total_duration_sec=total_duration_sec,
                current_raw_score=raw_scores_history[-1],
                current_smooth_score=smooth_scores_history[-1],
                scores_history=list(raw_scores_history),
                smoothed_history=list(smooth_scores_history),
                timestamps_history=list(timestamps_history),
                active_events=final_events,
                current_risk=final_risk,
                processing_fps=processing_fps,
                model_latency_ms=0.0,
                frames_processed=frames_processed,
                total_frames=total_frames,
                is_complete=True,
                status_banner=status_banner,
            )
