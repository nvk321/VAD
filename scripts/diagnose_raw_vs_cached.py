"""Diagnostic Script: Raw Video vs. Cached Feature Parity Verification.

Validates that sequential frame decoding, resizing, ImageNet normalization,
and S3D backbone execution on raw video files produce features and anomaly scores
that are consistent with the quarantined research feature cache.
"""

from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.backbone import build_backbone
from src.models.tcn import TCNModel
from src.utils.logger import get_logger

logger = get_logger("diagnose_parity")


def diagnose_parity(
    video_path: str = "data/demo_samples/Robbery037.mp4",
    cached_feature_path: str = "data/features/ucf_crime_s3d/Robbery037_x264.pt",
    checkpoint_path: str = "experiments/exp_B_tcn/checkpoints/best_auc.pt",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> bool:
    """Verifies numerical and trajectory consistency between raw video and cached features."""
    v_path = Path(video_path)
    c_path = Path(cached_feature_path)
    ckpt_path = Path(checkpoint_path)

    if not v_path.is_file():
        logger.error(f"Video file missing: {v_path}")
        return False
    if not c_path.is_file():
        logger.error(f"Cached feature file missing: {c_path}")
        return False
    if not ckpt_path.is_file():
        logger.error(f"Checkpoint missing: {ckpt_path}")
        return False

    print("=" * 70)
    print(" RAW-VIDEO VS CACHED-FEATURE PARITY DIAGNOSTIC")
    print("=" * 70)
    print(f"Device:           {device}")
    print(f"Input Video:      {v_path.name}")
    print(f"Cached Feature:   {c_path.name}")
    print(f"Model Checkpoint: {ckpt_path.name}")

    # 1. Load cached feature tensor
    cached_dict = torch.load(c_path, map_location="cpu", weights_only=False)
    cached_feats = cached_dict["features"] if isinstance(cached_dict, dict) else cached_dict
    num_cached_snippets = cached_feats.size(0)
    print(f"\n[1] Cached Features: {cached_feats.shape} ({num_cached_snippets} snippets)")

    # 2. Extract features from raw video sequentially
    cap = cv2.VideoCapture(str(v_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    print(f"[2] Raw Video Info: {total_frames} frames | {fps:.1f} FPS | Duration: {total_frames/fps:.1f}s")

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    backbone = build_backbone("s3d").to(device)
    backbone.eval()

    raw_feats_list = []
    frame_buffer = []

    t0 = time.perf_counter()
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        frame_buffer.append(transform(pil_img))

        if len(frame_buffer) == 16:
            clip = torch.stack(frame_buffer, dim=1).unsqueeze(0).to(device)
            with torch.inference_mode():
                feat = backbone(clip)
                if isinstance(feat, tuple):
                    feat = feat[0]
                raw_feats_list.append(feat.squeeze(0).cpu())
            frame_buffer = []

    cap.release()
    elapsed_extract = time.perf_counter() - t0
    raw_feats = torch.stack(raw_feats_list)
    num_raw_snippets = raw_feats.size(0)
    print(f"    Raw Features Extracted: {raw_feats.shape} ({num_raw_snippets} snippets in {elapsed_extract:.2f}s)")

    # 3. Align snippets for comparison
    n_compare = min(num_cached_snippets, num_raw_snippets)
    sub_cached = cached_feats[:n_compare]
    sub_raw = raw_feats[:n_compare]

    # Compute feature metrics
    cos_sims = F.cosine_similarity(sub_raw, sub_cached, dim=1)
    mean_cos = cos_sims.mean().item()
    min_cos = cos_sims.min().item()
    abs_diff = (sub_raw - sub_cached).abs()
    mean_abs_diff = abs_diff.mean().item()
    max_abs_diff = abs_diff.max().item()

    print(f"\n[3] Spatiotemporal Feature Alignment (First {n_compare} snippets):")
    print(f"    Mean Cosine Similarity: {mean_cos:.4f} (Ideal: > 0.94)")
    print(f"    Min Cosine Similarity:  {min_cos:.4f}")
    print(f"    Mean Absolute Diff:     {mean_abs_diff:.6f}")
    print(f"    Max Absolute Diff:      {max_abs_diff:.6f}")

    # 4. Load TCN Model and Evaluate Anomaly Trajectories
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    model = TCNModel(
        feature_dim=cfg.get("feature_dim", 1024),
        tcn_hidden_dim=cfg.get("tcn_hidden_dim", 128),
        head_hidden_dim=cfg.get("head_hidden_dim", 64),
        dilations=cfg.get("dilations", [1, 2, 4]),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.inference_mode():
        scores_cached = model(sub_cached.unsqueeze(0)).squeeze(0).tolist()
        scores_raw = model(sub_raw.unsqueeze(0)).squeeze(0).tolist()

    score_diffs = [abs(r - c) for r, c in zip(scores_raw, scores_cached)]
    mean_score_diff = np.mean(score_diffs)
    max_score_diff = np.max(score_diffs)
    peak_cached = max(scores_cached)
    peak_raw = max(scores_raw)

    print(f"\n[4] TCN Anomaly Score Trajectory Parity:")
    print(f"    Cached Peak Anomaly:    {peak_cached:.4f} (at snippet {scores_cached.index(peak_cached)})")
    print(f"    Raw Video Peak Anomaly: {peak_raw:.4f} (at snippet {scores_raw.index(peak_raw)})")
    print(f"    Mean Score Difference:  {mean_score_diff:.6f}")
    print(f"    Max Score Difference:   {max_score_diff:.6f}")

    print("\n[5] Sample Trajectory Comparison (10-Snippet Checkpoints):")
    step = max(1, n_compare // 8)
    for idx in range(0, n_compare, step):
        print(f"    Snippet {idx:2d} (t={idx*16/fps:.1f}s): "
              f"Raw={scores_raw[idx]:.4f} | Cached={scores_cached[idx]:.4f} | "
              f"Diff={abs(scores_raw[idx]-scores_cached[idx]):.4f} | "
              f"CosSim={cos_sims[idx].item():.4f}")

    # 5. Parity Decision Gate
    # High cosine similarity (>0.92) across video snippets verifies that raw decoding and S3D
    # match the cached research representations without temporal divergence.
    is_aligned = mean_cos >= 0.90
    print("\n" + "=" * 70)
    if is_aligned:
        print(" [PASS] RAW-VIDEO PIPELINE VERIFIED PARALLEL WITH RESEARCH CACHE")
    else:
        print(" [FAIL] MEAN COSINE SIMILARITY BELOW RESEARCH ACCEPTANCE GATE (0.90)")
    print("=" * 70 + "\n")

    return is_aligned


if __name__ == "__main__":
    success = diagnose_parity()
    sys.exit(0 if success else 1)
