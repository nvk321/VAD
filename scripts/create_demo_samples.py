"""Synthesize Ready-to-Use Demo Video Files from UCF-Crime Archive.

Extracts PNG frames directly from ufc-crime.zip using fast index lookup and
encodes them into browser-compatible H.264 MP4 videos using bundled FFmpeg.

Produces:
    data/demo_samples/
        Robbery037.mp4        (873 frames, 29.1s — armed robbery incident)
        Normal_Videos008.mp4  (257 frames, 8.6s  — calm street CCTV)
        Arson015.mp4          (1023 frames, 34.1s — structure fire incident)
        manifest.json         (precise measured metadata for all sample videos)
"""

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import get_logger

logger = get_logger("create_demo_samples")

ZIP_PATH = Path("ufc-crime.zip")
OUT_DIR = Path("data/demo_samples")

SAMPLES_CONFIG = [
    {
        "name": "Robbery037.mp4",
        "archive_prefix": "Train/Robbery/Robbery037_x264",
        "category": "Robbery",
        "label": 1,
        "description": "Armed commercial robbery incident",
    },
    {
        "name": "Normal_Videos008.mp4",
        "archive_prefix": "Train/NormalVideos/Normal_Videos008_x264",
        "category": "NormalVideos",
        "label": 0,
        "description": "Calm urban street surveillance",
    },
    {
        "name": "Arson015.mp4",
        "archive_prefix": "Train/Arson/Arson015_x264",
        "category": "Arson",
        "label": 1,
        "description": "Structure fire and arson incident",
    },
]


def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def create_demo_videos(force: bool = False):
    """Create sample H.264 MP4 video files for immediate interactive demonstration."""
    if not ZIP_PATH.is_file():
        logger.error(f"Archive not found: {ZIP_PATH}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    logger.info(f"Using FFmpeg engine at: {ffmpeg_exe}")

    manifest_data = {}
    manifest_path = OUT_DIR / "manifest.json"

    # Target resolution: 320x320 (bicubic scaled from 64x64) preserves aspect ratio
    # and prevents video codec macroblocking artifacts while remaining compact.
    target_w, target_h = 320, 320
    fps = 30.0

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        # Pre-build list of all zip entries for quick prefix lookup
        all_names = zf.namelist()

        for cfg in SAMPLES_CONFIG:
            out_path = OUT_DIR / cfg["name"]

            # Check if video already exists and is non-empty
            if not force and out_path.is_file() and out_path.stat().st_size > 50000:
                logger.info(f"Demo video already exists: {out_path} ({out_path.stat().st_size / (1024*1024):.2f} MB)")
                # If manifest entry missing, inspect and register
                cap = cv2.VideoCapture(str(out_path))
                f_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                manifest_data[cfg["name"]] = {
                    "source": "UCF-Crime",
                    "category": cfg["category"],
                    "label": cfg["label"],
                    "description": cfg["description"],
                    "frames": f_count,
                    "width": target_w,
                    "height": target_h,
                    "fps": fps,
                    "duration_sec": round(f_count / fps, 2),
                    "codec": "H.264 (libx264, yuv420p)",
                    "size_bytes": out_path.stat().st_size,
                    "sha256": compute_file_sha256(out_path),
                }
                continue

            logger.info(f"Synthesizing full-length demo video: {cfg['name']} from {cfg['archive_prefix']}...")

            prefix = cfg["archive_prefix"].rstrip("/") + "_"
            matching_names = [n for n in all_names if n.startswith(prefix) and n.endswith(".png")]

            def get_index(path_str):
                try:
                    return int(path_str.rsplit("_", 1)[1].split(".")[0])
                except Exception:
                    return 0

            matching_names.sort(key=get_index)
            total_frames = len(matching_names)

            if total_frames == 0:
                logger.error(f"No frames found for prefix: {prefix}")
                continue

            logger.info(f"Extracting {total_frames} frames ({total_frames / fps:.1f}s) for {cfg['name']}...")

            # FFmpeg rawvideo pipe command
            cmd = [
                ffmpeg_exe,
                "-y",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-s", f"{target_w}x{target_h}",
                "-pix_fmt", "bgr24",
                "-r", str(fps),
                "-i", "-",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "18",
                "-movflags", "+faststart",
                str(out_path),
            ]

            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            frames_written = 0
            t0 = time.perf_counter()

            for fname in matching_names:
                try:
                    raw_bytes = zf.read(fname)
                    pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                    # Clean bicubic upscale to target surveillance dimensions
                    pil_resized = pil_img.resize((target_w, target_h), Image.Resampling.BICUBIC)
                    frame_bgr = cv2.cvtColor(np.array(pil_resized), cv2.COLOR_RGB2BGR)
                    proc.stdin.write(frame_bgr.tobytes())
                    frames_written += 1
                except Exception as e:
                    logger.warning(f"Error reading frame {fname}: {e}")
                    continue

            proc.stdin.close()
            proc.wait()

            elapsed = time.perf_counter() - t0
            file_size_mb = out_path.stat().st_size / (1024 * 1024)
            logger.info(
                f"Generated {out_path}: {frames_written} frames in {elapsed:.1f}s | "
                f"{file_size_mb:.2f} MB | {frames_written / max(0.1, elapsed):.1f} FPS"
            )

            manifest_data[cfg["name"]] = {
                "source": "UCF-Crime",
                "category": cfg["category"],
                "label": cfg["label"],
                "description": cfg["description"],
                "frames": frames_written,
                "width": target_w,
                "height": target_h,
                "fps": fps,
                "duration_sec": round(frames_written / fps, 2),
                "codec": "H.264 (libx264, yuv420p)",
                "size_bytes": out_path.stat().st_size,
                "sha256": compute_file_sha256(out_path),
            }

    # Write manifest to disk
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    logger.info(f"Demo samples manifest saved: {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create demo sample videos from UCF-Crime archive")
    parser.add_argument("--force", action="store_true", help="Force regeneration of demo videos")
    args = parser.parse_args()

    create_demo_videos(force=args.force)
