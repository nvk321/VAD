"""Synthesize Ready-to-Use Demo Video Files from UCF-Crime Archive.

Extracts PNG frames directly from ufc-crime.zip using O(1) index access and
compiles them into standard playable MP4 video files using OpenCV cv2.VideoWriter.

Saves demo videos to:
    data/demo_samples/
        Robbery037.mp4
        Normal_Videos008.mp4
        Arson015.mp4
"""

import io
from pathlib import Path
import sys
import zipfile

import cv2
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
        "total_frames": 400,  # ~13.3 sec at 30 fps, captures the full armed robbery incident
        "stride": 10,
    },
    {
        "name": "Normal_Videos008.mp4",
        "archive_prefix": "Train/NormalVideos/Normal_Videos008_x264",
        "total_frames": 257,  # ~8.6 sec at 30 fps, clean street CCTV footage
        "stride": 10,
    },
    {
        "name": "Arson015.mp4",
        "archive_prefix": "Train/Arson/Arson015_x264",
        "total_frames": 400,  # ~13.3 sec at 30 fps, structure fire incident
        "stride": 10,
    },
]


def create_demo_videos():
    """Create sample MP4 video files for immediate interactive demonstration."""
    if not ZIP_PATH.is_file():
        logger.error(f"Archive not found: {ZIP_PATH}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        for cfg in SAMPLES_CONFIG:
            out_path = OUT_DIR / cfg["name"]
            if out_path.is_file() and out_path.stat().st_size > 10000:
                logger.info(f"Demo video already exists: {out_path} ({out_path.stat().st_size / (1024*1024):.2f} MB)")
                continue

            logger.info(f"Generating demo video: {cfg['name']} from {cfg['archive_prefix']}...")

            # Extract first image to determine video dimensions
            prefix = cfg["archive_prefix"].rstrip("/") + "_"
            first_name = f"{prefix}0.png"
            try:
                raw_bytes = zf.read(first_name)
                img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                w, h = img.size
            except Exception as e:
                logger.error(f"Failed to read first frame {first_name}: {e}")
                continue

            # Initialize OpenCV VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = 30.0
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

            frames_written = 0
            for i in range(cfg["total_frames"]):
                frame_name = f"{prefix}{i * cfg['stride']}.png"
                try:
                    raw_bytes = zf.read(frame_name)
                    pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                    # Convert RGB (PIL) to BGR (OpenCV)
                    frame_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    writer.write(frame_bgr)
                    frames_written += 1
                except KeyError:
                    # End of available frames in archive
                    break
                except Exception as e:
                    logger.warning(f"Error processing frame {frame_name}: {e}")
                    continue

            writer.release()
            file_size_mb = out_path.stat().st_size / (1024 * 1024)
            logger.info(f"Successfully generated {out_path}: {frames_written} frames, {file_size_mb:.2f} MB")


if __name__ == "__main__":
    create_demo_videos()
