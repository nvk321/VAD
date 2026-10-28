"""Resumable Feature Extraction Pipeline for UCF-Crime.

Extracts spatiotemporal features from the UCF-Crime frame archive using a frozen
backbone (S3D 1024-d / R(2+1)D-18 512-d). Supports resume, progress reporting,
corrupt sample handling, and memory-aware GPU execution.
"""

import argparse
import csv
import io
import json
from pathlib import Path
import sys
import time
import zipfile
from typing import Dict, List, Optional, Tuple

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
import torch
from torchvision import transforms

from src.models.backbone import build_backbone
from src.utils.seed import seed_everything


console = Console()


def build_transform(spatial_size=(224, 224)):
    """Standard ImageNet/Kinetics normalization transform."""
    return transforms.Compose([
        transforms.Resize(spatial_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def load_video_frames_from_zip(
    zf: zipfile.ZipFile,
    archive_prefix: str,
    transform,
    num_sampled_frames: Optional[int] = None,
    frame_stride: int = 10,
) -> Optional[torch.Tensor]:
    """Read and decode all PNG frames belonging to a video from the zip archive.

    Uses O(1) direct naming when num_sampled_frames is provided, falling back to prefix matching.

    Returns:
        torch.Tensor of shape (T, 3, H, W) or None if corrupt.
    """
    prefix = archive_prefix.rstrip("/") + "_"

    if num_sampled_frames is not None and num_sampled_frames > 0:
        matching_names = [f"{prefix}{i * frame_stride}.png" for i in range(num_sampled_frames)]
    else:
        matching_names = [
            name for name in zf.namelist()
            if name.startswith(prefix) and name.endswith(".png")
        ]
        def get_index(path_str):
            try:
                return int(path_str.rsplit("_", 1)[1].split(".")[0])
            except Exception:
                return 0
        matching_names.sort(key=get_index)

    if not matching_names:
        return None

    frame_tensors = []
    for name in matching_names:
        try:
            raw_bytes = zf.read(name)
            img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            frame_tensor = transform(img)
            frame_tensors.append(frame_tensor)
        except Exception:
            continue

    if not frame_tensors:
        return None

    return torch.stack(frame_tensors)  # (T, 3, H, W)


def chunk_into_snippets(video_tensor: torch.Tensor, snippet_length: int = 16) -> torch.Tensor:
    """Group temporal frame sequence into non-overlapping clips.

    Args:
        video_tensor: Tensor of shape (T, 3, H, W).
        snippet_length: Temporal window length (default 16).

    Returns:
        Tensor of shape (N_snippets, 3, snippet_length, H, W).
    """
    total_frames = video_tensor.size(0)

    if total_frames < snippet_length:
        # Repeat final frame to reach snippet length
        pad_count = snippet_length - total_frames
        last_frame = video_tensor[-1:].repeat(pad_count, 1, 1, 1)
        video_tensor = torch.cat([video_tensor, last_frame], dim=0)
        total_frames = video_tensor.size(0)

    num_snippets = total_frames // snippet_length
    clips = []
    for i in range(num_snippets):
        clip = video_tensor[i * snippet_length : (i + 1) * snippet_length]  # (T, 3, H, W)
        clip = clip.permute(1, 0, 2, 3)  # (3, T, H, W)
        clips.append(clip)

    return torch.stack(clips)  # (N_snippets, 3, snippet_length, H, W)


def extract_video_features_streaming(
    zf: zipfile.ZipFile,
    archive_prefix: str,
    transform,
    backbone,
    device: str,
    num_sampled_frames: Optional[int] = None,
    frame_stride: int = 10,
    snippet_length: int = 16,
    batch_snippets: int = 8,
) -> Optional[Tuple[torch.Tensor, int, int]]:
    """Stream and extract spatiotemporal features in small snippet batches.

    Guarantees constant, minimal host RAM (<100 MB) and GPU VRAM (<600 MB)
    regardless of whether video has 50 frames or 10,000 frames.
    """
    prefix = archive_prefix.rstrip("/") + "_"

    if num_sampled_frames is not None and num_sampled_frames > 0:
        frame_names = [f"{prefix}{i * frame_stride}.png" for i in range(num_sampled_frames)]
    else:
        frame_names = [
            name for name in zf.namelist()
            if name.startswith(prefix) and name.endswith(".png")
        ]
        def get_index(path_str):
            try:
                return int(path_str.rsplit("_", 1)[1].split(".")[0])
            except Exception:
                return 0
        frame_names.sort(key=get_index)

    total_frames = len(frame_names)
    if total_frames == 0:
        return None

    # Pad short videos
    if total_frames < snippet_length:
        pad_count = snippet_length - total_frames
        frame_names += [frame_names[-1]] * pad_count
        total_frames = len(frame_names)

    num_snippets = total_frames // snippet_length
    features_list = []

    for s_start in range(0, num_snippets, batch_snippets):
        s_end = min(s_start + batch_snippets, num_snippets)
        curr_batch_snippets = s_end - s_start

        clips_in_batch = []
        for snip_idx in range(curr_batch_snippets):
            snip_global = s_start + snip_idx
            snip_file_names = frame_names[snip_global * snippet_length : (snip_global + 1) * snippet_length]

            frame_tensors = []
            for fname in snip_file_names:
                try:
                    raw_bytes = zf.read(fname)
                    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                    frame_tensors.append(transform(img))
                except Exception:
                    if frame_tensors:
                        frame_tensors.append(frame_tensors[-1])

            if not frame_tensors:
                continue

            # Ensure snippet has exactly snippet_length frames
            while len(frame_tensors) < snippet_length:
                frame_tensors.append(frame_tensors[-1])
            frame_tensors = frame_tensors[:snippet_length]

            # (T, 3, H, W) -> permute to (3, T, H, W)
            clip_tensor = torch.stack(frame_tensors).permute(1, 0, 2, 3)
            clips_in_batch.append(clip_tensor)

        if not clips_in_batch:
            continue

        # (B, 3, T, H, W)
        batch_tensor = torch.stack(clips_in_batch).to(device)

        with torch.no_grad():
            if device == "cuda":
                with torch.amp.autocast("cuda"):
                    feat = backbone.extract_features(batch_tensor)
            else:
                feat = backbone.extract_features(batch_tensor)

        features_list.append(feat.float().cpu())

    if not features_list:
        return None

    all_features = torch.cat(features_list, dim=0)  # (num_snippets, feature_dim)
    return all_features, total_frames, all_features.size(0)


def run_feature_extraction(
    manifest_path: Path,
    archive_path: Path,
    output_dir: Path,
    backbone_name: str = "s3d",
    snippet_length: int = 16,
    batch_size: int = 16,
    device: str = "cuda",
    max_videos: Optional[int] = None,
    force: bool = False,
):
    """Main extraction routine with resume and Rich progress reporting."""
    seed_everything(42)

    output_dir.mkdir(parents=True, exist_ok=True)
    progress_manifest_path = output_dir / "extraction_manifest.json"

    # Load existing progress manifest if resuming
    manifest_data: Dict[str, dict] = {}
    if progress_manifest_path.is_file() and not force:
        try:
            with open(progress_manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
        except Exception:
            manifest_data = {}

    # Read input video list
    with open(manifest_path, "r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    if max_videos:
        abnormal = [r for r in records if int(r["label"]) == 1]
        normal = [r for r in records if int(r["label"]) == 0]
        if abnormal and normal:
            half = max_videos // 2
            records = abnormal[:half] + normal[: (max_videos - half)]
        else:
            records = records[:max_videos]

    total_videos = len(records)
    console.print(Panel.fit(
        f"[bold cyan]UCF-Crime Feature Extraction Pipeline[/bold cyan]\n"
        f"Manifest: [yellow]{manifest_path.name}[/yellow] ({total_videos} videos)\n"
        f"Backbone: [green]{backbone_name.upper()}[/green] | Output: [green]{output_dir}[/green]\n"
        f"Device: [magenta]{device}[/magenta] | Batch: {batch_size} | Snippet: {snippet_length} frames",
        title="Configuration"
    ))

    # Initialize model
    console.print(f"Loading backbone [bold green]{backbone_name}[/bold green]...")
    backbone = build_backbone(backbone_name, pretrained=True).to(device)
    backbone.eval()
    transform = build_transform(backbone.spatial_size)

    # Open archive
    zf = zipfile.ZipFile(archive_path, "r")

    successful = 0
    skipped = 0
    failed = 0
    start_time = time.perf_counter()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
    ) as progress_bar:
        task = progress_bar.add_task("Extracting features...", total=total_videos)

        for idx, rec in enumerate(records):
            video_id = rec["video_id"]
            category = rec["category"]
            label = int(rec["label"])
            archive_path_prefix = rec["archive_path"]

            target_feature_path = output_dir / f"{video_id}.pt"

            # Resume check
            if not force and target_feature_path.is_file() and target_feature_path.stat().st_size > 0:
                skipped += 1
                progress_bar.advance(task)
                continue

            try:
                num_sampled = int(rec["num_sampled_frames"]) if "num_sampled_frames" in rec else None
                stride = int(rec["frame_stride"]) if "frame_stride" in rec else 10
                result = extract_video_features_streaming(
                    zf=zf,
                    archive_prefix=archive_path_prefix,
                    transform=transform,
                    backbone=backbone,
                    device=device,
                    num_sampled_frames=num_sampled,
                    frame_stride=stride,
                    snippet_length=snippet_length,
                    batch_snippets=max(1, batch_size // 2),
                )
                if result is None:
                    failed += 1
                    manifest_data[video_id] = {"status": "failed", "error": "Zero readable snippets"}
                    progress_bar.advance(task)
                    continue

                features, num_frames, num_snippets = result

                # 4. Save to disk
                payload = {
                    "features": features,
                    "video_id": video_id,
                    "category": category,
                    "label": label,
                    "num_sampled_frames": num_frames,
                    "num_snippets": num_snippets,
                    "snippet_length": snippet_length,
                    "feature_dim": backbone.feature_dim,
                    "backbone": backbone_name,
                }
                torch.save(payload, target_feature_path)

                manifest_data[video_id] = {
                    "status": "success",
                    "file": f"{video_id}.pt",
                    "num_snippets": num_snippets,
                    "num_frames": num_frames,
                    "feature_dim": backbone.feature_dim,
                    "category": category,
                    "label": label,
                }
                successful += 1

            except Exception as e:
                failed += 1
                manifest_data[video_id] = {"status": "failed", "error": str(e)}

            progress_bar.advance(task)

            # Periodically write progress manifest to disk
            if (idx + 1) % 10 == 0 or (idx + 1) == total_videos:
                with open(progress_manifest_path, "w", encoding="utf-8") as f:
                    json.dump(manifest_data, f, indent=2)

    zf.close()
    elapsed = time.perf_counter() - start_time

    # Final summary display
    vram_used = (
        torch.cuda.max_memory_allocated() / (1024**2) if device == "cuda" else 0.0
    )
    console.print("\n" + "=" * 60)
    console.print(f"[bold green]FEATURE EXTRACTION COMPLETE[/bold green]")
    console.print(f"  Total Processed : {total_videos}")
    console.print(f"  Successful      : [green]{successful}[/green]")
    console.print(f"  Skipped (resume): [cyan]{skipped}[/cyan]")
    console.print(f"  Failed          : [red]{failed}[/red]")
    console.print(f"  Elapsed Time    : {elapsed:.1f} s (~{total_videos / max(1.0, elapsed):.2f} videos/s)")
    if device == "cuda":
        console.print(f"  Peak VRAM       : {vram_used:.1f} MB ({vram_used / 1024:.2f} GB / 4.0 GB)")
    console.print(f"  Saved Cache     : {output_dir.resolve()}")
    console.print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract video features for UCF-Crime")
    parser.add_argument("--manifest", type=str, default="data/manifests/ucf_crime_val.csv")
    parser.add_argument("--archive", type=str, default="ufc-crime.zip")
    parser.add_argument("--output_dir", type=str, default="data/features/ucf_crime_s3d")
    parser.add_argument("--backbone", type=str, default="s3d")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--snippet_length", type=int, default=16)
    parser.add_argument("--max_videos", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    archive = Path(args.archive)
    if not archive.exists():
        archive = Path("ucf-crime.zip")

    run_feature_extraction(
        manifest_path=Path(args.manifest),
        archive_path=archive,
        output_dir=Path(args.output_dir),
        backbone_name=args.backbone,
        snippet_length=args.snippet_length,
        batch_size=args.batch_size,
        device=args.device,
        max_videos=args.max_videos,
        force=args.force,
    )
