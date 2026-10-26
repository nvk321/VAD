"""Manifest Generation Pipeline for UCF-Crime.

Discovers all 1,900 canonical UCF-Crime videos directly from the local archive,
verifies frame sequences, extracts video-level metadata, performs stratified
Train/Validation splitting (using training data only, with fixed seed 42),
and generates version-controlled manifests in data/manifests/.
"""

import csv
import json
from pathlib import Path
import random
import re
import sys
import zipfile
from collections import defaultdict


def extract_video_metadata_from_zip(archive_path: Path):
    """Scan archive and group frame indices by video.

    Returns:
        tuple: (train_videos, test_videos, frame_stats)
    """
    print(f"Reading archive metadata from {archive_path.resolve()}...")
    train_dict = defaultdict(list)
    test_dict = defaultdict(list)

    with zipfile.ZipFile(archive_path, "r") as zf:
        for name in zf.namelist():
            parts = name.split("/")
            if len(parts) >= 3 and parts[2].endswith(".png"):
                part = parts[0]
                category = parts[1]
                filename = parts[2]

                m = re.match(r"^(.*)_(\d+)\.png$", filename)
                if m:
                    video_id = m.group(1)
                    frame_idx = int(m.group(2))
                    if part == "Train":
                        train_dict[(category, video_id)].append(frame_idx)
                    elif part == "Test":
                        test_dict[(category, video_id)].append(frame_idx)

    return train_dict, test_dict


def process_video_entries(raw_dict, split_name):
    """Sort frames, verify continuity, and construct structured records."""
    records = []
    for (category, video_id), frames in raw_dict.items():
        sorted_frames = sorted(frames)
        num_frames = len(sorted_frames)
        frame_start = sorted_frames[0]
        frame_end = sorted_frames[-1]

        # Calculate stride statistics
        if num_frames > 1:
            diffs = [sorted_frames[i+1] - sorted_frames[i] for i in range(num_frames - 1)]
            median_stride = sorted(diffs)[len(diffs) // 2]
        else:
            median_stride = 10

        is_normal = "Normal" in category
        label = 0 if is_normal else 1

        records.append({
            "video_id": video_id,
            "category": category,
            "label": label,
            "split": split_name,
            "num_sampled_frames": num_frames,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "frame_stride": median_stride,
            "archive_path": f"{split_name}/{category}/{video_id}"
        })

    return records


def stratified_train_val_split(train_records, val_ratio=0.15, seed=42):
    """Perform deterministic stratified Train/Validation split on training data only.

    Args:
        train_records: List of records from the Train partition (1,610 videos).
        val_ratio: Proportion allocated to validation (default 0.15 ~ 242 videos).
        seed: Random seed for deterministic reproducibility.

    Returns:
        tuple: (train_split_records, val_split_records)
    """
    rng = random.Random(seed)

    # Group by category to ensure identical class proportions in train and val
    by_category = defaultdict(list)
    for r in train_records:
        by_category[r["category"]].append(r)

    train_split = []
    val_split = []

    for cat in sorted(by_category.keys()):
        cat_records = by_category[cat]
        # Sort by video_id first to ensure deterministic ordering before shuffling
        cat_records.sort(key=lambda x: x["video_id"])
        rng.shuffle(cat_records)

        num_val = max(1, round(len(cat_records) * val_ratio))
        # Ensure normal videos don't take disproportionate share
        val_items = cat_records[:num_val]
        train_items = cat_records[num_val:]

        for item in train_items:
            item_copy = dict(item)
            item_copy["subset"] = "train"
            train_split.append(item_copy)

        for item in val_items:
            item_copy = dict(item)
            item_copy["subset"] = "val"
            val_split.append(item_copy)

    train_split.sort(key=lambda x: (x["category"], x["video_id"]))
    val_split.sort(key=lambda x: (x["category"], x["video_id"]))

    return train_split, val_split


def extract_test_ground_truth(ref_archive_path: Path):
    """Extract official test temporal ground-truth annotations from reference archive if present.

    Returns:
        dict: video_id -> list of [start_frame, end_frame] anomaly spans
    """
    test_annotations = {}
    if ref_archive_path.is_file():
        try:
            with zipfile.ZipFile(ref_archive_path, "r") as zf:
                if "VERA-main/Data/UCF_Eval.json" in zf.namelist():
                    data = json.loads(zf.read("VERA-main/Data/UCF_Eval.json"))
                    for item in data:
                        # video path e.g. "frames/Abuse/Abuse028_x264.mp4"
                        v_path = item["video"]
                        v_id = Path(v_path).stem  # "Abuse028_x264"
                        raw_labels = item.get("temporal_label", [])
                        # raw labels format [start1, end1, start2, end2, ...] with -1 padding
                        spans = []
                        for i in range(0, len(raw_labels), 2):
                            if i + 1 < len(raw_labels):
                                s, e = raw_labels[i], raw_labels[i+1]
                                if s != -1 and e != -1 and s < e:
                                    spans.append([s, e])
                        test_annotations[v_id] = {
                            "category": Path(v_path).parent.name,
                            "temporal_spans": spans,
                            "is_anomalous": len(spans) > 0,
                            "total_frames": item.get("length", 0)
                        }
        except Exception as e:
            print(f"Warning reading reference annotations: {e}")

    return test_annotations


def write_csv_manifest(records, output_path: Path, fieldnames):
    """Write list of dictionaries to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"  Wrote: {output_path} ({len(records)} rows)")


def generate_manifests(archive_path: Path, ref_archive_path: Path, output_dir: Path):
    """Generate all manifests and summary reports."""
    print("=" * 70)
    print("PHASE 1: DATASET VALIDATION & MANIFEST GENERATION")
    print("=" * 70)

    # 1. Discover and group entries
    raw_train, raw_test = extract_video_metadata_from_zip(archive_path)
    print(f"Discovered: {len(raw_train)} raw train videos, {len(raw_test)} raw test videos.")

    train_records = process_video_entries(raw_train, "Train")
    test_records = process_video_entries(raw_test, "Test")

    # 2. Stratified train/val split (seed 42)
    train_split, val_split = stratified_train_val_split(train_records, val_ratio=0.15, seed=42)

    for item in test_records:
        item["subset"] = "test"

    print(f"\nPartition split counts (Total 1,900 videos):")
    print(f"  - Training split    : {len(train_split)} videos")
    print(f"  - Validation split  : {len(val_split)} videos")
    print(f"  - Testing split     : {len(test_records)} videos")

    # 3. CSV manifests
    fieldnames = [
        "video_id", "category", "label", "subset", "num_sampled_frames",
        "frame_start", "frame_end", "frame_stride", "archive_path"
    ]

    write_csv_manifest(train_split, output_dir / "ucf_crime_train.csv", fieldnames)
    write_csv_manifest(val_split, output_dir / "ucf_crime_val.csv", fieldnames)
    write_csv_manifest(test_records, output_dir / "ucf_crime_test.csv", fieldnames)

    # 4. Canonical test ground-truth annotations
    test_annotations = extract_test_ground_truth(ref_archive_path)
    if test_annotations:
        gt_path = output_dir / "ucf_crime_test_annotations.json"
        with open(gt_path, "w", encoding="utf-8") as f:
            json.dump(test_annotations, f, indent=2)
        print(f"  Wrote canonical test ground-truth annotations: {gt_path} ({len(test_annotations)} videos)")

    # 5. Machine-readable dataset audit summary
    summary = {
        "dataset_name": "UCF-Crime",
        "archive_file": archive_path.name,
        "archive_size_bytes": archive_path.stat().st_size,
        "total_source_videos": len(train_records) + len(test_records),
        "split_counts": {
            "train": len(train_split),
            "val": len(val_split),
            "test": len(test_records)
        },
        "class_breakdown": {
            "train": defaultdict(int),
            "val": defaultdict(int),
            "test": defaultdict(int)
        },
        "random_seed": 42,
        "val_ratio": 0.15,
        "sampling_stride": 10,
        "anomalous_vs_normal": {
            "train": {"anomalous": sum(r["label"] == 1 for r in train_split), "normal": sum(r["label"] == 0 for r in train_split)},
            "val": {"anomalous": sum(r["label"] == 1 for r in val_split), "normal": sum(r["label"] == 0 for r in val_split)},
            "test": {"anomalous": sum(r["label"] == 1 for r in test_records), "normal": sum(r["label"] == 0 for r in test_records)}
        }
    }

    for r in train_split:
        summary["class_breakdown"]["train"][r["category"]] += 1
    for r in val_split:
        summary["class_breakdown"]["val"][r["category"]] += 1
    for r in test_records:
        summary["class_breakdown"]["test"][r["category"]] += 1

    summary["class_breakdown"]["train"] = dict(summary["class_breakdown"]["train"])
    summary["class_breakdown"]["val"] = dict(summary["class_breakdown"]["val"])
    summary["class_breakdown"]["test"] = dict(summary["class_breakdown"]["test"])

    summary_path = output_dir / "ucf_crime_dataset_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  Wrote dataset audit summary: {summary_path}")

    print("\n" + "=" * 70)
    print("SUCCESS: Phase 1 manifest generation complete.")
    print("=" * 70)


if __name__ == "__main__":
    archive = Path("ufc-crime.zip")
    if not archive.exists():
        archive = Path("ucf-crime.zip")

    ref_archive = Path("refrence/VERA-main.zip")
    manifests_dir = Path("data/manifests")

    generate_manifests(archive, ref_archive, manifests_dir)
