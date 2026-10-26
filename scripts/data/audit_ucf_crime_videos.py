"""Deep Audit Script for UCF-Crime Local Archive.

Inspects video-level organization, source video counts, canonical video ID mapping,
frame sampling stride, and anomaly/normal distribution inside ufc-crime.zip.
"""

from collections import defaultdict
from pathlib import Path
import re
import sys
import zipfile


def audit_ucf_crime(archive_path: Path) -> dict:
    """Perform a rigorous video-level audit of the UCF-Crime archive.

    Args:
        archive_path: Path to the ufc-crime.zip archive.

    Returns:
        dict: Detailed audit results.
    """
    if not archive_path.is_file():
        print(f"ERROR: Archive not found at {archive_path}")
        return {}

    print(f"Opening archive: {archive_path.resolve()} ({archive_path.stat().st_size / (1024**3):.2f} GiB)")
    video_frames = defaultdict(lambda: defaultdict(list))
    malformed_filenames = []
    sampling_strides = defaultdict(int)

    with zipfile.ZipFile(archive_path, "r") as zf:
        namelist = zf.namelist()
        total_entries = len(namelist)
        print(f"Read {total_entries:,} entries from central directory.")

        for name in namelist:
            parts = name.split("/")
            if len(parts) >= 3 and parts[2].endswith(".png"):
                part = parts[0]  # 'Train' or 'Test'
                cat = parts[1]   # Category name
                fn = parts[2]    # Filename e.g. Abuse028_x264_10.png

                m = re.match(r"^(.*)_(\d+)\.png$", fn)
                if m:
                    video_id = m.group(1)
                    frame_idx = int(m.group(2))
                    video_frames[part][(cat, video_id)].append(frame_idx)
                else:
                    malformed_filenames.append(name)

    print("\n" + "=" * 70)
    print("UCF-CRIME DETAILED VIDEO-LEVEL AUDIT REPORT")
    print("=" * 70)

    audit_summary = {}

    for part in ["Train", "Test"]:
        vids = video_frames[part]
        categories = set(c for c, _ in vids.keys())
        normal_vids = [v for (c, v) in vids.keys() if "Normal" in c]
        anomaly_vids = [v for (c, v) in vids.keys() if "Normal" not in c]

        print(f"\n[Partition: {part}]")
        print(f"  Total Unique Videos Represented : {len(vids):,}")
        print(f"  - Anomalous Videos              : {len(anomaly_vids):,} (across {len(categories - {'NormalVideos'})} crime classes)")
        print(f"  - Normal Videos                 : {len(normal_vids):,}")
        print(f"  Total Extracted Frames          : {sum(len(f) for f in vids.values()):,}")

        # Class breakdown
        class_video_counts = defaultdict(int)
        class_frame_counts = defaultdict(int)
        for (c, _), frames in vids.items():
            class_video_counts[c] += 1
            class_frame_counts[c] += len(frames)

        print("\n  Class Breakdown (Videos / Frames):")
        for cls in sorted(class_video_counts.keys()):
            v_cnt = class_video_counts[cls]
            f_cnt = class_frame_counts[cls]
            avg_f = f_cnt / v_cnt if v_cnt > 0 else 0
            print(f"    - {cls:<16}: {v_cnt:>4} videos | {f_cnt:>8,} frames | ~{avg_f:>6.1f} frames/vid")

        # Frame stride audit
        for (c, v), frames in list(vids.items())[:20]:
            frames_sorted = sorted(frames)
            if len(frames_sorted) >= 2:
                for i in range(len(frames_sorted) - 1):
                    diff = frames_sorted[i + 1] - frames_sorted[i]
                    sampling_strides[diff] += 1

        audit_summary[part] = {
            "total_videos": len(vids),
            "anomaly_videos": len(anomaly_vids),
            "normal_videos": len(normal_vids),
            "classes": sorted(categories),
            "class_video_counts": dict(class_video_counts),
        }

    print("\n[Frame Sampling Pattern Analysis]")
    top_strides = sorted(sampling_strides.items(), key=lambda x: x[1], reverse=True)[:5]
    for stride, count in top_strides:
        print(f"  Stride of {stride} frames: observed {count:,} times")

    print(f"\n[Filename & Video-ID Consistency]")
    print(f"  Malformed filenames: {len(malformed_filenames)}")
    if malformed_filenames:
        print(f"  Sample malformed: {malformed_filenames[:3]}")
    else:
        print("  100% of PNG files conform to canonical pattern: <Video_ID>_<frame_index>.png")

    print("\n" + "=" * 70)
    return audit_summary


if __name__ == "__main__":
    archive = Path("ufc-crime.zip")
    if not archive.exists():
        archive = Path("ucf-crime.zip")

    audit_summary = audit_ucf_crime(archive)
    if not audit_summary:
        sys.exit(1)
