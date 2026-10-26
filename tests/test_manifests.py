"""Unit test suite verifying UCF-Crime manifest integrity, zero data leakage, and label consistency."""

import csv
import json
from pathlib import Path


MANIFESTS_DIR = Path("data/manifests")


def test_manifest_files_exist():
    """Verify that all manifest files and summaries exist and are non-empty."""
    expected_files = [
        "ucf_crime_train.csv",
        "ucf_crime_val.csv",
        "ucf_crime_test.csv",
        "ucf_crime_test_annotations.json",
        "ucf_crime_dataset_summary.json",
    ]
    for fn in expected_files:
        p = MANIFESTS_DIR / fn
        assert p.is_file(), f"Missing manifest file: {p}"
        assert p.stat().st_size > 0, f"Manifest file is empty: {p}"


def test_row_counts_and_partition_sums():
    """Verify exact row counts: Train=1369, Val=241, Test=290 (Total=1900)."""
    with open(MANIFESTS_DIR / "ucf_crime_train.csv", "r", encoding="utf-8") as f:
        train_rows = list(csv.DictReader(f))
    with open(MANIFESTS_DIR / "ucf_crime_val.csv", "r", encoding="utf-8") as f:
        val_rows = list(csv.DictReader(f))
    with open(MANIFESTS_DIR / "ucf_crime_test.csv", "r", encoding="utf-8") as f:
        test_rows = list(csv.DictReader(f))

    assert len(train_rows) == 1369, f"Expected 1369 train rows, got {len(train_rows)}"
    assert len(val_rows) == 241, f"Expected 241 val rows, got {len(val_rows)}"
    assert len(test_rows) == 290, f"Expected 290 test rows, got {len(test_rows)}"
    assert len(train_rows) + len(val_rows) + len(test_rows) == 1900


def test_zero_data_leakage():
    """Verify strictly zero video_id overlap across train, val, and test splits."""
    def get_ids(filename):
        with open(MANIFESTS_DIR / filename, "r", encoding="utf-8") as f:
            return set(r["video_id"] for r in csv.DictReader(f))

    train_ids = get_ids("ucf_crime_train.csv")
    val_ids = get_ids("ucf_crime_val.csv")
    test_ids = get_ids("ucf_crime_test.csv")

    train_val_overlap = train_ids & val_ids
    train_test_overlap = train_ids & test_ids
    val_test_overlap = val_ids & test_ids

    assert len(train_val_overlap) == 0, f"Data leakage between Train and Val: {train_val_overlap}"
    assert len(train_test_overlap) == 0, f"Data leakage between Train and Test: {train_test_overlap}"
    assert len(val_test_overlap) == 0, f"Data leakage between Val and Test: {val_test_overlap}"


def test_label_and_category_consistency():
    """Verify that NormalVideos has label 0 and all 13 crime classes have label 1."""
    for fn in ["ucf_crime_train.csv", "ucf_crime_val.csv", "ucf_crime_test.csv"]:
        with open(MANIFESTS_DIR / fn, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["category"] == "NormalVideos":
                    assert row["label"] == "0", f"Normal video with non-zero label: {row}"
                else:
                    assert row["label"] == "1", f"Anomaly video with non-one label: {row}"
                assert int(row["num_sampled_frames"]) > 0, f"Zero frames in {row}"
                assert int(row["frame_stride"]) == 10, f"Unexpected stride in {row}"


def test_ground_truth_test_annotations():
    """Verify test ground-truth temporal spans structure and validity."""
    gt_path = MANIFESTS_DIR / "ucf_crime_test_annotations.json"
    with open(gt_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) == 290, f"Expected 290 test annotation entries, got {len(data)}"
    for v_id, meta in data.items():
        assert "temporal_spans" in meta, f"Missing temporal spans for {v_id}"
        assert "is_anomalous" in meta, f"Missing anomaly flag for {v_id}"
        for span in meta["temporal_spans"]:
            assert len(span) == 2, f"Invalid span length: {span}"
            start, end = span
            assert start >= 0 and end > start, f"Invalid temporal span range [{start}, {end}] in {v_id}"


if __name__ == "__main__":
    print("Running test_manifest_files_exist()...")
    test_manifest_files_exist()
    print("Running test_row_counts_and_partition_sums()...")
    test_row_counts_and_partition_sums()
    print("Running test_zero_data_leakage()...")
    test_zero_data_leakage()
    print("Running test_label_and_category_consistency()...")
    test_label_and_category_consistency()
    print("Running test_ground_truth_test_annotations()...")
    test_ground_truth_test_annotations()
    print("\nALL MANIFEST VERIFICATION TESTS PASSED SUCCESSFULLY!")
