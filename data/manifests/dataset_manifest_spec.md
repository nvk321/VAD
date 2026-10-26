# Dataset Manifest Specification & Stratification Protocol

## 1. Overview
This directory contains version-controlled, machine-readable manifest files generated directly from the local UCF-Crime archive (`ufc-crime.zip`) without unpacking raw media into Git.

---

## 2. Manifest Files Overview

| File | Type | Video Count | Description |
|---|---|---|---|
| `ucf_crime_train.csv` | CSV | 1,369 | Training partition (85% of training data, stratified) |
| `ucf_crime_val.csv` | CSV | 241 | Validation partition (15% of training data, stratified) |
| `ucf_crime_test.csv` | CSV | 290 | Canonical benchmark test partition |
| `ucf_crime_test_annotations.json` | JSON | 290 | Canonical temporal anomaly boundaries `[start, end]` for test videos |
| `ucf_crime_dataset_summary.json` | JSON | — | Machine-readable audit summary, class distributions, and checksums |

---

## 3. CSV Manifest Schema

| Field Name | Type | Description | Example |
|---|---|---|---|
| `video_id` | string | Canonical source video identifier | `Abuse028_x264` |
| `category` | string | Anomaly crime class or `NormalVideos` | `Abuse` |
| `label` | int | Binary weak label: `0` = Normal, `1` = Anomaly | `1` |
| `subset` | string | Dataset subset partition: `train`, `val`, `test` | `train` |
| `num_sampled_frames` | int | Total sampled PNG image frames available | `141` |
| `frame_start` | int | Zero-based index of first sampled frame | `0` |
| `frame_end` | int | Zero-based index of last sampled frame | `1400` |
| `frame_stride` | int | Sampling interval between adjacent frames | `10` |
| `archive_path` | string | Relative path within `ufc-crime.zip` | `Train/Abuse/Abuse028_x264` |

---

## 4. Stratification & Data Splitting Protocol

### 4.1 Strict Test Set Quarantine
- The canonical benchmark testing set consists of **290 videos** (140 anomalous videos across 13 crime classes + 150 normal videos).
- **The test set was completely quarantined** during validation split generation.
- Test-set annotations will NEVER be used to calibrate risk thresholds, tune TCN hyperparameters, or select smoothing parameters.

### 4.2 Training vs. Validation Split
- The validation split is drawn **solely from the 1,610 training videos**.
- Stratification is performed category-by-category across all 14 classes (13 crime categories + NormalVideos) using a deterministic random seed: `seed = 42`.
- Validation ratio: $15\%$ ($\approx 241$ videos).
- Resulting distribution:
  - **Train Split (1,369 videos)**: 689 anomalous videos, 680 normal videos ($\approx 50.3\%$ anomalous).
  - **Validation Split (241 videos)**: 121 anomalous videos, 120 normal videos ($\approx 50.2\%$ anomalous).
  - Every single crime category is proportionally represented in both Train and Val.

### 4.3 Zero Data Leakage Guarantee
Verified mathematically and confirmed by unit tests in `tests/test_manifests.py`:
$$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$$
$$\text{Total Videos} = |\text{Train}| + |\text{Val}| + |\text{Test}| = 1369 + 241 + 290 = 1900$$

---

## 5. Temporal Ground-Truth Test Annotations Schema
In `ucf_crime_test_annotations.json`:
```json
{
  "Abuse028_x264": {
    "category": "Abuse",
    "temporal_spans": [
      [165, 240]
    ],
    "is_anomalous": true,
    "total_frames": 1412
  }
}
```
- `temporal_spans`: List of `[start_frame, end_frame]` intervals indicating active anomaly duration in native 30 fps frames.
- Verified against canonical Sultani et al. (CVPR 2018) benchmark specifications.
