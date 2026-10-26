# DATASET_GUIDE.md — Dataset Strategy, Provenance & Management

## 1. Dataset Strategy & Selection Principles
In accordance with academic research standards, the dataset strategy adheres to the following decision rule:
> **"Use the minimum number of datasets necessary to make the research claim convincing and defensible."**

We do not indiscriminately download dozens of gigabytes of external data. Datasets are evaluated across 7 rigorous dimensions:
1. Research relevance to real-world CCTV surveillance.
2. Suitability for Weakly Supervised Multiple Instance Learning (MIL).
3. Annotation quality and native ground-truth structure.
4. Official academic availability and licensing.
5. Reproducibility and community benchmark comparability.
6. Local computational feasibility under a 4 GB GPU budget.
7. Contribution to specific experimental hypotheses.

---

## 2. Dataset Comparison & Candidate Evaluation Matrix

| Dataset | Modality | Scale / Hours | Supervision in Literature | Native Risk Labels? | Role in Project | Access / Status |
|---|---|---|---|---|---|---|
| **UCF-Crime** | RGB (CCTV) | 1,900 videos / 128 hrs | Weak (Video-level Train, Frame-level Test) | ❌ None (Binary) | **Primary Benchmark (Exp A–E)** | **Locally Available** (`ufc-crime.zip`, 11.8 GB) |
| **XD-Violence** | RGB + Audio | 4,754 videos / 212 hrs | Weak (Video-level Train, Frame-level Test) | ❌ None (Binary) | **Secondary Evaluation (Exp F)** | Evaluated (Optional expansion) |
| **ShanghaiTech** | RGB (Static) | 437 videos / 13 scenes | Unsupervised (Normal-only Train) | ❌ None | Localization Reference only | Not selected for primary MIL |
| **UBnormal** | Synthetic | 543 videos / 29 virtual scenes | Unsupervised / Open-set | ❌ None | Optional stretch | Deferred |
| **MSAD** | Multi-Scene | Varied | Multi-domain | ❌ None | Optional stretch | Deferred |

---

## 3. Primary Dataset: UCF-Crime Deep Dive

### 3.1 Dataset Overview
- **Citation**: Sultani, W., Chen, C., & Shah, M. *"Real-world Anomaly Detection in Surveillance Videos."* IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.
- **Classes**: 13 Real-world Crime Categories + Normal Activities:
  - *Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting, RoadAccidents, Robbery, Shooting, Shoplifting, Stealing, Vandalism, and NormalVideos*.
- **Standard Benchmark Split**:
  - **Training Set**: 1,610 videos (810 anomalous videos across 13 classes, 800 normal videos). Only video-level binary labels are used during training.
  - **Testing Set**: 290 videos (140 anomalous videos with frame-level temporal anomaly start/end annotations, 150 normal videos).
- **Standard Metric**: Frame-level Area Under the Receiver Operating Characteristic Curve (ROC-AUC) and Average Precision (PR-AUC).

### 3.2 Inspection of Local Archive (`ufc-crime.zip`)
- **Location**: `c:\Projects\VAD\ufc-crime.zip`
- **File Size**: 11,827,280,807 bytes (~11.01 GiB / 11.8 GB).
- **Internal Contents**: Total of **1,377,653 PNG image files**.
- **Directory Layout**:
  ```text
  Test/
    Abuse/            (e.g., Abuse028_x264_0.png, Abuse028_x264_10.png, ...)
    Arrest/, Arson/, Assault/, Burglary/, Explosion/, Fighting/, NormalVideos/
    RoadAccidents/, Robbery/, Shooting/, Shoplifting/, Stealing/, Vandalism/
  Train/
    Abuse/, Arrest/, Arson/, Assault/, Burglary/, Explosion/, Fighting/, NormalVideos/
    RoadAccidents/, Robbery/, Shooting/, Shoplifting/, Stealing/, Vandalism/
  ```
- **Sampling Rate**: Frames are pre-extracted and indexed at **10-frame intervals** (e.g., `_0.png`, `_10.png`, `_20.png`), representing an effective sampling rate of 3.0 fps from native 30 fps surveillance video.
- **Git Policy**: `ufc-crime.zip` and all extracted image frames are strictly excluded via `.gitignore`.

---

## 4. Inspection of Reference Archive & Annotation Provenance

### 4.1 Reference Archive Overview
- **Location**: `c:\Projects\VAD\refrence\VERA-main.zip` (244 KB).
- **Contents**: Research repository snapshot from the CVPR 2025 paper (*VERA: Explainable Video Anomaly Detection via Verbalized Learning of Vision-Language Models*).
- **Internal Files in `VERA-main/Data/`**:
  - `UCF_Eval.json`: 290 test video metadata entries with temporal ground-truth boundaries `[start_frame, end_frame, ...]` and video lengths.
  - `UCF_Instruct_train.json`: 1,449 train video metadata entries.
  - `UCF_Instruct_val.json`: 161 validation video metadata entries.
  - `name_gt_pair.json`: 800 test video frame-level ground-truth arrays for XD-Violence.
  - `xd_test_800.json`, `xd_train.json`, `xd_val.json`.

### 4.2 Annotation Provenance & Separation Policy
> [!IMPORTANT]
> **Strict Policy on External Reference Annotations**:
> 1. The primary dataset is the local **UCF-Crime** archive.
> 2. Ground-truth temporal boundaries for UCF-Crime testing originate from the official Sultani et al. (CVPR 2018) `Temporal_Anomaly_Annotation_for_Testing_Videos.txt`.
> 3. The `UCF_Eval.json` file inside `VERA-main.zip` is an external formatting of these canonical temporal boundaries.
> 4. We do NOT treat auxiliary verbalized instructions or VLM prompt files from `VERA-main` as ground truth.
> 5. We generate our own standalone, clean, verified JSON/CSV manifests in `data/manifests/` directly from verifiable benchmark annotations. The project does NOT rely on or import from `refrence/` in production code.

---

## 5. Industrial Data Management Architecture

To guarantee complete reproducibility without bloating the Git repository:

```text
data/
├── raw/               <-- Untracked: Local archive mount or extracted frames
├── interim/           <-- Untracked: Resized clips or sampled snippet sequences
├── processed/         <-- Untracked: Standardized video snippet tensors
├── features/          <-- Untracked: Extracted feature caches (*.pt / *.npy)
│   ├── ucf_crime_i3d_train/
│   └── ucf_crime_i3d_test/
└── manifests/         <-- TRACKED IN GIT: Lightweight metadata and split definitions
    ├── ucf_crime_train.csv
    ├── ucf_crime_val.csv
    ├── ucf_crime_test.csv
    └── ucf_crime_test_annotations.json
```

---

## 6. Manifest Generation & Verification Pipeline
In Phase 1, data verification and manifest generation will run as follows:
1. `scripts/data/verify_ucf_crime.py`: Safely checks archive integrity, validates file counts, and ensures zero missing video categories without extracting the entire 11.8 GB archive.
2. `scripts/data/generate_manifests.py`: Parses video IDs, class labels, frame indices, and ground-truth temporal boundaries, generating deterministic, version-controlled CSV manifests in `data/manifests/`.
