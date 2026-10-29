# Real-Time Video Anomaly Detection and Risk Assessment Using Deep Learning

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x%20CUDA-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Hardware: 4GB GPU](https://img.shields.io/badge/Target%20GPU-RTX%203050%20(4GB)-green.svg)]()

> **Undergraduate Deep Learning Research Project**  
> *A practical, computationally feasible integration of weakly supervised video anomaly detection, temporal modeling, event formation, and a transparent risk-oriented decision support layer.*

---

## 1. Problem Statement & Motivation
Automated video surveillance systems in modern public and commercial spaces continuously generate massive video volumes. Manual monitoring by human operators suffers from severe cognitive fatigue, high operational costs, and missed rare anomalies. 

While existing Video Anomaly Detection (VAD) algorithms can identify whether a short video segment is anomalous, standard systems exhibit two major operational shortcomings:
1. **Alert Fragmentation**: Instantaneous snippet anomaly scores fluctuate rapidly above and below detection thresholds, generating noisy on-off alarm bursts (`ALERT -> NORMAL -> ALERT`) that overwhelm security personnel.
2. **Absence of Actionable Prioritization**: Standard models produce raw probability values without incident-level context. Security teams need to know whether an anomaly is persistent enough to form a cohesive event and how urgently it should be investigated.
3. **No Ground-Truth Risk Labels**: Benchmark datasets (e.g., UCF-Crime) do **not** provide ground-truth severity labels (Low/Medium/High). Training a supervised classifier on arbitrary labels is mathematically invalid.

This project addresses these gaps by coupling weakly supervised anomaly detection with temporal persistence and a transparent, deterministic risk decision support layer.

---

## 2. Central Research Question
> **"Can a weakly supervised video anomaly detection system be extended with temporal persistence and a transparent risk-oriented decision layer to produce more stable and actionable incident-level alerts, without requiring additional ground-truth risk labels?"**

---

## 3. The Three-Task Architecture

```text
[ Task 1: Detection ]
Is the video snippet anomalous?
Output: Continuous snippet anomaly score s_t ∈ [0.0, 1.0]
       ↓
[ Task 2: Event Formation ]
Does the anomaly persist across time to constitute a meaningful event?
Output: Formed temporal event envelope E_k = (t_start, t_end, Δt)
       ↓
[ Task 3: Risk Prioritization ]
How should the detected event be prioritized for operational action?
Output: Deterministic risk score R and alert level (Low / Medium / High)
```

```text
Surveillance Video
        │
        ▼
Snippet Windowing (16 / 32 frames)
        │
        ▼
Frozen Pretrained Backbone (I3D / S3D / R(2+1)D-18)  ──►  Cached Features (data/features/)
        │
        ▼
Dilated Causal Temporal ConvNet (TCN)  (Strictly causal, zero future leakage)
        │
        ▼
Weakly Supervised MIL Anomaly Head  (Max-margin ranking loss + smoothness + sparsity)
        │
        ▼  Raw Anomaly Scores s_t ∈ [0, 1]
Temporal Smoothing Filter  (EMA / Moving Average / Gaussian)
        │
        ▼  Smoothed Scores s_t^(smooth)
Persistence & Event Formation  (Dual-threshold hysteresis + min duration τ_min)
        │
        ▼  Active Incidents (t_start, t_end, Δt, s_max)
Deterministic Risk Decision Layer  (Calibrated on Train/Val only)
        │
        ▼
Prioritized Incident Stream  (Low / Medium / High Alert)
```

---

## 4. Key Engineering & Research Principles
- **Two-Stage Decoupled Design**: Feature extraction using the deep 3D video backbone is decoupled from temporal sequence training. Features are cached to disk, enabling TCN + MIL training in under 15 minutes with **<1.0 GB VRAM** usage on our **4 GB RTX 3050 Laptop GPU**.
- **Transparent Deterministic Risk**: Risk priority is computed from peak anomaly confidence and temporal persistence ($R = w_1 s_{max} + w_2 \phi(\Delta t)$). It is an explicit, explainable operational decision aid—not an unverified "deep severity classifier".
- **Strict Scientific Honesty**: No fake results. Metrics are clearly demarcated as `TARGET`, `EXPECTED`, or `MEASURED`.

---

## 5. Dataset Strategy
- **Primary Benchmark**: **UCF-Crime** (1,900 surveillance videos across 13 crime classes and normal activities).
  - *Training*: 1,610 videos with video-level binary weak supervision.
  - *Testing*: 290 videos with frame-level ground-truth temporal boundaries.
  - *Local Data*: Evaluated from local archive `ufc-crime.zip` (11.8 GB, 1,377,653 PNG frames sampled at 10-frame intervals).
- **Secondary Benchmark**: **XD-Violence** (4,754 multi-modal videos; evaluated for cross-dataset robustness in Exp F).
- *Data Policy*: Raw video frames, zip archives, and extracted features are strictly ignored by Git. Reproducibility is maintained via version-controlled manifests in `data/manifests/`.

---

## 6. Project Structure

```text
c:\Projects\VAD\
├── .gitignore
├── pyproject.toml              # Modern Python dependency specifications
├── AGENTS.md                   # Agent system laws and operating guidelines
├── PROJECT_SPEC.md             # Formal research framing, problem, and gaps
├── ARCHITECTURE.md             # In-depth system architecture and math
├── RESEARCH_DECISIONS.md       # ADRs: Frozen Decisions vs Validated Parameters
├── DATASET_GUIDE.md            # Dataset analysis, provenance, and layouts
├── EXPERIMENT_PLAN.md          # 6-stage experimental hierarchy (Exp A–F)
├── IMPLEMENTATION_PLAN.md      # Phased engineering roadmap (Phase 0–8)
├── PROJECT_STATUS.md           # Live progress tracker and milestone ledger
├── CHANGELOG.md                # Release and modification history
├── CONTRIBUTING.md             # Development and coding standards
├── README.md                   # This document
├── configs/                    # YAML configuration files
│   ├── base.yaml               # Device, seeds, logging
│   ├── data.yaml               # Paths, snippet lengths, manifests
│   ├── model.yaml              # Backbone, TCN, MIL head architecture
│   ├── training.yaml           # Optimizers, learning rate, loss weights
│   └── evaluation.yaml         # Hysteresis thresholds, risk weights
├── data/
│   ├── raw/                    # (Untracked) Raw frames / archive mount
│   ├── interim/                # (Untracked) Temporary clips
│   ├── processed/              # (Untracked) Standardized video snippets
│   ├── features/               # (Untracked) Cached feature tensors (*.pt)
│   └── manifests/              # (Tracked) Train/Val/Test CSV manifests
├── src/                        # Modular source code
│   ├── data/                   # Dataset loaders and transforms
│   ├── models/                 # Backbone wrappers, TCN, MIL head
│   ├── postprocessing/         # Smoothing and hysteresis event formation
│   ├── risk/                   # Deterministic decision support layer
│   ├── evaluation/             # Frame ROC/PR and event-level metrics
│   └── utils/                  # Configuration, logging, seeding
├── scripts/                    # Standalone execution scripts
│   ├── data/                   # Verification and manifest generation
│   ├── train.py                # Model training pipeline
│   ├── evaluate.py             # Evaluation on test partitions
│   ├── infer.py                # Single-video inference
│   └── benchmark.py            # Latency, FPS, and VRAM profiling
└── tests/                      # Unit and integration tests
```

---

## 7. Quickstart & Environment Setup

### 7.1 Prerequisites
- Windows 11 (or Linux)
- Python 3.11.0
- NVIDIA GPU with CUDA support (tested on RTX 3050 Laptop GPU, 4 GB VRAM)

### 7.2 Installation
```powershell
# 1. Clone repository
git clone https://github.com/your-username/VAD.git
cd VAD

# 2. Create Python 3.11 virtual environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Install PyTorch with CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 7.3 Launching the Interactive Demonstration Web Application
```powershell
# Launch the multi-tab Gradio surveillance console
python app.py
```
Open **`http://127.0.0.1:7860`** in your browser.

- **Tab 1: Live Anomaly Detection**: Upload any `.mp4`/`.avi`/`.mov` or select a preset demo (`Robbery037`, `Normal_Videos008`, `Arson015`), toggle operational HUD, view progressive timeline, and download H.264 annotated video.
- **Tab 2: Benchmark Exploration**: Rapid evaluation of UCF-Crime test set features across 13 crime categories (< 50ms latency).
- **Tab 3: Research Dashboard**: Ablation tables, ROC/PR curves, and paper-ready comparison artifacts.
- **Tab 4: Architecture & Viva Guide**: Comprehensive methodology walkthrough, mathematical derivations, and live subsystem health check.

For operational and viva defense details, see [docs/DEMO_GUIDE.md](file:///C:/Projects/VAD/docs/DEMO_GUIDE.md) and [docs/DEMO_TEST_MATRIX.md](file:///C:/Projects/VAD/docs/DEMO_TEST_MATRIX.md).

### 7.4 Verifying Dataset Archive
```powershell
# Verify the integrity of the local UCF-Crime zip archive
python scripts/data/verify_ucf_crime.py
```

---

## 8. Experimental Roadmap
- **Experiment A**: Baseline Weakly Supervised MIL (I3D + MLP Head).
- **Experiment B**: Baseline + Dilated Causal TCN (Temporal sequence modeling ablation).
- **Experiment C**: TCN + Temporal Persistence & Event Formation (Alert stability ablation).
- **Experiment D**: Full Pipeline + Calibrated Risk Decision Layer (Incident prioritization).
- **Experiment E**: Hardware Efficiency Profiling (FPS, latency, peak VRAM on RTX 3050).
- **Experiment F**: Secondary Cross-Dataset Evaluation on XD-Violence.

---

## 9. License & Academic Integrity
This project is licensed under the MIT License. Reference materials in `refrence/` belong to their respective authors and are used solely for academic inspection.
