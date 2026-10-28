# CHANGELOG.md — Project Version & Modification Ledger

All notable changes, architectural decisions, and experiment milestones will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), adhering to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned for Phase 8 / Post-Capstone Extensions
- Final frozen test set evaluation (290 UCF-Crime benchmark test videos) once defense protocol is locked.
- Cross-dataset generalization benchmark on XD-Violence features (Exp F).

---

## [0.8.0] - 2026-10-28 — Phase 7: Efficiency Benchmark, Web Demonstration & Paper Artifacts (Exp E)

### Added
- **Hardware Efficiency Benchmark** (`scripts/benchmark.py`):
  - Benchmarked local NVIDIA GeForce RTX 3050 Laptop GPU (300 iterations).
  - S3D Backbone latency: 17.45 ms/snippet (57.3 clips/s).
  - TCN + MIL scoring latency: 0.0916 ms/snippet (10,914 snippets/s).
  - Complete End-to-End Latency: **20.23 ms/snippet**, **791.1 effective video FPS**, Peak VRAM: **1.08 GB** (27% of 4.0 GB budget).
- **Interactive Demonstration Web Interface** (`app.py`):
  - Built with Gradio; includes preloaded samples from all 13 crime categories and normal videos.
  - Interactive parameter controls for smoothing factor ($\alpha$), trigger thresholds ($T_{high}, T_{low}$), and persistence ($\tau_{min}$).
  - Live continuous anomaly timeline visualization, detected event table, and operational triage alert badge.
- **Inference & Triage Command-Line Utility** (`scripts/infer_video.py`):
  - Evaluates both TCN and MIL models on cached features, applies EMA smoothing, dual-threshold hysteresis, event formation, and deterministic risk scoring.
  - Generates Rich terminal incident table and publication-quality timeline plots.
- **Publication-Ready Figures & LaTeX Tables** (`scripts/generate_paper_artifacts.py`):
  - Generated high-resolution figures in `experiments/paper_artifacts/plots/`.
  - Generated LaTeX tables (`ablation.tex`, `results.tex`, `efficiency.tex`) and formal `summary_report.md`.

---

## [0.7.0] - 2026-10-28 — Phase 6: Transparent Deterministic Risk Decision Layer (Exp D)

### Added
- **Deterministic Risk Decision Layer** (`src/risk/decision_layer.py`):
  - Formulates operational risk score $R = w_{mag} \cdot s_{max} + w_{dur} \cdot \phi(\Delta t)$ ($w_{mag}=0.65, w_{dur}=0.35$).
  - Maps continuous score to transparent operational triage levels: Normal ($R < 0.25$), Low Risk ($0.25 \le R < 0.45$), Medium Risk ($0.45 \le R < 0.70$), High Risk ($R \ge 0.70$).
  - Evaluated on 241 validation videos: 174 Normal (72.2%), 14 Medium Risk (5.8%), 53 High Risk (22.0%), 0 Low Risk.
  - Produces human-readable explainability strings detailing exact mathematical contributions.
- **Automated Risk Unit Tests** (`tests/test_risk.py`):
  - Verified monotonicity, zero-duration baseline behavior, score bounding, and triage classification boundaries.

---

## [0.6.0] - 2026-10-28 — Phase 5: Temporal Persistence & Event Formation (Exp C)

### Added
- **Temporal Postprocessing Filters** (`src/postprocessing/smoothing.py`):
  - Exponential Moving Average (EMA), Moving Average, and Gaussian 1D filter implementations.
- **Event Formation Engine** (`src/postprocessing/event_formation.py`):
  - Dual-threshold hysteresis ($T_{high}=0.60, T_{low}=0.40$).
  - Temporal persistence filter ($\tau_{min}=1.5s$) and gap bridging ($\tau_{gap}=1.2s$).
- **Persistence Ablation Benchmark** (`scripts/evaluate_persistence_risk.py`):
  - Measured across 241 validation videos: reduced fragmented alert blocks from 131 to 74 consolidated incidents (**43.5% alert fragmentation reduction**), eliminating 193 chattering boundary toggles.
- **Automated Event Formation Tests** (`tests/test_event_formation.py`):
  - Validated smoothing filters, hysteresis state machine, gap bridging, and event aggregation.

---

## [0.5.0] - 2026-10-28 — Phase 4: Dilated Causal Temporal Convolutional Network (Exp B)

### Added
- **Causal Temporal Architecture** (`src/models/tcn.py`):
  - Implemented `CausalConv1d` with explicit left-padding $(K-1) \cdot d$ to prevent future leakage.
  - Implemented `ChannelLayerNorm` for sequence-causal and batch-size invariant normalization.
  - Built `TemporalResidualBlock` with dilated causal convolutions ($d \in [1, 2, 4]$) and `TCNModel`.
- **TCN Training & Controlled Ablation** (`scripts/train_tcn.py`):
  - Trained on balanced stratified UCF-Crime training set and evaluated on 241 validation videos.
  - **Val ROC-AUC: 0.8750** (Epoch 18), **Val PR-AUC (AP): 0.8705**.
  - **+4.97% absolute ROC-AUC gain** over baseline MIL head.
  - Peak VRAM: 44.1 MB. Checkpoints saved to `experiments/exp_B_tcn/checkpoints/`.
- **Strict Causality Verification Tests** (`tests/test_tcn.py`):
  - Validated mathematical zero future leakage (0.00000000 score difference when future inputs are perturbed).

---

## [0.4.0] - 2026-10-28 — Phase 3: Weakly Supervised MIL Baseline (Exp A)

### Added
- **Weakly Supervised MIL Baseline Architecture** (`src/models/mil_head.py`):
  - Canonical 3-layer MLP anomaly scoring head (1024-d $\to$ 512 $\to$ 64 $\to$ 1) with Dropout (0.6).
- **Ranking Loss with Regularization** (`src/losses/ranking_loss.py`):
  - Hinge ranking loss maximizing bag anomaly score margin, temporal smoothness ($\lambda_{smooth}=8\times 10^{-5}$), and sparsity ($\lambda_{sparse}=8\times 10^{-5}$).
- **Resumable Training Pipeline & Telemetry** (`scripts/train_mil.py`, `src/utils/experiment_tracker.py`):
  - Automatic checkpointing (`best_auc.pt`, `best_pr_auc.pt`, `last.pt`), TensorBoard telemetry, Rich terminal progress, and metrics CSV exports.
  - Executed smoke test (`experiments/smoke_test_mil/`) and full Experiment A.
  - **Val ROC-AUC: 0.8253**, **Val PR-AUC: 0.8236**, Peak VRAM: 43.8 MB, Epoch time: 0.50 s.
- **Automated MIL Tests** (`tests/test_mil.py`, `tests/test_ranking_loss.py`, `tests/test_training.py`, `tests/test_experiment_logging.py`):
  - 100% test coverage for shapes, gradient flows, hinge margins, and checkpoint resume.

---

## [0.3.0] - 2026-10-27 — Phase 2: Pretrained Backbone & Feature Extraction Pipeline

### Added
- **Backbone Benchmarking & Validation**:
  - `scripts/benchmark_backbone.py`: Empirical GPU benchmark on NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM).
  - Validated S3D (1024-d): 1084.5 MB peak VRAM (26% budget), 17.45 ms/clip latency, 57.3 clips/sec throughput.
  - Validated R(2+1)D-18 (512-d): 972.0 MB peak VRAM, 27.66 ms/clip latency, 36.1 clips/sec throughput.
- **Modular Backbone Interface**:
  - `src/models/backbone.py`: Unified `VideoBackbone` wrapper with architectural parameter freezing (`requires_grad = False`), input spatial dimension management, and dynamic feature dimensionality.
- **Resumable Feature Extraction Pipeline**:
  - `scripts/extract_features.py`: In-memory frame extraction from `ufc-crime.zip`, bilinear resize to 224x224, temporal chunking into 16-frame snippets, automatic GPU batching with OOM handling and dynamic batch halving.
  - Resumable state persistence via `extraction_manifest.json` and Rich terminal progress visualization.
  - Validated on 5-video validation smoke test (100% success, 0 corrupt, 8.9s elapsed, instant resume verified).
- **Feature Dataset Loaders**:
  - `src/data/feature_dataset.py`: `CachedFeatureDataset` for sequential evaluation, `interpolate_segments` for 32-segment fixed temporal pooling (Sultani protocol), and `MILPairDataset` for balanced abnormal/normal training pair generation.
- **Automated Feature Test Suite**:
  - `tests/test_features.py`: Unit tests validating `.pt` payload keys, non-NaN/non-Inf checks, segment interpolation properties, dataset loader behavior, and backbone instantiation. Passing 100%.

---

## [0.2.0] - 2026-10-26 — Phase 1: Dataset Validation & Manifest Generation

### Added
- **Dataset Audit Script**:
  - `scripts/data/audit_ucf_crime_videos.py`: Discovered and validated all 1,900 source videos from `ufc-crime.zip`.
- **Reproducible Dataset Manifests** (`data/manifests/`):
  - `ucf_crime_train.csv`: 1,369 videos (689 anomalous, 680 normal).
  - `ucf_crime_val.csv`: 241 videos (121 anomalous, 120 normal), stratified by category with seed 42.
  - `ucf_crime_test.csv`: 290 benchmark test videos (140 anomalous, 150 normal).
  - `ucf_crime_test_annotations.json`: Canonical temporal anomaly boundaries `[start, end]` for test evaluation.
  - `ucf_crime_dataset_summary.json`: Machine-readable audit report and checksums.
  - `dataset_manifest_spec.md`: Formal specification of schema, stratification protocol, and zero-leakage guarantee.
- **Manifest Test Suite**:
  - `tests/test_manifests.py`: Automated tests verifying manifest existence, row counts, 100% label consistency, zero data leakage ($\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$), and ground-truth temporal validity.

---

## [0.1.0] - 2026-10-26 — Phase 0: Scaffolding, Research Validation & Architecture Freeze

### Added
- **Core Governance & Research Documents**:
  - `AGENTS.md`: Operating guidelines, honest academic positioning, and source-of-truth hierarchy.
  - `PROJECT_SPEC.md`: Formal problem specification, academic logic, research question, 4 research gaps, and 3-task distinction (Detection, Event Formation, Risk Prioritization).
  - `ARCHITECTURE.md`: Two-stage decoupled pipeline, dilated causal TCN, MIL ranking loss, hysteresis event formation, and deterministic risk decision layer.
  - `RESEARCH_DECISIONS.md`: Formal ADRs establishing Frozen Architectural Decisions (ADR-001 through ADR-007) and Validated Implementation Parameters (VIP-001 through VIP-006).
  - `DATASET_GUIDE.md`: In-depth analysis of UCF-Crime (primary), XD-Violence (secondary), ShanghaiTech, annotation provenance, and data directory architecture.
  - `EXPERIMENT_PLAN.md`: 6-stage experimental hierarchy (Exp A through Exp F), metrics definitions, and benchmark tracking table with strict "no fake results" labeling.
  - `IMPLEMENTATION_PLAN.md`: Chronological 9-phase roadmap from scaffolding to interactive demo.
  - `PROJECT_STATUS.md`: Live milestone tracker and current state ledger.
  - `CONTRIBUTING.md` and `README.md`.
- **Configuration Hierarchy**:
  - `configs/base.yaml`: Random seeds, device auto-selection, logging configuration.
  - `configs/data.yaml`: Dataset paths, snippet parameters, split definitions.
  - `configs/model.yaml`: Backbone options, TCN dimensions, MIL head architecture.
  - `configs/training.yaml`: Optimizer, learning rates, ranking loss margin, sparsity/smoothness weights.
  - `configs/evaluation.yaml`: Frame-level and event-level metric configs, hysteresis thresholds, risk weights.
- **Project Structure**:
  - Created `.gitignore` excluding all raw data, archives, checkpoints, features, and reference files.
  - Created modular Python packages in `src/` (`data/`, `models/`, `postprocessing/`, `risk/`, `evaluation/`, `utils/`).
  - Created execution script structure in `scripts/` (`data/`, `train.py`, `evaluate.py`, `infer.py`, `benchmark.py`).
  - Created test suite structure in `tests/`.
  - Created data directory skeleton with `.gitkeep` files in `data/raw/`, `data/interim/`, `data/processed/`, `data/features/`, `data/manifests/`.
- **Dataset & Hardware Validation**:
  - Detected hardware: NVIDIA GeForce RTX 3050 Laptop GPU (4096 MiB VRAM), CUDA 13.1.
  - Verified local archive `ufc-crime.zip` (11.8 GB, 1,377,653 PNG frames across all 14 UCF-Crime categories in Test and Train).
  - Inspected reference archive `refrence/VERA-main.zip` (244 KB) and established strict provenance boundaries.
  - Configured dedicated Python 3.11 virtual environment (`.venv`).
  - Created dataset verification script: `scripts/data/verify_ucf_crime.py`.
