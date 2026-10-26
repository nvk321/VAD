# CHANGELOG.md — Project Version & Modification Ledger

All notable changes, architectural decisions, and experiment milestones will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), adhering to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned for Phase 3 (Weakly Supervised MIL Baseline)
- Implementation of baseline MIL anomaly head (`src/models/mil_head.py`).
- Implementation of ranking loss with temporal smoothness and sparsity regularization (`src/losses/ranking_loss.py`).
- Training pipeline with TensorBoard logging and model checkpointing (`scripts/train_mil.py`).
- Execution of Experiment A (Baseline MIL evaluation on validation split).

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
