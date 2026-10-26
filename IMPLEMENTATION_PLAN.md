# IMPLEMENTATION_PLAN.md — Phased Engineering Roadmap

## Phase 0: Project Scaffolding, Research Validation & Architecture Freeze
- **Objective**: Establish rock-solid engineering foundations, environment specifications, research decisions (ADRs), dataset strategies, and strict exclusion policies.
- **Deliverables**:
  - [x] Comprehensive `.gitignore` protecting all raw data, archives, features, and checkpoints.
  - [x] Dedicated Python 3.11 virtual environment (`.venv`).
  - [x] Standardized `pyproject.toml` dependency specification.
  - [x] Full suite of project governing documents (`AGENTS.md`, `PROJECT_SPEC.md`, `ARCHITECTURE.md`, `RESEARCH_DECISIONS.md`, `DATASET_GUIDE.md`, `EXPERIMENT_PLAN.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `README.md`).
  - [x] Modular directory skeleton (`configs/`, `data/`, `src/`, `scripts/`, `tests/`).
  - [x] Verification script for `ufc-crime.zip` (`scripts/data/verify_ucf_crime.py`).
  - [x] Architecture Freeze: Frozen Conceptual Pipeline vs. Validated Implementation Parameters.
- **Status**: **IN PROGRESS / FINALIZING REPORT**

---

## Phase 1: Data Pipeline, Verification & Manifest Generation
- **Objective**: Validate the local UCF-Crime dataset archive and generate reproducible, version-controlled metadata manifests without unpacking raw media into Git.
- **Tasks**:
  1. Execute `scripts/data/verify_ucf_crime.py` to confirm archive integrity, frame counts, and all 14 categories.
  2. Implement `scripts/data/generate_manifests.py` to index snippet sequences and parse official ground-truth temporal boundaries for testing into `data/manifests/ucf_crime_test_annotations.json`.
  3. Create deterministic stratified Train/Validation split manifests (`data/manifests/ucf_crime_train.csv` and `data/manifests/ucf_crime_val.csv`) ensuring zero data leakage.
- **Verification**: Run unit tests verifying manifest integrity and frame-range validity.

---

## Phase 2: Upstream Feature Extraction Pipeline (Backbone)
- **Objective**: Extract and cache spatiotemporal snippet representations using a frozen pretrained video backbone, optimized for the 4 GB GPU budget.
- **Tasks**:
  1. Implement `src/models/backbone.py` with support for pretrained I3D (Kinetics-400 RGB) and verified fallback (S3D / R(2+1)D-18).
  2. Implement `src/data/loader.py` for chunked video snippet batching (`torch.no_grad()`, mixed-precision fp16).
  3. Implement `scripts/extract_features.py` to extract and serialize snippet features to `data/features/`.
  4. Perform smoke test profiling extraction throughput (FPS) and peak VRAM allocation (<2.0 GB).
- **Verification**: Verify that output tensors have expected shape $(T, D_{feat})$ and contain valid non-NaN floats.

---

## Phase 3: Baseline Weakly Supervised MIL Model (Experiment A)
- **Objective**: Train and evaluate the canonical weakly supervised MIL baseline on cached features.
- **Tasks**:
  1. Implement `src/models/mil_head.py` (2-layer MLP anomaly scoring head).
  2. Implement `src/models/losses.py` (Max-margin ranking loss, temporal smoothness loss, sparsity penalty).
  3. Implement `scripts/train.py` with learning rate scheduling, gradient clipping, and checkpoint management.
  4. Implement `scripts/evaluate.py` calculating frame-level ROC-AUC and PR-AUC using standard trapezoidal integration.
  5. Execute Experiment A and log baseline metrics to `PROJECT_STATUS.md`.
- **Verification**: Baseline ROC-AUC meets target literature range (~75–77%).

---

## Phase 4: Temporal Model Integration (Experiment B)
- **Objective**: Integrate Dilated Causal TCN to model temporal context across feature sequences.
- **Tasks**:
  1. Implement `src/models/tcn.py` with causal convolutions, exponential dilation schedule, residual connections, and LayerNorm.
  2. Implement `src/models/complete_vad.py` chaining TCN and MIL head.
  3. Run hyperparameter grid on validation set: layers $L \in \{2, 3, 4\}$, channels $C \in \{64, 128, 256\}$.
  4. Train Exp B model and evaluate ROC-AUC/PR-AUC against Exp A baseline.
- **Verification**: Confirm strictly causal temporal receptive field and empirical gain over baseline.

---

## Phase 5: Temporal Persistence & Event Formation Engine (Experiment C)
- **Objective**: Build post-processing smoothing and hysteresis event formation to eliminate alert fragmentation.
- **Tasks**:
  1. Implement `src/postprocessing/smoothing.py` (EMA, Moving Average, Gaussian filter).
  2. Implement `src/postprocessing/event_formation.py` (Dual-threshold hysteresis $T_{high}, T_{low}$, min duration $\tau_{min}$).
  3. Implement event-level evaluation metrics in `src/evaluation/event_metrics.py` (Alert Fragmentation Index, tIoU, Detection Delay).
  4. Execute Experiment C ablation comparing smoothing algorithms and thresholding methods.
- **Verification**: Alert Fragmentation Index drops significantly compared to raw static thresholding.

---

## Phase 6: Deterministic Risk Decision Support Layer (Experiment D)
- **Objective**: Implement the transparent, deterministic risk layer and calibrate operational thresholds on validation data.
- **Tasks**:
  1. Implement `src/risk/decision_layer.py` combining peak anomaly evidence and temporal persistence:
     $$R = w_{mag} \cdot s_{max} + w_{dur} \cdot \phi(\Delta t)$$
  2. Implement automated threshold calibration deriving $\Theta_{base}, \Theta_{low}, \Theta_{high}$ from validation set distributions.
  3. Map active events to prioritized alert streams (Low / Medium / High).
  4. Execute Experiment D sensitivity analysis.
- **Verification**: Confirm zero test-set leakage in calibration and intuitive incident prioritization across test classes.

---

## Phase 7: Comprehensive Evaluation & Hardware Profiling (Experiments E & F)
- **Objective**: Profile computational efficiency and execute secondary cross-dataset validation.
- **Tasks**:
  1. Implement `scripts/benchmark.py` profiling end-to-end FPS, model latency (ms), and peak VRAM allocation (MB) using CUDA events on RTX 3050.
  2. Log empirical hardware efficiency metrics in `EXPERIMENT_PLAN.md`.
  3. (If compute permits) Evaluate trained model on XD-Violence test partition (Exp F).
- **Verification**: Confirmed memory usage remains well below 4 GB VRAM and model inference runs under 10 ms.

---

## Phase 8: Reporting, Documentation & Interactive Demo
- **Objective**: Consolidate findings into the academic report, presentation slides, and an interactive demonstration.
- **Tasks**:
  1. Generate publication-quality visualization figures (ROC curves, PR curves, temporal score timelines, event envelopes).
  2. Compile comprehensive final project report adhering to academic logic.
  3. Provide a lightweight CLI or web-based demo allowing live video upload, timeline visualization, and prioritized alert streaming.
