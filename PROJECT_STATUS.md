# PROJECT_STATUS.md — Live Project State & Progress Ledger

## 1. Current Snapshot
- **Current Phase**: **Phase 7 Complete — Research System, Controlled Ablation Suite, and Interactive Demo Validated**
- **Date**: 2026-10-28
- **Lead AI Agent**: Antigravity (DeepMind Advanced Agentic Coding)
- **Host System**: Windows 11, Python 3.11.0 runtime, `.venv` virtual environment active
- **Hardware Profile**: NVIDIA GeForce RTX 3050 Laptop GPU (4096 MiB VRAM, Driver 592.82, CUDA 13.1)
- **PyTorch Environment**: PyTorch 2.6.0+cu124, CUDA 12.4, GPU Active (`torch.cuda.is_available() == True`), Total VRAM 4.00 GB
- **Dataset Status**: **UCF-Crime 100% Validated (1,900 videos; 1,369 Train / 241 Val / 290 Test)**
- **Feature Cache**: **516 total extracted videos (241/241 Validation [100%], 260 Balanced Train across all 13 categories + Normal)**
- **Backbone Status**: **S3D (Separable 3D Inception, 1024-d) EMPIRICALLY VALIDATED (1.08 GB peak VRAM, 57.3 clips/s, 17.45 ms/clip)**
- **Baseline MIL (Exp A)**: **Val ROC-AUC: 0.8253, Val PR-AUC: 0.8236 [MEASURED]**
- **Causal TCN Model (Exp B)**: **Val ROC-AUC: 0.8750, Val PR-AUC: 0.8705 (+4.97% AUC) [MEASURED]**
- **Temporal Persistence (Exp C)**: **Alert Fragmentation Reduction: 43.5% (74 consolidated events vs 131 raw blocks) [MEASURED]**
- **Risk Decision Layer (Exp D)**: **174 Normal, 0 Low, 14 Medium, 53 High Risk on validation set [MEASURED]**
- **Efficiency Benchmark (Exp E)**: **20.23 ms snippet latency, 791.1 effective video FPS, 1.08 GB peak VRAM (27% budget) [MEASURED]**
- **Interactive Demonstration**: **Gradio Web App (`app.py`) Operational on `127.0.0.1:7860` [VERIFIED]**
- **Test Suite**: **32/32 Tests Passing (100% pass rate in 13.1s)**

---

## 2. Phase Completion Ledger

### Phase 0: Scaffolding & Architecture Freeze (✅ COMPLETED)
- Scaffolding test suite passed 100% (`tests/test_scaffolding.py`).
- Virtual environment verified with PyTorch 2.6.0+cu124 on RTX 3050.
- Governing laws and specifications established (`AGENTS.md`, `ARCHITECTURE.md`, `RESEARCH_DECISIONS.md`).

### Phase 1: Dataset Validation & Manifest Generation (✅ COMPLETED)
- 1,900 source videos audited with 0 corruption.
- Stratified Train/Val split: Train 1,369 (689 abn, 680 norm), Val 241 (121 abn, 120 norm), Test 290 strictly quarantined.
- Zero data leakage verified (`tests/test_manifests.py`).

### Phase 2: Pretrained Backbone & Resumable Feature Extraction (✅ COMPLETED)
- S3D (1024-d, 17.45 ms/clip, 57.3 clips/s) verified as primary spatiotemporal backbone.
- Resumable streaming extraction pipeline implemented (`scripts/extract_features.py`).
- 100% of validation videos extracted (241/241) and balanced stratified train set extracted (260 videos across all 13 anomaly categories + normal videos).

### Phase 3: Weakly Supervised MIL Baseline — Experiment A (✅ COMPLETED)
- Implemented canonical `MILHead` and `MILBaselineModel` (`src/models/mil_head.py`).
- Implemented `MILRankingLoss` (`src/losses/ranking_loss.py`) with hinge ranking loss, temporal smoothness ($\lambda_{smooth}=8\times 10^{-5}$), and sparsity ($\lambda_{sparse}=8\times 10^{-5}$).
- Live Rich dashboard, TensorBoard logging, and automatic checkpointing in `scripts/train_mil.py`.
- **Measured Metrics:**
  - Val ROC-AUC: **0.8253** (Epoch 29)
  - Val PR-AUC (AP): **0.8236**
  - Peak Training VRAM: **43.8 MB** (0.04 GB)
  - Training Epoch Time: **0.50 s**
  - Checkpoints: `experiments/exp_A_mil_baseline/checkpoints/` (`best_auc.pt`, `best_pr_auc.pt`, `last.pt`).

### Phase 4: Dilated Causal TCN Modeling — Experiment B (✅ COMPLETED)
- Implemented `CausalConv1d`, `ChannelLayerNorm`, `TemporalResidualBlock`, and `TCNModel` (`src/models/tcn.py`).
- Strict temporal causality (zero future leakage) mathematically tested and verified (`tests/test_tcn.py`).
- Evaluated comparative ablation against Experiment A on identical dataset split.
- **Measured Metrics:**
  - Val ROC-AUC: **0.8750** (Epoch 18) — **+4.97% improvement over MIL baseline**!
  - Val PR-AUC (AP): **0.8705** — **+4.69% improvement over MIL baseline**!
  - Peak Training VRAM: **44.1 MB** (0.04 GB)
  - Checkpoints: `experiments/exp_B_tcn/checkpoints/` (`best_auc.pt`, `best_pr_auc.pt`, `last.pt`).

### Phase 5: Temporal Persistence & Event Formation — Experiment C (✅ COMPLETED)
- Implemented temporal smoothing filters (`src/postprocessing/smoothing.py`): EMA ($\alpha=0.25$), Moving Average, Gaussian.
- Implemented `EventFormationEngine` (`src/postprocessing/event_formation.py`) with dual-threshold hysteresis ($T_{high}=0.60, T_{low}=0.40$), gap bridging ($\tau_{gap}=1.2s$), and minimum persistence duration ($\tau_{min}=1.5s$).
- **Measured Metrics across 241 Validation Videos:**
  - Raw Thresholded Blocks: 131 fragmented blocks (193 chattering toggles).
  - Persistence-Formed Events: 74 consolidated incidents.
  - **Alert Fragmentation Reduction: 43.5% [MEASURED]**.
  - Total Validated Event Duration: 1,955.1 s.

### Phase 6: Transparent Deterministic Risk Decision Layer — Experiment D (✅ COMPLETED)
- Implemented `RiskDecisionLayer` (`src/risk/decision_layer.py`) combining peak anomaly magnitude ($w_{mag}=0.65$) and normalized persistence ($w_{dur}=0.35$).
- Calibrated operational tiers on validation set: Normal ($R < 0.25$), Low Risk ($0.25 \le R < 0.45$), Medium Risk ($0.45 \le R < 0.70$), High Risk ($R \ge 0.70$).
- **Validation Distribution:** 174 Normal (72.2%), 14 Medium Risk (5.8%), 53 High Risk (22.0%), 0 Low Risk.
- Generates transparent, human-readable operational explainability breakdown for every incident.

### Phase 7: Efficiency Benchmark & Demonstration — Experiment E (✅ COMPLETED)
- Benchmarked local RTX 3050 GPU with 300 iterations (`scripts/benchmark.py`):
  - S3D Backbone Latency: 17.45 ms/snippet.
  - TCN + MIL Downstream Latency: 0.0916 ms/snippet (10,914 snippets/s).
  - Complete End-to-End Latency: **20.23 ms per snippet** (~**791.1 effective video FPS**).
  - Peak VRAM Consumption: **1.08 GB** (27% of 4.0 GB budget).
- Interactive Demonstration Interface:
  - Implemented Gradio application (`app.py`).
  - Provides sample selection across all 13 anomaly categories + normal videos, interactive parameter sliders, continuous anomaly score timeline plot, detected event table, and operational risk alert badge.
  - Smoke tested and verified with 100% success.
- Paper-Ready Artifacts:
  - Publication figures generated in `experiments/paper_artifacts/plots/`.
  - LaTeX tables generated in `experiments/paper_artifacts/tables/` (`ablation.tex`, `results.tex`, `efficiency.tex`).
  - Formal academic research summary report generated in `experiments/paper_artifacts/summary_report.md`.

---

## 3. Systematic Experimental Comparison Table

| Experiment | Architecture / Stage | Val ROC-AUC | Val PR-AUC | Fragmentation Red. | Latency / Snippet | Video FPS | Peak VRAM | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp A** | S3D (1024-d) + MIL Head | 0.8253 | 0.8236 | --- | 0.011 ms* | 93,023* | 43.8 MB | **MEASURED** |
| **Exp B** | S3D + Dilated Causal TCN + MIL | **0.8750** | **0.8705** | --- | 0.092 ms* | 10,914* | 44.1 MB | **MEASURED** |
| **Exp C** | S3D + TCN + Persistence | 0.8750 | 0.8705 | **-43.5% (74 events)** | 0.092 ms* | 10,914* | 44.1 MB | **MEASURED** |
| **Exp D** | (C) + Deterministic Risk Layer | 0.8750 | 0.8705 | 53 High / 14 Med | 0.093 ms* | 10,800* | 44.1 MB | **MEASURED** |
| **Exp E** | **Full Decoupled Pipeline (E2E)** | **0.8750** | **0.8705** | **-43.5%** | **20.23 ms** | **791.1 FPS** | **1.08 GB** | **MEASURED** |

*\*Downstream scoring latency excluding 3D video decoding. E2E includes full S3D feature extraction.*

---

## 4. Test Suite Verification
- **Total Tests:** 32 passed, 0 failed in 13.13 seconds.
- **Coverage:** Scaffolding, dataset manifests, feature caching, ranking loss, MIL model, causal TCN (zero future leakage), temporal smoothing, event formation, risk decision layer, checkpoint resume.
