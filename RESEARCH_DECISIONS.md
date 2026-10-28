# RESEARCH_DECISIONS.md — Architecture Decision Records (ADRs)

This document is the official architectural ledger for the project. Every structural design choice, paradigm shift, and validated parameter is recorded here in ADR format.

---

## SECTION I: FROZEN ARCHITECTURAL DECISIONS
These conceptual and structural architectural choices are **officially frozen**. They define the project identity and cannot be altered casually without an explicit research justification, empirical evidence, and formal ADR amendment.

---

### ADR-001: Weakly Supervised Multiple Instance Learning (MIL) Paradigm
- **Date**: 2026-10-26
- **Status**: **FROZEN**
- **Context**: Video Anomaly Detection in realistic surveillance scenarios suffers from the scarcity of frame-level training annotations. Full frame-level supervision is prohibitive to annotate at scale, while unsupervised normal-only reconstruction methods suffer from high false-positive rates on complex dynamic scenes.
- **Decision**: Adopt the Weakly Supervised Multiple Instance Learning (MIL) ranking framework (Sultani et al., 2018). Videos are treated as bags and temporal snippets as instances. Video-level labels indicate whether an anomaly occurs anywhere in the video.
- **Alternatives Considered**:
  - *Unsupervised / Reconstruction-based (Autoencoders, MemAE)*: High false alarms on unseen normal variations.
  - *Fully Supervised Frame-Level Detection*: Unrealistic data annotation cost; benchmark training sets do not provide frame-level labels.
- **Consequences**: Downstream anomaly heads must optimize bag-level ranking losses (max-margin between abnormal and normal bag peaks) supplemented with temporal smoothness and sparsity constraints.

---

### ADR-002: Two-Stage Decoupled Feature Extraction & Caching
- **Date**: 2026-10-26
- **Status**: **FROZEN**
- **Context**: The project operates under local hardware constraints: an **NVIDIA GeForce RTX 3050 Laptop GPU with 4096 MiB VRAM**. End-to-end backpropagation through multi-layer 3D spatiotemporal networks across thousands of video clips exhausts VRAM within a few iterations.
- **Decision**: Decouple the pipeline into:
  1. Offline spatiotemporal feature extraction using a frozen pretrained video backbone.
  2. Training downstream temporal sequence models directly on cached dense feature representations (`data/features/`).
- **Alternatives Considered**:
  - *End-to-End Fine-Tuning*: Infeasible on 4 GB VRAM; requires high-end multi-GPU clusters.
  - *2D Frame-by-Frame CNN (ResNet-50)*: Completely ignores temporal motion cues and optical flow dynamics.
- **Consequences**: Upstream feature extraction is an $O(N)$ offline precomputation step. Downstream training of TCN + MIL heads is blazing fast (<15 min on local GPU), uses <1.0 GB VRAM, and allows rapid hyperparameter exploration.

---

### ADR-003: Temporal Modeling via Dilated Causal Temporal Convolutional Networks (TCN)
- **Date**: 2026-10-26
- **Status**: **FROZEN**
- **Context**: Standard MIL models evaluate each video snippet independently, causing noisy, fluctuating anomaly scores. Temporal context is essential to understand whether an anomalous pattern develops progressively.
- **Decision**: Employ a **Dilated Causal TCN** with residual connections over the feature sequence. Convolutions must be strictly causal ($y_t$ depends only on $x_{\le t}$).
- **Alternatives Considered**:
  - *Recurrent Networks (LSTM/GRU)*: Slower training due to sequential backprop through time (BPTT); prone to gradient vanishing over long video sequences.
  - *Temporal Transformers*: Computationally expensive quadratic attention complexity; excessive memory footprint and high sample complexity for training from scratch under weak supervision.
- **Consequences**: Provides an exponentially growing temporal receptive field, zero future frame leakage (enabling realistic online streaming evaluation), and stable $O(1)$ parallel training.

---

### ADR-004: Post-Processing via Temporal Persistence & Hysteresis Event Formation
- **Date**: 2026-10-26
- **Status**: **FROZEN**
- **Context**: Instantaneous snippet anomaly scores fluctuate above and below single static thresholds, causing severe alert fragmentation (`ALERT -> NORMAL -> ALERT -> NORMAL`). Surveillance operators require cohesive incident envelopes.
- **Decision**: Implement a two-step post-processing pipeline:
  1. Temporal score smoothing over time.
  2. Dual-threshold hysteresis windowing ($T_{high}, T_{low}$) combined with a minimum temporal duration filter ($\tau_{min}$) to output discrete incident envelopes $E_k = (t_{start}, t_{end}, \Delta t)$.
- **Alternatives Considered**:
  - *Raw Static Thresholding*: Produces jagged, intermittent alarms and high false alarm rates.
  - *Connected Component Labeling without Smoothing*: Fails on intermittent single-frame dips.
- **Consequences**: Converts continuous snippet probabilities into operational incident entities, drastically reducing alert fragmentation.

---

### ADR-005: Transparent Deterministic Risk Decision Layer (No Pseudo-Supervised Classifier)
- **Date**: 2026-10-26
- **Status**: **FROZEN**
- **Context**: Standard VAD datasets (UCF-Crime, XD-Violence) do NOT contain ground-truth risk or severity labels (Low/Medium/High). Fabricating categorical labels and training a supervised classifier constitutes unscientific pseudo-learning.
- **Decision**: Define the risk layer as a **transparent, deterministic, heuristic decision support function** derived exclusively from:
  $$\text{Risk Evidence } R = w_{mag} \cdot s_{max} + w_{dur} \cdot \phi(\Delta t)$$
  mapped to operational priority tiers (Low / Medium / High). Thresholds must be calibrated strictly on training/validation video distributions.
- **Alternatives Considered**:
  - *Supervised Risk Classifier*: Rejected due to absence of ground-truth severity annotations.
  - *Category Prior-Based Weighting*: Rejected for initial core pipeline to avoid unverified heuristic bias.
- **Consequences**: The risk layer is clearly described as operational decision support rather than ground-truth severity classification.

---

### ADR-006: Primary Benchmark — UCF-Crime
- **Date**: 2026-10-26
- **Status**: **FROZEN**
- **Context**: Research must be benchmarked against a standard, widely cited, peer-reviewed academic dataset.
- **Decision**: UCF-Crime is frozen as the primary benchmark dataset. Local raw frames archive `ufc-crime.zip` (11.8 GB) provides 1,377,653 frames covering 13 crime classes and normal videos across Train and Test splits.
- **Consequences**: Ensures scientific comparability against Sultani et al. (2018), RTFM (2021), MIST (2021), and subsequent weakly supervised VAD literature.

---

### ADR-007: Baseline-First Experimental Progression
- **Date**: 2026-10-26
- **Status**: **FROZEN**
- **Context**: Adding complex modules without isolating their marginal contributions violates academic research standards.
- **Decision**: Mandate a strict 6-stage experimental hierarchy:
  - Exp A: Baseline Weakly Supervised MIL (No temporal modeling)
  - Exp B: Baseline + TCN (Evaluating temporal context contribution)
  - Exp C: TCN + Persistence & Event Formation (Evaluating alert stability)
  - Exp D: Full System + Calibrated Risk Layer (Evaluating decision support)
  - Exp E: Computational Efficiency Profiling (Empirical FPS, latency, VRAM)
  - Exp F: Secondary / Cross-Dataset Evaluation (XD-Violence, if compute allows)
- **Consequences**: Ensures clean ablation studies where every component justifies its inclusion.

---

## SECTION II: VALIDATED IMPLEMENTATION PARAMETERS
Unlike the frozen conceptual architecture, these parameters must NOT be hardcoded prematurely. They are subject to validation through literature standards, compatibility smoke tests, and empirical tuning.

---

### VIP-001: Video Backbone Selection & Feature Dimensionality
- **Status**: **EMPIRICALLY VALIDATED (S3D Primary / R(2+1)D-18 Fallback)**
- **Empirical GPU Smoke Benchmark (RTX 3050 Laptop GPU, 4096 MiB VRAM)**:
  - **S3D (Primary)**:
    - Weights: Official PyTorch torchvision `KINETICS400_V1` (`s3d-d76dad2f.pth`, 32 MB)
    - Output Feature Dimension: $D_{feat} = 1024$
    - Input Shape: $(B, 3, 16, 224, 224)$
    - Peak GPU VRAM: **1084.5 MB** (~26% of 4GB budget, safe headroom)
    - Inference Latency: **17.45 ms/clip**
    - Throughput: **57.3 clips/sec** (real-time extraction capable)
    - Architecture: Separable 3D Inception (matches I3D 1024-d spatiotemporal representations with separable conv efficiency)
  - **R(2+1)D-18 (Verified Fallback)**:
    - Weights: `KINETICS400_V1` (`r2plus1d_18-91a641e6.pth`, 120 MB)
    - Output Feature Dimension: $D_{feat} = 512$
    - Peak GPU VRAM: 972.0 MB
    - Latency: 27.66 ms/clip | Throughput: 36.1 clips/sec
- **Selection Decision**: S3D adopted as the primary frozen backbone. Downstream models dynamically consume $D_{feat} = 1024$.
- **Validation Rule**: Model heads must read $D_{feat}$ dynamically from feature metadata rather than assuming a hardcoded constant.

---

### VIP-002: Temporal Snippet Length ($T_{clip}$)
- **Status**: **UNDER VALIDATION**
- **Starting Hypothesis**: $T_{clip} = 16$ frames.
- **Justification**: 16 frames at standard CCTV frame rates (15-30 fps) captures 0.5 to 1.0 second of coherent motion dynamics. Extensively validated in C3D/I3D/RTFM literature. Halves extraction memory compared to 32 frames.
- **Candidate Alternative**: $T_{clip} = 32$ frames (to be compared if temporal granularity demands broader context).

---

### VIP-003: TCN Structural Hyperparameters (Depth & Dilation Schedule)
- **Status**: **EMPIRICALLY VALIDATED IN EXP B**
- **Architecture**: 3-layer Dilated Causal TCN with $d \in [1, 2, 4]$, $C_{hidden}=128$, $k=3$, and `ChannelLayerNorm`.
- **Measured Result**: Achieved **0.8750 ROC-AUC** and **0.8705 PR-AUC** (+4.97% AUC improvement over MIL baseline) with zero future leakage.

---

### VIP-004: Temporal Smoothing Algorithm
- **Status**: **EMPIRICALLY VALIDATED IN EXP C**
- **Validated Setting**: Causal Exponential Moving Average (EMA) with $\alpha = 0.25$.
- **Measured Result**: Causal smoothing smoothly filters frame-level jitter without introducing future leakage.

---

### VIP-005: Event Formation Hysteresis Thresholds & Minimum Duration
- **Status**: **EMPIRICALLY VALIDATED ON VALIDATION SET**
- **Validated Setting**: $T_{high} = 0.60$, $T_{low} = 0.40$, $\tau_{min} = 1.5\text{ s}$ (~3 snippets), $\tau_{gap} = 1.2\text{ s}$.
- **Measured Result**: Reduced alert fragmentation by **43.5%** (from 131 raw blocks to 74 consolidated incidents) across 241 validation videos.

---

### VIP-006: Risk Decision Weights & Operational Thresholds
- **Status**: **EMPIRICALLY VALIDATED ON VALIDATION SET**
- **Formulation**: $R = 0.65 \cdot s_{max} + 0.35 \cdot \min(1.0, \Delta t / 8.0\text{ s})$.
- **Operational Tiers**: High Risk ($R \ge 0.70$), Medium Risk ($0.45 \le R < 0.70$), Low Risk ($0.25 \le R < 0.45$), Normal ($R < 0.25$).
- **Measured Result**: Classified 174 Normal, 14 Medium Risk, 53 High Risk on the validation set.

---

## SECTION III: NEW ARCHITECTURE DECISION RECORDS (ADRs 008-011)

### ADR-008: Channel-Wise Layer Normalization for Causal Sequence Modeling
- **Date**: 2026-10-28
- **Status**: **ACCEPTED & IMPLEMENTED**
- **Context**: In sequence modeling with 1D dilated convolutions, standard `BatchNorm1d` introduces batch-size dependency and collapses during single-video streaming validation ($B=1$). Furthermore, standard `GroupNorm(1, C)` averages over time $T$, leaking future information to past frames and violating strict causality.
- **Decision**: Implement `ChannelLayerNorm(C)`, applying `nn.LayerNorm(C)` across the channel dimension for each temporal snippet $t$ independently: $(B, C, T) \to (B, T, C) \to \text{LayerNorm} \to (B, C, T)$.
- **Consequences**: Mathematically guarantees zero future leakage (verified via unit test with 0.00000000 max error) and makes training/validation completely invariant to batch size.

---

### ADR-009: Stratified Categorical Sampling for Downstream Training
- **Date**: 2026-10-28
- **Status**: **ACCEPTED & IMPLEMENTED**
- **Context**: UCF-Crime exhibits class imbalance across its 13 anomaly categories. Sampling videos strictly sequentially caused training batches to contain only a single category.
- **Decision**: Pre-extract a balanced stratified training subset of 260 videos (10 videos per each of the 13 anomaly categories = 130 abnormal bags, plus 130 normal bags), while retaining 100% of the official validation set (241 videos).
- **Consequences**: Models train on rich categorical diversity, boosting generalizability across violence, property crime, and traffic incidents.

---

### ADR-010: Causal Exponential Moving Average for Online Surveillance Smoothing
- **Date**: 2026-10-28
- **Status**: **ACCEPTED & IMPLEMENTED**
- **Context**: Real-time CCTV surveillance requires streaming anomaly predictions. Bidirectional filters (moving average, Gaussian) require waiting for future frames, introducing a latency buffer.
- **Decision**: Deploy causal Exponential Moving Average ($s_t = \alpha s_t + (1-\alpha)s_{t-1}$) as the primary real-time smoothing filter with $\alpha=0.25$.
- **Consequences**: Zero latency buffer during online streaming while successfully attenuating instantaneous frame fluctuations.

---

### ADR-011: Saturated Persistence Penalty in Operational Risk Evidence
- **Date**: 2026-10-28
- **Status**: **ACCEPTED & IMPLEMENTED**
- **Context**: Linear duration weighting unbounded over time allows minor low-confidence anomalies to artificially become "High Risk" merely by being long.
- **Decision**: Apply persistence saturation $\phi(\Delta t) = \min(1.0, \Delta t / \Delta t_{sat})$ with $\Delta t_{sat} = 8.0\text{ seconds}$.
- **Consequences**: An incident that persists beyond 8 seconds reaches maximum duration risk contribution ($0.35$), but cannot exceed $0.35$ unless supported by high anomaly magnitude ($s_{max}$).

