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
- **Status**: **VALIDATED PARAMETER TO BE TUNED IN EXP B**
- **Architecture Requirement**: Strictly causal 1D convolutions with residual additions (Frozen Concept ADR-003).
- **Flexible Parameters (VIP)**: Exact depth $L$ and dilation schedule are implementation parameters, not frozen constants:
  - Candidate 1 (Default): $L = 3$ layers with dilations $d = [1, 2, 4]$
  - Candidate 2 (Deep): $L = 4$ layers with dilations $d = [1, 2, 4, 8]$
  - Candidate 3 (Compact): $L = 2$ layers with dilations $d = [1, 2]$
  - Hidden channels: $C_{hidden} \in \{64, 128, 256\}$
  - Kernel size: $k = 3$ (causal padding $p = (k - 1) \cdot d$)
  - Dropout: $p_{drop} = 0.2$
- **Optimization Criterion**: Maximize temporal receptive field while maintaining parameter efficiency and inference latency $< 5 \text{ ms}$ per video sequence.

---

### VIP-004: Temporal Smoothing Algorithm
- **Status**: **TO BE COMPARED IN EXP C**
- **Candidates**:
  1. *Exponential Moving Average (EMA)*: Causal, smooth, minimal latency overhead ($\alpha \in [0.1, 0.4]$).
  2. *Moving Average (MA)*: Symmetric sliding window ($W \in \{5, 11, 21\}$ snippets).
  3. *Gaussian 1D Smoothing*: $\sigma \in [1.0, 3.0]$.
- **Selection Metric**: False alert fragmentation reduction vs. temporal onset detection delay.

---

### VIP-005: Event Formation Hysteresis Thresholds & Minimum Duration
- **Status**: **TO BE CALIBRATED ON VALIDATION SET**
- **Hypothesis**: $T_{high} \in [0.55, 0.70]$, $T_{low} \in [0.35, 0.45]$, $\tau_{min} \in [16, 32]$ frames.
- **Rule**: Must be calibrated to minimize false alarms on validated normal videos while preserving true event boundaries. Never tune on test annotations.

---

### VIP-006: Risk Decision Weights & Operational Thresholds
- **Status**: **TO BE CALIBRATED ON VALIDATION SET**
- **Formulation**: $R = w_{mag} \cdot s_{max} + w_{dur} \cdot \min(1.0, \Delta t / \Delta t_{sat})$.
- **Hypothesis**: $w_{mag} = 0.65$, $w_{dur} = 0.35$, $\Delta t_{sat} = 150$ frames (~5-10 sec).
- **Threshold Calibration**: Derive $(\Theta_{base}, \Theta_{low}, \Theta_{high})$ from empirical quantiles of validation set anomalies.
