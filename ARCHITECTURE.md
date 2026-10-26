# ARCHITECTURE.md — System Architecture & Engineering Blueprint

## 1. Executive Summary & Design Paradigm
The system implements a computationally feasible, modular pipeline for **Video Anomaly Detection and Risk Assessment** under undergraduate hardware constraints (**4 GB VRAM, NVIDIA GeForce RTX 3050 Laptop GPU**).

To prevent memory exhaustion during training and eliminate the prohibitive cost of backpropagating through deep 3D video backbones, the system is strictly decoupled into a **Two-Stage Modular Architecture**:
- **Stage 1 (Upstream Feature Extraction)**: Pretrained, frozen spatiotemporal backbone extracts dense feature representation vectors per temporal snippet. These representations are cached to disk (`data/features/`).
- **Stage 2 (Downstream Temporal Modeling & MIL Scoring)**: A lightweight Dilated Causal Temporal Convolutional Network (TCN) models temporal dependencies across the feature sequence, driving a Multiple Instance Learning (MIL) anomaly head trained with ranking loss.
- **Stage 3 (Post-Processing & Event Formation)**: Temporal smoothing and dual-threshold hysteresis windowing aggregate snippet scores into stable, contiguous incident envelopes ($[t_{start}, t_{end}]$).
- **Stage 4 (Risk Decision Support Layer)**: A deterministic, transparent heuristic evaluates peak anomaly evidence and temporal persistence to prioritize active incidents into operational tiers (Low / Medium / High).

---

## 2. End-to-End System Architecture

```text
+---------------------------------------------------------------------------------------------------+
|                                       SYSTEM DATA FLOW                                            |
+---------------------------------------------------------------------------------------------------+

                                    Surveillance Video Stream
                                                │
                                                ▼
                                    Temporal Snippet Windowing
                                (Clip length T_clip: 16 / 32 frames)
                                                │
                                                ▼
                                  Frozen Pretrained Video Backbone
                              (I3D / S3D / R(2+1)D-18 Feature Extractor)
                                                │
                                                ▼
                                   Cached Feature Sequence
                              X = [x_1, x_2, ..., x_T] ∈ R^(T × D_feat)
                                                │
                                                ▼
                                Dilated Causal ConvNet (TCN)
                          (Multi-scale receptive field, zero future leakage)
                                                │
                                                ▼
                                    Temporal Latent Sequence
                              H = [h_1, h_2, ..., h_T] ∈ R^(T × D_hidden)
                                                │
                                                ▼
                                   Weakly Supervised MIL Head
                                  (MLP with Sigmoidal Output)
                                                │
                                                ▼
                                   Raw Snippet Anomaly Scores
                                  S = [s_1, s_2, ..., s_T], s_t ∈ [0, 1]
                                                │
                    ════════════════════════════╪════════════════════════════
                                                ▼
                                    TASK 1: ANOMALY DETECTION
                                (Evaluated via Frame-Level ROC-AUC / PR-AUC)
                    ════════════════════════════╪════════════════════════════
                                                │
                                                ▼
                                   Temporal Smoothing Filter
                                  (Moving Average / EMA / Gaussian)
                                                │
                                                ▼
                                    Smoothed Anomaly Scores
                                            s_t^(smooth)
                                                │
                                                ▼
                                  Persistence & Event Formation
                             (Dual-threshold hysteresis + min duration τ_min)
                                                │
                    ════════════════════════════╪════════════════════════════
                                                ▼
                                   TASK 2: EVENT FORMATION
                           Active Event: E_k = (t_start, t_end, Δt, s_max)
                            (Evaluated via Alert Stability & Fragmentation)
                    ════════════════════════════╪════════════════════════════
                                                │
                                                ▼
                               Deterministic Risk Decision Layer
                                Risk Evidence R = w_1 * s_max + w_2 * f(Δt)
                                 (Thresholds calibrated on Train/Val only)
                                                │
                    ════════════════════════════╪════════════════════════════
                                                ▼
                                  TASK 3: RISK PRIORITIZATION
                                 ┌──────────────┼──────────────┐
                                 ▼              ▼              ▼
                             LOW ALERT    MEDIUM ALERT    HIGH ALERT
                              (R < θ_1)    (θ_1 ≤ R < θ_2)   (R ≥ θ_2)
                    ════════════════════════════════════════════════════════
```

---

## 3. Detailed Component Specifications

### 3.1 Stage 1: Pretrained Video Feature Backbone
- **Objective**: Map high-dimensional video pixels $(T_{clip} \times 3 \times H \times W)$ to a compact, semantically rich spatiotemporal vector $x_t \in \mathbb{R}^{D_{feat}}$.
- **Frozen Invariant**: The backbone weights are strictly frozen during downstream training. Backpropagation through the 3D backbone is forbidden.
- **Candidate Backbones**:
  - *Primary Candidate*: **I3D (Inflated 3D ConvNet)** pretrained on Kinetics-400 (RGB stream). Yields $D_{feat} = 1024$ (RGB-only, after global spatial-temporal average pooling).
  - *Verified Lightweight Fallback*: **S3D (Separable 3D)** or **R(2+1)D-18** from `torchvision.models.video`. S3D yields $D_{feat} = 1024$; R(2+1)D-18 yields $D_{feat} = 512$.
- **Validation Rule**: The exact feature dimension $D_{feat}$ is parameter-flexible ($1024$ for standard RGB I3D/S3D, $512$ for R(2+1)D, $2048$ for dual-stream). Model heads dynamically configure their input dimension based on the extracted feature manifest.

### 3.2 Stage 2: Temporal Sequence Modeling (Dilated Causal TCN)
- **Objective**: Capture long-range context across consecutive video snippets to distinguish transient outliers from genuine anomalies.
- **Causality Requirement**: Convolutions must be **strictly causal** ($y_t$ depends only on $x_{\le t}$). Future frames cannot leak into current predictions, ensuring real-world streaming validity.
- **Dilation Structure**: Dilations grow exponentially ($d \in \{1, 2, 4, 8, \dots\}$) across layers, providing an expansive receptive field with minimal parameters and fast $O(1)$ parallel evaluation during training.
- **Residual Blocks**: Each block employs dilated causal 1D convolution $\rightarrow$ LayerNorm / BatchNorm $\rightarrow$ ReLU / GELU $\rightarrow$ Dropout $\rightarrow$ Residual addition.

### 3.3 Stage 3: Weakly Supervised Multiple Instance Learning (MIL) Head
- **Problem Formulation**: In weakly supervised VAD, a training video is labeled only at the video level:
  $$Y = \begin{cases} 1 & \text{if video contains at least one anomalous snippet} \\ 0 & \text{if video is completely normal} \end{cases}$$
- **MIL Bag Formulation**: A video is a "bag" of snippet instances: $\mathcal{B} = \{x_1, x_2, \dots, x_M\}$.
  - An abnormal video contains at least one anomalous snippet: $\max_{t} s_t \approx 1$.
  - A normal video contains no anomalous snippets: $\max_{t} s_t \approx 0$.
- **MIL Ranking Loss with Regularization**:
  $$\mathcal{L}_{total} = \mathcal{L}_{ranking} + \lambda_{smooth} \mathcal{L}_{smooth} + \lambda_{sparse} \mathcal{L}_{sparse}$$
  - **Ranking Loss (Hinge)**: Enforces that the peak score in an abnormal bag exceeds the peak score in a normal bag by margin $m$:
    $$\mathcal{L}_{ranking} = \max\left(0, 1 - \max_{i \in \mathcal{B}_a} s_i + \max_{j \in \mathcal{B}_n} s_j\right)$$
  - **Temporal Smoothness Loss**: Penalizes high-frequency variance between adjacent snippet scores:
    $$\mathcal{L}_{smooth} = \sum_{t=1}^{M-1} (s_{t+1} - s_t)^2$$
  - **Sparsity Loss**: Reflects that anomalies in abnormal videos are typically rare and brief:
    $$\mathcal{L}_{sparse} = \sum_{t=1}^{M} s_t$$

### 3.4 Stage 4: Temporal Smoothing & Event Formation Engine
- **Objective**: Transform noisy, oscillating frame-level scores into continuous incident intervals.
- **Smoothing Candidates** (to be empirically compared):
  1. *Moving Average Filter*: Uniform sliding window.
  2. *Exponential Moving Average (EMA)*: Asymmetric causal smoothing $s_t^{(EMA)} = \alpha s_t + (1 - \alpha) s_{t-1}^{(EMA)}$.
  3. *Gaussian 1D Filter*: Smooth temporal kernel.
- **Event Formation Logic (Dual-Threshold Hysteresis)**:
  - Event Trigger (Onset): Triggered when smoothed score crosses high threshold: $s_t \ge T_{high}$.
  - Event Sustained: Maintained while $s_t \ge T_{low}$ (where $T_{low} < T_{high}$).
  - Event Termination (Offset): Terminates when $s_t < T_{low}$ continuously for $\tau_{gap}$ frames.
  - Duration Filtering: Events with duration $\Delta t < \tau_{min}$ are discarded as transient noise.

### 3.5 Stage 5: Deterministic Risk Decision Support Layer
- **Core Conceptual Input**:
  $$\text{Risk Evidence } R = w_{mag} \cdot s_{max} + w_{dur} \cdot \phi(\Delta t)$$
  where:
  - $s_{max} = \max_{t \in [t_{start}, t_{end}]} s_t$: Peak anomaly confidence within the formed incident.
  - $\phi(\Delta t) = \min(1.0, \Delta t / \Delta t_{sat})$: Normalized temporal persistence up to saturation limit $\Delta t_{sat}$.
  - $w_{mag} + w_{dur} = 1.0$: Weighting parameters ($w_{mag}, w_{dur} \in [0, 1]$).
- **Calibrated Operational Tiers**:
  $$Alert(R) = \begin{cases} \text{High Risk} & \text{if } R \ge \Theta_{high} \\ \text{Medium Risk} & \text{if } \Theta_{low} \le R < \Theta_{high} \\ \text{Low Risk} & \text{if } \Theta_{base} \le R < \Theta_{low} \\ \text{Normal} & \text{if } R < \Theta_{base} \end{cases}$$
- **Calibration Protocol**: Thresholds $\Theta_{base}, \Theta_{low}, \Theta_{high}$ are calibrated strictly on training/validation video distributions (e.g., using empirical quantiles of validated normal vs. anomalous clips). **Test set annotations are strictly quarantined from threshold selection.**

---

## 4. Hardware Budget & Engineering Feasibility (4 GB VRAM)
- **Local GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM).
- **Memory Footprint Allocation**:
  - Feature extraction: Done in batches of 16-32 snippets with `torch.no_grad()` $\rightarrow$ Peak VRAM ~1.8 GB.
  - TCN + MIL Training: Operates entirely on cached 1024-dim tensors with batch size 32-64 $\rightarrow$ Peak VRAM < 1.0 GB.
  - Head Parameters: Total trainable parameters $\approx 50\text{K} - 250\text{K}$ $\rightarrow$ Model size $< 2 \text{ MB}$.
- **Zero OOM Risk**: Decoupling feature extraction from training guarantees complete stability on 4 GB hardware.
