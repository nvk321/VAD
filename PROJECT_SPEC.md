# PROJECT_SPEC.md — Project Specification & Research Framing

## 1. Project Overview
- **Project Title**: Real-Time Video Anomaly Detection and Risk Assessment Using Deep Learning
- **Academic Context**: Undergraduate Capstone Research in Deep Learning, Computer Vision, and Decision Support Systems
- **Domain**: Video Anomaly Detection (VAD) + Temporal Sequence Modeling + Multiple Instance Learning (MIL) + Heuristic Risk Assessment
- **Computational Environment**: Single Local GPU (NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB VRAM), CUDA 13.1 driver, Windows 11, Python 3.11.0 runtime

---

## 2. Real-World Motivation & Problem Statement
Modern automated security infrastructure relies heavily on Closed-Circuit Television (CCTV) surveillance. Thousands of surveillance cameras continuously capture hours of video feed across municipal, commercial, and transport facilities. 

However, operational monitoring faces critical challenges:
1. **Human Fatigue & Scalability**: Manual video analysis does not scale to dozens of continuous streams; human observers experience severe cognitive fatigue within 20 minutes, leading to missed critical incidents.
2. **Alert Fragmentation in Existing VAD**: Existing automated VAD models evaluate individual frames or short snippets, generating noisy, fluctuating instantaneous anomaly scores. A single real-world incident can trigger a series of fragmented on-off alarms (`ALERT -> NORMAL -> ALERT -> NORMAL`), overwhelming security operators.
3. **Lack of Actionable Incident Prioritization**: Real-world operations do not just require a binary "is this 0.5-second snippet anomalous?" signal. Operators require **incident-level understanding**:
   - *Has a persistent event started?*
   - *How long has it persisted?*
   - *How urgently should security personnel respond?*
4. **Absence of Ground-Truth Severity Labels**: Standard public benchmarks (e.g., UCF-Crime, XD-Violence) provide only binary video-level labels (Train) and frame-level temporal boundary annotations (Test). They **do not provide native ground-truth risk or severity labels** (Low/Medium/High). Arbitrarily training a supervised severity classifier on fabricated labels produces mathematically invalid models.

---

## 3. Academic Framing & Research Methodology
This project strictly adheres to academic rigor:
```text
Real-World Problem (CCTV overload, alert noise, unprioritized incidents)
        ↓
Existing Solutions (Reconstruction-based VAD, 3D-CNNs, Weakly Supervised MIL)
        ↓
Literature Review (Sultani et al., RTFM, MIST, Sultani-C3D, I3D backbones, TCNs)
        ↓
Limitations of Existing Solutions (Flickering scores, lack of event formation, no prioritization)
        ↓
Research Gap (Gaps A, B, C, D)
        ↓
Research Question
        ↓
Proposed Solution (Two-stage modular VAD + Persistence + Deterministic Risk Layer)
        ↓
Implementation (Python 3.11, PyTorch, Causal Dilated TCN, MIL ranking loss)
        ↓
Baseline Comparison (Standard Weakly Supervised MIL without temporal modeling)
        ↓
Ablation Studies (Evaluating +TCN, +Smoothing, +Persistence, +Risk)
        ↓
Evaluation (Frame-level ROC-AUC, PR-AUC, Event-level IoU, Fragmentation, Latency/FPS/VRAM)
        ↓
Conclusion, Honest Limitations & Future Work
```

---

## 4. The Central Project Story
- **Existing VAD Paradigm**:
  $$\text{Video} \longrightarrow \text{Anomaly Model} \longrightarrow \text{Instantaneous Anomaly Score } s_t \in [0, 1]$$
  *Limitation*: High score variance, flickering alerts, zero operational incident prioritization.

- **Our Proposed System Paradigm**:
  $$\text{Video} \longrightarrow \text{Weakly Supervised VAD} \longrightarrow s_t \longrightarrow \text{Temporal Smoothing} \longrightarrow \text{Persistence \& Event Formation} \longrightarrow \text{Transparent Risk Decision} \longrightarrow \text{Prioritized Alert (Low/Med/High)}$$

---

## 5. Formal Research Question
> **"Can a weakly supervised video anomaly detection system be extended with temporal persistence and a transparent risk-oriented decision layer to produce more stable and actionable incident-level alerts, without requiring additional ground-truth risk labels?"**

---

## 6. Targeted Research Gaps
- **Gap A: Anomaly Score $\rightarrow$ Actionable Decision**: Bridging the disconnect between continuous snippet scores and discrete, prioritized incident alerts.
- **Gap B: Handling the Absence of Risk Ground Truth**: Formulating a transparent, deterministic decision layer grounded in peak anomaly evidence and temporal persistence, rather than training an unscientific pseudo-supervised risk classifier.
- **Gap C: Event Continuity & Alert Stability**: Transforming volatile snippet scores into continuous incident envelopes ($[t_{start}, t_{end}]$) to drastically suppress false alert fragmentation.
- **Gap D: Computational Efficiency on Constrained Hardware**: Designing a two-stage decoupled pipeline (cached frozen backbone features + lightweight TCN/MIL head) capable of training and inference within a **4 GB VRAM** local budget.

---

## 7. The Three-Task Distinction
The pipeline explicitly decouples three separate operational tasks:
1. **Task 1: Anomaly Detection**
   - Question: *Is this snippet anomalous?*
   - Model: Frozen Video Backbone + Dilated Causal TCN + MIL Head.
   - Output: Snippet-level score $s_t \in [0.0, 1.0]$.
2. **Task 2: Event Formation**
   - Question: *Does the anomaly persist sufficiently to constitute an incident?*
   - Module: Temporal smoothing + Hysteresis thresholding ($T_{high}, T_{low}$) + Minimum duration filter ($\tau_{min}$).
   - Output: Active incident envelope $E_k = (t_{start}, t_{end}, \Delta t)$.
3. **Task 3: Risk Prioritization**
   - Question: *How urgently should security personnel respond to this active incident?*
   - Module: Deterministic risk scoring function $R = f(s_{peak}, \Delta t)$ mapped to operational categories (Low / Medium / High).
   - Output: Prioritized alert event stream with human-interpretable rationale.

---

## 8. Scope, Boundaries & Non-Goals
### Scope (What We Do):
- Train and evaluate a weakly supervised MIL VAD pipeline on the UCF-Crime benchmark.
- Model temporal dynamics using causal dilated temporal convolutions (no temporal leakage into the future).
- Implement post-processing temporal smoothing (EMA / Moving Average / Gaussian).
- Build a deterministic hysteresis-based event formation engine.
- Formulate a transparent risk prioritization layer calibrated on training/validation distributions.
- Quantitatively measure detection quality (ROC-AUC, PR-AUC), event stability (alert fragmentation count, temporal IoU), and hardware efficiency (FPS, latency in ms, peak VRAM in MB).

### Non-Goals (What We Explicitly DO NOT Claim or Attempt):
- We do NOT claim to predict validated medical, legal, or criminological severity.
- We do NOT claim that Low/Medium/High are native ground-truth labels in UCF-Crime.
- We do NOT train heavy multi-billion parameter foundation models end-to-end on local hardware.
- We do NOT claim real-time 30+ FPS capability before performing empirical hardware profiling.
- We do NOT claim architectural novelty over I3D, TCN, or Multiple Instance Learning.

---

## 9. Success Criteria
1. **Detection Performance**: Achieve competitive frame-level ROC-AUC on UCF-Crime consistent with published weakly supervised literature baselines.
2. **Alert Stability**: Measurably reduce alert fragmentation (number of false transitions per anomalous event) compared to raw thresholding.
3. **Resource Feasibility**: Training of the TCN + MIL head completes in under 15 minutes on the local 4 GB RTX 3050 GPU using cached features, consuming <1.5 GB VRAM.
4. **Reproducibility**: Entire experiment pipeline reproducible from configuration files with fixed random seeds.
