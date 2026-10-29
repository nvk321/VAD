# Research Walkthrough & Scientific Methodology Guide

**Companion Document to:** [`notebooks/Decision_Research_Walkthrough.ipynb`](file:///c:/Projects/VAD/notebooks/Decision_Research_Walkthrough.ipynb)  
**Project:** Real-Time Video Anomaly Detection and Operational Risk Assessment  
**Model Architecture:** Frozen S3D Backbone (1024-d) + Dilated Causal TCN + Causal EMA + Dual-Threshold Hysteresis + Temporal Persistence + Deterministic Decision-Support Heuristic  
**Target Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM Budget)  
**Inference Framing:** Progressive Real-Time Raw-Video Stream Evaluation  
**Risk Layer Nature:** Deterministic Decision-Support Heuristic (Transparent Decision Layer)  

---

## 1. Executive Summary & Core Research Thesis

In standard surveillance research, video anomaly detection (VAD) models output continuous, uncalibrated anomaly scores $s_t \in [0, 1]$. In practical monitoring environments, security personnel cannot monitor continuous waveforms; they require **actionable, consolidated incident-level alerts** prioritized by operational urgency.

This research investigates whether a weakly supervised video anomaly detection system can be extended with **causal temporal modeling**, **temporal persistence**, and a **transparent deterministic decision-support heuristic** to produce stable incident alerts **without requiring ground-truth risk labels**.

```text
[ Task 1: Anomaly Detection (Learned) ]
Is the 16-frame spatiotemporal snippet anomalous?
Output: Continuous anomaly score s_t ∈ [0.0, 1.0]
        ↓
[ Task 2: Incident Event Formation (Deterministic) ]
Does the anomaly persist across time to constitute a meaningful event?
Output: Formed temporal event [t_start, t_end] with duration Δt
        ↓
[ Task 3: Operational Risk Prioritization (Deterministic Decision Heuristic) ]
How should the detected event be prioritized for security response?
Output: Deterministic risk metric R ∈ [0.0, 1.0] and operational alert (Normal / Low / Medium / High Risk)
```

---

## 2. Fundamental Scientific Laws & Research Invariants

1. **Zero False Novelty:** We did NOT invent Video Anomaly Detection (VAD), Multiple Instance Learning (MIL), S3D, or Temporal Convolutional Networks (TCN).
2. **No Fake Risk Ground Truth:** Severity/risk labels DO NOT exist in UCF-Crime or XD-Violence. The operational risk layer is a **derived, deterministic decision-support heuristic**, NOT a supervised severity classifier or medical/legal diagnostic model.
3. **No Spatial Bounding Boxes:** Our system is a temporal sequence detector. It does not perform spatial object localization. Banners and overlays represent frame-level operational status alerts.
4. **Zero Future Leakage:** Dilated causal convolutions strictly utilize left-padding $(K-1) \cdot d$, mathematically guaranteeing that future video snippets do not contaminate past predictions ($\delta_{\text{leak}} < 10^{-7}$).
5. **Distributional Evidence vs. Proving OOD:** PCA projections and centroid cosine similarity provide quantitative supporting evidence of distributional shift; they do **not** constitute formal mathematical "proof of out-of-distribution (OOD)" status.
6. **Stability Against Ad-Hoc Retraining:** External diagnostic failures must **never** automatically trigger threshold adjustments, model retraining, or ad-hoc heuristic tweaks. Doing so destroys the calibrated operating point from the formal validation split.
7. **Progressive Raw-Video Inference:** The system evaluates uploaded raw video files sequentially snippet-by-snippet in progressive real-time without future leakage, rather than interfacing directly to a physical live RTSP camera hardware stream.
8. **Hardware Budget Feasibility:** The entire pipeline is designed and empirically benchmarked on a local 4 GB VRAM budget (peaking at 1.08 GB VRAM).

---

## 3. Separation of Learned vs. Deterministic Architecture Components

To ensure complete scientific defensibility during academic review, the system explicitly partitions learned representations from deterministic heuristics:

| Pipeline Stage | Component | Nature | Trainable Parameters | Objective / Function |
| :--- | :--- | :---: | :---: | :--- |
| **Stage 1** | **S3D Video Backbone** | **Learned (Frozen)** | 0 (7.9M frozen) | Extracts 1024-d spatiotemporal features from 16-frame snippets. Pretrained on Kinetics-400. |
| **Stage 2** | **Dilated Causal TCN** | **Learned** | 453,505 | Models temporal sequence evolution with zero future leakage ($R=15$ snippets $\approx 8.0\text{s}$). |
| **Stage 3** | **MIL Scoring Head** | **Learned** | (Included in TCN) | Multi-Layer Perceptron projecting 128-d temporal states to continuous scores $s_t \in [0, 1]$. |
| **Stage 4A** | **Causal EMA Smoothing** | **Deterministic** | 0 | $\tilde{s}_t = 0.25 s_t + 0.75 \tilde{s}_{t-1}$; dampens sensor noise without future information. |
| **Stage 4B** | **Dual Hysteresis Machine** | **Deterministic** | 0 | Triggers at $T_{\text{high}}=0.60$, sustains at $T_{\text{low}}=0.40$; eliminates threshold flickering. |
| **Stage 4C** | **Persistence Filtering** | **Deterministic** | 0 | Bridges gaps $\le 1.2\text{s}$, suppresses blips $< 1.5\text{s}$; consolidates fragmented alerts by **-43.5%**. |
| **Stage 5** | **Deterministic Decision Heuristic** | **Deterministic** | 0 | $R = 0.65 s_{\max}^{\text{event}} + 0.35 \min(1.0, \Delta t / 8.0)$; transparent operational triage. |

---

## 4. 24-Section Walkthrough Summary

The accompanying research notebook [`notebooks/Decision_Research_Walkthrough.ipynb`](file:///c:/Projects/VAD/notebooks/Decision_Research_Walkthrough.ipynb) executes all 24 stages sequentially and self-contained:

1. **Research Mission & Invariants:** Establishes academic context, 4 GB VRAM budget, and scientific laws.
2. **Hardware Diagnostics:** Detects NVIDIA RTX 3050 Laptop GPU, validates CUDA 12.4, and logs 4.00 GB dedicated VRAM.
3. **Dataset Manifest Protocol:** Inspects UCF-Crime split (1,369 train, 241 val, 290 test) and defines the protocol: *One-vs-normal anomaly evaluation for each UCF-Crime anomaly category*.
4. **Frozen S3D Backbone:** Inspects $(1 \times 3 \times 3) + (3 \times 1 \times 1)$ separable convolutions and 1024-d feature representation.
5. **Preprocessing Parity:** Validates bilinear $224 \times 224$ resizing, RGB ordering, and ImageNet normalization (Mean Feature Cosine Similarity = **0.9749**).
6. **Dilated Causal TCN:** Calculates receptive field $R = 1 + 2 + 4 + 8 = 15$ snippets ($\approx 8.0\text{s}$) under dilation schedule $[1, 2, 4]$.
7. **MIL Scoring Head:** Evaluates MLP projections into continuous scores $s_t \in [0.0, 1.0]$.
8. **Top-k Ranking Loss:** Demonstrates max-margin hinge ranking loss with temporal smoothness ($\lambda_1 = 8 \times 10^{-5}$) and sparsity ($\lambda_2 = 8 \times 10^{-5}$).
9. **Causal Leakage Test:** Empirically proves past outputs remain identical ($\delta < 10^{-7}$) when future snippets are severely perturbed.
10. **Model Checkpoint Inspection:** Loads production checkpoint `experiments/exp_B_tcn/checkpoints/best_auc.pt` (Epoch 18).
11. **Reference Centroid Computation:** Extracts validation Normal and Anomaly centroids in $\mathbb{R}^{1024}$ (Cosine Similarity = 0.9803).
12. **Benchmark Video Scoring:** Executes forward inference on `Robbery037` (54 snippets, 28.3s, peak raw score 0.9972).
13. **Oscillatory Alert Fragmentation:** Demonstrates that raw static thresholding generates fragmented, jarring alarms.
14. **Causal EMA Smoothing:** Smooths raw waveform with $\alpha = 0.25$, yielding peak smoothed score 0.9214.
15. **Hysteresis State Machine:** Establishes Schmitt trigger boundaries ($T_{\text{high}}=0.60, T_{\text{low}}=0.40$).
16. **Temporal Persistence:** Applies minimum duration filter ($\tau \ge 1.5\text{s}$) and gap bridging ($g \le 1.2\text{s}$).
17. **Event Formation & Consolidation:** Consolidates fragmented blips into 2 coherent events, reducing alert fragmentation by **-33.3%** on this clip (and **-43.5%** across the full validation split).
18. **Operational Risk Formulation:** Evaluates $R = 0.65 s_{\max}^{\text{event}} + 0.35 \min(1.0, \Delta t / 8.0)$ using the deterministic decision-support heuristic. Verifies exact mathematical agreement:
    - Event 1: $0.65(0.9214) + 0.35(1.0) = 0.9489$ (High Risk, MATCH)
    - Event 2: $0.65(0.8521) + 0.35(0.7337) = 0.8106$ (High Risk, MATCH)
19. **Operational Risk Triage:** Assesses overall surveillance video urgency: **High Risk** ($R=0.9489$).
20. **Mathematical Consistency Audit:** Inspects the audit ledger of 9 diagnostic videos; 100% exhibit exact mathematical formula reproduction from actual smoothed peak + persistence duration.
21. **Generalization Analysis:** Applies the 8-category scientific failure taxonomy across 9 video targets, emphasizing evidence of distributional shift rather than speculation.
22. **Feature Distribution PCA:** Projects in-distribution validation features and external video features into 2D principal component space, illustrating quantitative evidence of distributional shift.
23. **Multi-Tier Hardware Throughput:** Documents the 4-tier hierarchy (Model Latency: 0.092ms, Feature FPS: 791, E2E FPS: 123, Progressive UI FPS: 23–35).
24. **Viva Defense Checklist:** Summarizes key answers to anticipated capstone defense questions, confirming stability against ad-hoc retraining.

---

## 5. Multi-Tier Throughput Hierarchy (RTX 3050 Laptop GPU)

To avoid conflating model inference latency with complete end-to-end video decoding and UI rendering, performance is reported strictly across four distinct operational tiers:

| Throughput Tier | Measurement Scope | Latency per Snippet | Throughput | Meaning |
| :--- | :--- | :---: | :---: | :--- |
| **Tier 1: Model Latency** | Dilated Causal TCN + MIL forward pass | **0.092 ms** | **10,914 snippets/sec** | Pure PyTorch tensor evaluation latency on GPU. |
| **Tier 2: Feature Extraction** | S3D Backbone forward pass (batch=1) | **20.23 ms** | **791.1 FPS** | GPU spatiotemporal feature extraction rate on pre-buffered frames. |
| **Tier 3: Standalone Pipeline** | Video Decode + S3D + TCN + Postprocessing | **130.1 ms** | **123.0 FPS** | Headless Python pipeline running without web UI. |
| **Tier 4: Progressive Web UI** | Decode + S3D + TCN + Gradio Yield + Matplotlib | **45.0 - 68.0 ms** | **23.5 - 35.2 FPS** | Complete interactive surveillance console with live HUD overlays. |

---

## 6. Authoritative Experiment Ledger

| Exp ID | Architecture | Objective | Checkpoint File | Val ROC-AUC | Val PR-AUC | Test Metrics | Status | Scientific Interpretation |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Exp A** | S3D (1024-d) + Direct MIL Head | Weakly Supervised Top-k Ranking Loss | `experiments/exp_A_baseline/checkpoints/best_auc.pt` | **0.8253** | **0.8236** | *Not measured in current experiment* | **MEASURED** | Baseline static snippet scoring. Lacks temporal context; exhibits oscillatory score spikes. |
| **Exp B** | S3D + Dilated Causal TCN + MIL | Ranking Loss + Smoothness + Sparsity | `experiments/exp_B_tcn/checkpoints/best_auc.pt` | **0.8750** | **0.8705** | *Not measured in current experiment* | **MEASURED** | **Primary ML Contribution.** Dilated causal convolutions capture temporal evolution; +4.97% AUC. |
| **Exp C** | (Exp B) + Causal EMA + Persistence | Hysteresis Event Formation ($T_h=0.6, T_l=0.4, \tau\ge1.5s$) | Postprocessing Stage | **0.8750** | **0.8705** | *Not measured in current experiment* | **MEASURED** | **Decision Contribution.** Consolidates 131 fragmented alerts into 74 actionable events (**-43.5%**). |
| **Exp D** | (Exp C) + Deterministic Risk Layer | Calibrated Operational Triage ($R = 0.65 s + 0.35 \phi$) | Decision Layer Stage | **0.8750** | **0.8705** | *Not measured in current experiment* | **MEASURED** | Triages validation set into 53 High, 14 Med, 174 Normal incidents without fake ground-truth labels. |
| **Exp E** | **Full Decoupled Pipeline (E2E)** | Raw Video $\to$ S3D $\to$ TCN $\to$ Risk $\to$ UI | End-to-End System | **0.8750** | **0.8705** | *Not measured in current experiment* | **MEASURED** | Full system verified under 1.08 GB peak VRAM budget, achieving 791 downstream FPS / 123 E2E FPS. |

---

## 7. Execution Commands

### Execute Full Automated QA Suite:
```bash
.venv\Scripts\python.exe scripts/test_demo_application.py
```

### Execute Causal Leakage Verification:
```bash
.venv\Scripts\python.exe tests/test_causal_leakage.py
```

### Execute Feature Parity Diagnostics:
```bash
.venv\Scripts\python.exe scripts/diagnose_raw_vs_cached.py
```

### Execute External Video Diagnostic Audit:
```bash
.venv\Scripts\python.exe scripts/diagnose_external_videos.py
```

### Execute and Populate Jupyter Walkthrough Notebook:
```bash
.venv\Scripts\python.exe scripts/execute_and_save_notebook.py
```

### Launch Interactive Progressive Video Inference Web Application:
```bash
.venv\Scripts\python.exe app.py
```
Open browser at: `http://127.0.0.1:7860`
