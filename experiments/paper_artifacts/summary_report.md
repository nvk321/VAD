# Academic Research Milestone Report
## Real-Time Video Anomaly Detection and Operational Risk Assessment Using Deep Learning

**Date:** 2026-10-28  
**Hardware Environment:** NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM Profile)  
**Dataset:** UCF-Crime (1,900 videos; 1,369 Train / 241 Validation / 290 Test Quarantined)  
**Primary Backbone:** Frozen S3D Inception (1024-d features)  

---

### 1. Executive Summary & Core Research Findings

This research investigates whether a weakly supervised video anomaly detection pipeline can be extended with **causal temporal modeling**, **temporal persistence filtering**, and a **transparent deterministic risk layer** to generate stable, actionable incident-level alerts without requiring ground-truth risk labels.

Across controlled, reproducible experiments conducted strictly within local 4 GB VRAM constraints:
1. **Experiment A (MIL Baseline)**: Validated that frozen S3D spatiotemporal representations paired with a multiple-instance learning ranking head achieve **0.8253 ROC-AUC** and **0.8236 PR-AUC** on UCF-Crime.
2. **Experiment B (Dilated Causal TCN + MIL)**: Introducing dilated causal convolutions ($d \in [1, 2, 4]$) with channel-wise layer normalization yielded **0.8750 ROC-AUC** and **0.8705 PR-AUC**, demonstrating a decisive **+4.97% AUC improvement** from explicit temporal context without future leakage.
3. **Experiment C (Temporal Persistence & Event Formation)**: Evaluated across all 241 validation videos. Dual-threshold hysteresis ($T_{high}=0.60, T_{low}=0.40$) and minimum persistence duration ($\tau_{min}=1.5s$) **reduced alert fragmentation by 43.5%** (from 131 raw thresholded blocks to 74 consolidated incidents) and eliminated 193 chattering transitions.
4. **Experiment D (Transparent Risk Decision Support Layer)**: Deterministic heuristic combining peak magnitude ($w_{mag}=0.65$) and normalized persistence ($w_{dur}=0.35$) successfully stratified incidents into operational tiers: 174 Normal (72.2%), 14 Medium Risk (5.8%), and 53 High Risk (22.0%), providing transparent, explainable triage support.
5. **Experiment E (Computational Efficiency)**: Total end-to-end processing latency is **20.23 ms per 16-frame snippet**, achieving **791.1 effective video FPS** on local hardware with a peak memory footprint of **1.08 GB VRAM** (27% of the 4 GB hardware ceiling).


---

### 2. Experimental Progression Matrix

| Phase / Experiment | Architecture | Val ROC-AUC | Val PR-AUC | Fragmentation Red. | Peak VRAM | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Exp A: MIL Baseline** | S3D (1024-d) + MIL Head | 0.8253 | 0.8236 | Baseline (131 blocks) | 43.8 MB | **MEASURED** |
| **Exp B: Causal TCN** | S3D + Dilated Causal TCN + MIL | **0.8750** | **0.8705** | Baseline (131 blocks) | 44.1 MB | **MEASURED** |
| **Exp C: Persistence** | S3D + TCN + EMA + Hysteresis | 0.8750 | 0.8705 | **-43.5% (74 events)** | 44.1 MB | **MEASURED** |
| **Exp D: Risk Layer** | (C) + Deterministic Decision Support | 0.8750 | 0.8705 | 53 High / 14 Med / 0 Low | 44.1 MB | **MEASURED** |
| **Exp E: Efficiency** | Full Decoupled E2E Pipeline | 0.8750 | 0.8705 | E2E Latency: 20.23 ms | **1.08 GB** | **MEASURED** |

---

### 3. Generated Academic Artifacts
- **Figures:**
  - `figure1_training_curves_comparison.png`
  - `figure2_roc_pr_comparison.png`
  - `persistence_ablation.png`
  - `risk_distribution.png`
  - `qualitative_anomaly_timeline.png`
  - `latency_breakdown.png`
- **LaTeX Tables:**
  - `tables/ablation.tex`
  - `tables/results.tex`
  - `tables/efficiency.tex`
- **Interactive Demonstration:**
  - `app.py` (Gradio web application running on port 7860)
