# EXPERIMENT_PLAN.md — Experimental Protocol & Ablation Matrix

## 1. Experimental Philosophy & Integrity Rules
This document formalizes the experimental design for **Real-Time Video Anomaly Detection and Risk Assessment**.

### Mandatory Scientific Standards:
1. **Preserve Clean Baselines**: Every component must answer: *"Does this addition provide measurable value over the simpler baseline?"*
2. **Honesty & No Fake Results**: NEVER report synthetic or unmeasured performance numbers. All metrics must be categorized as:
   - **PUBLISHED**: Results cited directly from peer-reviewed literature.
   - **TARGET**: Realistic engineering and research goals.
   - **EXPECTED**: Theoretical or literature-guided estimates.
   - **MEASURED**: Empirically produced runs verified with logged checkpoints and manifests (currently **TBD / Not Yet Measured**).
3. **Data Quarantine**: Test-set annotations are strictly quarantined during hyperparameter tuning and risk threshold calibration. Validation splits are derived solely from training partitions.

---

## 2. Experimental Hierarchy & Ablation Matrix

```text
                                  [ EXP A: BASELINE MIL ]
                                  Frozen I3D + MLP Head
                                            │
                                            ▼
                               [ EXP B: BASELINE + TCN ]
                           Frozen I3D + Dilated Causal TCN
                               (Ablation: Temporal Modeling)
                                            │
                                            ▼
                       [ EXP C: TCN + PERSISTENCE & EVENT FORMATION ]
                             + Smoothing + Hysteresis Engine
                             (Ablation: Event Continuity)
                                            │
                                            ▼
                               [ EXP D: FULL VAD-RISK SYSTEM ]
                                + Calibrated Decision Layer
                             (Ablation: Actionable Prioritization)
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    ▼                                               ▼
      [ EXP E: EFFICIENCY PROFILING ]               [ EXP F: SECONDARY EVALUATION ]
     Latency, FPS, VRAM on RTX 3050                  XD-Violence Generalization
```

---

## 3. Detailed Experiment Specifications

### Experiment A: Standard Weakly Supervised MIL Baseline
- **Objective**: Establish the empirical lower bound using canonical weakly supervised MIL without temporal context modeling.
- **Architecture**: Frozen video backbone (I3D, $D_{feat} = 1024$) $\rightarrow$ Snippet Feature Sequence $\rightarrow$ 2-layer MLP Anomaly Head $\rightarrow$ Sigmoidal Score $s_t$.
- **Loss**: Standard max-margin MIL ranking loss with sparsity and smoothness penalties.
- **Evaluated Metric**: Frame-level ROC-AUC and PR-AUC on UCF-Crime test set.
- **Research Question Addressed**: What is the baseline detection accuracy before introducing sequence-level temporal convolutions?

### Experiment B: Baseline + Dilated Causal TCN
- **Objective**: Measure the marginal contribution of temporal sequence modeling.
- **Architecture**: Frozen video backbone $\rightarrow$ Dilated Causal TCN ($L$ layers, dilation schedule $[1, 2, 4, 8]$) $\rightarrow$ 2-layer MLP Anomaly Head $\rightarrow s_t$.
- **Loss**: Same MIL ranking loss as Exp A.
- **Ablation Studies**:
  - TCN depth ($L \in \{2, 3, 4\}$).
  - Hidden channel capacity ($C_{hidden} \in \{64, 128, 256\}$).
  - Causal vs. Non-Causal convolution (verifying zero future leakage impact).
- **Evaluated Metric**: Frame-level ROC-AUC, PR-AUC, and snippet-level score variance.
- **Hypothesis**: TCN temporal receptive field suppresses isolated single-frame score spikes and improves ROC-AUC by 1.5–3.0%.

### Experiment C: TCN + Temporal Persistence & Event Formation
- **Objective**: Quantify the impact of temporal smoothing and hysteresis event formation on alert stability and false alarm suppression.
- **Architecture**: Exp B Model $\rightarrow$ Temporal Smoothing Filter $\rightarrow$ Hysteresis Windowing ($T_{high}, T_{low}$) $\rightarrow$ Min Duration Filter ($\tau_{min}$) $\rightarrow$ Active Incidents $[t_{start}, t_{end}]$.
- **Ablation Studies**:
  - Smoothing Filter: Moving Average vs. Exponential Moving Average (EMA) vs. Gaussian 1D.
  - Thresholding: Static Single Threshold ($T$) vs. Dual-Threshold Hysteresis ($T_{high}, T_{low}$).
  - Duration Threshold: $\tau_{min} \in \{8, 16, 24, 32\}$ frames.
- **Evaluated Metrics**:
  - Alert Fragmentation Index ($AFI = \frac{\text{Total Triggered Alert Segments}}{\text{True Ground-Truth Events}}$).
  - Event-level Precision, Recall, and F1-score (at temporal IoU threshold $IoU \ge 0.5$).
  - Mean Onset Detection Delay ($\delta_t = t_{alert\_start} - t_{gt\_start}$).

### Experiment D: Full System with Transparent Risk Decision Layer
- **Objective**: Evaluate the behavior, sensitivity, and prioritization efficacy of the deterministic risk decision layer.
- **Formulation**: $R = w_{mag} \cdot s_{max} + w_{dur} \cdot \min(1.0, \Delta t / \Delta t_{sat})$.
- **Operational Tiers**: Low Risk, Medium Risk, High Risk.
- **Calibration Protocol**: Thresholds $(\Theta_{base}, \Theta_{low}, \Theta_{high})$ calibrated using empirical 50th, 75th, and 90th percentiles of validation set anomalous event scores.
- **Ablation Studies**:
  - Magnitude vs. Duration weighting ($w_{mag} \in [0.5, 0.8]$, $w_{dur} = 1 - w_{mag}$).
  - Saturation limit $\Delta t_{sat} \in \{90, 150, 300\}$ frames.
- **Evaluated Metrics**:
  - Alert distribution across crime categories.
  - Sensitivity analysis across varying calibration percentiles.
  - Qualitative verification of incident priority ranking.

### Experiment E: Computational Efficiency & Latency Profiling
- **Objective**: Empirically profile the system on the physical target hardware (**NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB VRAM**).
- **Measurement Protocol**:
  - Warmup runs: 50 iterations.
  - Timed benchmark: 500 iterations with CUDA synchronization (`torch.cuda.synchronize()`).
- **Measured Metrics**:
  1. *Feature Extraction Throughput*: Frames processed per second (FPS) with frozen backbone.
  2. *TCN + MIL Head Inference Latency*: Wall-clock time (milliseconds) per video feature sequence.
  3. *Post-Processing & Risk Computation Latency*: Milliseconds per video.
  4. *End-to-End Pipeline FPS*: Measured with simulated live streaming input.
  5. *Peak GPU Memory Allocation*: Measured via `torch.cuda.max_memory_allocated()`.

### Experiment F: Secondary Robustness Evaluation (XD-Violence)
- **Objective**: Assess out-of-domain generalization on the violence-focused XD-Violence benchmark.
- **Execution Condition**: Conditional upon training time and local compute bandwidth.
- **Evaluated Metric**: Average Precision (AP / PR-AUC) on XD-Violence test partition.

---

## 4. Benchmark Tracking Table

| Exp ID | Configuration | Frame ROC-AUC (%) | Frame PR-AUC (%) | Alert Frag. Index | Mean Delay (s) | Model Latency (ms) | Peak VRAM (MB) | Status |
|---|---|---|---|---|---|---|---|---|
| *Literature* | Sultani et al. (C3D) | 75.41 (Published) | — | High (Published) | — | — | — | Baseline Reference |
| *Literature* | RTFM (I3D, ICCV 2021) | 84.30 (Published) | — | — | — | — | — | SOTA Reference |
| **Exp A** | Baseline MIL (I3D) | TARGET: ~77.0 | TARGET: ~25.0 | EXPECTED: High | 0.0 | TBD | TBD | **Not Yet Executed** |
| **Exp B** | Baseline + TCN | TARGET: ~79.5 | TARGET: ~28.0 | EXPECTED: Mod. | TBD | TBD | TBD | **Not Yet Executed** |
| **Exp C** | TCN + Persistence | TARGET: ~80.0 | TARGET: ~29.0 | TARGET: < 1.3 | TARGET: < 0.8s | TBD | TBD | **Not Yet Executed** |
| **Exp D** | Full + Risk Layer | TARGET: ~80.0 | TARGET: ~29.0 | TARGET: < 1.2 | TARGET: < 0.8s | TBD | TBD | **Not Yet Executed** |
| **Exp E** | Efficiency Profile | — | — | — | — | TARGET: < 5 ms | TARGET: < 1.5 GB | **Not Yet Executed** |
| **Exp F** | Cross-Dataset (XD) | TARGET: TBD | TARGET: TBD | — | — | TBD | TBD | **Deferred to Phase 7** |
