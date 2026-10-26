# AGENTS.md — Agent Operating Guidelines & System Laws

## 1. Project Identity & Mission
This repository houses the academic research project:
**"Real-Time Video Anomaly Detection and Risk Assessment Using Deep Learning"**

The project investigates whether a weakly supervised video anomaly detection system can be extended with temporal persistence and a transparent risk-oriented decision layer to produce stable, actionable incident-level alerts without requiring ground-truth risk labels.

Target Audience & Context: Undergraduate Deep Learning capstone project; must be technically credible, reproducible, experimentally defensible, and feasible under a local **4 GB VRAM GPU constraint (NVIDIA GeForce RTX 3050 Laptop GPU)**.

---

## 2. Core Scientific Positioning & Honest Claims
Future agents MUST strictly adhere to honest academic positioning:
- **NO False Novelty**: We did NOT invent Video Anomaly Detection (VAD), Multiple Instance Learning (MIL), I3D, C3D, or Temporal Convolutional Networks (TCN).
- **NO State-of-the-Art Claims**: Do not claim the system is automatically SOTA.
- **NO Unmeasured Efficiency Claims**: NEVER claim "30 FPS real-time" or sub-millisecond latency until empirical benchmarks have been executed and logged on local hardware.
- **NO Fake Ground Truth**: Low/Medium/High risk labels DO NOT exist in UCF-Crime or XD-Violence. The risk layer is a **derived, heuristic, transparent decision support layer**, NOT a supervised severity classifier or medical/legal/security diagnostic model.
- **NO Fake Results**: Never write placeholder metrics (e.g., `AUC = 92.4%`, `FPS = 30`) that look like measured results. Explicitly label values as `TARGET`, `EXPECTED`, or `MEASURED (TBD)`.

---

## 3. The Central Project Story & Three-Task Distinction
Every architecture diagram, code module, and evaluation report must clearly separate three distinct tasks:

```text
[ Task 1: Detection ]
Is the video snippet anomalous?
Output: Continuous anomaly score s_t ∈ [0.0, 1.0]
       ↓
[ Task 2: Event Formation ]
Does the anomaly persist across time to constitute a meaningful event?
Output: Formed temporal event with start t_s, end t_e, and persistence duration Δt
       ↓
[ Task 3: Risk Prioritization ]
How should the detected event be prioritized for operational action?
Output: Deterministic risk score R and alert level (Low / Medium / High)
```

---

## 4. Source-of-Truth Hierarchy
When resolving design decisions or conflicts, adhere strictly to this precedence order:
1. **Verified Project Decisions & User Directives**
2. **`RESEARCH_DECISIONS.md`** (Architecture Decision Records)
3. **`ARCHITECTURE.md` / `PROJECT_SPEC.md`**
4. **`PROJECT_STATUS.md`**
5. **Empirical Experiment Results** (`experiments/`)
6. **Peer-Reviewed Scientific Literature & Official Dataset Docs**
7. **Agent Assumptions** (Least authoritative — must be validated)

---

## 5. Architectural Invariants (Frozen Concepts)
The following conceptual pipeline is **FROZEN** and must not be altered:
1. Video temporal snippet sampling.
2. Frozen pretrained video backbone (extracting spatiotemporal feature representations).
3. Dilated causal Temporal Convolutional Network (TCN) modeling temporal evolution with zero future leakage.
4. Weakly supervised Multiple Instance Learning (MIL) anomaly head trained with ranking loss.
5. Post-processing via temporal smoothing.
6. Temporal persistence measurement and event formation.
7. Transparent, deterministic risk decision support layer.

### Implementation Flexibility (Parameters to Validate, NOT Prematurely Freeze):
Do NOT freeze the following without empirical or literature validation:
- Exact backbone weights (I3D Kinetics-400 vs S3D vs R(2+1)D).
- Exact feature dimension ($D_{feat} \in \{512, 1024, 2048\}$ depending on backbone).
- Exact snippet length (16 vs 32 frames).
- TCN layer count, channel count, and dilation schedule.
- Smoothing algorithm (Moving Average vs EMA vs Gaussian).
- Event formation thresholds and minimum persistence duration $\tau_{min}$.
- Risk scoring weights and calibrated operational thresholds.

---

## 6. Prohibited Behaviors for Future Agents
- ❌ **DO NOT switch architectures casually**: No switching to Vision-Language Models (VLMs), multi-billion parameter foundation models, or heavy Transformers without explicit authorization and ADR logging.
- ❌ **DO NOT vibe-code into an unrelated domain**: This is NOT an object detector, a facial recognizer, a surveillance chatbot, or a generic CCTV classifier.
- ❌ **DO NOT commit datasets or large files**: Raw videos, extracted frames, feature caches (`*.npy`, `*.pt`), checkpoints (`*.pth`), and archives (`*.zip`) MUST remain untracked.
- ❌ **DO NOT pollute global Python**: Always use the project virtual environment (`.venv\Scripts\python.exe`).
- ❌ **DO NOT invent arbitrary risk constants**: Calibrate risk thresholds using training/validation distributions only; NEVER use test-set labels.
- ❌ **DO NOT delete or silently alter reference files**: Material in `refrence/` is for inspection only. Do not copy internal reference names or code into project source files.

---

## 7. Workflow Before and After Coding
- **Before coding**: Review `PROJECT_STATUS.md`, `RESEARCH_DECISIONS.md`, and `ARCHITECTURE.md`. Verify the environment and 4 GB VRAM budget.
- **Before modifying dependencies**: Verify necessity and compatibility in `pyproject.toml`.
- **After major milestones**: Update `PROJECT_STATUS.md`, log changes in `CHANGELOG.md`, and record new ADRs in `RESEARCH_DECISIONS.md`.
