# Real-Time Video Anomaly Detection & Operational Risk Assessment
## Demo Operations Guide & Viva Reference Manual

This document provides complete instructions for executing, evaluating, demonstrating, and troubleshooting the end-to-end Video Anomaly Detection (VAD) and Operational Risk Assessment application.

---

### 1. Quick Launch

From the repository root on Windows PowerShell:

```powershell
# Activate local virtual environment
.\.venv\Scripts\Activate.ps1

# Launch the Gradio multi-tab demonstration console
python app.py
```

The application starts a local web server at:
**`http://127.0.0.1:7860`**

Open this URL in Google Chrome, Microsoft Edge, or Mozilla Firefox.

---

### 2. Main Live Demonstration Workflow (Tab 1)

The primary interface provides a production-grade surveillance workflow designed for real-world video uploads:

```text
RAW VIDEO (.mp4, .avi, .mov)
           ↓
   Validate Container & FPS
           ↓
  Sequential Frame Decoding
           ↓
 16-Frame Temporal Snippet Buffer
           ↓
 Preprocessing (224x224, ImageNet Norm)
           ↓
   Frozen S3D Backbone (1024-d)
           ↓
  Dilated Causal TCN (No Future Leakage)
           ↓
   Online Causal EMA Smoothing (α=0.25)
           ↓
 Dual-Threshold Persistence Hysteresis (0.60 / 0.40, τ ≥ 1.5s)
           ↓
 Deterministic Risk Decision Layer (R = 0.65·s_max + 0.35·φ)
           ↓
 Progressive Real-Time UI + Downloadable H.264 Artifacts
```

#### Step-by-Step Operation:
1. **Choose Input**:
   - **Upload Video**: Drag and drop any `.mp4`, `.avi`, or `.mov` file (e.g., CCTV footage, mobile video, security cameras).
   - **OR Select Demo Example**: Choose one of the bundled full-length UCF-Crime benchmark videos:
     - `Robbery037`: Full 873 frames (29.1s) armed robbery at a convenience store counter.
     - `Normal_Videos008`: Full 257 frames (8.6s) calm suburban street surveillance.
     - `Arson015`: Full 1023 frames (34.1s) structural fire and arson incident.
2. **Detection Overlay**:
   - Keep `☑ Show Detection Overlay` checked to render a semi-transparent operational telemetry HUD (Time, Score, Smoothed Score, Risk Tier, Event Status, Active Duration).
   - Uncheck to receive a clean, unannotated video copy.
3. **Start Detection**:
   - Click `▶ START DETECTION`.
   - The video decodes frame by frame; the anomaly curve grows live as each 16-frame snippet is processed.
   - When anomalous behavior sustains across time, an incident card appears and highlights the timeline.
4. **Control During Inference**:
   - `⏹ STOP`: Cleanly interrupts inference, releases video capture and GPU resources, and saves partial results without application crashes.
   - `🔄 RESET`: Clears video, plots, tables, and telemetry to prepare for a fresh run.
5. **Inspect & Download Artifacts**:
   - View HTML5 surveillance playback directly in browser (H.264 / `yuv420p` + `+faststart`).
   - Download generated files: Annotated MP4, Clean MP4, Scores CSV, Incident Log CSV, Timeline PNG, Machine Summary JSON, and Reproducibility Manifest JSON.

---

### 3. Multi-Tab Architecture Overview

| Tab | Name | Purpose | Data Source |
|---|---|---|---|
| **Tab 1** | **🎥 Live Anomaly Detection** | Real-world streaming detection, sequential decoding, online causal EMA, incident formation, and live telemetry | Raw video files (.mp4/.avi/.mov) |
| **Tab 2** | **🧪 Benchmark Examples** | Rapid academic exploration of UCF-Crime test set videos across 13 anomaly classes | Pre-cached S3D features |
| **Tab 3** | **📊 Research & Evaluation Dashboard** | Empirical ablation analysis, ROC/PR curves, event fragmentation reduction, and paper artifacts | Validation experiment results |
| **Tab 4** | **⚙️ System Architecture & Methodology** | Interactive subsystem health checks, mathematical formulations, and viva defense guidance | Live system diagnostics |

---

### 4. Critical Scientific Boundary & Viva Defense

During an academic viva or project evaluation, be prepared to answer key questions regarding system claims:

#### Q1: "Does this system detect weapons, guns, or recognize specific crimes?"
> **No.** The model performs **purely temporal video anomaly detection**. It maps spatiotemporal motion features through a causal Temporal Convolutional Network to compute an anomaly score $s(t) \in [0, 1]$. It does **not** perform object detection, spatial bounding boxes, facial recognition, or legal classification.

#### Q2: "Why do you claim ~25-35 FPS here, but the research paper reports 791 FPS?"
> **The distinction is fundamental:**
> - **791 FPS** is the **downstream TCN + postprocessing throughput** evaluated on pre-extracted feature tensors (Tab 2 benchmark mode).
> - **25-35 FPS** is the **end-to-end raw-video pipeline throughput** on a laptop GPU (NVIDIA RTX 3050 4 GB), which includes OpenCV frame decoding, resizing, ImageNet normalization, 3D spatiotemporal convolution through the frozen S3D backbone, TCN forward pass, causal smoothing, and progressive Matplotlib rendering.

#### Q3: "What is the ground truth for an uploaded video?"
> **Always UNKNOWN.** Unless a video is an official benchmark video with peer-reviewed temporal annotations, the system strictly marks Ground Truth as `UNKNOWN`. We never guess ground truth from file names or visual appearances.

#### Q4: "How is Risk calculated? Is it a trained classifier?"
> **No.** The risk layer is a **deterministic, transparent decision support heuristic**:
> $$R = 0.65 \cdot s_{\max} + 0.35 \cdot \min\left(1.0, \frac{\Delta t}{\tau_{\text{sat}}}\right)$$
> It combines the peak anomaly magnitude ($s_{\max}$) with temporal persistence ($\Delta t$), mapping into four operational tiers:
> - **High Risk** ($R \ge 0.70$): Sustained high-magnitude anomaly; immediate dispatch recommended.
> - **Medium Risk** ($0.45 \le R < 0.70$): Moderate anomaly or brief high anomaly; operator verification needed.
> - **Low Risk** ($0.25 \le R < 0.45$): Isolated transient spike below persistence threshold.
> - **Normal** ($R < 0.25$): Baseline surveillance; non-anomalous.

---

### 5. Multi-Run Directory Structure

Every analysis session receives a dedicated, collision-free run directory:

```text
outputs/demo/YYYYMMDD_HHMMSS_<slug>_<hash>/
├── input/
│   └── original_video.mp4        # Exact uploaded video copy
├── annotated/
│   └── <slug>_annotated.mp4      # H.264 browser-compatible MP4 with HUD
├── clean/
│   └── <slug>_clean.mp4          # Unannotated H.264 stream
├── scores/
│   ├── scores.csv                # Snippet-by-snippet scores & EMA
│   └── incidents.csv             # Formed incident table with durations
├── plots/
│   └── timeline.png              # High-DPI publication timeline figure
├── reports/
│   └── summary.json              # Machine-readable evaluation report
├── telemetry/
│   └── performance.json          # Decoding, inference, and VRAM profile
└── run.json                      # Full reproducibility manifest
```

---

### 6. Troubleshooting Common Edge Cases

| Issue | Cause | Solution |
|---|---|---|
| **`0:00 / NaN:NaN` in Browser Player** | Using obsolete OpenCV `mp4v` codec or missing `moov` atom | Automatically resolved: System uses bundled FFmpeg 7.1 with `libx264`, `yuv420p`, and `-movflags +faststart`. |
| **`WinError 10054` in Terminal** | Gradio client disconnects or browser refreshes mid-stream | Handled gracefully: `VideoStreamInference` catches `GeneratorExit` and `ConnectionResetError`, automatically releasing VideoCapture. |
| **Out of Memory (CUDA OOM)** | Multiple models loaded or memory leak across runs | Handled: Model singleton (`get_streaming_inference_engine`) loads S3D + TCN once into GPU (~37 MB VRAM). Memory is reused across consecutive runs. |
| **Zero Decodable Frames** | Corrupt container or unsupported codec | Validated upfront: Produces clean UI error card without crashing application. |
| **Short Video (< 16 frames)** | Video shorter than minimum temporal receptive field | Supported: Last available frame is replicated to pad snippet buffer to 16 frames. |
