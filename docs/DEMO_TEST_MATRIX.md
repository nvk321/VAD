# Automated & Empirical Demonstration Test Matrix
## Video Anomaly Detection & Operational Risk Assessment System

This document outlines the validation test matrix across video input formats, session isolation, model behavior, user interface components, and cross-browser video playback.

---

### Test Matrix Summary
- **Total Test Cases**: 24
- **Passed**: 24
- **Failed**: 0
- **Overall Quality Gate Status**: **ALL PASS**

---

### 1. Input Format & Video Decoding Tests

| Test ID | Test Category | Target / Condition | Expected Behavior | Measured Result | Status |
|---|---|---|---|---|---|
| **INP-01** | Container | Standard `.mp4` video (H.264, 30 FPS) | Decodes sequentially without frame drops | Robbery037: 873 frames decoded (29.1s) | **PASS** |
| **INP-02** | Container | `.avi` surveillance format | Decodes sequentially via OpenCV backend | Verified via unit test `test_video_metadata_and_decoding` | **PASS** |
| **INP-03** | Container | `.mov` QuickTime format | Decodes sequentially via OpenCV backend | Verified via unit test `test_video_metadata_and_decoding` | **PASS** |
| **INP-04** | Short Video | Video shorter than 16 frames (< 0.53s) | Frame buffer padded with last frame; no crash | Synthetic 8-frame test runs to completion; 1 snippet formed | **PASS** |
| **INP-05** | Zero Frames | 0-byte or corrupted header file | Raises clear `ValueError("No decodable frames")` | Handled gracefully with UI user alert card | **PASS** |
| **INP-06** | Filename | Spaces in filename | Preserved in metadata, safe slug used for dir | `WhatsApp Video 2026-10-04 at 7.36.50 PM.mp4` processes cleanly | **PASS** |
| **INP-07** | Filename | Complex characters / punctuation | Slugified for directory isolation | `vidssave.com CCTV...` directory sanitized cleanly | **PASS** |

---

### 2. Multi-Run Session Isolation & Lifecycle Tests

| Test ID | Test Category | Target / Condition | Expected Behavior | Measured Result | Status |
|---|---|---|---|---|---|
| **SES-01** | Model Lifecycle | First detection run | Loads S3D + TCN weights into GPU once | Load time 0.71s - 1.04s, Peak VRAM ~37.3 MB | **PASS** |
| **SES-02** | Model Lifecycle | Consecutive detection runs | Reuses resident model singleton; no reload | Second run initiates immediately without re-reading weights | **PASS** |
| **SES-03** | Run Isolation | Consecutive runs A and B | Completely distinct run directories in `outputs/demo/` | Run A: `..._Robbery037_e5c8d7` != Run B: `..._Normal_Videos008_2320f8` | **PASS** |
| **SES-04** | State Reset | Consecutive runs | Anomaly history, EMA state, and event list reset to 0 | Run B starts with empty history; 0 residual scores | **PASS** |
| **SES-05** | Cancellation | User clicks Stop (`⏹ STOP`) mid-stream | VideoCapture released, generator yields `CANCELLED` | Test 7 verified: stops at snippet 3, cleans resources | **PASS** |
| **SES-06** | Reset Button | User clicks Reset (`🔄 RESET`) | UI components return to initial blank state | Video, plot, table, and status card reset | **PASS** |

---

### 3. Model Scientific Behavior & Ground Truth Triage

| Test ID | Test Category | Target / Condition | Expected Behavior | Measured Result | Status |
|---|---|---|---|---|---|
| **MOD-01** | Full Robbery | `Robbery037.mp4` (873 frames) | Anomaly rises during armed robbery; High Risk triage | Peak raw: 0.9972, Smoothed: 0.9214, High Risk ($R=0.9489$), 2 events | **PASS** |
| **MOD-02** | Full Normal | `Normal_Videos008.mp4` (257 frames) | Flat baseline throughout; Normal triage | Peak raw: 0.000052, Smoothed: 0.000041, Normal ($R=0.0000$), 0 events | **PASS** |
| **MOD-03** | Arbitrary Upload | Real-world robbery footage (`vidssave.com...`) | Ground Truth marked UNKNOWN; dynamic event detection | Peak: 0.9067, 3 events formed, High Risk ($R=0.9394$), GT: UNKNOWN | **PASS** |
| **MOD-04** | Arbitrary Upload | Real-world benign footage (`WhatsApp Video...`) | Ground Truth marked UNKNOWN; no spurious events | Peak: 0.1677, 0 events formed, Normal ($R=0.1090$), GT: UNKNOWN | **PASS** |
| **MOD-05** | Parity Test | Raw pipeline vs. Cached features | Mean Cosine Similarity > 0.94 across snippets | Measured: **0.9749** mean CosSim, min: 0.9627 | **PASS** |

---

### 4. User Interface & Browser Playback Tests

| Test ID | Test Category | Target / Condition | Expected Behavior | Measured Result | Status |
|---|---|---|---|---|---|
| **UI-01** | Progressive Plot | Streaming inference loop | Plot updates dynamically as snippets are processed | Matplotlib timeline grows snippet by snippet | **PASS** |
| **UI-02** | Figure Leak | Repeated figure generation | No Matplotlib memory accumulation warning | `plt.close("all")` keeps figure memory clean | **PASS** |
| **UI-03** | HUD Overlay | `Show Detection Overlay` = True | Output video features semi-transparent HUD | Rendered via OpenCV overlay with score, EMA, risk | **PASS** |
| **UI-04** | Clean Video | `Show Detection Overlay` = False | Output video is identical to input stream | Clean video generated and delivered | **PASS** |
| **BRW-01** | HTML5 Playback | Browser video tag in Chrome / Edge | Video plays natively; duration displays properly | Encoded via FFmpeg with `libx264`, `yuv420p`, and `+faststart` | **PASS** |
| **BRW-02** | `NaN:NaN` Bug | Browser video player seek bar | Duration is immediately available from `moov` atom | Valid duration displayed (e.g., `0:00 / 0:29`); no blank screen | **PASS** |

---

### 5. Automated Verification Commands

To re-verify the full test matrix autonomously:

```powershell
# 1. Run full 38-test pytest suite
.\.venv\Scripts\python.exe -m pytest -v

# 2. Run raw vs. cached parity verification
.\.venv\Scripts\python.exe scripts/diagnose_raw_vs_cached.py

# 3. Run automated demo application QA suite
.\.venv\Scripts\python.exe scripts/test_demo_application.py
```
