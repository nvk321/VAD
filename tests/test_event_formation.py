"""Unit tests for temporal persistence and event formation engine."""

from pathlib import Path
import sys

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from src.postprocessing.event_formation import AnomalyEvent, EventFormationEngine, evaluate_persistence_ablation
from src.postprocessing.smoothing import exponential_moving_average, gaussian_smoothing, moving_average


def test_smoothing_filters():
    """Verify that moving average, EMA, and Gaussian smoothing behave predictably."""
    raw = np.array([0.1, 0.1, 0.9, 0.1, 0.1, 0.8, 0.8, 0.8, 0.1], dtype=np.float32)

    ma = moving_average(raw, window_size=3)
    assert len(ma) == len(raw)
    assert ma.min() >= 0.0 and ma.max() <= 1.0

    ema = exponential_moving_average(raw, alpha=0.3)
    assert len(ema) == len(raw)
    assert ema.min() >= 0.0 and ema.max() <= 1.0

    gauss = gaussian_smoothing(raw, sigma=1.0)
    assert len(gauss) == len(raw)
    assert gauss.min() >= 0.0 and gauss.max() <= 1.0


def test_hysteresis_and_persistence_filtering():
    """Verify dual-threshold hysteresis triggers and short blip rejection."""
    # 20 snippets (~10.6s at 0.533s/snippet)
    # Snippet indices:
    # 0..2: normal (0.1)
    # 3: momentary spike (0.85) for 1 snippet (0.53s) -> should be discarded as transient blip!
    # 4..6: normal (0.1)
    # 7..12: sustained anomaly (0.75, 0.70, 0.65, 0.55, 0.50, 0.45) -> 6 snippets (3.2s) -> kept!
    # 13..19: normal (0.1)
    scores = np.array([
        0.1, 0.1, 0.1,
        0.85,
        0.1, 0.1, 0.1,
        0.75, 0.70, 0.65, 0.55, 0.50, 0.45,
        0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1
    ], dtype=np.float32)

    engine = EventFormationEngine(
        threshold_high=0.60,
        threshold_low=0.40,
        min_persistence_sec=1.5,   # Requires at least ~3 snippets
        max_gap_sec=1.0,
        snippet_duration_sec=0.533,
    )

    events = engine.form_events(scores)

    # The momentary spike at index 3 (1 snippet = 0.53s < 1.5s) must be discarded!
    # Only the sustained event (indices 7..12) should form!
    assert len(events) == 1, f"Expected 1 valid event, found {len(events)}"

    e = events[0]
    assert e.start_snippet == 7
    assert e.end_snippet == 12
    assert e.duration_sec >= 3.0
    assert e.peak_score >= 0.75
    assert e.mean_score > 0.55


def test_gap_bridging():
    """Verify that adjacent candidate intervals separated by <= max_gap are merged."""
    # Sustained anomaly with a 1-snippet dip in between:
    # 0..4: anomaly (0.75)
    # 5: brief dip below threshold_low (0.35) for 1 snippet
    # 6..9: anomaly continues (0.70)
    # 10..15: normal (0.1)
    scores = np.array([
        0.75, 0.75, 0.75, 0.75, 0.75,
        0.35,
        0.70, 0.70, 0.70, 0.70,
        0.1, 0.1, 0.1, 0.1, 0.1, 0.1
    ], dtype=np.float32)

    engine = EventFormationEngine(
        threshold_high=0.60,
        threshold_low=0.40,
        min_persistence_sec=1.0,
        max_gap_sec=1.2,  # 1.2s bridges up to 2 snippets
        snippet_duration_sec=0.533,
    )

    events = engine.form_events(scores)
    assert len(events) == 1, f"Gap should be bridged into 1 event, got {len(events)}"
    assert events[0].start_snippet == 0
    assert events[0].end_snippet == 9


def test_persistence_ablation_metrics():
    """Verify evaluate_persistence_ablation outputs valid metrics and fragmentation reduction."""
    raw = np.array([0.1, 0.6, 0.2, 0.7, 0.1, 0.8, 0.8, 0.8, 0.1, 0.9, 0.1])
    smoothed = exponential_moving_average(raw, alpha=0.3)

    ablation = evaluate_persistence_ablation(raw, smoothed, threshold=0.5)

    assert "raw_alert_blocks" in ablation
    assert "raw_chatter_toggles" in ablation
    assert "formed_events_count" in ablation
    assert "fragmentation_reduction_pct" in ablation
    assert ablation["raw_alert_blocks"] >= ablation["formed_events_count"]


if __name__ == "__main__":
    test_smoothing_filters()
    test_hysteresis_and_persistence_filtering()
    test_gap_bridging()
    test_persistence_ablation_metrics()
    print("ALL EVENT FORMATION TESTS PASSED!")
