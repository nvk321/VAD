"""Unit tests for deterministic risk decision support layer."""

from pathlib import Path
import sys

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.postprocessing.event_formation import AnomalyEvent
from src.risk.decision_layer import RiskAssessment, RiskDecisionLayer


def test_risk_decision_layer_calibration():
    """Verify threshold ordering and weight constraints."""
    with pytest.raises(ValueError):
        # Weights must sum to 1.0
        RiskDecisionLayer(weight_magnitude=0.8, weight_duration=0.5)

    with pytest.raises(ValueError):
        # Thresholds must be ordered theta_base < theta_low < theta_high
        RiskDecisionLayer(theta_high=0.5, theta_low=0.6, theta_base=0.2)


def test_risk_level_assessments():
    """Verify tier assignment across high, medium, low, and normal events."""
    layer = RiskDecisionLayer(
        weight_magnitude=0.65,
        weight_duration=0.35,
        duration_saturation_sec=8.0,
        theta_high=0.70,
        theta_low=0.45,
        theta_base=0.25,
    )

    # 1. High Risk Event: High magnitude (0.90) and long persistence (8.0s -> saturation=1.0)
    # R = 0.65*0.90 + 0.35*1.0 = 0.585 + 0.35 = 0.935 >= 0.70 -> High Risk
    high_event = AnomalyEvent(
        event_id=1,
        start_snippet=10,
        end_snippet=25,
        start_sec=5.3,
        end_sec=13.3,
        duration_sec=8.0,
        peak_score=0.90,
        mean_score=0.82,
    )
    assess_high = layer.assess_event(high_event)
    assert assess_high.risk_level == "High Risk"
    assert assess_high.risk_score >= 0.70
    assert "High Risk" in assess_high.explanation

    # 2. Medium Risk Event: Moderate magnitude (0.65), medium persistence (4.0s -> saturation=0.5)
    # R = 0.65*0.65 + 0.35*0.5 = 0.4225 + 0.175 = 0.5975 in [0.45, 0.70) -> Medium Risk
    med_event = AnomalyEvent(
        event_id=2,
        start_snippet=5,
        end_snippet=12,
        start_sec=2.6,
        end_sec=6.6,
        duration_sec=4.0,
        peak_score=0.65,
        mean_score=0.58,
    )
    assess_med = layer.assess_event(med_event)
    assert assess_med.risk_level == "Medium Risk"
    assert 0.45 <= assess_med.risk_score < 0.70

    # 3. Low Risk Event: Low magnitude (0.45), brief persistence (1.6s -> saturation=0.2)
    # R = 0.65*0.45 + 0.35*0.2 = 0.2925 + 0.07 = 0.3625 in [0.25, 0.45) -> Low Risk
    low_event = AnomalyEvent(
        event_id=3,
        start_snippet=0,
        end_snippet=3,
        start_sec=0.0,
        end_sec=1.6,
        duration_sec=1.6,
        peak_score=0.45,
        mean_score=0.42,
    )
    assess_low = layer.assess_event(low_event)
    assert assess_low.risk_level == "Low Risk"
    assert 0.25 <= assess_low.risk_score < 0.45

    # 4. Normal video with zero events
    assess_normal = layer.assess_video(events=[], overall_peak_score=0.15)
    assert assess_normal.risk_level == "Normal"
    assert assess_normal.risk_score < 0.25


if __name__ == "__main__":
    test_risk_decision_layer_calibration()
    test_risk_level_assessments()
    print("ALL RISK TESTS PASSED!")
