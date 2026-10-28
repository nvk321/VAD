"""Transparent Deterministic Risk Decision Support Layer.

Stage 5 Component:
Transforms detected incident events into prioritized operational alert levels
(Low Risk, Medium Risk, High Risk) without requiring ground-truth risk labels.

Combines peak anomaly magnitude and normalized temporal persistence:
    R = w_mag * s_max + w_dur * min(1.0, duration / duration_sat)

Decision Rules:
- Transparent & Heuristic: Fully explainable breakdown of risk drivers.
- Calibrated on Validation Distributions: Never tuned on the quarantined test set.
- Decision Support: Clearly documented as operational triage support, NOT diagnostic severity ground-truth.
"""

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from src.postprocessing.event_formation import AnomalyEvent


@dataclass
class RiskAssessment:
    """Operational risk assessment for a detected event or overall video."""

    risk_level: str  # "Normal", "Low Risk", "Medium Risk", "High Risk"
    risk_score: float  # Continuous R in [0.0, 1.0]
    peak_score: float  # s_max
    duration_sec: float  # Delta t
    persistence_ratio: float  # phi(Delta t)
    magnitude_contribution: float  # w_mag * s_max
    duration_contribution: float  # w_dur * phi(Delta t)
    event_id: Optional[int] = None
    explanation: str = ""

    def to_dict(self) -> Dict[str, Union[str, float, int, None]]:
        """Return dictionary representation."""
        return asdict(self)


class RiskDecisionLayer:
    """Deterministic, transparent decision support layer for operational incident triage."""

    def __init__(
        self,
        weight_magnitude: float = 0.65,
        weight_duration: float = 0.35,
        duration_saturation_sec: float = 8.0,
        theta_high: float = 0.70,
        theta_low: float = 0.45,
        theta_base: float = 0.25,
    ):
        if not np.isclose(weight_magnitude + weight_duration, 1.0, atol=1e-3):
            raise ValueError(f"Weights must sum to 1.0. Got {weight_magnitude} + {weight_duration} = {weight_magnitude + weight_duration}")
        if not (theta_base < theta_low < theta_high):
            raise ValueError(f"Thresholds must satisfy theta_base < theta_low < theta_high. Got {theta_base}, {theta_low}, {theta_high}")

        self.w_mag = weight_magnitude
        self.w_dur = weight_duration
        self.duration_sat = duration_saturation_sec
        self.theta_high = theta_high
        self.theta_low = theta_low
        self.theta_base = theta_base

    def assess_event(self, event: AnomalyEvent) -> RiskAssessment:
        """Compute operational risk assessment for a specific formed event."""
        s_max = float(event.peak_score)
        duration = float(event.duration_sec)

        # Persistence saturation: phi(Delta t) in [0.0, 1.0]
        phi = min(1.0, duration / max(0.1, self.duration_sat))

        # Composite risk evidence
        mag_contrib = self.w_mag * s_max
        dur_contrib = self.w_dur * phi
        risk_score = min(1.0, max(0.0, mag_contrib + dur_contrib))

        # Classify operational tier
        if risk_score >= self.theta_high:
            tier = "High Risk"
            urgency = "Immediate operational dispatch recommended"
        elif risk_score >= self.theta_low:
            tier = "Medium Risk"
            urgency = "Operator review and surveillance camera verification needed"
        elif risk_score >= self.theta_base:
            tier = "Low Risk"
            urgency = "Non-critical alert logged for background review"
        else:
            tier = "Normal"
            urgency = "Anomaly score below baseline operational threshold"

        explanation = (
            f"{tier} (R={risk_score:.3f}): Peak anomaly score {s_max:.3f} "
            f"(contributes {mag_contrib:.3f}), persistent across {duration:.1f}s "
            f"(saturation {phi*100:.0f}%, contributes {dur_contrib:.3f}). {urgency}."
        )

        return RiskAssessment(
            risk_level=tier,
            risk_score=round(risk_score, 4),
            peak_score=round(s_max, 4),
            duration_sec=round(duration, 2),
            persistence_ratio=round(phi, 4),
            magnitude_contribution=round(mag_contrib, 4),
            duration_contribution=round(dur_contrib, 4),
            event_id=event.event_id,
            explanation=explanation,
        )

    def assess_video(
        self,
        events: List[AnomalyEvent],
        overall_peak_score: float = 0.0,
    ) -> RiskAssessment:
        """Compute top-level operational risk summary for an entire video stream.

        If events are formed, prioritizes the maximum risk event.
        If no events formed, returns baseline Normal assessment.
        """
        if not events:
            return RiskAssessment(
                risk_level="Normal",
                risk_score=round(self.w_mag * overall_peak_score, 4),
                peak_score=round(overall_peak_score, 4),
                duration_sec=0.0,
                persistence_ratio=0.0,
                magnitude_contribution=round(self.w_mag * overall_peak_score, 4),
                duration_contribution=0.0,
                event_id=None,
                explanation="No persistent anomaly events detected. Normal activity.",
            )

        # Assess all events and pick the highest risk event
        assessments = [self.assess_event(e) for e in events]
        top_assessment = max(assessments, key=lambda a: a.risk_score)
        return top_assessment
