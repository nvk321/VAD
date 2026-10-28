"""Postprocessing package: Temporal smoothing, hysteresis thresholding, and event formation."""

from src.postprocessing.event_formation import AnomalyEvent, EventFormationEngine, evaluate_persistence_ablation
from src.postprocessing.smoothing import exponential_moving_average, gaussian_smoothing, moving_average

__all__ = [
    "moving_average",
    "exponential_moving_average",
    "gaussian_smoothing",
    "AnomalyEvent",
    "EventFormationEngine",
    "evaluate_persistence_ablation",
]
