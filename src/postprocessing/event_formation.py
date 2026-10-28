"""Temporal Persistence and Event Formation Engine.

Stage 4 Component:
Converts continuous snippet anomaly scores into stable, actionable incident intervals
using dual-threshold hysteresis windowing, temporal persistence filtering, and gap bridging.

Distinguishes transient video artifacts (e.g. compression noise, lens flare) from
meaningful prolonged anomalies.
"""

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from src.postprocessing.smoothing import exponential_moving_average, moving_average


@dataclass
class AnomalyEvent:
    """Represents a formed continuous anomaly event."""

    event_id: int
    start_snippet: int
    end_snippet: int
    start_sec: float
    end_sec: float
    duration_sec: float
    peak_score: float
    mean_score: float

    def to_dict(self) -> Dict[str, Union[int, float]]:
        """Return dictionary representation."""
        return asdict(self)


class EventFormationEngine:
    """Forms stable incident intervals from continuous anomaly scores.

    Algorithm:
    1. Dual-Threshold Hysteresis:
       - An incident candidate is TRIGGERED when s(t) >= threshold_high.
       - The incident is SUSTAINED while s(t) >= threshold_low (where threshold_low < threshold_high).
       - The incident ENDS when s(t) < threshold_low.
    2. Gap Bridging:
       - If two adjacent incidents are separated by less than max_gap_sec, merge them.
    3. Minimum Persistence Filtering:
       - Discard candidate events with duration_sec < min_persistence_sec (removes momentary blips).
    """

    def __init__(
        self,
        threshold_high: float = 0.60,
        threshold_low: float = 0.40,
        min_persistence_sec: float = 1.0,
        max_gap_sec: float = 1.5,
        snippet_duration_sec: float = 0.533,  # 16 frames / 30 fps ~= 0.533s
    ):
        if threshold_low >= threshold_high:
            raise ValueError(f"threshold_low ({threshold_low}) must be strictly less than threshold_high ({threshold_high})")

        self.threshold_high = threshold_high
        self.threshold_low = threshold_low
        self.min_persistence_sec = min_persistence_sec
        self.max_gap_sec = max_gap_sec
        self.snippet_duration_sec = snippet_duration_sec

    def form_events(
        self,
        scores: Union[np.ndarray, torch.Tensor],
    ) -> List[AnomalyEvent]:
        """Form events from smoothed or raw anomaly scores.

        Args:
            scores: 1D array of continuous anomaly scores s(t) in [0.0, 1.0].

        Returns:
            List of validated AnomalyEvent objects.
        """
        if isinstance(scores, torch.Tensor):
            s = scores.detach().cpu().numpy().flatten()
        else:
            s = np.asarray(scores, dtype=np.float32).flatten()

        T = len(s)
        if T == 0:
            return []

        # Step 1: Detect candidate intervals using dual-threshold hysteresis
        candidates: List[Tuple[int, int]] = []
        in_event = False
        event_start = 0

        for t in range(T):
            score = s[t]

            if not in_event:
                if score >= self.threshold_high:
                    in_event = True
                    # Trace backwards to find when score first exceeded threshold_low
                    back_t = t
                    while back_t > 0 and s[back_t - 1] >= self.threshold_low:
                        back_t -= 1
                    event_start = back_t
            else:
                if score < self.threshold_low:
                    in_event = False
                    candidates.append((event_start, t - 1))

        if in_event:
            candidates.append((event_start, T - 1))

        if not candidates:
            return []

        # Step 2: Gap bridging (merge intervals separated by <= max_gap_snippets)
        max_gap_snippets = int(round(self.max_gap_sec / self.snippet_duration_sec))
        merged: List[Tuple[int, int]] = []
        curr_start, curr_end = candidates[0]

        for next_start, next_end in candidates[1:]:
            gap = next_start - curr_end - 1
            if gap <= max_gap_snippets:
                # Merge
                curr_end = next_end
            else:
                merged.append((curr_start, curr_end))
                curr_start, curr_end = next_start, next_end
        merged.append((curr_start, curr_end))

        # Step 3: Minimum persistence duration filtering
        min_snippets = int(round(self.min_persistence_sec / self.snippet_duration_sec))
        events: List[AnomalyEvent] = []
        event_counter = 1

        for start_idx, end_idx in merged:
            num_snippets = end_idx - start_idx + 1
            duration_sec = num_snippets * self.snippet_duration_sec

            if duration_sec >= self.min_persistence_sec or num_snippets >= min_snippets:
                segment_scores = s[start_idx : end_idx + 1]
                peak_score = float(np.max(segment_scores))
                mean_score = float(np.mean(segment_scores))

                events.append(
                    AnomalyEvent(
                        event_id=event_counter,
                        start_snippet=start_idx,
                        end_snippet=end_idx,
                        start_sec=round(start_idx * self.snippet_duration_sec, 2),
                        end_sec=round((end_idx + 1) * self.snippet_duration_sec, 2),
                        duration_sec=round(duration_sec, 2),
                        peak_score=round(peak_score, 4),
                        mean_score=round(mean_score, 4),
                    )
                )
                event_counter += 1

        return events


def evaluate_persistence_ablation(
    raw_scores: np.ndarray,
    smoothed_scores: np.ndarray,
    threshold: float = 0.50,
    engine: Optional[EventFormationEngine] = None,
) -> Dict[str, Union[int, float]]:
    """Compare raw thresholding against persistence-based event formation.

    Answers Experiment C research question:
    Does temporal persistence reduce alert fragmentation and chatter compared to raw thresholding?
    """
    if engine is None:
        engine = EventFormationEngine()

    # 1. Raw thresholding baseline (no smoothing, no hysteresis, no duration filter)
    raw_binary = (raw_scores >= threshold).astype(int)
    # Count transitions (chattering): number of times alert toggles 0 -> 1 or 1 -> 0
    raw_toggles = int(np.sum(np.abs(np.diff(raw_binary))))
    raw_active_snippets = int(np.sum(raw_binary))

    # Raw contiguous blocks
    raw_events_count = 0
    in_block = False
    for v in raw_binary:
        if v == 1 and not in_block:
            in_block = True
            raw_events_count += 1
        elif v == 0 and in_block:
            in_block = False

    # 2. Persistence-based event formation
    formed_events = engine.form_events(smoothed_scores)
    formed_event_count = len(formed_events)
    total_formed_duration = sum(e.duration_sec for e in formed_events)

    return {
        "raw_alert_blocks": raw_events_count,
        "raw_chatter_toggles": raw_toggles,
        "raw_active_snippets": raw_active_snippets,
        "formed_events_count": formed_event_count,
        "total_event_duration_sec": round(total_formed_duration, 2),
        "fragmentation_reduction_pct": round(
            max(0.0, (raw_events_count - formed_event_count) / max(1, raw_events_count) * 100.0), 1
        ),
    }
