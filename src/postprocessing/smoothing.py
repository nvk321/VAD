"""Temporal Smoothing Algorithms for Video Anomaly Scores.

Stage 4 Component:
Transforms raw snippet anomaly scores s(t) into temporally coherent signals,
suppressing high-frequency frame jitter before event formation.
"""

from typing import Union
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d


def moving_average(
    scores: Union[np.ndarray, torch.Tensor],
    window_size: int = 7,
) -> np.ndarray:
    """Apply uniform sliding window moving average filter.

    Args:
        scores: 1D array or tensor of snippet anomaly scores.
        window_size: Window size (must be odd, e.g. 5, 7, 11).

    Returns:
        Smoothed 1D numpy array with identical length.
    """
    if isinstance(scores, torch.Tensor):
        scores_arr = scores.detach().cpu().numpy().flatten()
    else:
        scores_arr = np.asarray(scores, dtype=np.float32).flatten()

    if len(scores_arr) < window_size:
        return scores_arr.copy()

    # Uniform box filter with edge reflection
    pad_width = window_size // 2
    padded = np.pad(scores_arr, pad_width, mode="edge")
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(padded, kernel, mode="valid")
    return np.clip(smoothed, 0.0, 1.0)


def exponential_moving_average(
    scores: Union[np.ndarray, torch.Tensor],
    alpha: float = 0.25,
) -> np.ndarray:
    """Apply causal Exponential Moving Average (EMA).

    Causal formulation: s_t^{(EMA)} = alpha * s_t + (1 - alpha) * s_{t-1}^{(EMA)}
    Ensures zero future information leakage during online streaming.

    Args:
        scores: 1D array or tensor of snippet anomaly scores.
        alpha: Smoothing coefficient in (0.0, 1.0]. Higher values track raw scores more closely.

    Returns:
        Smoothed 1D numpy array with identical length.
    """
    if isinstance(scores, torch.Tensor):
        scores_arr = scores.detach().cpu().numpy().flatten()
    else:
        scores_arr = np.asarray(scores, dtype=np.float32).flatten()

    if len(scores_arr) == 0:
        return scores_arr.copy()

    smoothed = np.zeros_like(scores_arr)
    smoothed[0] = scores_arr[0]

    for t in range(1, len(scores_arr)):
        smoothed[t] = alpha * scores_arr[t] + (1.0 - alpha) * smoothed[t - 1]

    return np.clip(smoothed, 0.0, 1.0)


def gaussian_smoothing(
    scores: Union[np.ndarray, torch.Tensor],
    sigma: float = 1.5,
) -> np.ndarray:
    """Apply 1D Gaussian kernel smoothing.

    Args:
        scores: 1D array or tensor of snippet anomaly scores.
        sigma: Standard deviation of Gaussian kernel.

    Returns:
        Smoothed 1D numpy array with identical length.
    """
    if isinstance(scores, torch.Tensor):
        scores_arr = scores.detach().cpu().numpy().flatten()
    else:
        scores_arr = np.asarray(scores, dtype=np.float32).flatten()

    if len(scores_arr) == 0:
        return scores_arr.copy()

    smoothed = gaussian_filter1d(scores_arr, sigma=sigma, mode="nearest")
    return np.clip(smoothed, 0.0, 1.0)
