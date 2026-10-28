"""Evaluation package for ROC-AUC, PR-AUC, and benchmark scoring."""

from src.evaluation.metrics import compute_roc_pr_curves, compute_video_level_metrics

__all__ = [
    "compute_video_level_metrics",
    "compute_roc_pr_curves",
]
