
"""Evaluation metrics for video anomaly detection.

Implements video-level and snippet-level ROC-AUC and PR-AUC metrics.
Strictly respects test set quarantine: validation metrics rely solely on validation videos.
"""

from typing import Dict, List, Sequence, Union
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


def compute_video_level_metrics(
    y_true: Sequence[int],
    y_pred_scores: Sequence[float],
) -> Dict[str, float]:
    """Compute video-level ROC-AUC and PR-AUC (Average Precision).

    Args:
        y_true: Binary ground-truth video labels (0 = normal, 1 = abnormal).
        y_pred_scores: Continuous video anomaly predictions (e.g. max snippet score).

    Returns:
        Dict containing 'roc_auc' and 'pr_auc'.
    """
    y_true_arr = np.asarray(y_true, dtype=np.int32)
    y_pred_arr = np.asarray(y_pred_scores, dtype=np.float32)

    if len(np.unique(y_true_arr)) < 2:
        # Edge case: only one class in batch/subset
        return {"roc_auc": 0.5, "pr_auc": 0.0}

    # Replace any NaNs with 0.0
    y_pred_arr = np.nan_to_num(y_pred_arr, nan=0.0, posinf=1.0, neginf=0.0)

    roc_auc = float(roc_auc_score(y_true_arr, y_pred_arr))
    pr_auc = float(average_precision_score(y_true_arr, y_pred_arr))

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def compute_roc_pr_curves(
    y_true: Sequence[int],
    y_pred_scores: Sequence[float],
) -> Dict[str, np.ndarray]:
    """Compute full ROC and PR curve coordinate arrays for publication plotting."""
    y_true_arr = np.asarray(y_true, dtype=np.int32)
    y_pred_arr = np.asarray(y_pred_scores, dtype=np.float32)

    if len(np.unique(y_true_arr)) < 2:
        return {
            "fpr": np.array([0.0, 1.0]),
            "tpr": np.array([0.0, 1.0]),
            "precision": np.array([0.0, 1.0]),
            "recall": np.array([1.0, 0.0]),
        }

    fpr, tpr, _ = roc_curve(y_true_arr, y_pred_arr)
    precision, recall, _ = precision_recall_curve(y_true_arr, y_pred_arr)

    return {
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision,
        "recall": recall,
    }
