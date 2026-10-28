"""Weakly Supervised Multiple Instance Learning (MIL) Ranking Loss.

Implements the canonical max-margin hinge ranking loss with temporal smoothness
and sparsity regularization (Sultani et al., CVPR 2018).
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MILRankingLoss(nn.Module):
    """Canonical MIL Ranking Loss for Weakly Supervised Video Anomaly Detection.

    Loss Formulation:
        L_total = L_ranking + lambda_smooth * L_smooth + lambda_sparse * L_sparse

    Where:
        - L_ranking = max(0, margin - max(s_abnormal) + max(s_normal))
        - L_smooth  = sum_{t=1}^{T-1} (s_abnormal[t] - s_abnormal[t+1])^2 / (T - 1)
        - L_sparse  = sum_{t=1}^T s_abnormal[t] / T
    """

    def __init__(
        self,
        margin: float = 1.0,
        lambda_smooth: float = 8e-5,
        lambda_sparse: float = 8e-5,
    ):
        super().__init__()
        self.margin = margin
        self.lambda_smooth = lambda_smooth
        self.lambda_sparse = lambda_sparse

    def forward(
        self,
        scores_abnormal: torch.Tensor,
        scores_normal: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the composite MIL ranking loss.

        Args:
            scores_abnormal: Anomaly scores for abnormal video bags.
                             Shape: (B, T) where B is batch size, T is number of segments.
            scores_normal:   Anomaly scores for normal video bags.
                             Shape: (B, T) where B is batch size, T is number of segments.

        Returns:
            total_loss: Scalar torch.Tensor with active gradient graph for backprop.
            loss_components: Dict containing float values of each loss component.
        """
        if scores_abnormal.dim() == 1:
            scores_abnormal = scores_abnormal.unsqueeze(0)
        if scores_normal.dim() == 1:
            scores_normal = scores_normal.unsqueeze(0)

        if scores_abnormal.shape != scores_normal.shape:
            raise ValueError(
                f"Abnormal and normal score shapes must match. Got {scores_abnormal.shape} vs {scores_normal.shape}"
            )

        # 1. Max-margin ranking loss: compare strongest abnormal vs strongest normal snippet
        # In Sultani et al., max over snippets represents bag-level anomaly presence
        max_abnormal, _ = torch.max(scores_abnormal, dim=1)  # (B,)
        max_normal, _ = torch.max(scores_normal, dim=1)      # (B,)

        # Hinge loss: max(0, margin - max_abn + max_norm)
        ranking_loss = torch.mean(F.relu(self.margin - max_abnormal + max_normal))

        # 2. Temporal smoothness regularization on abnormal bags
        # Encourages smooth transitions between consecutive snippets: (s_t - s_{t+1})^2
        diff = scores_abnormal[:, :-1] - scores_abnormal[:, 1:]  # (B, T - 1)
        smoothness_loss = torch.mean(torch.sum(diff ** 2, dim=1) / max(1, scores_abnormal.size(1) - 1))

        # 3. Temporal sparsity regularization on abnormal bags
        # Anomalies are rare events; penalizes overly pervasive high anomaly scores
        sparsity_loss = torch.mean(torch.mean(scores_abnormal, dim=1))

        # Composite total loss
        total_loss = (
            ranking_loss
            + self.lambda_smooth * smoothness_loss
            + self.lambda_sparse * sparsity_loss
        )

        loss_components = {
            "total_loss": total_loss.item(),
            "ranking_loss": ranking_loss.item(),
            "smoothness_loss": smoothness_loss.item(),
            "sparsity_loss": sparsity_loss.item(),
        }

        return total_loss, loss_components
