"""Weakly Supervised Multiple Instance Learning (MIL) Anomaly Head.

Implements the canonical snippet scoring architecture (Sultani et al., CVPR 2018):
Maps spatiotemporal feature vectors to continuous anomaly scores in [0.0, 1.0].
Supports variable-length sequences or fixed 32-segment bags.
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn


class MILHead(nn.Module):
    """Multi-Layer Perceptron (MLP) Anomaly Scoring Head for MIL.

    Maps snippet feature representations directly to a continuous anomaly score s_t in [0.0, 1.0].
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 64,
        output_dim: int = 1,
        dropout: float = 0.2,
        activation: str = "sigmoid",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout
        self.activation_name = activation.lower()

        # Multi-layer scoring network (canonical Sultani et al. design)
        # Fully connected layers with non-linear activation and dropout regularization
        mid_dim = 512 if input_dim >= 512 else max(hidden_dim, input_dim)
        self.fc1 = nn.Linear(input_dim, mid_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(mid_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(hidden_dim, output_dim)


        if self.activation_name == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: '{activation}'. Expected 'sigmoid'.")

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier / Kaiming normal initialization for stable MIL training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass computing continuous anomaly scores.

        Args:
            x: Tensor of shape (B, T, D) or (T, D) where:
               - B = batch size
               - T = temporal sequence length (e.g. 32 segments or raw snippets)
               - D = feature dimension (e.g. 1024 for S3D)

        Returns:
            scores: Tensor of shape (B, T) or (T,) with values in [0.0, 1.0].
        """
        is_2d = x.dim() == 2
        if is_2d:
            x = x.unsqueeze(0)  # (1, T, D)

        if x.dim() != 3:
            raise ValueError(f"Expected 2D or 3D input tensor, got shape {tuple(x.shape)}")

        if x.size(-1) != self.input_dim:
            raise ValueError(
                f"Input feature dimension {x.size(-1)} does not match expected input_dim {self.input_dim}"
            )

        # Forward through MLP layers
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        out = self.fc3(out)
        scores = self.activation(out)  # (B, T, 1)

        scores = scores.squeeze(-1)  # (B, T)

        if is_2d:
            scores = scores.squeeze(0)  # (T,)

        return scores

    def count_parameters(self) -> Dict[str, int]:
        """Return total and trainable parameter count."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total_parameters": total, "trainable_parameters": trainable}


class MILBaselineModel(nn.Module):
    """Complete Experiment A Baseline Model: Feature Input -> MIL Head -> Anomaly Scores.

    Explicitly excludes TCN, persistence, and risk modules (Experiment A constraint).
    """

    def __init__(
        self,
        feature_dim: int = 1024,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.head = MILHead(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            dropout=dropout,
            activation="sigmoid",
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute snippet-level anomaly scores.

        Args:
            features: Tensor of shape (B, T, D) or (T, D).

        Returns:
            scores: Tensor of shape (B, T) or (T,) in [0.0, 1.0].
        """
        return self.head(features)

    def get_summary(self) -> Dict[str, Union[int, str]]:
        """Return model metadata for experiment tracking and paper logging."""
        params = self.head.count_parameters()
        return {
            "model_type": "MILBaseline (Exp A)",
            "temporal_model": "None (Instance-level scoring)",
            "feature_dim": self.feature_dim,
            "hidden_dim": self.head.hidden_dim,
            "dropout": self.head.dropout_rate,
            "total_parameters": params["total_parameters"],
            "trainable_parameters": params["trainable_parameters"],
        }
