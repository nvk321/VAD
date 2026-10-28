"""Dilated Causal Temporal Convolutional Network (TCN) for Video Anomaly Detection.

Implements Stage 2 of the frozen architecture:
Spatiotemporal Features -> Dilated Causal TCN -> MIL Head -> Anomaly Scores

Key Properties:
- Strictly Causal: Zero future leakage (prediction at t depends only on tau <= t).
- Dilated Receptive Field: Exponential dilations (1, 2, 4) capture extended temporal context.
- Residual Connections: Preserves gradient flow through temporal blocks.
- Lightweight: ~100k parameters, easily feasible within 4 GB VRAM budget.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.mil_head import MILHead


class CausalConv1d(nn.Module):
    """1D Convolution with strict causal padding to guarantee zero future leakage.

    Input shape:  (B, C, T)
    Output shape: (B, C_out, T)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.causal_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.causal_padding,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with causal trimming of future positions."""
        out = self.conv(x)
        if self.causal_padding > 0:
            # Discard the last causal_padding elements to keep sequence length T
            out = out[:, :, :-self.causal_padding]
        return out


class ChannelLayerNorm(nn.Module):
    """LayerNorm across channels for each time step independently.

    Guarantees strict temporal causality (zero future leakage) by ensuring
    normalization at time step t depends solely on features at time step t.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.ln = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (B, C, T) -> (B, T, C) -> LayerNorm -> (B, C, T)
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class TemporalResidualBlock(nn.Module):
    """Residual block with dilated causal convolutions, layer normalization, and dropout.

    Architecture:
        x -> CausalConv1d -> ChannelLayerNorm -> ReLU -> Dropout
          -> CausalConv1d -> ChannelLayerNorm -> ReLU -> Dropout
          + Residual(x) -> ReLU -> out
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.norm1 = ChannelLayerNorm(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation=dilation)
        self.norm2 = ChannelLayerNorm(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # 1x1 conv if input and output dimensions mismatch
        if in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = None

        self.relu_out = nn.ReLU()
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Kaiming normal for ReLU and identity norm."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual block. Input shape: (B, C, T)."""
        residual = x if self.residual_conv is None else self.residual_conv(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        return self.relu_out(out + residual)


class DilatedCausalTCN(nn.Module):
    """Multi-layer Dilated Causal Temporal Convolutional Network.

    Processes temporal sequence of spatiotemporal feature embeddings with zero future leakage.
    Input shape:  (B, T, D_feat) or (B, D_feat, T)
    Output shape: (B, T, C_hidden)
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 128,
        kernel_size: int = 3,
        dilations: Optional[List[int]] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4]

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.dropout_rate = dropout

        # Initial linear projection to map high-dim backbone features into hidden_dim
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.proj_norm = ChannelLayerNorm(hidden_dim)
        self.proj_relu = nn.ReLU()



        # Stack of dilated causal residual blocks
        layers = []
        for d in dilations:
            layers.append(
                TemporalResidualBlock(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=d,
                    dropout=dropout,
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (B, T, input_dim) or (T, input_dim).

        Returns:
            Tensor of shape (B, T, hidden_dim) or (T, hidden_dim).
        """
        is_2d = x.dim() == 2
        if is_2d:
            x = x.unsqueeze(0)  # (1, T, input_dim)

        if x.dim() != 3:
            raise ValueError(f"Expected 2D or 3D tensor, got shape {tuple(x.shape)}")

        # Convert (B, T, D) -> (B, D, T) for 1D convolution
        x_conv = x.transpose(1, 2)

        out = self.input_proj(x_conv)
        out = self.proj_norm(out)
        out = self.proj_relu(out)

        out = self.network(out)

        # Convert back: (B, C, T) -> (B, T, C)
        out = out.transpose(1, 2)

        if is_2d:
            out = out.squeeze(0)

        return out

    def count_parameters(self) -> Dict[str, int]:
        """Return total and trainable parameter count."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total_parameters": total, "trainable_parameters": trainable}


class TCNModel(nn.Module):
    """Complete Experiment B Model: S3D Features -> Dilated Causal TCN -> MIL Head -> Scores.

    Evaluated in Experiment B to test whether explicit causal temporal modeling
    outperforms instance-level MIL baseline (Experiment A).
    """

    def __init__(
        self,
        feature_dim: int = 1024,
        tcn_hidden_dim: int = 128,
        head_hidden_dim: int = 64,
        kernel_size: int = 3,
        dilations: Optional[List[int]] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4]

        self.feature_dim = feature_dim
        self.tcn_hidden_dim = tcn_hidden_dim
        self.head_hidden_dim = head_hidden_dim
        self.dilations = dilations

        self.tcn = DilatedCausalTCN(
            input_dim=feature_dim,
            hidden_dim=tcn_hidden_dim,
            kernel_size=kernel_size,
            dilations=dilations,
            dropout=dropout,
        )

        self.head = MILHead(
            input_dim=tcn_hidden_dim,
            hidden_dim=head_hidden_dim,
            output_dim=1,
            dropout=dropout,
            activation="sigmoid",
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass computing continuous anomaly scores s_t in [0.0, 1.0].

        Args:
            features: Tensor of shape (B, T, D) or (T, D).

        Returns:
            scores: Tensor of shape (B, T) or (T,) in [0.0, 1.0].
        """
        temporal_features = self.tcn(features)  # (B, T, tcn_hidden_dim)
        scores = self.head(temporal_features)    # (B, T)
        return scores

    def get_summary(self) -> Dict[str, Union[int, str, List[int]]]:
        """Return model metadata for experiment tracking and paper logging."""
        tcn_params = self.tcn.count_parameters()
        head_params = self.head.count_parameters()
        total = tcn_params["total_parameters"] + head_params["total_parameters"]
        trainable = tcn_params["trainable_parameters"] + head_params["trainable_parameters"]
        return {
            "model_type": "TCN_MIL (Exp B)",
            "temporal_model": f"DilatedCausalTCN (dilations={self.dilations})",
            "feature_dim": self.feature_dim,
            "tcn_hidden_dim": self.tcn_hidden_dim,
            "head_hidden_dim": self.head_hidden_dim,
            "dilations": self.dilations,
            "total_parameters": total,
            "trainable_parameters": trainable,
        }
