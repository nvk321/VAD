"""Modular Video Backbone Interface for Spatiotemporal Feature Extraction.

Supports S3D (1024-d Inception-based separable 3D, primary) and R(2+1)D-18 (512-d, fallback),
providing unified feature extraction, parameter freezing, and zero future leakage.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torchvision.models.video as v


class VideoBackbone(nn.Module):
    """Unified modular wrapper for frozen pretrained video feature extractors."""

    SUPPORTED_MODELS: Dict[str, dict] = {
        "s3d": {
            "feature_dim": 1024,
            "weights": "KINETICS400_V1",
            "spatial_size": (224, 224),
            "description": "Separable 3D Inception (I3D variant, 1024-dim)",
        },
        "r2plus1d_18": {
            "feature_dim": 512,
            "weights": "KINETICS400_V1",
            "spatial_size": (112, 112),
            "description": "Factorized Spatiotemporal ResNet-18 (512-dim)",
        },
    }

    def __init__(self, model_name: str = "s3d", pretrained: bool = True):
        super().__init__()
        model_name = model_name.lower()
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported backbone: '{model_name}'. Choose from: {list(self.SUPPORTED_MODELS.keys())}"
            )

        self.model_name = model_name
        self.meta = self.SUPPORTED_MODELS[model_name]
        self.feature_dim: int = self.meta["feature_dim"]
        self.spatial_size: Tuple[int, int] = self.meta["spatial_size"]
        self.snippet_length: int = 16

        weights = self.meta["weights"] if pretrained else None

        if model_name == "s3d":
            base_model = v.s3d(weights=weights)
            self.features = base_model.features
            self.avgpool = base_model.avgpool
        elif model_name == "r2plus1d_18":
            base_model = v.r2plus1d_18(weights=weights)
            self.stem = base_model.stem
            self.layer1 = base_model.layer1
            self.layer2 = base_model.layer2
            self.layer3 = base_model.layer3
            self.layer4 = base_model.layer4
            self.avgpool = base_model.avgpool

        # Mandatory architectural freeze: all parameters non-trainable
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract pooled 1D feature representation for a batch of video snippets.

        Args:
            x: Input tensor of shape (B, 3, T, H, W) where T is snippet length (e.g. 16).

        Returns:
            torch.Tensor of shape (B, feature_dim).
        """
        if self.model_name == "s3d":
            h = self.features(x)
            pooled = self.avgpool(h)
            return torch.flatten(pooled, 1)
        elif self.model_name == "r2plus1d_18":
            h = self.stem(x)
            h = self.layer1(h)
            h = self.layer2(h)
            h = self.layer3(h)
            h = self.layer4(h)
            pooled = self.avgpool(h)
            return torch.flatten(pooled, 1)

    @torch.no_grad()
    def extract_features(self, clips: torch.Tensor) -> torch.Tensor:
        """Convenience evaluation method running with torch.no_grad()."""
        return self(clips)


def build_backbone(model_name: str = "s3d", pretrained: bool = True) -> VideoBackbone:
    """Factory function for instantiating the modular video backbone."""
    return VideoBackbone(model_name=model_name, pretrained=pretrained)
