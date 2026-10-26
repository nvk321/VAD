"""Unit tests for feature extraction, feature caching, and dataset loaders."""

from pathlib import Path
import sys

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.data.feature_dataset import CachedFeatureDataset, interpolate_segments
from src.models.backbone import build_backbone


def test_interpolate_segments():
    """Verify segment interpolation preserves dimension and outputs exact target segments."""
    # Test case 1: T > num_segments (downsampling)
    feat_long = torch.randn(50, 1024)
    interp_50 = interpolate_segments(feat_long, num_segments=32)
    assert interp_50.shape == (32, 1024), f"Expected (32, 1024), got {interp_50.shape}"

    # Test case 2: T < num_segments (upsampling)
    feat_short = torch.randn(10, 1024)
    interp_10 = interpolate_segments(feat_short, num_segments=32)
    assert interp_10.shape == (32, 1024), f"Expected (32, 1024), got {interp_10.shape}"

    # Test case 3: T == num_segments (identity)
    feat_exact = torch.randn(32, 1024)
    interp_exact = interpolate_segments(feat_exact, num_segments=32)
    assert torch.allclose(feat_exact, interp_exact), "Interpolation with same size should be identity"


def test_smoke_test_cached_features():
    """Verify saved .pt files in data/features/smoke_test."""
    smoke_dir = Path("data/features/smoke_test")
    assert smoke_dir.is_dir(), f"Smoke test directory {smoke_dir} not found"

    pt_files = list(smoke_dir.glob("*.pt"))
    assert len(pt_files) == 5, f"Expected 5 .pt files, found {len(pt_files)}"

    for pt_file in pt_files:
        data = torch.load(pt_file, map_location="cpu", weights_only=True)
        assert "features" in data, f"Missing 'features' in {pt_file.name}"
        assert "video_id" in data, f"Missing 'video_id' in {pt_file.name}"
        assert "category" in data, f"Missing 'category' in {pt_file.name}"
        assert "label" in data, f"Missing 'label' in {pt_file.name}"
        assert "feature_dim" in data, f"Missing 'feature_dim' in {pt_file.name}"

        feat = data["features"]
        assert isinstance(feat, torch.Tensor), f"Features should be tensor in {pt_file.name}"
        assert feat.dim() == 2, f"Features should be 2D (num_snippets, feature_dim), got {feat.shape}"
        assert feat.size(-1) == 1024, f"Expected feature_dim 1024, got {feat.size(-1)}"
        assert not torch.isnan(feat).any(), f"NaNs found in features of {pt_file.name}"
        assert not torch.isinf(feat).any(), f"Infs found in features of {pt_file.name}"


def test_cached_feature_dataset_loader():
    """Verify CachedFeatureDataset loading directly from directory and with manifest."""
    smoke_dir = Path("data/features/smoke_test")
    dataset_raw = CachedFeatureDataset(features_dir=smoke_dir)
    assert len(dataset_raw) == 5, f"Expected 5 samples, got {len(dataset_raw)}"

    item = dataset_raw[0]
    assert "features" in item
    assert "label" in item
    assert "video_id" in item
    assert "category" in item
    assert "raw_num_snippets" in item
    assert item["feature_dim"] == 1024
    assert item["features"].dim() == 2

    # Test with 32-segment interpolation
    dataset_32 = CachedFeatureDataset(features_dir=smoke_dir, num_segments=32)
    item_32 = dataset_32[0]
    assert item_32["features"].shape == (32, 1024), f"Expected (32, 1024), got {item_32['features'].shape}"


def test_backbone_factory_cpu():
    """Verify backbone interface instantiation and property checks."""
    s3d = build_backbone("s3d", pretrained=False)
    assert s3d.feature_dim == 1024
    assert s3d.snippet_length == 16
    assert s3d.spatial_size == (224, 224)

    r2plus1d = build_backbone("r2plus1d_18", pretrained=False)
    assert r2plus1d.feature_dim == 512
    assert r2plus1d.snippet_length == 16


if __name__ == "__main__":
    print("Running test_interpolate_segments()...")
    test_interpolate_segments()
    print("Running test_smoke_test_cached_features()...")
    test_smoke_test_cached_features()
    print("Running test_cached_feature_dataset_loader()...")
    test_cached_feature_dataset_loader()
    print("Running test_backbone_factory_cpu()...")
    test_backbone_factory_cpu()
    print("All feature and backbone tests PASSED successfully!")
