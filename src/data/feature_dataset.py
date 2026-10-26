"""Dataset and DataLoader utilities for cached spatiotemporal video features.

Provides:
- CachedFeatureDataset: Sequential video feature loader for evaluation and inference.
- MILPairDataset: Balanced abnormal/normal pair dataset for weakly supervised MIL training.
- 32-segment temporal feature interpolation (standard Sultani et al. protocol).
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def interpolate_segments(features: torch.Tensor, num_segments: int = 32) -> torch.Tensor:
    """Interpolate a variable-length snippet feature sequence into a fixed number of segments.

    Standard benchmark protocol (Sultani et al., CVPR 2018; RTFM, ICCV 2021):
    Input shape: (T, D) where T = number of snippets, D = feature dimension (e.g. 1024).
    Output shape: (num_segments, D) = (32, D).

    Uses linear 1D interpolation along the temporal dimension.
    """
    if features.dim() != 2:
        raise ValueError(f"Expected 2D tensor (T, D), got shape {tuple(features.shape)}")

    T, D = features.shape

    if T == num_segments:
        return features

    # Permute to (1, D, T) for 1D linear interpolation
    feat_transposed = features.transpose(0, 1).unsqueeze(0)  # (1, D, T)
    interpolated = F.interpolate(
        feat_transposed,
        size=num_segments,
        mode="linear",
        align_corners=False
    )  # (1, D, num_segments)
    return interpolated.squeeze(0).transpose(0, 1)  # (num_segments, D)


class CachedFeatureDataset(Dataset):
    """Dataset for sequential loading of pre-extracted video feature files (.pt).

    Used for validation, testing, and continuous timeline evaluation.
    """

    def __init__(
        self,
        features_dir: Union[str, Path],
        manifest_path: Optional[Union[str, Path]] = None,
        num_segments: Optional[int] = None,
        filter_label: Optional[int] = None,
    ):
        """
        Args:
            features_dir: Path to directory containing cached .pt files.
            manifest_path: Optional CSV manifest path (e.g. ucf_crime_val.csv).
            num_segments: If specified, temporal sequence is interpolated to this length.
            filter_label: If specified, filter dataset to only label (0=normal, 1=abnormal).
        """
        self.features_dir = Path(features_dir)
        self.num_segments = num_segments
        self.filter_label = filter_label
        self.samples: List[Dict[str, Union[str, int, Path]]] = []

        if not self.features_dir.is_dir():
            raise FileNotFoundError(f"Features directory does not exist: {self.features_dir}")

        if manifest_path is not None:
            manifest_path = Path(manifest_path)
            if not manifest_path.is_file():
                raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

            with open(manifest_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    label = int(row["label"])
                    if self.filter_label is not None and label != self.filter_label:
                        continue
                    feat_file = self.features_dir / f"{row['video_id']}.pt"
                    if feat_file.is_file():
                        self.samples.append({
                            "video_id": row["video_id"],
                            "feature_path": feat_file,
                            "label": label,
                            "category": row.get("category", "Unknown"),
                        })
        else:
            # Discover directly from .pt files in directory
            pt_files = sorted(self.features_dir.glob("*.pt"))
            for feat_file in pt_files:
                video_id = feat_file.stem
                self.samples.append({
                    "video_id": video_id,
                    "feature_path": feat_file,
                    "label": -1,  # inferred from file payload if loaded
                    "category": "Unknown",
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int]]:
        sample_meta = self.samples[idx]
        feat_path = Path(sample_meta["feature_path"])

        data = torch.load(feat_path, map_location="cpu", weights_only=True)
        features = data["features"]  # (T, D)

        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)

        if torch.isnan(features).any() or torch.isinf(features).any():
            raise ValueError(f"Corrupted features (NaN/Inf) found in {feat_path}")

        raw_num_snippets = features.size(0)

        if self.num_segments is not None:
            processed_features = interpolate_segments(features, self.num_segments)
        else:
            processed_features = features

        label = int(data.get("label", sample_meta["label"]))
        category = str(data.get("category", sample_meta["category"]))
        video_id = str(data.get("video_id", sample_meta["video_id"]))

        return {
            "features": processed_features,
            "label": label,
            "video_id": video_id,
            "category": category,
            "raw_num_snippets": raw_num_snippets,
            "feature_dim": features.size(-1),
        }


class MILPairDataset(Dataset):
    """Paired Abnormal/Normal Dataset for weakly supervised Multiple Instance Learning (MIL).

    Yields balanced pairs:
    (abnormal_bag, normal_bag) where each bag has shape (num_segments, feature_dim),
    typically (32, 1024).
    """

    def __init__(
        self,
        features_dir: Union[str, Path],
        manifest_path: Union[str, Path],
        num_segments: int = 32,
        seed: int = 42,
    ):
        self.abnormal_dataset = CachedFeatureDataset(
            features_dir=features_dir,
            manifest_path=manifest_path,
            num_segments=num_segments,
            filter_label=1,
        )
        self.normal_dataset = CachedFeatureDataset(
            features_dir=features_dir,
            manifest_path=manifest_path,
            num_segments=num_segments,
            filter_label=0,
        )

        if len(self.abnormal_dataset) == 0:
            raise ValueError("No abnormal feature files found matching manifest!")
        if len(self.normal_dataset) == 0:
            raise ValueError("No normal feature files found matching manifest!")

        self.num_segments = num_segments
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        # Standard MIL epoch length is determined by the maximum class count
        return max(len(self.abnormal_dataset), len(self.normal_dataset))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        abn_idx = idx % len(self.abnormal_dataset)
        # Random normal sample pairing per epoch iteration
        norm_idx = self.rng.integers(0, len(self.normal_dataset))

        abn_item = self.abnormal_dataset[abn_idx]
        norm_item = self.normal_dataset[norm_idx]

        return abn_item["features"], norm_item["features"]
