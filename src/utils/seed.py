"""Reproducibility utilities for setting fixed random seeds across all engines."""

import os
import random
import numpy as np


def seed_everything(seed: int = 42, deterministic_torch: bool = True) -> None:
    """Seed Python built-in random, NumPy, and PyTorch (if available).

    Args:
        seed: The integer seed to set.
        deterministic_torch: Whether to enforce deterministic CUDNN algorithms.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
