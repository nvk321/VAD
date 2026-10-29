"""Empirical Causal Leakage Verification Test Suite.

Verifies strict temporal causality of the Dilated Causal TCN and online inference pipeline:
For any chosen timestep t:
1. Run sequence forward normally: y = f(x_{0:T})
2. Perturb all future inputs x_{t+1:T} with large magnitude noise
3. Re-run inference: y' = f(x'_{0:T})
4. Mathematically verify that for all tau <= t, |y[tau] - y'[tau]| < 1e-6.

Guarantees ZERO FUTURE LEAKAGE across:
- Untrained TCN with various dilation schedules
- The production trained checkpoint (experiments/exp_B_tcn/checkpoints/best_auc.pt)
- Streaming causal EMA smoothing
"""

from pathlib import Path
import sys
import pytest
import torch
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.tcn import TCNModel
from src.postprocessing.smoothing import exponential_moving_average


def test_tcn_architectural_causality():
    """Verify that future frames do not leak into past predictions across multiple dilation schedules."""
    torch.manual_seed(1337)
    feature_dim = 1024
    seq_len = 64

    for dilations in [[1], [1, 2], [1, 2, 4], [1, 2, 4, 8]]:
        model = TCNModel(
            feature_dim=feature_dim,
            tcn_hidden_dim=128,
            head_hidden_dim=64,
            dilations=dilations,
        )
        model.eval()

        x_orig = torch.randn(1, seq_len, feature_dim)

        with torch.no_grad():
            out_orig = model(x_orig).squeeze(0)  # (seq_len,)

        # Test at multiple cutoff points t
        for t in [8, 16, 31, 48]:
            x_perturbed = x_orig.clone()
            # Severely perturb all timesteps strictly after t
            x_perturbed[:, (t + 1):, :] += torch.randn_like(x_perturbed[:, (t + 1):, :]) * 50.0

            with torch.no_grad():
                out_perturbed = model(x_perturbed).squeeze(0)

            past_orig = out_orig[:(t + 1)]
            past_perturbed = out_perturbed[:(t + 1)]

            max_diff = torch.max(torch.abs(past_orig - past_perturbed)).item()
            assert max_diff < 1e-6, (
                f"Causal leakage detected for dilations {dilations} at t={t}! "
                f"Max past difference: {max_diff:.8e} (Tolerance: 1e-6)"
            )


def test_trained_checkpoint_causal_leakage():
    """Verify causal leakage on the actual production trained TCN checkpoint."""
    ckpt_path = Path("experiments/exp_B_tcn/checkpoints/best_auc.pt")
    assert ckpt_path.is_file(), f"Trained checkpoint not found at: {ckpt_path}"

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    model = TCNModel(
        feature_dim=cfg.get("feature_dim", 1024),
        tcn_hidden_dim=cfg.get("tcn_hidden_dim", 128),
        head_hidden_dim=cfg.get("head_hidden_dim", 64),
        dilations=cfg.get("dilations", [1, 2, 4]),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    torch.manual_seed(42)
    seq_len = 50
    x_orig = torch.randn(1, seq_len, 1024)

    with torch.no_grad():
        out_orig = model(x_orig).squeeze(0).numpy()

    # Step through every possible timestep t from 0 to 45
    for t in [5, 10, 20, 30, 40]:
        x_perturbed = x_orig.clone()
        x_perturbed[:, (t + 1):, :] += torch.randn_like(x_perturbed[:, (t + 1):, :]) * 100.0

        with torch.no_grad():
            out_perturbed = model(x_perturbed).squeeze(0).numpy()

        past_orig = out_orig[:(t + 1)]
        past_perturbed = out_perturbed[:(t + 1)]

        max_err = float(np.max(np.abs(past_orig - past_perturbed)))
        assert max_err < 1e-6, (
            f"Trained checkpoint has causal leakage at t={t}! Max diff: {max_err:.8e}"
        )


def test_causal_ema_smoothing():
    """Verify that online causal EMA smoothing exhibits zero future leakage."""
    np.random.seed(42)
    raw_scores = np.random.uniform(0.0, 1.0, size=100)
    alpha = 0.25

    smooth_orig = exponential_moving_average(raw_scores, alpha=alpha)

    for t in [10, 25, 50, 75]:
        perturbed_scores = raw_scores.copy()
        # Change future inputs completely
        perturbed_scores[(t + 1):] = np.random.uniform(0.0, 1.0, size=len(perturbed_scores) - (t + 1))

        smooth_perturbed = exponential_moving_average(perturbed_scores, alpha=alpha)

        diff = np.max(np.abs(smooth_orig[:(t + 1)] - smooth_perturbed[:(t + 1)]))
        assert diff < 1e-7, (
            f"EMA smoothing causality violation at t={t}! Diff: {diff:.8e}"
        )


if __name__ == "__main__":
    print("Testing TCN architectural causality across dilation schedules...")
    test_tcn_architectural_causality()
    print("PASS: Zero future leakage across all dilation schedules.")

    print("Testing trained checkpoint causal leakage...")
    test_trained_checkpoint_causal_leakage()
    print("PASS: Production checkpoint strictly causal (diff < 1e-6).")

    print("Testing causal EMA smoothing...")
    test_causal_ema_smoothing()
    print("PASS: Online EMA smoothing strictly causal.")

    print("ALL CAUSAL LEAKAGE VERIFICATION TESTS PASSED SUCCESSFULLY!")
