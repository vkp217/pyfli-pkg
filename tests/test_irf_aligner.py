"""
Tests for IRFAligner — decay/IRF cube alignment via threshold-based rise
detection, RMSE-window refinement, and debiased low-threshold estimation.
"""

import numpy as np
import pytest

from pyfli.data_cc.irfAligner import IRFAligner
from pyfli.simulator.irf_sim.irf_generator import IRFGenerator

T_LASER = 12.5
NUM_BINS = 256
MU_IRF = 60


def _make_pixel_shifted_cube(shifts, mu_irf=MU_IRF, sigma=0.15, num_bins=NUM_BINS):
    """
    Builds an (H, W, num_bins) decay/irf pair from Gaussian pulses, where
    decay's peak is offset from irf's peak by shifts[i, j] bins.
    """
    H, W = shifts.shape
    decay = np.zeros((H, W, num_bins))
    irf = np.zeros((H, W, num_bins))
    for i in range(H):
        for j in range(W):
            decay[i, j, :] = 500 * IRFGenerator.gaussianIRF(
                mu=mu_irf + shifts[i, j], T=T_LASER, num_bins=num_bins, sigma=sigma
            )
            irf[i, j, :] = 200 * IRFGenerator.gaussianIRF(
                mu=mu_irf, T=T_LASER, num_bins=num_bins, sigma=sigma
            )
    return decay, irf


# ─────────────────────────────────────────────────────────────────────────────
# __init__
# ─────────────────────────────────────────────────────────────────────────────


class TestInit:
    def test_default_gate_delay(self):
        decay, irf = _make_pixel_shifted_cube(np.zeros((2, 2)))
        aligner = IRFAligner(decay, irf)
        assert np.isclose(aligner.gate_delay, T_LASER / NUM_BINS)
        assert np.isclose(aligner.dt, aligner.gate_delay)

    def test_explicit_gate_delay_overrides_default(self):
        decay, irf = _make_pixel_shifted_cube(np.zeros((2, 2)))
        aligner = IRFAligner(decay, irf, gate_delay=0.04)
        assert aligner.gate_delay == 0.04
        assert aligner.dt == 0.04

    def test_independent_noise_windows_accepted(self):
        decay, irf = _make_pixel_shifted_cube(np.zeros((2, 2)))
        aligner = IRFAligner(decay, irf, decay_noise_bins=(0, 8), irf_noise_bins=(2, 6))
        assert aligner.decay.shape == decay.shape

    def test_background_subtracted_and_clipped_nonnegative(self):
        decay, irf = _make_pixel_shifted_cube(np.zeros((2, 2)))
        decay[:, :, :5] += 3.0  # inject a baseline offset
        aligner = IRFAligner(decay, irf)
        assert aligner.decay.min() >= 0

    def test_noise_baseline_warning(self):
        H, W = 2, 2
        rng = np.random.default_rng(0)
        decay = rng.uniform(
            9, 11, size=(H, W, NUM_BINS)
        )  # no real peak, all noise-scale
        irf = rng.uniform(9, 11, size=(H, W, NUM_BINS))
        with pytest.warns(UserWarning, match="noise baseline"):
            IRFAligner(decay, irf)


# ─────────────────────────────────────────────────────────────────────────────
# estimate_shift
# ─────────────────────────────────────────────────────────────────────────────


class TestEstimateShift:
    def test_recovers_known_integer_shifts(self):
        rng = np.random.default_rng(0)
        shifts_true = rng.integers(-15, 16, size=(4, 4)).astype(float)
        decay, irf = _make_pixel_shifted_cube(shifts_true)
        aligner = IRFAligner(decay, irf)
        shifts_est = aligner.estimate_shift()
        assert np.max(np.abs(shifts_est - shifts_true)) < 0.5

    def test_dead_pixel_falls_back_to_zero_with_warning(self):
        decay, irf = _make_pixel_shifted_cube(np.zeros((2, 2)))
        decay[0, 0, :] = 0  # dead pixel
        aligner = IRFAligner(decay, irf)
        with pytest.warns(UserWarning, match="no detectable rise"):
            shifts = aligner.estimate_shift()
        assert shifts[0, 0] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# apply_fourier_shift / apply_circular_shift / align
# ─────────────────────────────────────────────────────────────────────────────


class TestApplyShifts:
    def test_fourier_and_circular_agree_for_integer_shifts(self):
        shifts = np.array([[5.0, -3.0], [0.0, 10.0]])
        decay, irf = _make_pixel_shifted_cube(np.zeros((2, 2)))
        aligner = IRFAligner(decay, irf)
        fourier_result = aligner.apply_fourier_shift(shifts)
        circular_result = aligner.apply_circular_shift(shifts)
        assert np.allclose(fourier_result, circular_result, atol=1e-6)

    def test_align_moves_peak_to_match_decay(self):
        shifts_true = np.array([[8.0]])
        decay, irf = _make_pixel_shifted_cube(shifts_true)
        aligner = IRFAligner(decay, irf)
        aligned_irf, _ = aligner.align(method="fourier")
        decay_peak = np.argmax(aligner.decay[0, 0, :])
        aligned_peak = np.argmax(aligned_irf[0, 0, :])
        assert abs(int(decay_peak) - int(aligned_peak)) <= 1

    def test_align_circular_method(self):
        shifts_true = np.array([[8.0]])
        decay, irf = _make_pixel_shifted_cube(shifts_true)
        aligner = IRFAligner(decay, irf)
        aligned_irf, _ = aligner.align(method="circular")
        decay_peak = np.argmax(aligner.decay[0, 0, :])
        aligned_peak = np.argmax(aligned_irf[0, 0, :])
        assert abs(int(decay_peak) - int(aligned_peak)) <= 1


# ─────────────────────────────────────────────────────────────────────────────
# align_pixel
# ─────────────────────────────────────────────────────────────────────────────


class TestAlignPixel:
    def test_matches_vectorized_estimate_shift(self):
        rng = np.random.default_rng(1)
        shifts_true = rng.integers(-10, 11, size=(3, 3)).astype(float)
        decay, irf = _make_pixel_shifted_cube(shifts_true)
        aligner = IRFAligner(decay, irf)
        full_shifts = aligner.estimate_shift()
        _, pixel_shift = aligner.align_pixel(1, 2)
        assert np.isclose(pixel_shift, full_shifts[1, 2], atol=1e-6)

    def test_dead_pixel_warns_and_returns_zero(self):
        decay, irf = _make_pixel_shifted_cube(np.zeros((2, 2)))
        decay[0, 0, :] = 0
        aligner = IRFAligner(decay, irf)
        with pytest.warns(UserWarning, match="No detectable rise"):
            _, shift = aligner.align_pixel(0, 0)
        assert shift == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# estimate_shift_rmse / estimate_shift_rmse_pixel
# ─────────────────────────────────────────────────────────────────────────────


class TestEstimateShiftRMSE:
    def test_recovers_known_shifts_with_auto_window(self):
        rng = np.random.default_rng(2)
        shifts_true = rng.integers(-15, 16, size=(3, 3)).astype(float)
        decay, irf = _make_pixel_shifted_cube(shifts_true)
        aligner = IRFAligner(decay, irf)
        shifts_est = aligner.estimate_shift_rmse(left=10, right=7, max_shift=20)
        assert np.max(np.abs(shifts_est - shifts_true)) < 0.5

    def test_global_bin_window_also_works(self):
        shifts_true = np.array([[5.0]])
        decay, irf = _make_pixel_shifted_cube(shifts_true)
        aligner = IRFAligner(decay, irf)
        shifts_est = aligner.estimate_shift_rmse(bin_window=(45, 75), max_shift=20)
        assert abs(shifts_est[0, 0] - 5.0) < 1.0

    def test_dead_pixel_falls_back_to_zero(self):
        decay, irf = _make_pixel_shifted_cube(np.zeros((2, 2)))
        decay[0, 0, :] = 0
        aligner = IRFAligner(decay, irf)
        with pytest.warns(UserWarning, match="no detectable decay"):
            shifts = aligner.estimate_shift_rmse()
        assert shifts[0, 0] == 0.0


class TestEstimateShiftRMSEPixel:
    def test_matches_vectorized_method(self):
        rng = np.random.default_rng(3)
        shifts_true = rng.integers(-10, 11, size=(2, 2)).astype(float)
        decay, irf = _make_pixel_shifted_cube(shifts_true)
        aligner = IRFAligner(decay, irf)
        full = aligner.estimate_shift_rmse(left=10, right=7, max_shift=20)
        probe = aligner.estimate_shift_rmse_pixel(0, 1, left=10, right=7, max_shift=20)
        assert np.isclose(probe["best_shift"], full[0, 1], atol=1e-6)

    def test_returns_expected_keys_and_shapes(self):
        decay, irf = _make_pixel_shifted_cube(np.array([[3.0]]))
        aligner = IRFAligner(decay, irf)
        probe = aligner.estimate_shift_rmse_pixel(0, 0, max_shift=15)
        for key in (
            "shift_candidates",
            "rmse",
            "best_shift",
            "window",
            "decay_trace",
            "scaled_irf_trace",
            "shifted_irf",
        ):
            assert key in probe
        assert probe["shift_candidates"].shape == probe["rmse"].shape
        assert probe["decay_trace"].shape == (NUM_BINS,)
        assert probe["shifted_irf"].shape == (NUM_BINS,)


# ─────────────────────────────────────────────────────────────────────────────
# estimate_shift_debiased
# ─────────────────────────────────────────────────────────────────────────────


class TestEstimateShiftDebiased:
    def test_runs_and_returns_finite_shifts(self):
        rng = np.random.default_rng(4)
        shifts_true = rng.integers(-10, 11, size=(3, 3)).astype(float)
        decay, irf = _make_pixel_shifted_cube(shifts_true)
        aligner = IRFAligner(decay, irf)
        shifts_est = aligner.estimate_shift_debiased()
        assert np.all(np.isfinite(shifts_est))

    def test_deterministic_for_fixed_inputs(self):
        shifts_true = np.array([[4.0]])
        decay, irf = _make_pixel_shifted_cube(shifts_true)
        aligner = IRFAligner(decay, irf)
        s1 = aligner.estimate_shift_debiased(low_fraction=0.02, smooth_window=3)
        s2 = aligner.estimate_shift_debiased(low_fraction=0.02, smooth_window=3)
        assert np.array_equal(s1, s2)

    def test_dead_pixel_falls_back_to_zero_with_warning(self):
        decay, irf = _make_pixel_shifted_cube(np.zeros((2, 2)))
        decay[0, 0, :] = 0
        aligner = IRFAligner(decay, irf)
        with pytest.warns(UserWarning, match="no detectable rise"):
            shifts = aligner.estimate_shift_debiased()
        assert shifts[0, 0] == 0.0
