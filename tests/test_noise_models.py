"""
Tests for NoiseEngine — the simulator noise model collection.
"""

import numpy as np
import pytest

from pyfli.scripts.simulator.noise_models import NoiseEngine


@pytest.fixture
def clean_signal():
    rng = np.random.default_rng(0)
    return rng.uniform(10.0, 100.0, size=(256,))


@pytest.fixture
def clean_2d():
    rng = np.random.default_rng(1)
    return rng.uniform(5.0, 50.0, size=(8, 64))


# ─────────────────────────────────────────────────────────────────────────────
# Poisson noise
# ─────────────────────────────────────────────────────────────────────────────

class TestApplyPoisson:
    def test_shape_preserved(self, clean_signal):
        out = NoiseEngine.apply_poisson(clean_signal)
        assert out.shape == clean_signal.shape

    def test_output_non_negative(self, clean_signal):
        out = NoiseEngine.apply_poisson(clean_signal)
        assert np.all(out >= 0)

    def test_dtype_float64(self, clean_signal):
        out = NoiseEngine.apply_poisson(clean_signal)
        assert out.dtype == np.float64

    def test_2d_shape_preserved(self, clean_2d):
        out = NoiseEngine.apply_poisson(clean_2d)
        assert out.shape == clean_2d.shape

    def test_mean_close_to_lambda(self, rng):
        # E[Poisson(λ)] = λ; with many draws the sample mean converges
        signal = np.full(10_000, 20.0)
        out = NoiseEngine.apply_poisson(signal)
        assert abs(out.mean() - 20.0) < 1.0  # within 5 % of mean


# ─────────────────────────────────────────────────────────────────────────────
# Dark Count Rate (DCR)
# ─────────────────────────────────────────────────────────────────────────────

class TestApplyDCR:
    def test_shape_preserved(self, clean_signal):
        out = NoiseEngine.apply_dcr(clean_signal, dcr_level=0.5)
        assert out.shape == clean_signal.shape

    def test_output_at_least_as_large_as_input_on_average(self):
        # DCR only adds, so the mean output ≥ mean input
        sig = np.full(5_000, 10.0)
        out = NoiseEngine.apply_dcr(sig, dcr_level=1.0)
        assert out.mean() >= sig.mean() - 0.5  # slight slack for randomness

    def test_zero_dcr_equals_input(self, clean_signal):
        # dcr_level=0 → Poisson(0) → always 0 → output == input
        out = NoiseEngine.apply_dcr(clean_signal, dcr_level=0)
        np.testing.assert_array_equal(out, clean_signal)

    def test_2d_shape_preserved(self, clean_2d):
        out = NoiseEngine.apply_dcr(clean_2d)
        assert out.shape == clean_2d.shape


# ─────────────────────────────────────────────────────────────────────────────
# Read noise (Gaussian)
# ─────────────────────────────────────────────────────────────────────────────

class TestApplyReadNoise:
    def test_shape_preserved(self, clean_signal):
        out = NoiseEngine.apply_read_noise(clean_signal, sigma_read=1.5)
        assert out.shape == clean_signal.shape

    def test_noise_mean_near_zero(self):
        sig = np.zeros(20_000)
        out = NoiseEngine.apply_read_noise(sig, sigma_read=2.0)
        assert abs(out.mean()) < 0.1  # near-zero mean

    def test_noise_std_close_to_sigma(self):
        sig = np.zeros(20_000)
        sigma = 3.0
        out = NoiseEngine.apply_read_noise(sig, sigma_read=sigma)
        assert abs(out.std() - sigma) < 0.1

    def test_2d_shape_preserved(self, clean_2d):
        out = NoiseEngine.apply_read_noise(clean_2d)
        assert out.shape == clean_2d.shape


# ─────────────────────────────────────────────────────────────────────────────
# Timing jitter
# ─────────────────────────────────────────────────────────────────────────────

class TestApplyJitter:
    def test_length_preserved(self, clean_signal):
        out = NoiseEngine.apply_jitter(clean_signal, max_shift=5)
        assert len(out) == len(clean_signal)

    def test_zero_max_shift_returns_original(self, clean_signal):
        out = NoiseEngine.apply_jitter(clean_signal, max_shift=0)
        np.testing.assert_array_equal(out, clean_signal)

    def test_output_has_zeros_at_boundary_for_positive_shift(self):
        # Force a known positive shift by seeding
        np.random.seed(0)
        sig = np.ones(20)
        # Run several times; at least one should shift and pad with zeros
        has_zero = False
        for _ in range(30):
            out = NoiseEngine.apply_jitter(sig, max_shift=3)
            if out[0] == 0:
                has_zero = True
                break
        assert has_zero, "Expected at least one positive-shift result with leading zeros"

    def test_total_energy_conserved_modulo_shift(self, clean_signal):
        # Jitter moves photons; the count of non-zero bins changes, but
        # the sum of the non-zero slice stays close (boundary bins are zero).
        out = NoiseEngine.apply_jitter(clean_signal, max_shift=2)
        # Sum should be within the range of the original signal (up to boundary loss)
        assert out.sum() <= clean_signal.sum() + 1e-9
