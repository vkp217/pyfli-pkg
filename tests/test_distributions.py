"""
Tests for ParameterSampler — the simulator parameter distribution helpers.
"""

import numpy as np
import pytest

from pyfli.scripts.simulator.distributions import ParameterSampler


# ─────────────────────────────────────────────────────────────────────────────
# Quantum efficiency sampling
# ─────────────────────────────────────────────────────────────────────────────

class TestSampleQE:
    def test_iccd_range(self):
        for _ in range(200):
            qe = ParameterSampler.sample_qe("ICCD")
            assert 0.15 <= qe <= 0.35, f"ICCD QE out of range: {qe}"

    def test_spad_range(self):
        for _ in range(200):
            qe = ParameterSampler.sample_qe("SPAD")
            assert 0.70 <= qe <= 0.90, f"SPAD QE out of range: {qe}"

    def test_case_insensitive(self):
        # Lower-case input should still work
        for _ in range(50):
            qe = ParameterSampler.sample_qe("iccd")
            assert 0.15 <= qe <= 0.35


# ─────────────────────────────────────────────────────────────────────────────
# Noise parameter sampling
# ─────────────────────────────────────────────────────────────────────────────

class TestSampleNoiseParams:
    def test_iccd_returns_dict_with_read_sigma(self):
        params = ParameterSampler.sample_noise_params(bit_depth=8, sensor_type="ICCD")
        assert "read_sigma" in params
        assert params["read_sigma"] > 0

    def test_spad_read_sigma_zero(self):
        params = ParameterSampler.sample_noise_params(bit_depth=12, sensor_type="SPAD")
        assert params["read_sigma"] == 0.0

    def test_iccd_read_sigma_scales_with_bit_depth(self):
        low = ParameterSampler.sample_noise_params(bit_depth=8, sensor_type="ICCD")
        high = ParameterSampler.sample_noise_params(bit_depth=16, sensor_type="ICCD")
        # Higher bit depth → higher read sigma in expectation
        # Run many samples to check average ordering
        low_vals = [
            ParameterSampler.sample_noise_params(8, "ICCD")["read_sigma"]
            for _ in range(200)
        ]
        high_vals = [
            ParameterSampler.sample_noise_params(16, "ICCD")["read_sigma"]
            for _ in range(200)
        ]
        assert np.mean(high_vals) > np.mean(low_vals)


# ─────────────────────────────────────────────────────────────────────────────
# Beta sampling
# ─────────────────────────────────────────────────────────────────────────────

class TestSampleBeta:
    def test_output_in_offset_to_offset_plus_scale(self):
        scale, offset = 0.9, 0.05
        for _ in range(500):
            v = ParameterSampler.sample_beta(5, 5, scale=scale, offset=offset)
            assert offset <= v <= offset + scale, f"sample_beta out of range: {v}"

    def test_default_scale_and_offset(self):
        for _ in range(200):
            v = ParameterSampler.sample_beta(2, 2)
            assert 0.0 <= v <= 1.0

    def test_mean_closer_to_0_5_for_symmetric_beta(self):
        samples = [ParameterSampler.sample_beta(5, 5) for _ in range(2000)]
        assert abs(np.mean(samples) - 0.5) < 0.05


class TestBetaSample:
    def test_clipped_to_epsilon(self):
        eps = 1e-4
        for _ in range(500):
            v = ParameterSampler.beta_sample(1, 1, scale=1.0, clip_eps=eps)
            assert eps <= v <= 1.0 - eps, f"beta_sample out of clip range: {v}"

    def test_scale_applied(self):
        scale = 5.0
        for _ in range(200):
            v = ParameterSampler.beta_sample(5, 5, scale=scale)
            assert 0.0 <= v <= scale


# ─────────────────────────────────────────────────────────────────────────────
# Truncated normal sampling
# ─────────────────────────────────────────────────────────────────────────────

class TestTruncatedNormal:
    def test_within_bounds(self):
        lower, upper = 0.01, 5.0
        for _ in range(500):
            v = ParameterSampler.truncated_normal(mu=1.0, sigma=0.5,
                                                   lower=lower, upper=upper)
            assert lower <= v <= upper, f"truncated_normal out of range: {v}"

    def test_custom_bounds(self):
        for _ in range(200):
            v = ParameterSampler.truncated_normal(mu=2.0, sigma=0.3,
                                                   lower=1.0, upper=3.0)
            assert 1.0 <= v <= 3.0


# ─────────────────────────────────────────────────────────────────────────────
# Stretch / squeeze mapping
# ─────────────────────────────────────────────────────────────────────────────

class TestStretchSqueeze:
    def test_zero_maps_to_epsilon(self):
        eps = 0.05
        result = ParameterSampler.stretch_squeeze(0.0, eps)
        assert abs(result - eps) < 1e-12

    def test_one_maps_to_one_minus_epsilon(self):
        eps = 0.05
        result = ParameterSampler.stretch_squeeze(1.0, eps)
        assert abs(result - (1.0 - eps)) < 1e-12

    def test_half_maps_to_half(self):
        result = ParameterSampler.stretch_squeeze(0.5, 0.1)
        assert abs(result - 0.5) < 1e-12

    def test_output_bounded(self):
        eps = 0.1
        for s in np.linspace(0.0, 1.0, 50):
            v = ParameterSampler.stretch_squeeze(s, eps)
            assert eps <= v <= 1.0 - eps
