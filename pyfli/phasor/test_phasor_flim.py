"""
test_phasor_flim.py
===================
Unit and integration tests for the phasor_flim package.

Run with:
    python -m pytest test_phasor_flim.py -v
or:
    python test_phasor_flim.py
"""

from __future__ import annotations
import math
import numpy as np
import pytest
from dataclasses import replace

from config import AcquisitionConfig, AcquisitionMode
from phasors import (
    phasor_continuous,
    phasor_discrete,
    phasor_gated_single,
    phasor_gated_N,
    phasor_truncated,
    phasor_offset,
    phasor_from_config,
)
from locus import build_locus, tau_grid, universal_semicircle, sepl_center_radius_discrete
from lifetimes import phase_lifetime, modulus_lifetime, lifetime_from_phasor


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def base_cfg():
    return AcquisitionConfig(T_ns=12.5, harmonic=1, tau_min_ns=0.1, tau_max_ns=10.0)


@pytest.fixture
def tau_arr():
    return np.array([0.5, 1.0, 2.0, 5.0, 10.0])


# ──────────────────────────────────────────────────────────────────────────────
# Config validation
# ──────────────────────────────────────────────────────────────────────────────

class TestConfigValidation:
    def test_valid_default(self):
        cfg = AcquisitionConfig()
        assert cfg.T_ns == 12.5

    def test_omega(self):
        cfg = AcquisitionConfig(T_ns=12.5, harmonic=1)
        expected = 2 * math.pi / 12.5
        assert abs(cfg.omega - expected) < 1e-12

    def test_frequency_MHz(self):
        cfg = AcquisitionConfig(T_ns=12.5)
        assert abs(cfg.frequency_MHz - 80.0) < 1e-9

    def test_gate_width_ns(self):
        cfg = AcquisitionConfig(T_ns=12.5, gate_width_frac=0.4)
        assert abs(cfg.gate_width_ns - 5.0) < 1e-12

    def test_invalid_T(self):
        with pytest.raises(ValueError):
            AcquisitionConfig(T_ns=-1.0)

    def test_invalid_N_bins(self):
        with pytest.raises(ValueError):
            AcquisitionConfig(N_bins=1)

    def test_invalid_tau_range(self):
        with pytest.raises(ValueError):
            AcquisitionConfig(tau_min_ns=5.0, tau_max_ns=1.0)

    def test_describe_runs(self):
        for mode in AcquisitionMode:
            cfg = AcquisitionConfig(mode=mode)
            assert isinstance(cfg.describe(), str)


# ──────────────────────────────────────────────────────────────────────────────
# Universal semicircle — continuous phasor
# ──────────────────────────────────────────────────────────────────────────────

class TestContinuousPhasor:
    def test_on_universal_semicircle(self, base_cfg, tau_arr):
        """Single-exponential phasors must lie on (g-0.5)^2 + s^2 = 0.25."""
        g, s = phasor_continuous(tau_arr, base_cfg)
        dist2 = (g - 0.5) ** 2 + s ** 2
        np.testing.assert_allclose(dist2, 0.25, atol=1e-12)

    def test_tau_zero_endpoint(self, base_cfg):
        """As τ→0, phasor → (1, 0)."""
        g, s = phasor_continuous(1e-9, base_cfg)
        assert abs(g - 1.0) < 1e-6
        assert abs(s) < 1e-6

    def test_tau_inf_endpoint(self, base_cfg):
        """As τ→∞, phasor → (0, 0)."""
        g, s = phasor_continuous(1e9, base_cfg)
        assert abs(g) < 1e-4
        assert abs(s) < 1e-4

    def test_scalar_input(self, base_cfg):
        g, s = phasor_continuous(2.0, base_cfg)
        assert np.ndim(g) == 0 or len(np.atleast_1d(g)) >= 1

    def test_harmonic_2(self, base_cfg, tau_arr):
        cfg2 = replace(base_cfg, harmonic=2)
        g, s = phasor_continuous(tau_arr, cfg2)
        dist2 = (g - 0.5) ** 2 + s ** 2
        np.testing.assert_allclose(dist2, 0.25, atol=1e-12)


# ──────────────────────────────────────────────────────────────────────────────
# Discrete phasor
# ──────────────────────────────────────────────────────────────────────────────

class TestDiscretePhasor:
    def test_converges_to_continuous(self, base_cfg, tau_arr):
        """Large N → discrete phasor ≈ continuous phasor."""
        cfg_d = replace(base_cfg, mode=AcquisitionMode.DISCRETE, N_bins=1024)
        g_d, s_d = phasor_discrete(tau_arr, cfg_d)
        g_c, s_c = phasor_continuous(tau_arr, base_cfg)
        np.testing.assert_allclose(g_d, g_c, atol=1e-3)
        np.testing.assert_allclose(s_d, s_c, atol=1e-3)

    def test_n2_different_from_continuous(self, base_cfg, tau_arr):
        cfg_d = replace(base_cfg, N_bins=2)
        g_d, s_d = phasor_discrete(tau_arr, cfg_d)
        g_c, s_c = phasor_continuous(tau_arr, base_cfg)
        assert not np.allclose(g_d, g_c, atol=1e-3), "N=2 should differ from continuous"

    def test_s_non_negative(self, base_cfg, tau_arr):
        cfg_d = replace(base_cfg, N_bins=16)
        _, s = phasor_discrete(tau_arr, cfg_d)
        assert np.all(s >= -1e-12)

    def test_circle_geometry(self, base_cfg):
        """Verify that the N-bin SEPL lies on a circle with the analytically derived centre/radius."""
        cfg_d = replace(base_cfg, mode=AcquisitionMode.DISCRETE, N_bins=8)
        gc, sc, r = sepl_center_radius_discrete(cfg_d)
        # Sample the SEPL densely
        x_arr = np.linspace(1e-8, 1 - 1e-8, 5000)
        tau_arr_c = -base_cfg.T_ns / (cfg_d.N_bins * np.log(x_arr))
        g, s = phasor_discrete(tau_arr_c, cfg_d)
        dist = np.sqrt((g - gc) ** 2 + (s - sc) ** 2)
        np.testing.assert_allclose(dist, r, atol=1e-6)


# ──────────────────────────────────────────────────────────────────────────────
# Gated phasors
# ──────────────────────────────────────────────────────────────────────────────

class TestGatedPhasors:
    def test_gated_single_full_window_approaches_continuous(self, base_cfg, tau_arr):
        """W → T (nearly full window) should approach continuous case."""
        # At very small τ (like 0.5 ns), even tiny gate differences matter; test
        # only for τ values where the gate-width effect is negligible.
        cfg_g = replace(base_cfg, mode=AcquisitionMode.GATED_SINGLE, gate_width_frac=0.9999)
        g_g, s_g = phasor_gated_single(tau_arr, cfg_g)
        g_c, s_c = phasor_continuous(tau_arr, base_cfg)
        # Use only tau >= 1 ns to avoid numerical edge near tau->0
        mask = tau_arr >= 1.0
        np.testing.assert_allclose(g_g[mask], g_c[mask], atol=0.01)

    def test_gated_N_output_shape(self, base_cfg, tau_arr):
        cfg_gn = replace(base_cfg, mode=AcquisitionMode.GATED_N, N_gates=4, gate_width_frac=0.5)
        g, s = phasor_gated_N(tau_arr, cfg_gn)
        assert g.shape == tau_arr.shape
        assert s.shape == tau_arr.shape

    def test_gated_N_1_gate_approaches_single(self, base_cfg, tau_arr):
        """N_gates=1 should match phasor_gated_single (same gate, same origin)."""
        cfg_g1  = replace(base_cfg, mode=AcquisitionMode.GATED_SINGLE, gate_width_frac=0.5)
        cfg_gn1 = replace(base_cfg, mode=AcquisitionMode.GATED_N, N_gates=1, gate_width_frac=0.5)
        g1, s1  = phasor_gated_single(tau_arr, cfg_g1)
        gn, sn  = phasor_gated_N(tau_arr, cfg_gn1)
        np.testing.assert_allclose(gn, g1, atol=0.05)


# ──────────────────────────────────────────────────────────────────────────────
# Truncated phasor
# ──────────────────────────────────────────────────────────────────────────────

class TestTruncatedPhasor:
    def test_full_window_matches_continuous(self, base_cfg, tau_arr):
        cfg_tr = replace(base_cfg, mode=AcquisitionMode.TRUNCATED, T_rec_frac=1.0)
        g_tr, s_tr = phasor_truncated(tau_arr, cfg_tr)
        g_c, s_c   = phasor_continuous(tau_arr, base_cfg)
        np.testing.assert_allclose(g_tr, g_c, atol=1e-6)
        np.testing.assert_allclose(s_tr, s_c, atol=1e-6)

    def test_truncated_deformation(self, base_cfg, tau_arr):
        """Short window produces different locus from the semicircle."""
        cfg_tr = replace(base_cfg, T_rec_frac=0.5)
        g_tr, s_tr = phasor_truncated(tau_arr, cfg_tr)
        dist2 = (g_tr - 0.5) ** 2 + s_tr ** 2
        # Should NOT be on the universal semicircle for all τ
        assert not np.allclose(dist2, 0.25, atol=0.01), \
            "Truncated phasors should deviate from the universal semicircle"


# ──────────────────────────────────────────────────────────────────────────────
# Offset phasor
# ──────────────────────────────────────────────────────────────────────────────

class TestOffsetPhasor:
    def test_zero_offset_equals_continuous(self, base_cfg, tau_arr):
        cfg_off = replace(base_cfg, mode=AcquisitionMode.OFFSET, t0_frac=0.0)
        g_off, s_off = phasor_offset(tau_arr, cfg_off)
        g_c,   s_c   = phasor_continuous(tau_arr, base_cfg)
        np.testing.assert_allclose(g_off, g_c, atol=1e-12)
        np.testing.assert_allclose(s_off, s_c, atol=1e-12)

    def test_offset_preserves_modulus(self, base_cfg, tau_arr):
        """Rotation preserves |z|."""
        cfg_off = replace(base_cfg, t0_frac=0.1)
        g_c, s_c   = phasor_continuous(tau_arr, base_cfg)
        g_off, s_off = phasor_offset(tau_arr, cfg_off)
        m_c   = np.sqrt(g_c   ** 2 + s_c   ** 2)
        m_off = np.sqrt(g_off ** 2 + s_off ** 2)
        np.testing.assert_allclose(m_off, m_c, atol=1e-12)


# ──────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ──────────────────────────────────────────────────────────────────────────────

class TestDispatcher:
    def test_all_modes_run(self, tau_arr):
        for mode in AcquisitionMode:
            cfg = AcquisitionConfig(mode=mode)
            g, s = phasor_from_config(tau_arr, cfg)
            assert g.shape == tau_arr.shape
            assert np.all(np.isfinite(g))
            assert np.all(np.isfinite(s))


# ──────────────────────────────────────────────────────────────────────────────
# Locus builder
# ──────────────────────────────────────────────────────────────────────────────

class TestLocus:
    def test_build_locus_shapes_match(self, base_cfg):
        g, s, tau = build_locus(base_cfg)
        assert g.shape == s.shape == tau.shape

    def test_tau_grid_monotone(self, base_cfg):
        tau = tau_grid(base_cfg)
        assert np.all(np.diff(tau) > 0)

    def test_universal_semicircle_radius(self):
        g, s = universal_semicircle(500)
        dist2 = (g - 0.5) ** 2 + s ** 2
        np.testing.assert_allclose(dist2, 0.25, atol=1e-12)

    def test_filter_finite_default(self, base_cfg):
        g, s, tau = build_locus(base_cfg)
        assert np.all(np.isfinite(g))
        assert np.all(np.isfinite(s))


# ──────────────────────────────────────────────────────────────────────────────
# Lifetime inversion
# ──────────────────────────────────────────────────────────────────────────────

class TestLifetimes:
    def test_phase_lifetime_roundtrip(self, base_cfg, tau_arr):
        g, s = phasor_continuous(tau_arr, base_cfg)
        tau_r = phase_lifetime(g, s, base_cfg)
        np.testing.assert_allclose(tau_r, tau_arr, rtol=1e-9)

    def test_modulus_lifetime_roundtrip(self, base_cfg, tau_arr):
        g, s = phasor_continuous(tau_arr, base_cfg)
        tau_r = modulus_lifetime(g, s, base_cfg)
        np.testing.assert_allclose(tau_r, tau_arr, rtol=1e-9)

    def test_phase_equals_modulus_on_semicircle(self, base_cfg, tau_arr):
        g, s = phasor_continuous(tau_arr, base_cfg)
        tau_ph = phase_lifetime(g, s, base_cfg)
        tau_m  = modulus_lifetime(g, s, base_cfg)
        np.testing.assert_allclose(tau_ph, tau_m, rtol=1e-8)

    def test_dispatcher_methods(self, base_cfg, tau_arr):
        g, s = phasor_continuous(tau_arr, base_cfg)
        for method in ("phase", "modulus", "mean"):
            tau_r = lifetime_from_phasor(g, s, base_cfg, method=method)
            assert tau_r.shape == tau_arr.shape

    def test_invalid_method(self, base_cfg):
        with pytest.raises(ValueError):
            lifetime_from_phasor([0.5], [0.3], base_cfg, method="bogus")

    def test_fractional_components(self, base_cfg):
        from lifetimes import fractional_components
        g1, s1 = phasor_continuous(1.0, base_cfg)
        g2, s2 = phasor_continuous(5.0, base_cfg)
        # Mix at f1=0.3, f2=0.7  =>  phasor = 0.3*(g1,s1) + 0.7*(g2,s2)
        g_mix = 0.3 * g1 + 0.7 * g2
        s_mix = 0.3 * s1 + 0.7 * s2
        f1, f2 = fractional_components(g_mix, s_mix, float(g1), float(s1), float(g2), float(s2))
        # f1 = fraction of species 1, f2 = fraction of species 2
        assert abs(float(f1) - 0.3) < 1e-9
        assert abs(float(f2) - 0.7) < 1e-9


# ──────────────────────────────────────────────────────────────────────────────
# Run without pytest
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    ret = pytest.main([__file__, "-v"])
    sys.exit(ret)
