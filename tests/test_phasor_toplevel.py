"""
test_phasor_toplevel.py
=======================
Comprehensive tests for the phasor module accessed through the top-level
pyfli namespace (i.e. ``from pyfli import phasor_gated_N`` etc.).

Covers:
  - All six acquisition modes (config, phasor coordinates, locus)
  - Lifetime inversion (phase, modulus, mean, dispatcher)
  - Fractional component decomposition for two-species mixtures
  - Locus builder (build_locus, build_loci, tau_grid, universal_semicircle)
  - Discrete SEPL circle geometry (sepl_center_radius_discrete)
  - Plot helpers (smoke-tests: no crash, correct return types)

Reference:
    Michalet X. AIP Advances 11, 035331 (2021).
    https://doi.org/10.1063/5.0027834
"""

from __future__ import annotations

import math
import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
from dataclasses import replace

from pyfli import (
    AcquisitionConfig,
    AcquisitionMode,
    phasor_continuous,
    phasor_discrete,
    phasor_gated_single,
    phasor_gated_N,
    phasor_truncated,
    phasor_offset,
    phasor_from_config,
    build_locus,
    build_loci,
    tau_grid,
    universal_semicircle,
    sepl_center_radius_discrete,
    phase_lifetime,
    modulus_lifetime,
    lifetime_from_phasor,
    phase_lifetime_gated,
    fractional_components,
    plot_phasor,
    plot_locus_comparison,
    plot_discrete_N_sweep,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def base_cfg():
    return AcquisitionConfig(T_ns=12.5, harmonic=1, tau_min_ns=0.1, tau_max_ns=10.0)


@pytest.fixture
def tau_arr():
    return np.array([0.5, 1.0, 2.0, 5.0, 10.0])


# ─────────────────────────────────────────────────────────────────────────────
# AcquisitionConfig
# ─────────────────────────────────────────────────────────────────────────────

class TestAcquisitionConfig:
    def test_default_period(self):
        cfg = AcquisitionConfig()
        assert cfg.T_ns == 12.5

    def test_omega_formula(self):
        cfg = AcquisitionConfig(T_ns=12.5, harmonic=1)
        assert abs(cfg.omega - 2 * math.pi / 12.5) < 1e-12

    def test_frequency_mhz(self):
        assert abs(AcquisitionConfig(T_ns=12.5).frequency_MHz - 80.0) < 1e-9

    def test_gate_width_ns(self):
        cfg = AcquisitionConfig(T_ns=10.0, gate_width_frac=0.3)
        assert abs(cfg.gate_width_ns - 3.0) < 1e-12

    def test_t_rec_ns(self):
        cfg = AcquisitionConfig(T_ns=10.0, T_rec_frac=0.8)
        assert abs(cfg.T_rec_ns - 8.0) < 1e-12

    def test_t0_ns(self):
        cfg = AcquisitionConfig(T_ns=10.0, t0_frac=0.1)
        assert abs(cfg.t0_ns - 1.0) < 1e-12

    def test_invalid_T_negative(self):
        with pytest.raises(ValueError):
            AcquisitionConfig(T_ns=-1.0)

    def test_invalid_harmonic_zero(self):
        with pytest.raises(ValueError):
            AcquisitionConfig(harmonic=0)

    def test_invalid_N_bins_one(self):
        with pytest.raises(ValueError):
            AcquisitionConfig(N_bins=1)

    def test_invalid_gate_width_zero(self):
        with pytest.raises(ValueError):
            AcquisitionConfig(gate_width_frac=0.0)

    def test_invalid_gate_width_over_one(self):
        with pytest.raises(ValueError):
            AcquisitionConfig(gate_width_frac=1.1)

    def test_invalid_T_rec_frac(self):
        with pytest.raises(ValueError):
            AcquisitionConfig(T_rec_frac=0.0)

    def test_invalid_tau_range(self):
        with pytest.raises(ValueError):
            AcquisitionConfig(tau_min_ns=5.0, tau_max_ns=1.0)

    def test_describe_all_modes(self):
        for mode in AcquisitionMode:
            cfg = AcquisitionConfig(mode=mode)
            text = cfg.describe()
            assert isinstance(text, str)
            assert mode.name in text

    def test_all_modes_exist(self):
        expected = {"CONTINUOUS", "DISCRETE", "GATED_SINGLE", "GATED_N", "TRUNCATED", "OFFSET"}
        actual = {m.name for m in AcquisitionMode}
        assert expected == actual


# ─────────────────────────────────────────────────────────────────────────────
# Continuous phasor
# ─────────────────────────────────────────────────────────────────────────────

class TestPhasorContinuous:
    def test_lies_on_universal_semicircle(self, base_cfg, tau_arr):
        g, s = phasor_continuous(tau_arr, base_cfg)
        dist2 = (g - 0.5) ** 2 + s ** 2
        np.testing.assert_allclose(dist2, 0.25, atol=1e-12)

    def test_s_non_negative(self, base_cfg, tau_arr):
        _, s = phasor_continuous(tau_arr, base_cfg)
        assert np.all(s >= 0)

    def test_tau_zero_limit(self, base_cfg):
        g, s = phasor_continuous(1e-9, base_cfg)
        assert abs(g - 1.0) < 1e-6
        assert abs(s) < 1e-6

    def test_tau_inf_limit(self, base_cfg):
        g, s = phasor_continuous(1e9, base_cfg)
        assert abs(g) < 1e-4
        assert abs(s) < 1e-4

    def test_harmonic2_still_on_semicircle(self, base_cfg, tau_arr):
        cfg2 = replace(base_cfg, harmonic=2)
        g, s = phasor_continuous(tau_arr, cfg2)
        np.testing.assert_allclose((g - 0.5) ** 2 + s ** 2, 0.25, atol=1e-12)

    def test_scalar_input(self, base_cfg):
        g, s = phasor_continuous(2.0, base_cfg)
        assert np.isscalar(g) or np.ndim(g) == 0 or len(np.atleast_1d(g)) == 1

    def test_output_shapes(self, base_cfg, tau_arr):
        g, s = phasor_continuous(tau_arr, base_cfg)
        assert g.shape == tau_arr.shape
        assert s.shape == tau_arr.shape


# ─────────────────────────────────────────────────────────────────────────────
# Discrete phasor
# ─────────────────────────────────────────────────────────────────────────────

class TestPhasorDiscrete:
    def test_large_N_converges_to_continuous(self, base_cfg, tau_arr):
        cfg_d = replace(base_cfg, mode=AcquisitionMode.DISCRETE, N_bins=1024)
        g_d, s_d = phasor_discrete(tau_arr, cfg_d)
        g_c, s_c = phasor_continuous(tau_arr, base_cfg)
        np.testing.assert_allclose(g_d, g_c, atol=5e-3)
        np.testing.assert_allclose(s_d, s_c, atol=5e-3)

    def test_small_N_differs_from_continuous(self, base_cfg, tau_arr):
        cfg_d = replace(base_cfg, N_bins=2)
        g_d, _ = phasor_discrete(tau_arr, cfg_d)
        g_c, _ = phasor_continuous(tau_arr, base_cfg)
        assert not np.allclose(g_d, g_c, atol=1e-3)

    def test_s_non_negative(self, base_cfg, tau_arr):
        cfg_d = replace(base_cfg, N_bins=16)
        _, s = phasor_discrete(tau_arr, cfg_d)
        assert np.all(s >= -1e-12)

    def test_output_shapes(self, base_cfg, tau_arr):
        cfg_d = replace(base_cfg, mode=AcquisitionMode.DISCRETE, N_bins=32)
        g, s = phasor_discrete(tau_arr, cfg_d)
        assert g.shape == tau_arr.shape
        assert s.shape == tau_arr.shape

    def test_output_finite(self, base_cfg, tau_arr):
        cfg_d = replace(base_cfg, mode=AcquisitionMode.DISCRETE, N_bins=64)
        g, s = phasor_discrete(tau_arr, cfg_d)
        assert np.all(np.isfinite(g))
        assert np.all(np.isfinite(s))


# ─────────────────────────────────────────────────────────────────────────────
# Gated single phasor
# ─────────────────────────────────────────────────────────────────────────────

class TestPhasorGatedSingle:
    def test_full_window_near_continuous(self, base_cfg, tau_arr):
        cfg_g = replace(base_cfg, mode=AcquisitionMode.GATED_SINGLE, gate_width_frac=0.9999)
        g_g, s_g = phasor_gated_single(tau_arr, cfg_g)
        g_c, s_c = phasor_continuous(tau_arr, base_cfg)
        mask = tau_arr >= 1.0
        np.testing.assert_allclose(g_g[mask], g_c[mask], atol=0.01)

    def test_output_shapes(self, base_cfg, tau_arr):
        cfg_g = replace(base_cfg, mode=AcquisitionMode.GATED_SINGLE, gate_width_frac=0.5)
        g, s = phasor_gated_single(tau_arr, cfg_g)
        assert g.shape == tau_arr.shape
        assert s.shape == tau_arr.shape

    def test_output_finite(self, base_cfg, tau_arr):
        cfg_g = replace(base_cfg, gate_width_frac=0.5)
        g, s = phasor_gated_single(tau_arr, cfg_g)
        assert np.all(np.isfinite(g))
        assert np.all(np.isfinite(s))


# ─────────────────────────────────────────────────────────────────────────────
# Gated N phasor
# ─────────────────────────────────────────────────────────────────────────────

class TestPhasorGatedN:
    def test_output_shapes(self, base_cfg, tau_arr):
        cfg_gn = replace(base_cfg, mode=AcquisitionMode.GATED_N, N_gates=4, gate_width_frac=0.5)
        g, s = phasor_gated_N(tau_arr, cfg_gn)
        assert g.shape == tau_arr.shape
        assert s.shape == tau_arr.shape

    def test_output_finite(self, base_cfg, tau_arr):
        cfg_gn = replace(base_cfg, mode=AcquisitionMode.GATED_N, N_gates=4, gate_width_frac=0.5)
        g, s = phasor_gated_N(tau_arr, cfg_gn)
        assert np.all(np.isfinite(g))
        assert np.all(np.isfinite(s))

    def test_many_gates_s_non_negative(self, base_cfg, tau_arr):
        # For N_gates=4 with the fundamental harmonic, s should be >= 0
        cfg_gn = replace(base_cfg, mode=AcquisitionMode.GATED_N,
                         N_gates=4, gate_width_frac=0.25)
        _, s = phasor_gated_N(tau_arr, cfg_gn)
        assert np.all(s >= -1e-10)

    @pytest.mark.parametrize("n_gates", [2, 4, 8])
    def test_various_gate_counts(self, base_cfg, tau_arr, n_gates):
        cfg_gn = replace(base_cfg, mode=AcquisitionMode.GATED_N,
                         N_gates=n_gates, gate_width_frac=0.4)
        g, s = phasor_gated_N(tau_arr, cfg_gn)
        assert g.shape == tau_arr.shape
        assert np.all(np.isfinite(g))


# ─────────────────────────────────────────────────────────────────────────────
# Truncated phasor
# ─────────────────────────────────────────────────────────────────────────────

class TestPhasorTruncated:
    def test_full_window_matches_continuous(self, base_cfg, tau_arr):
        cfg_tr = replace(base_cfg, mode=AcquisitionMode.TRUNCATED, T_rec_frac=1.0)
        g_tr, s_tr = phasor_truncated(tau_arr, cfg_tr)
        g_c, s_c = phasor_continuous(tau_arr, base_cfg)
        np.testing.assert_allclose(g_tr, g_c, atol=1e-6)
        np.testing.assert_allclose(s_tr, s_c, atol=1e-6)

    def test_short_window_deviates_from_semicircle(self, base_cfg, tau_arr):
        cfg_tr = replace(base_cfg, T_rec_frac=0.5)
        g_tr, s_tr = phasor_truncated(tau_arr, cfg_tr)
        dist2 = (g_tr - 0.5) ** 2 + s_tr ** 2
        assert not np.allclose(dist2, 0.25, atol=0.01)

    def test_output_finite(self, base_cfg, tau_arr):
        cfg_tr = replace(base_cfg, T_rec_frac=0.7)
        g, s = phasor_truncated(tau_arr, cfg_tr)
        assert np.all(np.isfinite(g))
        assert np.all(np.isfinite(s))


# ─────────────────────────────────────────────────────────────────────────────
# Offset phasor
# ─────────────────────────────────────────────────────────────────────────────

class TestPhasorOffset:
    def test_zero_offset_equals_continuous(self, base_cfg, tau_arr):
        cfg_off = replace(base_cfg, mode=AcquisitionMode.OFFSET, t0_frac=0.0)
        g_off, s_off = phasor_offset(tau_arr, cfg_off)
        g_c, s_c = phasor_continuous(tau_arr, base_cfg)
        np.testing.assert_allclose(g_off, g_c, atol=1e-12)
        np.testing.assert_allclose(s_off, s_c, atol=1e-12)

    def test_offset_preserves_modulus(self, base_cfg, tau_arr):
        cfg_off = replace(base_cfg, t0_frac=0.15)
        g_c, s_c = phasor_continuous(tau_arr, base_cfg)
        g_off, s_off = phasor_offset(tau_arr, cfg_off)
        m_c = np.sqrt(g_c ** 2 + s_c ** 2)
        m_off = np.sqrt(g_off ** 2 + s_off ** 2)
        np.testing.assert_allclose(m_off, m_c, atol=1e-12)

    @pytest.mark.parametrize("t0", [0.05, 0.1, 0.2, 0.3])
    def test_various_offsets_finite(self, base_cfg, tau_arr, t0):
        cfg_off = replace(base_cfg, mode=AcquisitionMode.OFFSET, t0_frac=t0)
        g, s = phasor_offset(tau_arr, cfg_off)
        assert np.all(np.isfinite(g))
        assert np.all(np.isfinite(s))


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher — phasor_from_config
# ─────────────────────────────────────────────────────────────────────────────

class TestPhasorFromConfig:
    def test_all_modes_run_and_finite(self, tau_arr):
        for mode in AcquisitionMode:
            cfg = AcquisitionConfig(mode=mode)
            g, s = phasor_from_config(tau_arr, cfg)
            assert g.shape == tau_arr.shape, f"Shape mismatch for {mode}"
            assert np.all(np.isfinite(g)), f"Non-finite g for {mode}"
            assert np.all(np.isfinite(s)), f"Non-finite s for {mode}"

    def test_continuous_dispatch_matches_direct(self, base_cfg, tau_arr):
        cfg = replace(base_cfg, mode=AcquisitionMode.CONTINUOUS)
        g1, s1 = phasor_from_config(tau_arr, cfg)
        g2, s2 = phasor_continuous(tau_arr, base_cfg)
        np.testing.assert_allclose(g1, g2)
        np.testing.assert_allclose(s1, s2)

    def test_discrete_dispatch_matches_direct(self, base_cfg, tau_arr):
        cfg = replace(base_cfg, mode=AcquisitionMode.DISCRETE, N_bins=64)
        g1, s1 = phasor_from_config(tau_arr, cfg)
        g2, s2 = phasor_discrete(tau_arr, cfg)
        np.testing.assert_allclose(g1, g2)

    def test_gated_n_dispatch_matches_direct(self, base_cfg, tau_arr):
        cfg = replace(base_cfg, mode=AcquisitionMode.GATED_N, N_gates=4, gate_width_frac=0.5)
        g1, s1 = phasor_from_config(tau_arr, cfg)
        g2, s2 = phasor_gated_N(tau_arr, cfg)
        np.testing.assert_allclose(g1, g2)


# ─────────────────────────────────────────────────────────────────────────────
# Locus builder
# ─────────────────────────────────────────────────────────────────────────────

class TestLocus:
    def test_build_locus_shapes_consistent(self, base_cfg):
        g, s, tau = build_locus(base_cfg)
        assert g.shape == s.shape == tau.shape

    def test_build_locus_all_finite(self, base_cfg):
        g, s, tau = build_locus(base_cfg)
        assert np.all(np.isfinite(g))
        assert np.all(np.isfinite(s))

    def test_build_locus_continuous_on_semicircle(self, base_cfg):
        cfg = replace(base_cfg, mode=AcquisitionMode.CONTINUOUS)
        g, s, _ = build_locus(cfg)
        np.testing.assert_allclose((g - 0.5) ** 2 + s ** 2, 0.25, atol=1e-10)

    def test_build_loci_returns_list(self, base_cfg):
        cfgs = [
            replace(base_cfg, mode=AcquisitionMode.CONTINUOUS),
            replace(base_cfg, mode=AcquisitionMode.DISCRETE, N_bins=32),
        ]
        results = build_loci(cfgs)
        assert len(results) == 2
        for g, s, tau in results:
            assert g.shape == s.shape == tau.shape

    def test_tau_grid_monotone(self, base_cfg):
        tau = tau_grid(base_cfg)
        assert np.all(np.diff(tau) > 0)

    def test_tau_grid_bounds(self, base_cfg):
        tau = tau_grid(base_cfg)
        assert tau[0] >= base_cfg.tau_min_ns
        assert tau[-1] <= base_cfg.tau_max_ns

    def test_custom_tau_passed_to_build_locus(self, base_cfg):
        custom_tau = np.array([0.5, 1.0, 2.0, 5.0])
        g, s, tau_out = build_locus(base_cfg, tau=custom_tau)
        np.testing.assert_array_equal(tau_out, custom_tau)


# ─────────────────────────────────────────────────────────────────────────────
# Universal semicircle
# ─────────────────────────────────────────────────────────────────────────────

class TestUniversalSemicircle:
    def test_on_circle_equation(self):
        g, s = universal_semicircle(500)
        dist2 = (g - 0.5) ** 2 + s ** 2
        np.testing.assert_allclose(dist2, 0.25, atol=1e-12)

    def test_s_non_negative(self):
        _, s = universal_semicircle(300)
        assert np.all(s >= 0)

    def test_endpoints(self):
        g, s = universal_semicircle(300)
        assert abs(g[0] - 1.0) < 1e-10
        assert abs(s[0]) < 1e-10
        assert abs(g[-1]) < 1e-10
        assert abs(s[-1]) < 1e-10

    def test_custom_n_pts(self):
        g, s = universal_semicircle(100)
        assert len(g) == 100
        assert len(s) == 100


# ─────────────────────────────────────────────────────────────────────────────
# Discrete SEPL circle geometry
# ─────────────────────────────────────────────────────────────────────────────

class TestSeplCenterRadiusDiscrete:
    def test_returns_three_floats(self, base_cfg):
        cfg_d = replace(base_cfg, mode=AcquisitionMode.DISCRETE, N_bins=8)
        gc, sc, r = sepl_center_radius_discrete(cfg_d)
        assert isinstance(gc, float)
        assert isinstance(sc, float)
        assert isinstance(r, float)

    def test_gc_is_half(self, base_cfg):
        cfg_d = replace(base_cfg, N_bins=16)
        gc, _, _ = sepl_center_radius_discrete(cfg_d)
        assert abs(gc - 0.5) < 1e-12

    def test_sepl_lies_on_derived_circle(self, base_cfg):
        cfg_d = replace(base_cfg, mode=AcquisitionMode.DISCRETE, N_bins=8)
        gc, sc, r = sepl_center_radius_discrete(cfg_d)
        x_arr = np.linspace(1e-8, 1 - 1e-8, 3000)
        tau_test = -base_cfg.T_ns / (cfg_d.N_bins * np.log(x_arr))
        g, s = phasor_discrete(tau_test, cfg_d)
        dist = np.sqrt((g - gc) ** 2 + (s - sc) ** 2)
        np.testing.assert_allclose(dist, r, atol=1e-6)

    def test_large_N_radius_approaches_half(self, base_cfg):
        cfg_d = replace(base_cfg, N_bins=4096)
        _, _, r = sepl_center_radius_discrete(cfg_d)
        assert abs(r - 0.5) < 1e-3


# ─────────────────────────────────────────────────────────────────────────────
# Lifetime inversion
# ─────────────────────────────────────────────────────────────────────────────

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
        tau_m = modulus_lifetime(g, s, base_cfg)
        np.testing.assert_allclose(tau_ph, tau_m, rtol=1e-8)

    @pytest.mark.parametrize("method", ["phase", "modulus", "mean"])
    def test_dispatcher_all_methods(self, base_cfg, tau_arr, method):
        g, s = phasor_continuous(tau_arr, base_cfg)
        tau_r = lifetime_from_phasor(g, s, base_cfg, method=method)
        assert tau_r.shape == tau_arr.shape
        assert np.all(np.isfinite(tau_r))

    def test_invalid_method_raises(self, base_cfg):
        with pytest.raises(ValueError):
            lifetime_from_phasor([0.5], [0.3], base_cfg, method="bogus")

    def test_modulus_zero_at_origin(self, base_cfg):
        tau_r = modulus_lifetime(np.array([0.0]), np.array([0.0]), base_cfg)
        assert np.all(np.isnan(tau_r) | (tau_r == 0.0))

    def test_phase_lifetime_positive_for_positive_tau(self, base_cfg, tau_arr):
        g, s = phasor_continuous(tau_arr, base_cfg)
        tau_ph = phase_lifetime(g, s, base_cfg)
        assert np.all(tau_ph > 0)


# ─────────────────────────────────────────────────────────────────────────────
# Fractional components
# ─────────────────────────────────────────────────────────────────────────────

class TestFractionalComponents:
    def test_two_species_30_70_split(self, base_cfg):
        g1, s1 = phasor_continuous(1.0, base_cfg)
        g2, s2 = phasor_continuous(5.0, base_cfg)
        g_mix = 0.3 * g1 + 0.7 * g2
        s_mix = 0.3 * s1 + 0.7 * s2
        f1, f2 = fractional_components(
            g_mix, s_mix, float(g1), float(s1), float(g2), float(s2)
        )
        assert abs(float(f1) - 0.3) < 1e-9
        assert abs(float(f2) - 0.7) < 1e-9

    def test_pure_species_1_gives_f1_one(self, base_cfg):
        g1, s1 = phasor_continuous(1.0, base_cfg)
        g2, s2 = phasor_continuous(5.0, base_cfg)
        f1, f2 = fractional_components(g1, s1, float(g1), float(s1), float(g2), float(s2))
        assert abs(float(f1) - 1.0) < 1e-9
        assert abs(float(f2) - 0.0) < 1e-9

    def test_pure_species_2_gives_f2_one(self, base_cfg):
        g1, s1 = phasor_continuous(1.0, base_cfg)
        g2, s2 = phasor_continuous(5.0, base_cfg)
        f1, f2 = fractional_components(g2, s2, float(g1), float(s1), float(g2), float(s2))
        assert abs(float(f1) - 0.0) < 1e-9
        assert abs(float(f2) - 1.0) < 1e-9

    def test_fractions_sum_to_one(self, base_cfg):
        g1, s1 = phasor_continuous(0.5, base_cfg)
        g2, s2 = phasor_continuous(3.0, base_cfg)
        tau_vals = np.array([0.8, 1.5, 2.5])
        g_mix_arr, s_mix_arr = phasor_continuous(tau_vals, base_cfg)
        f1, f2 = fractional_components(
            g_mix_arr, s_mix_arr, float(g1), float(s1), float(g2), float(s2)
        )
        np.testing.assert_allclose(f1 + f2, 1.0, atol=1e-9)

    def test_coincident_species_raises(self, base_cfg):
        g, s = phasor_continuous(2.0, base_cfg)
        with pytest.raises(ValueError):
            fractional_components(g, s, float(g), float(s), float(g), float(s))


# ─────────────────────────────────────────────────────────────────────────────
# Phase lifetime gated (corrected estimator)
# ─────────────────────────────────────────────────────────────────────────────

class TestPhaseLifetimeGated:
    def test_returns_array(self, base_cfg):
        pytest.importorskip("scipy")
        cfg_g = replace(base_cfg, mode=AcquisitionMode.GATED_SINGLE, gate_width_frac=0.5)
        tau_input = np.array([1.0, 2.0, 4.0])
        g, s = phasor_gated_single(tau_input, cfg_g)
        tau_corr = phase_lifetime_gated(g, s, cfg_g)
        assert tau_corr.shape == tau_input.shape

    def test_corrected_closer_to_truth_than_naive(self, base_cfg):
        pytest.importorskip("scipy")
        cfg_g = replace(base_cfg, mode=AcquisitionMode.GATED_SINGLE, gate_width_frac=0.5)
        tau_true = np.array([2.0, 4.0])
        g, s = phasor_gated_single(tau_true, cfg_g)
        tau_naive = phase_lifetime(g, s, cfg_g)
        tau_corr = phase_lifetime_gated(g, s, cfg_g)
        err_naive = np.abs(tau_naive - tau_true)
        err_corr = np.abs(tau_corr - tau_true)
        assert np.all(err_corr <= err_naive + 0.01)


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers (smoke tests — verify no crash and correct return type)
# ─────────────────────────────────────────────────────────────────────────────

class TestPlotHelpers:
    def test_plot_phasor_returns_fig_ax(self, base_cfg):
        import matplotlib.pyplot as plt
        fig, ax = plot_phasor(base_cfg)
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plot_phasor_into_existing_ax(self, base_cfg):
        import matplotlib.pyplot as plt
        fig_ext, ax_ext = plt.subplots()
        fig, ax = plot_phasor(base_cfg, ax=ax_ext)
        assert ax is ax_ext
        plt.close(fig_ext)

    def test_plot_phasor_all_modes(self):
        import matplotlib.pyplot as plt
        for mode in AcquisitionMode:
            cfg = AcquisitionConfig(mode=mode)
            fig, ax = plot_phasor(cfg)
            assert fig is not None
            plt.close(fig)

    def test_plot_locus_comparison(self, base_cfg):
        import matplotlib.pyplot as plt
        cfgs = [
            replace(base_cfg, mode=AcquisitionMode.CONTINUOUS),
            replace(base_cfg, mode=AcquisitionMode.DISCRETE, N_bins=32),
            replace(base_cfg, mode=AcquisitionMode.TRUNCATED, T_rec_frac=0.7),
        ]
        fig, ax = plot_locus_comparison(cfgs)
        assert fig is not None
        plt.close(fig)

    def test_plot_discrete_N_sweep(self, base_cfg):
        import matplotlib.pyplot as plt
        fig, ax = plot_discrete_N_sweep(base_cfg, N_values=[4, 16, 64])
        assert fig is not None
        plt.close(fig)

    def test_plot_phasor_no_universal_no_ticks(self, base_cfg):
        import matplotlib.pyplot as plt
        fig, ax = plot_phasor(base_cfg, show_universal=False, show_ticks=False)
        assert fig is not None
        plt.close(fig)
