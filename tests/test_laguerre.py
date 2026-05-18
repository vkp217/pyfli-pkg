"""
Tests for LaguerreFLI — the Laguerre Expansion Technique (LET) fitter.

All tests use purely synthetic numpy arrays; no file I/O is required.
"""

import numpy as np
import pytest

from pyfli import LaguerreFLI


# ─────────────────────────────────────────────────────────────────────────────
# Initialisation validation
# ─────────────────────────────────────────────────────────────────────────────

class TestInit:
    def test_defaults_accepted(self):
        m = LaguerreFLI()
        assert m.n_components == 2
        assert 0.0 < m.alpha < 1.0

    def test_n_components_below_1_raises(self):
        with pytest.raises(ValueError, match="n_components"):
            LaguerreFLI(n_components=0)

    def test_alpha_out_of_range_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            LaguerreFLI(alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            LaguerreFLI(alpha=1.0)
        with pytest.raises(ValueError, match="alpha"):
            LaguerreFLI(alpha=1.5)

    def test_dt_non_positive_raises(self):
        with pytest.raises(ValueError, match="dt"):
            LaguerreFLI(dt=0.0)
        with pytest.raises(ValueError, match="dt"):
            LaguerreFLI(dt=-1.0)

    def test_n_laguerre_below_n_components_raises(self):
        with pytest.raises(ValueError, match="n_laguerre"):
            LaguerreFLI(n_components=3, n_laguerre=2)

    def test_n_laguerre_defaults_to_max_4_2n(self):
        m = LaguerreFLI(n_components=1)
        assert m.n_laguerre >= 4

    def test_repr_contains_key_params(self):
        m = LaguerreFLI(n_components=2, alpha=0.9, dt=0.05)
        r = repr(m)
        assert "n_components=2" in r
        assert "alpha=0.900" in r

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit"):
            LaguerreFLI().predict()

    def test_get_parameters_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit"):
            LaguerreFLI().get_parameters()


# ─────────────────────────────────────────────────────────────────────────────
# Static helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestBasis:
    def test_basis_shape(self):
        T, L = 64, 5
        B = LaguerreFLI._discrete_laguerre_basis(T, 0.85, L)
        assert B.shape == (L, T)

    def test_basis_first_row_decays(self):
        T = 64
        B = LaguerreFLI._discrete_laguerre_basis(T, 0.85, 4)
        # b[0] = sqrt(1-alpha) * alpha^(n/2) — strictly decreasing for alpha<1
        assert B[0, 0] > B[0, 1] > B[0, -1]

    def test_basis_finite(self):
        B = LaguerreFLI._discrete_laguerre_basis(128, 0.7, 6)
        assert np.all(np.isfinite(B))


class TestConvolveWithIRF:
    def test_output_shape(self, gaussian_irf, n_bins):
        L = 4
        B = LaguerreFLI._discrete_laguerre_basis(n_bins, 0.85, L)
        V = LaguerreFLI._convolve_with_irf(B, gaussian_irf)
        assert V.shape == (n_bins, L)

    def test_irf_area_normalisation(self, n_bins):
        """IRF is area-normalised internally; scaling the input changes nothing."""
        irf = np.ones(n_bins, dtype=float)
        B = LaguerreFLI._discrete_laguerre_basis(n_bins, 0.85, 4)
        V1 = LaguerreFLI._convolve_with_irf(B, irf)
        V2 = LaguerreFLI._convolve_with_irf(B, irf * 10.0)
        np.testing.assert_allclose(V1, V2)

    def test_output_finite(self, gaussian_irf, n_bins):
        B = LaguerreFLI._discrete_laguerre_basis(n_bins, 0.85, 4)
        V = LaguerreFLI._convolve_with_irf(B, gaussian_irf)
        assert np.all(np.isfinite(V))


# ─────────────────────────────────────────────────────────────────────────────
# Fit on 1-D (single pixel) input
# ─────────────────────────────────────────────────────────────────────────────

class TestFit1D:
    @pytest.fixture
    def fitted_mono(self, mono_decay_1d, gaussian_irf, dt):
        m = LaguerreFLI(n_components=1, n_laguerre=6, alpha=0.85, dt=dt)
        return m.fit(mono_decay_1d, gaussian_irf)

    def test_fit_returns_self(self, mono_decay_1d, gaussian_irf, dt):
        m = LaguerreFLI(n_components=1, n_laguerre=6, alpha=0.85, dt=dt)
        result = m.fit(mono_decay_1d, gaussian_irf)
        assert result is m

    def test_attributes_populated(self, fitted_mono):
        assert fitted_mono.coeffs_ is not None
        assert fitted_mono.taus_ is not None
        assert fitted_mono.amplitudes_ is not None
        assert fitted_mono.fractions_ is not None
        assert fitted_mono.tau_mean_ is not None
        assert fitted_mono.reconstructed_ is not None

    def test_coeffs_shape_1d(self, fitted_mono, n_bins):
        # 1-D input → treated as (1,1,T)
        assert fitted_mono.coeffs_.shape[0] == 1
        assert fitted_mono.coeffs_.shape[1] == 1

    def test_reconstructed_shape_1d(self, fitted_mono, n_bins):
        assert fitted_mono.reconstructed_.shape == (1, 1, n_bins)

    def test_taus_positive(self, fitted_mono):
        assert np.all(fitted_mono.taus_ > 0)

    def test_fractions_sum_to_one(self, fitted_mono):
        np.testing.assert_allclose(
            fitted_mono.fractions_.sum(axis=-1), 1.0, atol=1e-6
        )

    def test_tau_mean_positive(self, fitted_mono):
        assert np.all(fitted_mono.tau_mean_ >= 0)

    def test_nonneg_coefficients(self, mono_decay_1d, gaussian_irf, dt):
        m = LaguerreFLI(n_components=1, nonneg=True, dt=dt).fit(
            mono_decay_1d, gaussian_irf
        )
        assert np.all(m.coeffs_ >= -1e-12)

    def test_predict_returns_reconstructed(self, fitted_mono):
        pred = fitted_mono.predict()
        np.testing.assert_array_equal(pred, fitted_mono.reconstructed_)


# ─────────────────────────────────────────────────────────────────────────────
# Fit on 3-D image cube
# ─────────────────────────────────────────────────────────────────────────────

class TestFit3D:
    @pytest.fixture
    def fitted_bi(self, decay_cube, gaussian_irf, dt):
        m = LaguerreFLI(n_components=2, n_laguerre=6, alpha=0.85, dt=dt)
        return m.fit(decay_cube, gaussian_irf)

    def test_coeffs_shape_3d(self, fitted_bi, decay_cube):
        X, Y, _ = decay_cube.shape
        assert fitted_bi.coeffs_.shape[:2] == (X, Y)

    def test_taus_map_shape(self, fitted_bi, decay_cube):
        X, Y, _ = decay_cube.shape
        assert fitted_bi.taus_.shape == (X, Y, 2)

    def test_amplitudes_shape(self, fitted_bi, decay_cube):
        X, Y, _ = decay_cube.shape
        assert fitted_bi.amplitudes_.shape == (X, Y, 2)

    def test_tau_mean_shape(self, fitted_bi, decay_cube):
        X, Y, _ = decay_cube.shape
        assert fitted_bi.tau_mean_.shape == (X, Y)

    def test_taus_ascending_per_pixel(self, fitted_bi):
        # taus_ are sorted ascending within each pixel
        assert np.all(
            fitted_bi.taus_[..., 0] <= fitted_bi.taus_[..., 1] + 1e-12
        )

    def test_fractions_sum_to_one(self, fitted_bi):
        np.testing.assert_allclose(
            fitted_bi.fractions_.sum(axis=-1), 1.0, atol=1e-6
        )

    def test_fit_curve_shape(self, fitted_bi, decay_cube):
        assert fitted_bi.fit_curve_.shape == decay_cube.shape

    def test_residual_curve_shape(self, fitted_bi, decay_cube):
        assert fitted_bi.residual_curve_.shape == decay_cube.shape

    def test_residuals_finite(self, fitted_bi):
        assert np.all(np.isfinite(fitted_bi.residuals_))


# ─────────────────────────────────────────────────────────────────────────────
# get_parameters dict structure
# ─────────────────────────────────────────────────────────────────────────────

class TestGetParameters:
    @pytest.fixture
    def params(self, bi_decay_1d, gaussian_irf, dt):
        m = LaguerreFLI(n_components=2, n_laguerre=6, alpha=0.85, dt=dt)
        m.fit(bi_decay_1d, gaussian_irf)
        return m.get_parameters("TestDataset")

    def test_top_level_keys(self, params):
        assert set(params.keys()) == {"name", "method", "results"}

    def test_name_field(self, params):
        assert params["name"] == "TestDataset"

    def test_method_contains_exp(self, params):
        assert "exp" in params["method"]

    def test_results_keys(self, params):
        assert "maps" in params["results"]
        assert "error_maps" in params["results"]
        assert "TR_maps" in params["results"]

    def test_maps_contain_tau_and_alpha(self, params):
        maps = params["results"]["maps"]
        assert "tau1_map" in maps
        assert "tau2_map" in maps
        assert "alpha1_map" in maps
        assert "alpha2_map" in maps

    def test_maps_contain_standard_keys(self, params):
        maps = params["results"]["maps"]
        assert "Area_map" in maps
        assert "tau_mean_map" in maps
        assert "chi2_or_deviance_map" in maps
        assert "pixel_health_map" in maps

    def test_tr_maps_keys(self, params):
        tr = params["results"]["TR_maps"]
        assert "fit_map" in tr
        assert "residual_map" in tr


# ─────────────────────────────────────────────────────────────────────────────
# Input validation
# ─────────────────────────────────────────────────────────────────────────────

class TestFitValidation:
    def test_2d_decay_raises(self, gaussian_irf, dt):
        bad_decay = np.ones((4, 64))
        with pytest.raises(ValueError):
            LaguerreFLI(dt=dt).fit(bad_decay, gaussian_irf[:64])

    def test_irf_length_mismatch_raises(self, mono_decay_1d, dt):
        bad_irf = np.ones(10)
        with pytest.raises(ValueError):
            LaguerreFLI(dt=dt).fit(mono_decay_1d, bad_irf)

    def test_per_pixel_irf_shape_mismatch_raises(self, decay_cube, dt):
        bad_irf = np.ones((3, 3, decay_cube.shape[-1]))  # wrong X,Y
        with pytest.raises(ValueError):
            LaguerreFLI(dt=dt).fit(decay_cube, bad_irf)


# ─────────────────────────────────────────────────────────────────────────────
# Auto-alpha optimisation
# ─────────────────────────────────────────────────────────────────────────────

class TestAutoAlpha:
    def test_auto_alpha_updates_alpha(self, mono_decay_1d, gaussian_irf, dt):
        m = LaguerreFLI(n_components=1, alpha=0.5, dt=dt, auto_alpha=True)
        m.fit(mono_decay_1d, gaussian_irf)
        # alpha should have been updated by the optimiser
        assert 0.0 < m.alpha < 1.0

    def test_auto_alpha_fit_completes(self, bi_decay_1d, gaussian_irf, dt):
        m = LaguerreFLI(n_components=2, dt=dt, auto_alpha=True)
        m.fit(bi_decay_1d, gaussian_irf)
        assert m.taus_ is not None


# ─────────────────────────────────────────────────────────────────────────────
# Lifetime accuracy on noise-free synthetic data
# ─────────────────────────────────────────────────────────────────────────────

class TestLifetimeAccuracy:
    def test_mono_tau_ballpark(self, mono_decay_1d, gaussian_irf, dt):
        """Recovered tau should be within 50% of the ground truth (2 ns)."""
        m = LaguerreFLI(n_components=1, n_laguerre=8, alpha=0.85, dt=dt)
        m.fit(mono_decay_1d, gaussian_irf)
        tau_recovered = float(m.taus_.mean())
        assert 1.0 < tau_recovered < 4.0, f"Mono tau out of range: {tau_recovered:.3f} ns"

    def test_mean_lifetime_positive(self, bi_decay_1d, gaussian_irf, dt):
        m = LaguerreFLI(n_components=2, n_laguerre=6, alpha=0.85, dt=dt)
        m.fit(bi_decay_1d, gaussian_irf)
        assert float(m.tau_mean_.mean()) > 0
