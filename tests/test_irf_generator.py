"""
Tests for IRFGenerator — synthetic IRF trace generation for simulator and
alignment testing.
"""

import numpy as np
import pytest

from pyfli.simulator.irf_sim.irf_generator import IRFGenerator

T_LASER = 12.5
NUM_BINS = 256
GATE_DELAY = T_LASER / NUM_BINS
MU = 60


# ─────────────────────────────────────────────────────────────────────────────
# gaussianIRF
# ─────────────────────────────────────────────────────────────────────────────


class TestGaussianIRF:
    def test_peak_location_and_height(self):
        irf = IRFGenerator.gaussianIRF(mu=MU, T=T_LASER, num_bins=NUM_BINS, sigma=0.1)
        assert irf.shape == (NUM_BINS,)
        assert int(np.argmax(irf)) == MU
        assert np.isclose(irf.max(), 1.0)

    def test_fwhm_matches_theory(self):
        sigma_ns = 0.15
        irf = IRFGenerator.gaussianIRF(
            mu=MU, T=T_LASER, num_bins=NUM_BINS, sigma=sigma_ns
        )
        sigma_bins = sigma_ns / GATE_DELAY
        measured_fwhm = np.sum(irf >= 0.5)
        expected_fwhm = 2.355 * sigma_bins
        assert abs(measured_fwhm - expected_fwhm) <= 1.5

    def test_sigma_widens_pulse(self):
        narrow = IRFGenerator.gaussianIRF(mu=MU, sigma=0.05)
        wide = IRFGenerator.gaussianIRF(mu=MU, sigma=0.4)
        assert np.sum(narrow >= 0.5) < np.sum(wide >= 0.5)

    def test_fractional_mu_supported(self):
        irf = IRFGenerator.gaussianIRF(mu=60.5, sigma=0.2)
        # symmetric around the fractional peak -> neighbors should be ~equal
        assert np.isclose(irf[60], irf[61], atol=1e-9)

    def test_3d_broadcast(self):
        H, W = 4, 5
        cube = IRFGenerator.gaussianIRF(mu=MU, H=H, W=W)
        assert cube.shape == (H, W, NUM_BINS)
        assert np.allclose(cube, cube[0, 0, :])

    @pytest.mark.parametrize("kwargs", [{"H": 5}, {"W": 5}])
    def test_h_w_must_both_be_given(self, kwargs):
        with pytest.raises(ValueError):
            IRFGenerator.gaussianIRF(mu=MU, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# expdecay
# ─────────────────────────────────────────────────────────────────────────────


class TestExpDecay:
    def test_peak_location_and_height(self):
        irf = IRFGenerator.expdecay(mu=MU, T=T_LASER, num_bins=NUM_BINS, tau=0.1)
        assert irf.shape == (NUM_BINS,)
        assert int(np.argmax(irf)) == MU
        assert np.isclose(irf.max(), 1.0)

    def test_causal_zero_before_peak(self):
        irf = IRFGenerator.expdecay(mu=MU, tau=0.1)
        assert np.all(irf[:MU] == 0)

    def test_matches_closed_form_exponential(self):
        tau_ns = 0.1
        irf = IRFGenerator.expdecay(mu=MU, T=T_LASER, num_bins=NUM_BINS, tau=tau_ns)
        n = np.arange(1, 10)
        expected = np.exp(-n * GATE_DELAY / tau_ns)
        assert np.allclose(irf[MU + n], expected, atol=1e-6)

    def test_larger_tau_decays_slower(self):
        fast = IRFGenerator.expdecay(mu=MU, tau=0.05)
        slow = IRFGenerator.expdecay(mu=MU, tau=0.4)
        assert slow[MU + 5] > fast[MU + 5]

    def test_3d_broadcast(self):
        H, W = 3, 3
        cube = IRFGenerator.expdecay(mu=MU, H=H, W=W)
        assert cube.shape == (H, W, NUM_BINS)
        assert np.allclose(cube, cube[0, 0, :])


# ─────────────────────────────────────────────────────────────────────────────
# gaussianExpIRF (EMG)
# ─────────────────────────────────────────────────────────────────────────────


class TestGaussianExpIRF:
    def test_peak_normalized(self):
        irf = IRFGenerator.gaussianExpIRF(
            mu=MU, T=T_LASER, num_bins=NUM_BINS, sigma=0.1, tau=0.1
        )
        assert irf.shape == (NUM_BINS,)
        assert np.isclose(irf.max(), 1.0)

    def test_peak_at_or_after_mu(self):
        emg = IRFGenerator.gaussianExpIRF(mu=MU, sigma=0.1, tau=0.1)
        assert int(np.argmax(emg)) >= MU

    def test_broader_tail_than_pure_gaussian(self):
        gauss = IRFGenerator.gaussianIRF(mu=MU, sigma=0.1)
        emg = IRFGenerator.gaussianExpIRF(mu=MU, sigma=0.1, tau=0.1)
        tail_bin = MU + 5
        assert emg[tail_bin] > gauss[tail_bin]

    def test_3d_broadcast(self):
        H, W = 2, 2
        cube = IRFGenerator.gaussianExpIRF(mu=MU, H=H, W=W)
        assert cube.shape == (H, W, NUM_BINS)
        assert np.allclose(cube, cube[0, 0, :])
