"""Tests for pyfli.analysis.utils.compute_detailed_results."""

import numpy as np
import pytest

from pyfli.analysis.utils import compute_detailed_results


@pytest.fixture
def irf_delta():
    """Near-delta IRF so convolution stays close to identity."""
    bins = 64
    irf = np.zeros(bins, dtype=np.float32)
    irf[0] = 1.0
    return irf


def test_requires_either_tau_or_biexp_inputs(irf_delta):
    with pytest.raises(ValueError):
        compute_detailed_results(freq_acq=80.0, binned_irf=irf_delta)


def test_rejects_both_input_styles(irf_delta):
    H, W = 2, 2
    tau1 = np.full((H, W), 0.5, dtype=np.float32)
    tau2 = np.full((H, W), 1.5, dtype=np.float32)
    alpha1 = np.full((H, W), 0.5, dtype=np.float32)
    tau = np.full((H, W), 1.0, dtype=np.float32)
    with pytest.raises(ValueError):
        compute_detailed_results(
            tau1=tau1,
            tau2=tau2,
            alpha1=alpha1,
            tau=tau,
            freq_acq=80.0,
            binned_irf=irf_delta,
        )


def test_rejects_unknown_model_type(irf_delta):
    H, W = 2, 2
    tau = np.full((H, W), 1.0, dtype=np.float32)
    with pytest.raises(ValueError):
        compute_detailed_results(
            tau=tau, freq_acq=80.0, binned_irf=irf_delta, model_type="triexponential"
        )


def test_tau_only_mono_branch_skips_classifier_and_matches_input(irf_delta):
    H, W = 3, 3
    tau = np.full((H, W), 0.8, dtype=np.float32)
    decay = np.ones((H, W, irf_delta.shape[-1]), dtype=np.float32)

    out = compute_detailed_results(
        tau=tau,
        freq_acq=80.0,
        binned_irf=irf_delta,
        binned_decay=decay,
        model_type="mono-exponential",
    )
    maps = out["results"]["maps"]
    tr_maps = out["results"]["TR_maps"]

    assert set(maps) >= {
        "photon_count_map",
        "tau_map",
        "R2_map",
        "chi2_map",
        "reduced_chi2_map",
    }
    assert set(tr_maps) == {"fit_map", "residual_map", "sdf_map", "convolved_map"}
    np.testing.assert_allclose(maps["tau_map"], tau)


def test_biexponential_sdf_uses_per_tau_normalization(irf_delta):
    """Regression test for the 1/tau mixture-weight bug: sdf must match
    forward_model.decay_kernel's convention, (a1/tau1)*exp(-t/tau1) +
    ((1-a1)/tau2)*exp(-t/tau2), not the un-normalized a1*exp(...) + (1-a1)*exp(...)."""
    H, W, bins = 2, 2, irf_delta.shape[-1]
    tau1 = np.full((H, W), 0.4, dtype=np.float32)
    tau2 = np.full((H, W), 1.6, dtype=np.float32)
    alpha1 = np.full((H, W), 0.3, dtype=np.float32)
    freq_acq = 80.0

    out = compute_detailed_results(
        tau1=tau1,
        tau2=tau2,
        alpha1=alpha1,
        freq_acq=freq_acq,
        binned_irf=irf_delta,
        model_type="bi-exponential",
    )
    sdf = out["results"]["TR_maps"]["sdf_map"]

    t = np.linspace(0, 1000.0 / freq_acq, bins, endpoint=False, dtype=np.float32)
    expected = (0.3 / 0.4) * np.exp(-t / 0.4) + (0.7 / 1.6) * np.exp(-t / 1.6)
    np.testing.assert_allclose(sdf[0, 0], expected, rtol=1e-4)


def test_photon_count_map_scaled_by_dt(irf_delta):
    H, W, bins = 2, 2, irf_delta.shape[-1]
    tau = np.full((H, W), 1.0, dtype=np.float32)
    decay = np.full((H, W, bins), 2.0, dtype=np.float32)
    freq_acq = 80.0

    out = compute_detailed_results(
        tau=tau,
        freq_acq=freq_acq,
        binned_irf=irf_delta,
        binned_decay=decay,
        model_type="mono-exponential",
    )
    t = np.linspace(0, 1000.0 / freq_acq, bins, endpoint=False, dtype=np.float32)
    dt = float(t[1] - t[0])
    expected_photon_count = decay.sum(axis=-1) * dt
    np.testing.assert_allclose(
        out["results"]["maps"]["photon_count_map"], expected_photon_count
    )
