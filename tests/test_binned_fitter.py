"""Tests for pyfli.solver.FLIBinner and pyfli.solver.BinnedFLIFitter."""

import numpy as np
import pytest

from pyfli.solver import BinnedFLIFitter, FLIBinner


class FakeCPUProcessor:
    """Duck-types FLICPUProcessor: has process_image, not fit_image."""

    def __init__(self, freq=80.0):
        self.freq = freq
        self.process_image_calls = []
        self.save_results_calls = []

    def process_image(self, image_cube, irf_cube, mask, data_name, **kwargs):
        self.process_image_calls.append(
            {
                "image_cube": image_cube,
                "irf_cube": irf_cube,
                "mask": mask,
                "data_name": data_name,
                **kwargs,
            }
        )
        return {"results": {"maps": {"tau1_map": np.zeros((2, 2))}}}

    def save_results(self, dataset, folder):
        self.save_results_calls.append((dataset, folder))


class FakeGPUProcessor:
    """Duck-types FLIGPUProcessor: has fit_image, not process_image."""

    def __init__(self, freq=80.0):
        self.freq = freq
        self.fit_image_calls = []

    def fit_image(self, image_cube, irf_cube, mask, data_name, **kwargs):
        self.fit_image_calls.append(
            {
                "image_cube": image_cube,
                "irf_cube": irf_cube,
                "mask": mask,
                "data_name": data_name,
                **kwargs,
            }
        )
        return {"results": {"maps": {"tau1_map": np.zeros((2, 2))}}}


class UnsupportedProcessor:
    def __init__(self, freq=80.0):
        self.freq = freq


# --------------------------------------------------------------------------- #
# FLIBinner
# --------------------------------------------------------------------------- #


def test_flibinner_zero_radius_is_identity():
    rng = np.random.default_rng(0)
    img = rng.poisson(10, size=(4, 4, 8)).astype(np.float32)
    irf = rng.poisson(5, size=(4, 4, 8)).astype(np.float32)

    binner = FLIBinner(bin_radius=0)
    b_img, b_irf = binner.apply_binning(img, irf)

    np.testing.assert_allclose(b_img, img)
    np.testing.assert_allclose(b_irf, irf)


def test_flibinner_sums_neighborhood():
    # Single hot pixel at the center of a 5x5 image; radius=1 should sum the
    # 3x3 neighborhood, so every pixel touching the hot one picks up its value.
    img = np.zeros((5, 5, 1), dtype=np.float32)
    img[2, 2, 0] = 9.0
    irf = np.zeros((5, 5, 1), dtype=np.float32)

    binner = FLIBinner(bin_radius=1)
    b_img, _ = binner.apply_binning(img, irf)

    # Center pixel's 3x3 neighborhood (rows/cols 1..3) all see the hot pixel once.
    expected_nonzero = {(r, c) for r in range(1, 4) for c in range(1, 4)}
    nonzero = {(r, c) for r in range(5) for c in range(5) if b_img[r, c, 0] != 0}
    assert nonzero == expected_nonzero
    assert b_img[2, 2, 0] == 9.0
    assert b_img[1, 1, 0] == 9.0


def test_flibinner_get_binned_data_matches_apply_binning():
    img = np.ones((3, 3, 2), dtype=np.float32)
    irf = np.ones((3, 3, 2), dtype=np.float32)
    binner = FLIBinner(bin_radius=1)

    returned = binner.apply_binning(img, irf)
    stored = binner.get_binned_data()

    np.testing.assert_allclose(returned[0], stored[0])
    np.testing.assert_allclose(returned[1], stored[1])


# --------------------------------------------------------------------------- #
# BinnedFLIFitter
# --------------------------------------------------------------------------- #


def test_binned_fitter_dispatches_to_cpu_process_image():
    proc = FakeCPUProcessor()
    fitter = BinnedFLIFitter(proc, bin_radius=2)
    img = np.zeros((3, 3, 4))
    irf = np.zeros((3, 3, 4))

    dataset = fitter.fit(
        b_img=img, b_irf=irf, estimator="LEAST_SQUARES", data_name="test_ds", n_jobs=4
    )

    assert len(proc.process_image_calls) == 1
    call = proc.process_image_calls[0]
    assert call["estimator"] == "least_squares"
    assert call["n_jobs"] == 4
    assert call["data_name"] == "test_ds"
    assert dataset["name"] == "test_ds_Binned_R2"
    assert dataset["bin_radius"] == 2


def test_binned_fitter_dispatches_to_gpu_fit_image():
    proc = FakeGPUProcessor()
    fitter = BinnedFLIFitter(proc, bin_radius=1)
    img = np.zeros((3, 3, 4))
    irf = np.zeros((3, 3, 4))

    dataset = fitter.fit(
        b_img=img, b_irf=irf, estimator="poisson", data_name="gpu_ds", n_jobs=4
    )

    assert len(proc.fit_image_calls) == 1
    call = proc.fit_image_calls[0]
    assert call["mode"] == "POISSON"
    assert "n_jobs" not in call  # CPU-specific kwarg must be stripped for GPU path
    assert dataset["name"] == "gpu_ds_Binned_R1"
    assert dataset["bin_radius"] == 1


def test_binned_fitter_rejects_unsupported_processor():
    fitter = BinnedFLIFitter(UnsupportedProcessor(), bin_radius=1)
    with pytest.raises(TypeError):
        fitter.fit(b_img=np.zeros((2, 2, 2)), b_irf=np.zeros((2, 2, 2)))


def test_binned_fitter_save_results_delegates_to_processor():
    proc = FakeCPUProcessor()
    fitter = BinnedFLIFitter(proc, bin_radius=1)
    dataset = {"results": {}}

    fitter.save_results(dataset, folder="out")

    assert proc.save_results_calls == [(dataset, "out")]


def test_binned_fitter_save_results_noop_when_dataset_none():
    proc = FakeCPUProcessor()
    fitter = BinnedFLIFitter(proc, bin_radius=1)

    fitter.save_results(None, folder="out")

    assert proc.save_results_calls == []
