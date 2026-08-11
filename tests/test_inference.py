"""
Tests for pyfli.bayes_utils.inference.BiPipeline.

No real Keras checkpoint/model is used -- `loaded_model` is a fake object with
a `.sample()` method returning deterministic per-pixel values (constant across
posterior samples, so median == value and MAD == 0 exactly), which lets patch
stitching, masking, and output_samples construction be checked exactly without
any file I/O or TensorFlow/Keras runtime cost.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

# inference.py imports keras at module level (the "tf" extra) even though
# none of these tests touch a real Keras model -- skip cleanly rather than
# erroring collection when it isn't installed.
pytest.importorskip("keras")

from pyfli.bayes_utils.inference import BiPipeline

_KEY_BASE = {"tau1": 0.8, "tau2": 2.5, "alpha1": 0.4, "tau": 1.2}
_KEY_STEP = {"tau1": 0.05, "tau2": 0.05, "alpha1": 0.02, "tau": 0.05}


class _FakeModel:
    """Stand-in for a loaded BayesFlow/Keras model's `.sample()` API."""

    def __init__(self, keys):
        self.keys = keys
        self.calls = []

    def sample(self, conditions, num_samples, batch_size):
        self.calls.append(
            {
                "conditions": conditions,
                "num_samples": num_samples,
                "batch_size": batch_size,
            }
        )
        n_pixels = conditions["decay"].shape[0]
        out = {}
        for key in self.keys:
            base, step = _KEY_BASE[key], _KEY_STEP[key]
            vals = (base + step * np.arange(n_pixels)).astype(np.float32)
            # Constant across samples -> median == vals, MAD == 0 exactly.
            samples = np.broadcast_to(
                vals[:, None, None], (n_pixels, num_samples, 1)
            ).copy()
            out[key] = samples
        return out


class _FakeSaver:
    def __init__(self, save_dir="."):
        self.save_dir = save_dir
        self.saved = []
        self.logs = []

    def save_npy(self, name, data):
        self.saved.append((name, data))

    def log(self, msg):
        self.logs.append(msg)


def _value_for(key, flat_idx):
    return _KEY_BASE[key] + _KEY_STEP[key] * flat_idx


@pytest.fixture
def decay_irf(n_bins):
    H, W = 4, 4
    decay = (
        np.random.default_rng(0).poisson(5.0, size=(H, W, n_bins)).astype(np.float32)
    )
    # Shared (N_BINS,) IRF -- same convention ParameterToDecayReconstruction/
    # compute_detailed_results accept.
    irf = np.ones(n_bins, dtype=np.float32) / n_bins
    return decay, irf


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------


def test_rejects_unknown_model_type():
    with pytest.raises(ValueError, match="MODEL_TYPE"):
        BiPipeline("tri-exponential", model_weights="whatever.keras")


@pytest.mark.parametrize("bad_weights", [None, ""])
def test_rejects_missing_model_weights(bad_weights):
    with pytest.raises(ValueError, match="model_weights"):
        BiPipeline("bi-exponential", model_weights=bad_weights)


def test_keys_match_model_type():
    pipeline = BiPipeline("bi-exponential", model_weights="whatever.keras")
    assert pipeline.keys == ["tau1", "tau2", "alpha1"]
    pipeline_mono = BiPipeline("mono-exponential", model_weights="whatever.keras")
    assert pipeline_mono.keys == ["tau"]


def test_load_model_missing_checkpoint_raises(tmp_path):
    pipeline = BiPipeline(
        "bi-exponential", model_weights=str(tmp_path / "missing.keras")
    )
    with pytest.raises(FileNotFoundError):
        pipeline.load_model()


# ---------------------------------------------------------------------------
# run_inference: single patch covering the whole image
# ---------------------------------------------------------------------------


def test_run_inference_single_patch_fully_masked_in(decay_irf):
    decay, irf = decay_irf
    H, W = decay.shape[:2]
    pipeline = BiPipeline(
        "bi-exponential",
        model_weights="whatever.keras",
        patch_size=(H, W),
        num_samples=4,
    )
    pipeline.loaded_model = _FakeModel(pipeline.keys)

    mask = np.ones((H, W), dtype=bool)
    output_maps, output_uncertainties, output_samples = pipeline.run_inference(
        decay, irf, mask
    )

    assert len(pipeline.loaded_model.calls) == 1
    for key in pipeline.keys:
        expected = (_KEY_BASE[key] + _KEY_STEP[key] * np.arange(H * W)).reshape(H, W)
        np.testing.assert_allclose(output_maps[key], expected, rtol=1e-6)
        np.testing.assert_allclose(
            output_uncertainties[key], np.zeros((H, W)), atol=1e-9
        )
        assert output_samples[key].shape == (H, W, 4)
        for s in range(4):
            np.testing.assert_allclose(
                output_samples[key][:, :, s], expected, rtol=1e-6
            )


def test_run_inference_rejects_mismatched_mask_shape(decay_irf):
    decay, irf = decay_irf
    pipeline = BiPipeline(
        "bi-exponential", model_weights="whatever.keras", patch_size=(4, 4)
    )
    pipeline.loaded_model = _FakeModel(pipeline.keys)
    with pytest.raises(ValueError, match="mask shape"):
        pipeline.run_inference(decay, irf, np.ones((3, 3), dtype=bool))


def test_run_inference_rejects_bad_irf_ndim(decay_irf):
    decay, irf = decay_irf
    H, W = decay.shape[:2]
    pipeline = BiPipeline(
        "bi-exponential", model_weights="whatever.keras", patch_size=(H, W)
    )
    pipeline.loaded_model = _FakeModel(pipeline.keys)
    bad_irf = irf[:, None]  # 2-D, neither shared nor per-pixel
    with pytest.raises(ValueError, match="irf must be"):
        pipeline.run_inference(decay, bad_irf, np.ones((H, W), dtype=bool))


def test_run_inference_shared_1d_irf_matches_equivalent_per_pixel_cube(decay_irf):
    """A shared (N_BINS,) IRF and a (H, W, N_BINS) cube broadcasting the same
    trace to every pixel must produce identical patch_irf batches, and hence
    identical outputs -- both paths go through the same model.sample() call."""
    decay, irf_1d = decay_irf
    H, W = decay.shape[:2]
    irf_cube = np.broadcast_to(irf_1d, (H, W, irf_1d.shape[-1])).copy()

    pipeline_1d = BiPipeline(
        "bi-exponential",
        model_weights="whatever.keras",
        patch_size=(2, 2),
        num_samples=3,
    )
    pipeline_1d.loaded_model = _FakeModel(pipeline_1d.keys)
    pipeline_cube = BiPipeline(
        "bi-exponential",
        model_weights="whatever.keras",
        patch_size=(2, 2),
        num_samples=3,
    )
    pipeline_cube.loaded_model = _FakeModel(pipeline_cube.keys)

    mask = np.ones((H, W), dtype=bool)
    maps_1d, unc_1d, samples_1d = pipeline_1d.run_inference(decay, irf_1d, mask)
    maps_cube, unc_cube, samples_cube = pipeline_cube.run_inference(
        decay, irf_cube, mask
    )

    for key in pipeline_1d.keys:
        np.testing.assert_array_equal(maps_1d[key], maps_cube[key])
        np.testing.assert_array_equal(unc_1d[key], unc_cube[key])
        np.testing.assert_array_equal(samples_1d[key], samples_cube[key])

    # And every recorded patch_irf batch really was the broadcast IRF trace.
    for call in pipeline_1d.loaded_model.calls:
        patch_irf = call["conditions"]["irf"]
        for row in patch_irf:
            np.testing.assert_array_equal(row, irf_1d)


# ---------------------------------------------------------------------------
# run_inference: multi-patch stitching, skip-fully-masked, post-hoc masking
# ---------------------------------------------------------------------------


def test_run_inference_multi_patch_stitching_and_masking(decay_irf):
    decay, irf = decay_irf
    H, W = decay.shape[:2]  # 4x4
    pipeline = BiPipeline(
        "bi-exponential",
        model_weights="whatever.keras",
        patch_size=(2, 2),
        num_samples=3,
    )
    pipeline.loaded_model = _FakeModel(pipeline.keys)

    # Patch (0,0) [rows0-1,cols0-1] fully masked out -> skipped.
    # Patch (1,0) [rows2-3,cols0-1] partially masked -> (2,1) excluded post-hoc.
    # Patches (0,1) and (1,1) fully valid.
    mask = np.array(
        [
            [False, False, True, True],
            [False, False, True, True],
            [True, False, True, True],
            [True, True, True, True],
        ]
    )

    output_maps, output_uncertainties, output_samples = pipeline.run_inference(
        decay, irf, mask
    )

    # 4 total patches, 1 fully-masked-out (skipped) -> 3 model calls.
    assert len(pipeline.loaded_model.calls) == 3
    for call in pipeline.loaded_model.calls:
        assert call["num_samples"] == 3
        assert call["conditions"]["decay"].shape == (4, decay.shape[-1])

    key = "tau1"
    expected = np.zeros((H, W), dtype=np.float32)
    # patch (0,1): rows0-1,cols2-3, local row-major flat idx 0..3
    expected[0, 2], expected[0, 3] = _value_for(key, 0), _value_for(key, 1)
    expected[1, 2], expected[1, 3] = _value_for(key, 2), _value_for(key, 3)
    # patch (1,0): rows2-3,cols0-1 -- local idx1 (global (2,1)) is masked out -> 0
    expected[2, 0] = _value_for(key, 0)
    expected[2, 1] = 0.0
    expected[3, 0] = _value_for(key, 2)
    expected[3, 1] = _value_for(key, 3)
    # patch (1,1): rows2-3,cols2-3, fully valid
    expected[2, 2], expected[2, 3] = _value_for(key, 0), _value_for(key, 1)
    expected[3, 2], expected[3, 3] = _value_for(key, 2), _value_for(key, 3)
    # patch (0,0): fully skipped -> stays 0 (already the case in `expected`)

    np.testing.assert_allclose(output_maps[key], expected, rtol=1e-6)
    np.testing.assert_allclose(output_uncertainties[key], np.zeros((H, W)), atol=1e-9)

    for s in range(3):
        np.testing.assert_allclose(output_samples[key][:, :, s], expected, rtol=1e-6)

    # The masked-out pixel and the skipped patch are indistinguishable by
    # value alone (both 0.0) -- this is the gap BestParamFitSelector's
    # bool_mask passthrough exists to address downstream.
    assert output_maps[key][0, 0] == 0.0
    assert output_maps[key][2, 1] == 0.0


def test_run_inference_mono_exponential(decay_irf):
    decay, irf = decay_irf
    H, W = decay.shape[:2]
    pipeline = BiPipeline(
        "mono-exponential",
        model_weights="whatever.keras",
        patch_size=(H, W),
        num_samples=2,
    )
    pipeline.loaded_model = _FakeModel(pipeline.keys)
    output_maps, output_uncertainties, output_samples = pipeline.run_inference(
        decay, irf, np.ones((H, W), dtype=bool)
    )
    assert set(output_maps) == {"tau"}
    assert output_samples["tau"].shape == (H, W, 2)


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def test_save_outputs_calls_saver_with_samples(decay_irf):
    decay, irf = decay_irf
    H, W = decay.shape[:2]
    pipeline = BiPipeline(
        "bi-exponential", model_weights="whatever.keras", patch_size=(H, W)
    )
    pipeline.loaded_model = _FakeModel(pipeline.keys)
    output_maps, output_uncertainties, output_samples = pipeline.run_inference(
        decay, irf, np.ones((H, W), dtype=bool)
    )

    saver = _FakeSaver()
    pipeline.save_outputs(saver, output_maps, output_uncertainties, output_samples)

    names = [name for name, _ in saver.saved]
    assert "Bi Direct_Output_bi-exponential" in names
    assert "Bi output_uncertainties_bi-exponential" in names
    assert "Bi output_samples_bi-exponential" in names


def test_save_outputs_without_samples_skips_that_save(decay_irf):
    decay, irf = decay_irf
    H, W = decay.shape[:2]
    pipeline = BiPipeline("bi-exponential", model_weights="whatever.keras")
    saver = _FakeSaver()
    pipeline.save_outputs(
        saver,
        {k: np.zeros((H, W)) for k in pipeline.keys},
        {k: np.zeros((H, W)) for k in pipeline.keys},
    )
    names = [name for name, _ in saver.saved]
    assert not any("samples" in n for n in names)


def test_save_detailed(decay_irf):
    pipeline = BiPipeline("bi-exponential", model_weights="whatever.keras")
    saver = _FakeSaver()
    pipeline.save_detailed(saver, {"fake": "result"})
    assert saver.saved[0][0] == "Bi bi_Output"


# ---------------------------------------------------------------------------
# compute_detailed (real pyfli.analysis.utils.compute_detailed_results)
# ---------------------------------------------------------------------------


def test_compute_detailed_bi_exponential(decay_irf):
    decay, irf = decay_irf
    H, W = decay.shape[:2]
    pipeline = BiPipeline(
        "bi-exponential", model_weights="whatever.keras", patch_size=(H, W)
    )
    pipeline.loaded_model = _FakeModel(pipeline.keys)
    output_maps, _, _ = pipeline.run_inference(decay, irf, np.ones((H, W), dtype=bool))

    detailed = pipeline.compute_detailed(output_maps, freq=80.0, decay=decay, irf=irf)
    assert detailed["name"] == "Bi_bi"
    assert "tau1_map" in detailed["results"]["maps"]
    assert "fit_map" in detailed["results"]["TR_maps"]


def test_compute_detailed_mono_exponential(decay_irf):
    decay, irf = decay_irf
    H, W = decay.shape[:2]
    pipeline = BiPipeline(
        "mono-exponential", model_weights="whatever.keras", patch_size=(H, W)
    )
    pipeline.loaded_model = _FakeModel(pipeline.keys)
    output_maps, _, _ = pipeline.run_inference(decay, irf, np.ones((H, W), dtype=bool))

    detailed = pipeline.compute_detailed(output_maps, freq=80.0, decay=decay, irf=irf)
    assert detailed["name"] == "Bi_mono"


# ---------------------------------------------------------------------------
# Orchestration (run())
# ---------------------------------------------------------------------------


def test_run_full_pipeline(decay_irf, tmp_path):
    decay, irf = decay_irf
    H, W = decay.shape[:2]
    pipeline = BiPipeline(
        "bi-exponential", model_weights="whatever.keras", patch_size=(H, W)
    )
    pipeline.loaded_model = _FakeModel(pipeline.keys)
    saver = _FakeSaver(save_dir=str(tmp_path))

    results = pipeline.run(
        decay=decay,
        irf=irf,
        mask=np.ones((H, W), dtype=bool),
        freq=80.0,
        saver=saver,
        save_outputs=True,
        compute_detailed=True,
        visualize=True,
    )

    assert set(results) == {
        "output_maps",
        "output_uncertainties",
        "output_samples",
        "detailed",
    }
    assert results["detailed"]["name"] == "Bi_bi"
    saved_names = [name for name, _ in saver.saved]
    assert "Bi Direct_Output_bi-exponential" in saved_names
    assert "Bi output_samples_bi-exponential" in saved_names
    assert "Bi bi_Output" in saved_names


def test_run_save_outputs_without_saver_raises(decay_irf):
    decay, irf = decay_irf
    H, W = decay.shape[:2]
    pipeline = BiPipeline(
        "bi-exponential", model_weights="whatever.keras", patch_size=(H, W)
    )
    pipeline.loaded_model = _FakeModel(pipeline.keys)
    with pytest.raises(ValueError, match="saver"):
        pipeline.run(
            decay=decay,
            irf=irf,
            mask=np.ones((H, W), dtype=bool),
            freq=80.0,
            saver=None,
            save_outputs=True,
            compute_detailed=False,
        )


def test_run_skips_detailed_and_save_when_disabled(decay_irf):
    decay, irf = decay_irf
    H, W = decay.shape[:2]
    pipeline = BiPipeline(
        "bi-exponential", model_weights="whatever.keras", patch_size=(H, W)
    )
    pipeline.loaded_model = _FakeModel(pipeline.keys)
    results = pipeline.run(
        decay=decay,
        irf=irf,
        mask=np.ones((H, W), dtype=bool),
        freq=80.0,
        saver=None,
        save_outputs=False,
        compute_detailed=False,
    )
    assert results["detailed"] is None
