"""Tests for pyfli.analysis.FactorAnalysis."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pyfli.analysis import FactorAnalysis


class FakeSaver:
    def __init__(self):
        self.saved = []

    def save_plot(self, name, fig=None, dpi=300, close=True):
        self.saved.append((name, fig, close))


@pytest.fixture
def fa_inputs():
    rng = np.random.default_rng(0)
    H, W, T = 6, 6, 32
    decay = rng.poisson(20, size=(H, W, T)).astype(float)
    mask = np.ones((H, W), dtype=bool)

    def fake_dataset(bias):
        return {
            "tau1_map": np.full((H, W), 1.0 + bias),
            "chi2_map": np.full((H, W), 1.0),
        }

    def fake_fitset(bias):
        fit_map = decay * (1 + bias)
        return {"fit_map": fit_map, "residual_map": decay - fit_map}

    all_datasets = [fake_dataset(0.0), fake_dataset(0.1)]
    all_fitset = [fake_fitset(0.0), fake_fitset(0.05)]
    method_names = ["method_a", "method_b"]
    return decay, mask, all_datasets, all_fitset, method_names


def test_factor_analysis_importable_from_subpackage():
    assert callable(FactorAnalysis)


def test_default_factors_registered(fa_inputs):
    decay, mask, all_datasets, all_fitset, method_names = fa_inputs
    fa = FactorAnalysis(decay, None, mask, all_datasets, all_fitset, method_names)
    assert "total_photons" in fa.list_factors()
    assert "peak_counts" in fa.list_factors()
    assert "peak_to_mean_ratio" in fa.list_factors()
    np.testing.assert_allclose(
        fa.factor_values("peak_to_mean_ratio"),
        decay.max(axis=-1) / (decay.mean(axis=-1) + 1e-8),
    )


def test_mismatched_lengths_raise(fa_inputs):
    decay, mask, all_datasets, all_fitset, method_names = fa_inputs
    with pytest.raises(ValueError):
        FactorAnalysis(decay, None, mask, all_datasets, all_fitset, method_names[:1])


def test_missing_fitset_key_raises(fa_inputs):
    decay, mask, all_datasets, all_fitset, method_names = fa_inputs
    bad_fitset = [{"fit_map": all_fitset[0]["fit_map"]}, all_fitset[1]]
    with pytest.raises(ValueError):
        FactorAnalysis(decay, None, mask, all_datasets, bad_fitset, method_names)


def test_analyze_and_plot_datasets(fa_inputs):
    decay, mask, all_datasets, all_fitset, method_names = fa_inputs
    fa = FactorAnalysis(decay, None, mask, all_datasets, all_fitset, method_names)
    df, edges = fa.analyze(
        factor_key="total_photons", target_keys=["tau1_map"], n_bins=3
    )
    assert not df.empty
    assert set(df["method"]) == set(method_names)
    fig, axes = fa.plot(df)
    assert fig is not None
    plt.close(fig)


def test_analyze_fitset_targets(fa_inputs):
    decay, mask, all_datasets, all_fitset, method_names = fa_inputs
    fa = FactorAnalysis(decay, None, mask, all_datasets, all_fitset, method_names)
    df, edges = fa.analyze(
        factor_key="total_photons",
        target_source="fitset",
        target_keys=["residual_chi2"],
        n_bins=3,
    )
    assert not df.empty


def test_plot_with_saver_uses_default_name(fa_inputs):
    decay, mask, all_datasets, all_fitset, method_names = fa_inputs
    fa = FactorAnalysis(decay, None, mask, all_datasets, all_fitset, method_names)
    df, edges = fa.analyze(
        factor_key="total_photons", target_keys=["tau1_map"], n_bins=3
    )
    saver = FakeSaver()
    fig, axes = fa.plot(df, saver=saver)
    plt.close(fig)

    assert len(saver.saved) == 1
    name, saved_fig, close = saver.saved[0]
    assert name == "factor_analysis_total_photons_line"
    assert saved_fig is fig
    assert close is False


def test_plot_with_saver_and_explicit_name(fa_inputs):
    decay, mask, all_datasets, all_fitset, method_names = fa_inputs
    fa = FactorAnalysis(decay, None, mask, all_datasets, all_fitset, method_names)
    df, edges = fa.analyze(
        factor_key="total_photons", target_keys=["tau1_map"], n_bins=3
    )
    saver = FakeSaver()
    fig, axes = fa.plot(df, kind="box", saver=saver, name="my_custom_name")
    plt.close(fig)

    assert saver.saved == [("my_custom_name", fig, False)]


def test_plot_without_saver_does_not_save(fa_inputs):
    decay, mask, all_datasets, all_fitset, method_names = fa_inputs
    fa = FactorAnalysis(decay, None, mask, all_datasets, all_fitset, method_names)
    df, edges = fa.analyze(
        factor_key="total_photons", target_keys=["tau1_map"], n_bins=3
    )
    fig, axes = fa.plot(df)  # no saver passed -- must not raise
    plt.close(fig)


def test_recon_edge_nogap_emptybin(fa_inputs):
    decay, mask, all_datasets, all_fitset, method_names = fa_inputs
    H, W = mask.shape
    custom_factor = np.arange(H * W, dtype=float).reshape(H, W)
    n_bins = 4

    fa = FactorAnalysis(decay, None, mask, all_datasets, all_fitset, method_names)
    fa.add_factor("custom_linear", custom_factor)
    df, edges = fa.analyze(
        factor_key="custom_linear",
        target_keys=["tau1_map"],
        n_bins=n_bins,
        bin_mode="linear",
    )
    assert len(edges) == n_bins + 1

    target_bin = n_bins // 2
    lo, hi = edges[target_bin], edges[target_bin + 1]
    knockout = (custom_factor >= lo) & (custom_factor <= hi)
    all_datasets[0]["tau1_map"] = np.where(
        knockout, np.nan, all_datasets[0]["tau1_map"]
    )

    fa2 = FactorAnalysis(decay, None, mask, all_datasets, all_fitset, method_names)
    fa2.add_factor("custom_linear", custom_factor)
    df2, edges2 = fa2.analyze(
        factor_key="custom_linear",
        target_keys=["tau1_map"],
        n_bins=n_bins,
        bin_mode="linear",
    )

    method = method_names[0]
    present_bins = set(
        df2[(df2["method"] == method) & (df2["parameter"] == "tau1_map")]["bin_rank"]
    )
    assert target_bin not in present_bins

    reconstructed = fa2._reconstruct_edges(df2, method)
    assert len(reconstructed) == n_bins + 1
    np.testing.assert_allclose(reconstructed, edges2)

    fig, axes = fa2.plot(df2, target_keys=["tau1_map"], kind="box")
    plt.close(fig)


def test_plot_range_selection_grid_with_saver(fa_inputs):
    decay, mask, all_datasets, all_fitset, method_names = fa_inputs
    fa = FactorAnalysis(decay, None, mask, all_datasets, all_fitset, method_names)
    lo, hi = np.quantile(fa.factor_values("total_photons")[mask], [0.2, 0.9])
    saver = FakeSaver()

    fig, axes = fa.plot_range_selection_grid(
        "total_photons", (lo, hi), target_keys=["tau1_map"], saver=saver
    )
    plt.close(fig)

    assert len(saver.saved) == 1
    name, saved_fig, close = saver.saved[0]
    assert name == "range_selection_grid_total_photons"
    assert saved_fig is fig
    assert close is False
