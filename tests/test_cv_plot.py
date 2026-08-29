"""Tests for pyfli.data_vnp.CVPlot."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from pyfli.data_vnp import CVPlot


@pytest.fixture
def shot_noise_maps():
    """Synthetic (H, W) maps where tau's per-pixel noise scales as 1/sqrt(photon
    count), i.e. the ideal shot-noise-limited regime -- so cv = std/mean should
    decrease monotonically as photon count increases, giving a real, checkable
    relationship (not just "did it run")."""
    rng = np.random.default_rng(0)
    H, W = 40, 40
    tau_true = 1.2
    photon_map = rng.uniform(100, 8000, size=(H, W))
    noise_scale = tau_true / np.sqrt(photon_map)
    tau_map = tau_true + rng.normal(0, 1, size=(H, W)) * noise_scale
    tau2_map = 2.0 * tau_true + rng.normal(0, 1, size=(H, W)) * (2 * noise_scale)
    maps = {
        "tau_map": tau_map,
        "tau1_map": tau_map,
        "tau2_map": tau2_map,
        "photon_count_map": photon_map,
    }
    mask = np.ones((H, W), dtype=bool)
    cluster_mask = np.zeros((H, W), dtype=int)
    cluster_mask[: H // 2, :] = 1
    cluster_mask[H // 2 :, :] = 2
    return maps, mask, cluster_mask


def test_cv_plot_importable_from_subpackage():
    assert callable(CVPlot)


# ------------------------------------------------------------------ #
# compute()
# ------------------------------------------------------------------ #


def test_compute_pooled_basic_columns(shot_noise_maps):
    maps, mask, _ = shot_noise_maps
    df = CVPlot().compute(maps, tau_keys="tau_map", mask=mask, n_bins=8)
    assert not df.empty
    for col in ("cluster", "parameter", "bin_center", "mean", "std", "cv", "count"):
        assert col in df.columns
    assert (df["cluster"].isna()).all()
    assert (df["parameter"] == "tau_map").all()


def test_compute_cv_decreases_with_photon_count(shot_noise_maps):
    """The core physical claim this class exists to check: for shot-noise-limited
    data, cv should trend downward as photon count increases."""
    maps, mask, _ = shot_noise_maps
    df = CVPlot().compute(maps, tau_keys="tau_map", mask=mask, n_bins=8)
    df = df.sort_values("bin_center")
    # allow a little noise in the trend, but the last bin must be clearly below the
    # first -- this is the N**-0.5 scaling, not a flat/noisy line
    assert df["cv"].iloc[-1] < 0.5 * df["cv"].iloc[0]
    # overall trend should be (near-)monotonic: at most one increase out of 7 steps
    diffs = np.diff(df["cv"].to_numpy())
    assert (diffs > 0).sum() <= 1


def test_compute_multiple_tau_keys(shot_noise_maps):
    maps, mask, _ = shot_noise_maps
    df = CVPlot().compute(maps, tau_keys=["tau1_map", "tau2_map"], mask=mask, n_bins=5)
    assert set(df["parameter"].unique()) == {"tau1_map", "tau2_map"}


def test_compute_photon_map_overrides_photon_key(shot_noise_maps):
    maps, mask, _ = shot_noise_maps
    # a photon_map that disagrees with maps['photon_count_map'] should be the one
    # actually used for binning -- a clearly different, non-overlapping value range
    rng = np.random.default_rng(1)
    alt_photon_map = rng.uniform(2_000_000, 3_000_000, size=maps["tau_map"].shape)
    df = CVPlot().compute(
        maps, tau_keys="tau_map", photon_map=alt_photon_map, mask=mask, n_bins=4
    )
    assert df.attrs["photon_label"] == "photon_map"
    assert df["bin_center"].min() >= 2_000_000
    assert df["bin_center"].max() <= 3_000_000


def test_compute_missing_tau_key_raises(shot_noise_maps):
    maps, mask, _ = shot_noise_maps
    with pytest.raises(KeyError):
        CVPlot().compute(maps, tau_keys="does_not_exist_map", mask=mask)


def test_compute_missing_photon_key_raises(shot_noise_maps):
    maps, mask, _ = shot_noise_maps
    with pytest.raises(KeyError):
        CVPlot().compute(
            maps, tau_keys="tau_map", photon_key="does_not_exist_map", mask=mask
        )


def test_compute_tau_shape_mismatch_raises(shot_noise_maps):
    maps, mask, _ = shot_noise_maps
    bad_maps = dict(maps)
    bad_maps["tau_map"] = maps["tau_map"][:-1, :]
    with pytest.raises(ValueError):
        CVPlot().compute(bad_maps, tau_keys="tau_map", mask=mask)


def test_compute_mask_shape_mismatch_raises(shot_noise_maps):
    maps, _, _ = shot_noise_maps
    bad_mask = np.ones((3, 3), dtype=bool)
    with pytest.raises(ValueError):
        CVPlot().compute(maps, tau_keys="tau_map", mask=bad_mask)


def test_compute_cluster_mask_all_zero_raises(shot_noise_maps):
    maps, mask, cluster_mask = shot_noise_maps
    with pytest.raises(ValueError):
        CVPlot().compute(
            maps,
            tau_keys="tau_map",
            mask=mask,
            cluster_mask=np.zeros_like(cluster_mask),
        )


def test_compute_cluster_names_missing_label_raises(shot_noise_maps):
    maps, mask, cluster_mask = shot_noise_maps
    with pytest.raises(ValueError):
        CVPlot().compute(
            maps,
            tau_keys="tau_map",
            mask=mask,
            cluster_mask=cluster_mask,
            cluster_names={1: "A"},  # missing label 2
        )


def test_compute_invalid_bin_mode_raises(shot_noise_maps):
    maps, mask, _ = shot_noise_maps
    with pytest.raises(ValueError):
        CVPlot().compute(maps, tau_keys="tau_map", mask=mask, bin_mode="bogus")


def test_compute_invalid_bin_scope_raises(shot_noise_maps):
    maps, mask, _ = shot_noise_maps
    with pytest.raises(ValueError):
        CVPlot().compute(maps, tau_keys="tau_map", mask=mask, bin_scope="bogus")


def test_compute_cluster_default_names(shot_noise_maps):
    maps, mask, cluster_mask = shot_noise_maps
    df = CVPlot().compute(
        maps, tau_keys="tau_map", mask=mask, cluster_mask=cluster_mask, n_bins=4
    )
    assert set(df["cluster"].unique()) == {"cluster_1", "cluster_2"}
    assert df.attrs["has_cluster"] is True


def test_compute_bin_scope_pooled_shares_edges_across_clusters(shot_noise_maps):
    maps, mask, cluster_mask = shot_noise_maps
    df = CVPlot().compute(
        maps,
        tau_keys="tau_map",
        mask=mask,
        cluster_mask=cluster_mask,
        cluster_names={1: "A", 2: "B"},
        n_bins=4,
        bin_scope="pooled",
    )
    edges_a = df[df["cluster"] == "A"].sort_values("bin_rank")["bin_low"].to_numpy()
    edges_b = df[df["cluster"] == "B"].sort_values("bin_rank")["bin_low"].to_numpy()
    np.testing.assert_allclose(edges_a, edges_b)


# ------------------------------------------------------------------ #
# plot() / compute_and_plot()
# ------------------------------------------------------------------ #


def test_plot_pooled_multiple_targets(shot_noise_maps):
    maps, mask, _ = shot_noise_maps
    df = CVPlot().compute(maps, tau_keys=["tau1_map", "tau2_map"], mask=mask, n_bins=5)
    fig, axes = CVPlot().plot(df)
    assert fig is not None
    assert len(axes) == 1
    plt.close(fig)


def test_plot_cluster_grid(shot_noise_maps):
    maps, mask, cluster_mask = shot_noise_maps
    df = CVPlot().compute(
        maps,
        tau_keys=["tau1_map", "tau2_map"],
        mask=mask,
        cluster_mask=cluster_mask,
        cluster_names={1: "A", 2: "B"},
        n_bins=5,
    )
    fig, axes = CVPlot().plot(df, ncols=2)
    assert fig is not None
    assert axes.shape == (1, 2)
    plt.close(fig)


# ------------------------------------------------------------------ #
# ideal trend / power-law fit overlays
# ------------------------------------------------------------------ #


def test_ideal_trend_fit_recovers_known_amplitude():
    x = np.array([100.0, 400.0, 900.0, 1600.0, 2500.0])
    c_true = 2.0
    y = c_true / np.sqrt(x)  # noiseless, exact 1/sqrt(N) data
    c, y_pred = CVPlot._ideal_trend_fit(x, y)
    assert c == pytest.approx(c_true, rel=1e-9)
    np.testing.assert_allclose(y_pred, y, rtol=1e-9)


def test_ideal_trend_fit_too_few_points_returns_none():
    c, y_pred = CVPlot._ideal_trend_fit(np.array([]), np.array([]))
    assert c is None
    assert y_pred is None


def test_power_law_fit_recovers_known_parameters():
    x = np.array([100.0, 400.0, 900.0, 1600.0, 2500.0, 4000.0])
    a_true, b_true = 3.0, -0.5
    y = a_true * x**b_true  # noiseless power law
    a, b, r2, y_pred = CVPlot._power_law_fit(x, y)
    assert a == pytest.approx(a_true, rel=1e-6)
    assert b == pytest.approx(b_true, rel=1e-6)
    assert r2 == pytest.approx(1.0, abs=1e-9)
    np.testing.assert_allclose(y_pred, y, rtol=1e-6)


def test_power_law_fit_too_few_points_returns_none():
    a, b, r2, y_pred = CVPlot._power_law_fit(np.array([1.0]), np.array([1.0]))
    assert a is None and b is None and r2 is None and y_pred is None


def test_plot_show_ideal_trend_adds_dotted_reference_line(shot_noise_maps):
    maps, mask, _ = shot_noise_maps
    df = CVPlot().compute(maps, tau_keys="tau_map", mask=mask, n_bins=8)
    fig, axes = CVPlot().plot(df, show_ideal_trend=True)
    labels = [line.get_label() for line in axes[0].get_lines()]
    assert any("ideal" in lbl and "sqrt" in lbl for lbl in labels)
    plt.close(fig)


def test_plot_show_powerlaw_fit_adds_dashed_fit_line(shot_noise_maps):
    maps, mask, _ = shot_noise_maps
    df = CVPlot().compute(maps, tau_keys="tau_map", mask=mask, n_bins=8)
    fig, axes = CVPlot().plot(df, show_powerlaw_fit=True)
    labels = [line.get_label() for line in axes[0].get_lines()]
    assert any("fit:" in lbl and "R^2" in lbl for lbl in labels)
    plt.close(fig)


def test_plot_reference_overlays_off_by_default(shot_noise_maps):
    maps, mask, _ = shot_noise_maps
    df = CVPlot().compute(maps, tau_keys="tau_map", mask=mask, n_bins=8)
    fig, axes = CVPlot().plot(df)
    # exactly one line (the raw data) -- no reference/fit overlays unless asked for
    assert len(axes[0].get_lines()) == 1
    plt.close(fig)


def test_plot_cluster_grid_with_both_overlays(shot_noise_maps):
    maps, mask, cluster_mask = shot_noise_maps
    df = CVPlot().compute(
        maps,
        tau_keys="tau1_map",
        mask=mask,
        cluster_mask=cluster_mask,
        cluster_names={1: "A", 2: "B"},
        n_bins=8,
    )
    fig, axes = CVPlot().plot(df, show_ideal_trend=True, show_powerlaw_fit=True)
    # 2 clusters x (data + ideal + fit) = 6 lines on the single subplot
    assert len(axes[0, 0].get_lines()) == 6
    plt.close(fig)


def test_plot_empty_df_raises():
    with pytest.raises(ValueError):
        CVPlot().plot(pd.DataFrame())


def test_compute_and_plot_wrapper(shot_noise_maps):
    maps, mask, _ = shot_noise_maps
    df, fig, axes = CVPlot().compute_and_plot(maps, tau_keys="tau_map", mask=mask)
    assert not df.empty
    assert fig is not None
    plt.close(fig)


def test_save_path_writes_png(tmp_path, shot_noise_maps):
    maps, mask, _ = shot_noise_maps
    cv = CVPlot(save_path=str(tmp_path), fig_name="cv_test")
    df = cv.compute(maps, tau_keys="tau_map", mask=mask, n_bins=5)
    fig, _ = cv.plot(df)
    assert (tmp_path / "cv_test.png").exists()
    plt.close(fig)
