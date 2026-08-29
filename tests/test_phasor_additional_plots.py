"""
Tests for pyfli.phasor.phasorS.PhasorAdditionalPlots.

Covers:
  - the class really extends PhasorAnalyzer (inherited API still works, and the
    parent ``lifetime_to_phasor`` conversion is not shadowed by the plot method)
  - the ``_iso_proportion_levels`` / ``_distinct_colors`` static helpers
  - ``phasorlifetime_to_phasor``: 1x2 figure, mask handling, external axes, and
    the core guarantee that a pixel's color in the lifetime map is identical to
    its phasor point's color in the scatter
  - ``phasor_kde_to_px``: 1x2 figure, mask handling, single- vs multi-mode
    KDE colouring, and degenerate inputs

All data is synthetic; the Agg backend is used so nothing renders to screen.
"""

from __future__ import annotations

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from pyfli.phasor.phasorS import PhasorAdditionalPlots, PhasorAnalyzer

FREQ_HZ = 80e6


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _fli_cube(bimodal: bool):
    """Two circular regions in a small FLI cube.

    ``bimodal`` picks whether the two regions get distinct decays (two phasor
    clusters) or the same decay (one phasor cluster).
    """
    rng = np.random.default_rng(7)
    H, W, T = 40, 40, 96
    time_axis_ns = np.linspace(0.0, 12.5, T)
    t = time_axis_ns[None, None, :]
    yy, xx = np.mgrid[0:H, 0:W]
    blob_a = (yy - 12) ** 2 + (xx - 12) ** 2 < 8**2
    blob_b = (yy - 28) ** 2 + (xx - 27) ** 2 < 9**2
    mask = blob_a | blob_b

    if bimodal:
        tau1 = np.where(blob_a, 0.4, 0.6)
        tau2 = np.where(blob_a, 2.5, 3.5)
        a1 = np.where(blob_a, 0.7, 0.35)
    else:
        tau1 = np.full((H, W), 0.5)
        tau2 = np.full((H, W), 3.0)
        a1 = np.full((H, W), 0.5)

    decay = a1[..., None] * np.exp(-t / tau1[..., None]) + (1 - a1)[..., None] * np.exp(
        -t / tau2[..., None]
    )
    decay = decay * np.where(mask, 1.0, 0.0)[..., None]
    decay = np.clip(decay + rng.normal(0, 0.02, decay.shape), 0.0, None)

    irf = np.broadcast_to(
        np.exp(-((time_axis_ns - 0.7) ** 2) / (2 * 0.1**2)), (H, W, T)
    ).copy()

    analyzer = PhasorAdditionalPlots(
        frequency_hz=FREQ_HZ, time_axis_ns=time_axis_ns, n_harmonics=2
    )
    g, s = analyzer.create_phasor_cpu(decay)
    gc, sc = analyzer.calibrate(g, s, irf)
    return SimpleNamespace(
        analyzer=analyzer,
        Gc=gc,
        Sc=sc,
        decay=decay,
        mask=mask,
        blob_a=blob_a,
        blob_b=blob_b,
        shape=(H, W),
    )


@pytest.fixture
def bimodal():
    return _fli_cube(bimodal=True)


@pytest.fixture
def unimodal():
    return _fli_cube(bimodal=False)


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


# ─────────────────────────────────────────────────────────────────────────────
# Class / inheritance
# ─────────────────────────────────────────────────────────────────────────────


class TestInheritance:
    def test_is_phasor_analyzer_subclass(self):
        assert issubclass(PhasorAdditionalPlots, PhasorAnalyzer)

    def test_instance_is_a_phasor_analyzer(self, unimodal):
        assert isinstance(unimodal.analyzer, PhasorAnalyzer)

    def test_inherited_api_available(self, unimodal):
        an = unimodal.analyzer
        for name in (
            "compute_lifetime",
            "calibrate",
            "calibrate_pixelwise",
            "plot_phasor_diagram",
            "lifetime_to_phasor",
        ):
            assert callable(getattr(an, name))

    def test_parent_lifetime_to_phasor_not_shadowed(self, unimodal):
        """The plot method must not override the phasor-coordinate conversion:
        a single-exponential lifetime still maps onto the universal semicircle.
        """
        g, s = unimodal.analyzer.lifetime_to_phasor(np.array([0.5, 2.0, 5.0]), FREQ_HZ)
        dist2 = (np.asarray(g) - 0.5) ** 2 + np.asarray(s) ** 2
        np.testing.assert_allclose(dist2, 0.25, atol=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# Static helpers
# ─────────────────────────────────────────────────────────────────────────────


class TestHelpers:
    def test_iso_levels_ascending_unique_and_in_range(self):
        yy, xx = np.mgrid[-3:3:120j, -3:3:120j]
        dens = np.exp(-(xx**2 + yy**2))
        levels = PhasorAdditionalPlots._iso_proportion_levels(
            dens, np.linspace(0.05, 1.0, 4)
        )
        assert levels.ndim == 1 and levels.size >= 1
        assert np.all(np.diff(levels) > 0)
        assert levels.size == np.unique(levels).size
        assert levels.min() >= dens.min() and levels.max() <= dens.max()

    def test_iso_levels_zero_density_is_empty(self):
        out = PhasorAdditionalPlots._iso_proportion_levels(
            np.zeros((10, 10)), np.array([0.5])
        )
        assert out.size == 0

    def test_distinct_colors_count_and_bounds(self):
        cols = PhasorAdditionalPlots._distinct_colors(7, "tab10")
        assert len(cols) == 7
        arr = np.asarray(cols)
        assert arr.shape == (7, 3)
        assert arr.min() >= 0.0 and arr.max() <= 1.0

    def test_distinct_colors_are_actually_distinct_even_past_cmap(self):
        cols = PhasorAdditionalPlots._distinct_colors(25, "tab10")
        arr = np.round(np.asarray(cols), 4)
        assert len({tuple(row) for row in arr}) == 25

    def test_distinct_colors_zero(self):
        assert PhasorAdditionalPlots._distinct_colors(0, "tab10") == []


# ─────────────────────────────────────────────────────────────────────────────
# phasorlifetime_to_phasor
# ─────────────────────────────────────────────────────────────────────────────


class TestPhasorlifetimeToPhasor:
    def test_returns_figure_with_two_panels(self, bimodal):
        fig = bimodal.analyzer.phasorlifetime_to_phasor(
            bimodal.Gc, bimodal.Sc, boolean_mask=bimodal.mask
        )
        assert isinstance(fig, plt.Figure)
        # map panel + scatter panel (a colorbar axis may also be present)
        assert len(fig.axes) >= 2

    def test_masked_pixels_are_blanked_in_map(self, bimodal):
        fig = bimodal.analyzer.phasorlifetime_to_phasor(
            bimodal.Gc, bimodal.Sc, boolean_mask=bimodal.mask
        )
        arr = np.asarray(fig.axes[0].get_images()[0].get_array(), dtype=float)
        assert np.all(np.isnan(arr[~bimodal.mask]))
        assert np.isfinite(arr[bimodal.mask]).any()

    def test_pixel_and_phasor_point_colors_match_exactly(self, bimodal):
        an, gc, sc, mask = bimodal.analyzer, bimodal.Gc, bimodal.Sc, bimodal.mask
        cmap_scale = (0.0, 5.0)
        fig = an.phasorlifetime_to_phasor(
            gc, sc, boolean_mask=mask, cmap="jet", cmap_scale=cmap_scale
        )
        ax_map, ax_scatter = fig.axes[0], fig.axes[1]

        tau = np.asarray(an.compute_lifetime(gc[0], sc[0]), dtype=float)
        mask_flat = mask.ravel()
        g = np.ravel(gc[0])[mask_flat]
        s = np.ravel(sc[0])[mask_flat]
        tau_m = np.ravel(tau)[mask_flat]
        valid = np.isfinite(g) & np.isfinite(s) & np.isfinite(tau_m)
        full_idx = np.where(mask_flat)[0][valid]
        ys, xs = np.unravel_index(full_idx, mask.shape)

        scatter = ax_scatter.collections[0]
        face = scatter.get_facecolors()
        assert face.shape[0] == ys.size

        im = ax_map.get_images()[0]
        rendered = im.to_rgba(im.get_array())

        # sample a spread of points rather than all of them
        step = max(1, ys.size // 20)
        for k in range(0, ys.size, step):
            np.testing.assert_allclose(face[k], rendered[ys[k], xs[k]], atol=1e-6)

    def test_scatter_colors_follow_the_shared_normalize(self, unimodal):
        an, gc, sc, mask = unimodal.analyzer, unimodal.Gc, unimodal.Sc, unimodal.mask
        fig = an.phasorlifetime_to_phasor(
            gc, sc, boolean_mask=mask, cmap="viridis", cmap_scale=(0.0, 4.0)
        )
        tau = np.asarray(an.compute_lifetime(gc[0], sc[0]), dtype=float)
        mask_flat = mask.ravel()
        g = np.ravel(gc[0])[mask_flat]
        s = np.ravel(sc[0])[mask_flat]
        tau_m = np.ravel(tau)[mask_flat]
        valid = np.isfinite(g) & np.isfinite(s) & np.isfinite(tau_m)
        tau_v = tau_m[valid]

        sm = ScalarMappable(norm=Normalize(0.0, 4.0), cmap=plt.get_cmap("viridis"))
        expected = sm.to_rgba(tau_v)
        face = fig.axes[1].collections[0].get_facecolors()
        np.testing.assert_allclose(face, expected, atol=1e-6)

    def test_accepts_external_axes(self, unimodal):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        out = unimodal.analyzer.phasorlifetime_to_phasor(
            unimodal.Gc, unimodal.Sc, boolean_mask=unimodal.mask, axes=axes
        )
        assert out is fig

    def test_no_mask_uses_every_pixel(self, unimodal):
        fig = unimodal.analyzer.phasorlifetime_to_phasor(unimodal.Gc, unimodal.Sc)
        arr = np.asarray(fig.axes[0].get_images()[0].get_array(), dtype=float)
        # nothing masked out -> only compute_lifetime's own NaNs (few, if any)
        assert np.isfinite(arr).mean() > 0.5

    def test_empty_mask_does_not_crash(self, unimodal):
        fig = unimodal.analyzer.phasorlifetime_to_phasor(
            unimodal.Gc, unimodal.Sc, boolean_mask=np.zeros(unimodal.shape, bool)
        )
        assert isinstance(fig, plt.Figure)

    def test_harmonic_slice_selects_stack_layer(self, bimodal):
        fig = bimodal.analyzer.phasorlifetime_to_phasor(
            bimodal.Gc, bimodal.Sc, boolean_mask=bimodal.mask, harmonic=1
        )
        assert isinstance(fig, plt.Figure)

    def test_2d_input_without_harmonic_axis(self, bimodal):
        fig = bimodal.analyzer.phasorlifetime_to_phasor(
            bimodal.Gc[0], bimodal.Sc[0], boolean_mask=bimodal.mask
        )
        assert isinstance(fig, plt.Figure)


# ─────────────────────────────────────────────────────────────────────────────
# phasor_kde_to_px
# ─────────────────────────────────────────────────────────────────────────────


def _center_indices(fig):
    """Distinct KDE-center indices named in the overlay legend, or empty set."""
    leg_axes = [ax for ax in fig.axes if ax.get_legend() is not None]
    if not leg_axes:
        return set()
    idx = set()
    for txt in leg_axes[0].get_legend().get_texts():
        label = txt.get_text()
        if label.startswith("center "):
            idx.add(int(label.split()[1]))
    return idx


class TestPhasorKdeToPx:
    def test_returns_figure_with_two_panels(self, bimodal):
        fig = bimodal.analyzer.phasor_kde_to_px(
            bimodal.Gc, bimodal.Sc, bimodal.decay, boolean_mask=bimodal.mask
        )
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 2

    def test_overlay_panel_has_image_of_scene_shape(self, bimodal):
        fig = bimodal.analyzer.phasor_kde_to_px(
            bimodal.Gc, bimodal.Sc, bimodal.decay, boolean_mask=bimodal.mask
        )
        overlay = fig.axes[1].get_images()[0].get_array()
        assert overlay.shape[:2] == bimodal.shape
        assert overlay.shape[2] in (3, 4)

    def test_bimodal_cloud_yields_multiple_centers(self, bimodal):
        fig = bimodal.analyzer.phasor_kde_to_px(
            bimodal.Gc,
            bimodal.Sc,
            bimodal.decay,
            boolean_mask=bimodal.mask,
            kde_levels=3,
        )
        assert len(_center_indices(fig)) >= 2

    def test_unimodal_cloud_yields_single_center(self, unimodal):
        fig = unimodal.analyzer.phasor_kde_to_px(
            unimodal.Gc,
            unimodal.Sc,
            unimodal.decay,
            boolean_mask=unimodal.mask,
            kde_levels=3,
        )
        assert _center_indices(fig) == {1}

    def test_overlay_recolors_only_inside_the_mask(self, bimodal):
        an = bimodal.analyzer
        fig = an.phasor_kde_to_px(
            bimodal.Gc, bimodal.Sc, bimodal.decay, boolean_mask=bimodal.mask
        )
        rgb = np.asarray(fig.axes[1].get_images()[0].get_array(), dtype=float)[..., :3]
        # outside the mask the overlay is pure grayscale (R == G == B)
        out = rgb[~bimodal.mask]
        assert np.allclose(out[:, 0], out[:, 1]) and np.allclose(out[:, 1], out[:, 2])
        # inside, at least some pixels have been tinted (channels differ)
        inside = rgb[bimodal.mask]
        assert not np.allclose(inside[:, 0], inside[:, 1])

    def test_fill_alpha_controls_tint_strength(self, bimodal):
        an = bimodal.analyzer
        weak = an.phasor_kde_to_px(
            bimodal.Gc,
            bimodal.Sc,
            bimodal.decay,
            boolean_mask=bimodal.mask,
            fill_alpha=0.0,
        )
        rgb = np.asarray(weak.axes[1].get_images()[0].get_array(), dtype=float)[..., :3]
        inside = rgb[bimodal.mask]
        # alpha 0 -> no tint -> grayscale everywhere
        assert np.allclose(inside[:, 0], inside[:, 1], atol=1e-6)

    def test_accepts_external_axes(self, unimodal):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        out = unimodal.analyzer.phasor_kde_to_px(
            unimodal.Gc,
            unimodal.Sc,
            unimodal.decay,
            boolean_mask=unimodal.mask,
            axes=axes,
        )
        assert out is fig

    def test_tiny_mask_is_handled(self, unimodal):
        tiny = np.zeros(unimodal.shape, bool)
        tiny[0, 0] = tiny[0, 1] = True
        fig = unimodal.analyzer.phasor_kde_to_px(
            unimodal.Gc, unimodal.Sc, unimodal.decay, boolean_mask=tiny
        )
        assert isinstance(fig, plt.Figure)
        assert _center_indices(fig) == set()

    def test_empty_mask_is_handled(self, unimodal):
        fig = unimodal.analyzer.phasor_kde_to_px(
            unimodal.Gc,
            unimodal.Sc,
            unimodal.decay,
            boolean_mask=np.zeros(unimodal.shape, bool),
        )
        assert isinstance(fig, plt.Figure)

    def test_max_centers_caps_the_count(self, bimodal):
        fig = bimodal.analyzer.phasor_kde_to_px(
            bimodal.Gc,
            bimodal.Sc,
            bimodal.decay,
            boolean_mask=bimodal.mask,
            max_centers=1,
        )
        assert _center_indices(fig) <= {1}
