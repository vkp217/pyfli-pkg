"""
Plot a single pixel's posterior-sample decay reconstructions against its
measured decay, in the style of a posterior-predictive check: shaded credible-
interval bands plus a chosen central curve (best-fitting sample, median, or
mean), overlaid on the actual measured decay.

Belongs to :mod:`pyfli.bayes_utils`, downstream of
:class:`pyfli.bayes_utils.param_combinations.BestParamFitSelector` and
:class:`pyfli.reconstruction.ParameterToDecayReconstruction`.
"""

import numpy as np
import matplotlib.pyplot as plt

from pyfli.bayes_utils.param_combinations import BestParamFitSelector
from pyfli.reconstruction import ParameterToDecayReconstruction

#: Registry of {model_type: output_combination keys}, matching
#: BestParamFitSelector's own params dicts (no "_map" suffix).
_MODEL_PARAM_KEYS: dict[str, tuple[str, ...]] = {
    "bi-exponential": ("alpha1", "tau1", "tau2"),
    "mono-exponential": ("tau",),
}

#: Valid values for the `center` argument.
CENTERS: tuple[str, ...] = ("best", "median", "mean")

# House colors, matching pyfli.data_vnp.data_viewer.DataViewer._SERIES_COLORS
# (slot 1 = decay, slot 3 = fit/model) so this plot reads consistently with
# the rest of the package's decay-viewer figures.
_DECAY_COLOR = "#2a78d6"
_FIT_COLOR = "#1baf7a"


def _reconstruct_sample_stack(
    output_combination: dict[str, np.ndarray],
    pixel: tuple[int, int],
    irf: np.ndarray,
    decay_px: np.ndarray,
    freq_acq: float,
    model_type: str,
) -> np.ndarray:
    """
    Reconstruct every posterior sample's decay curve at one pixel, scaled to
    that pixel's measured photon count the same way
    :func:`pyfli.analysis.utils.compute_detailed_results` scales its fits
    (unit-amplitude reconstruction, then rescaled so its sum matches the
    measured decay's sum).

    Treats the NUM_SAMPLES axis as the reconstructor's "W" (pixel) axis, so
    every sample is batched through one vectorized reconstruction instead of
    looping in Python.

    Returns
    -------
    np.ndarray
        ``(NUM_SAMPLES, T)`` reconstructed decay curves.
    """
    x, y = pixel
    param_keys = _MODEL_PARAM_KEYS[model_type]
    num_samples = output_combination[param_keys[0]].shape[-1]

    recon_params = {
        f"{key}_map": output_combination[key][x, y, :][None, :] for key in param_keys
    }
    recon_params["photon_count_map"] = np.ones((1, num_samples), dtype=np.float32)

    irf_px = irf[x, y, :] if np.ndim(irf) == 3 else irf
    recon = ParameterToDecayReconstruction(model_type, freq_acq, irf=irf_px)

    unit = recon.reconstruct_unit_amplitude(recon_params)
    convolved = unit["convolved_map"]  # (1, NUM_SAMPLES, T)

    decay_rep = np.broadcast_to(decay_px, convolved.shape)
    scaled = recon.rescale_fit_to_measured_totals(convolved, decay_rep)
    return scaled[0]  # (NUM_SAMPLES, T)


def _select_best_sample_idx(
    output_combination: dict[str, np.ndarray],
    pixel: tuple[int, int],
    irf: np.ndarray,
    decay: np.ndarray,
    freq_acq: float,
    model_type: str,
    metric: str,
) -> int:
    """
    Pick the posterior sample that best fits this one pixel, by delegating to
    :class:`BestParamFitSelector` on a 1x1-pixel crop -- reuses its tested
    per-sample goodness-of-fit logic instead of duplicating it here.
    """
    x, y = pixel
    sub_combo = {k: v[x : x + 1, y : y + 1, :] for k, v in output_combination.items()}
    sub_decay = decay[x : x + 1, y : y + 1, :]
    sub_irf = irf[x : x + 1, y : y + 1, :] if np.ndim(irf) == 3 else irf

    selector = BestParamFitSelector(freq_acq, sub_irf, sub_decay, model_type=model_type)
    stacks = selector.evaluate_all_samples(sub_combo)
    selection = selector.select_best_combination(sub_combo, stacks, metric=metric)
    return int(selection["best_sample_idx"][0, 0])


def plot_pixel_posterior_fit(
    output_combination: dict[str, np.ndarray],
    decay: np.ndarray,
    irf: np.ndarray,
    freq_acq: float,
    pixel: tuple[int, int],
    model_type: str = "bi-exponential",
    center: str = "median",
    metric: str = "reduced_chi2",
    ci_levels: tuple[int, ...] = (92, 68),
    title: str | None = None,
    ax: "plt.Axes | None" = None,
):
    """
    Plot one pixel's posterior-sample decay reconstructions as nested
    credible-interval bands, a chosen central curve, and the measured decay.

    Parameters
    ----------
    output_combination : dict[str, np.ndarray]
        Posterior-sample parameter maps, e.g.
        ``{'tau1': (H,W,NUM_SAMPLES), 'tau2': (H,W,NUM_SAMPLES), 'alpha1': (H,W,NUM_SAMPLES)}``
        for bi-exponential, or ``{'tau': (H,W,NUM_SAMPLES)}`` for mono-exponential
        -- same shape convention as :class:`BestParamFitSelector`.
    decay : np.ndarray
        Measured decay, ``(H, W, T)``.
    irf : np.ndarray
        IRF, ``(T,)`` (shared) or ``(H, W, T)`` (per-pixel).
    freq_acq : float
        Acquisition frequency (MHz), i.e. ``freq[1]``.
    pixel : tuple[int, int]
        ``(x, y)`` pixel to plot.
    model_type : str
        ``"bi-exponential"`` or ``"mono-exponential"``.
    center : str
        Which curve to draw as the central line: ``"median"`` or ``"mean"``
        across posterior samples, or ``"best"`` (the single sample that
        optimizes ``metric`` at this pixel, via
        :meth:`BestParamFitSelector.select_best_combination`).
    metric : str
        Only used when ``center="best"``; one of
        :attr:`BestParamFitSelector.METRICS` (``"chi2"``, ``"reduced_chi2"``,
        ``"RMSE"``, ``"R2"``).
    ci_levels : tuple[int, ...]
        Nested credible-interval widths to shade, e.g. ``(92, 68)`` shades a
        92% and a 68% band (percentiles ``(4, 96)`` and ``(16, 84)`` of the
        per-bin sample distribution).
    title : str | None
        Axes title; defaults to ``f"Pixel ({x}, {y})"``.
    ax : matplotlib.axes.Axes | None
        Axes to draw into. If omitted, a new figure/axes is created and shown.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
    """
    if model_type not in _MODEL_PARAM_KEYS:
        raise ValueError(
            f"Unknown model_type: {model_type!r}; expected one of "
            f"{tuple(_MODEL_PARAM_KEYS)}"
        )
    if center not in CENTERS:
        raise ValueError(f"Unknown center: {center!r}; expected one of {CENTERS}")

    x, y = pixel
    decay_px = np.asarray(decay)[x, y, :].astype(np.float64)

    stack = _reconstruct_sample_stack(
        output_combination, pixel, irf, decay_px, freq_acq, model_type
    )

    if center == "median":
        center_curve = np.median(stack, axis=0)
        center_label = "Posterior median"
    elif center == "mean":
        center_curve = np.mean(stack, axis=0)
        center_label = "Posterior mean"
    else:
        best_idx = _select_best_sample_idx(
            output_combination, pixel, irf, decay, freq_acq, model_type, metric
        )
        center_curve = stack[best_idx]
        center_label = f"Best sample ({metric})"

    t = np.arange(stack.shape[-1])

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    else:
        fig = ax.figure

    # Nested washes of the same hue: widest interval lightest, narrowest
    # darkest, so the bands read as one distribution rather than competing
    # series -- narrowest drawn last so it sits on top.
    band_handles = []
    alphas = np.linspace(0.12, 0.30, len(ci_levels))
    for level, alpha in zip(sorted(ci_levels, reverse=True), alphas):
        half_width = (100 - level) / 2.0
        lo = np.percentile(stack, half_width, axis=0)
        hi = np.percentile(stack, 100 - half_width, axis=0)
        band = ax.fill_between(t, lo, hi, color=_FIT_COLOR, alpha=alpha, linewidth=0)
        band_handles.append((band, f"{level}% credible interval"))

    (center_line,) = ax.plot(t, center_curve, color=_FIT_COLOR, lw=2)
    (decay_line,) = ax.plot(t, decay_px, color=_DECAY_COLOR, lw=2)

    ax.set_xlabel("Time Bin")
    ax.set_ylabel("Counts")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title(title or f"Pixel ({x}, {y})")

    handles = [h for h, _ in band_handles] + [center_line, decay_line]
    labels = [lbl for _, lbl in band_handles] + [center_label, "Decay (measured)"]
    ax.legend(handles, labels, loc="best", fontsize=8, frameon=False)

    if own_fig:
        plt.show()
    return fig, ax
