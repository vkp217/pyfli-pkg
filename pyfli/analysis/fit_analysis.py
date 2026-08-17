"""
Plot fitted parameter maps, diagnostics, classifier summaries, and two-dimensional
comparisons.

This module belongs to :mod:`pyfli.analysis` and is part of PyFLI post-processing,
diagnostics, statistical comparison, and result-loading utilities for fitted FLI/FLIM
datasets. Public API includes functions :func:`plot_fitting_maps`,
:func:`plot_diagnostics`, :func:`plot_pixel_evidence`,
:func:`plot_statistical_comparison`, :func:`plot_2d_analysis`, and
:func:`run_mono_bi_classifier`.
"""

from typing import Any

from ..data_vnp import (
    DataViewer,
    Plotter,
    ColorProcessor,
    MonoBiClassifier,
    plot_2d_subplots,
)
from ..data_cc import Normalization
from ..data_text import MessageDisplay
from .utils import plot_pixel_diagnostic, random_true_pixel


# Per-key default thresholds reflecting the physical valid range of each parameter.
# alpha maps are fractions [0, 1]; lifetime maps use a generous 5 ns upper bound.
# Override via the per_key_thresholds argument in plot_statistical_comparison /
# plot_2d_analysis when your fluorophore has longer or shorter lifetimes.
DEFAULT_KEY_THRESHOLDS = {
    "tau_map": (0.0, 5.0),  # mono-exp apparent lifetime (ns)
    "tau1_map": (0.0, 5.0),  # bi-exp short component (ns)
    "tau2_map": (0.0, 5.0),  # bi-exp long component (ns)
    "alpha1_map": (0.0, 1.0),  # fraction — strictly [0, 1]
    "alpha2_map": (0.0, 1.0),
    "mean_lifetime": (0.0, 5.0),  # amplitude-weighted mean (ns)
}

_DEFAULT_COLORS = [
    "#5DADE2",
    "#EC7063",
    "#58D68D",
    "#F4D03F",
    "#AF7AC5",
    "#EB984E",
    "#48C9B0",
    "#52BE80",
    "#AAB7B8",
    "#F1948A",
    "#BB8FCE",
    "#7FB3D5",
    "#76D7C4",
]


def _resolve_threshold(map_key: str, per_key_thresholds: Any) -> Any:
    """Return the threshold for map_key, checking user overrides first."""
    if per_key_thresholds and map_key in per_key_thresholds:
        return per_key_thresholds[map_key]
    return DEFAULT_KEY_THRESHOLDS.get(map_key, (0.0, 5.0))


def plot_fitting_maps(
    all_datasets: Any,
    names: Any,
    map_keys: Any,
    v_ranges: Any | None = None,
    saver: Any | None = None,
    cmap: str | None = None,
) -> None:
    """Plot parameter maps for every fitting result.

    Parameters
    ----------
    all_datasets : list[dict]   from load_fitting_results()
    names        : list[str]
    map_keys     : list[str]    keys to extract, e.g. ['tau_map'] or
                                ['alpha1_map', 'tau1_map', 'tau2_map']
    v_ranges     : list[tuple] or None
                   Display range per map, e.g. [(0, 1.5)] or [(0,1),(0,2),(0,2)].
                   Pass None to let DataViewer auto-scale.
    saver        : DataSaver or None
    cmap         : colormap — defaults to jet with zero→black
    """
    if cmap is None:
        cmap = ColorProcessor().lowest_zero("jet")
    n_cols = len(map_keys)

    for ds, name in zip(all_datasets, names):
        data_list = [ds[k] for k in map_keys]
        data_names = [f"{k}{name}" for k in map_keys]
        viewer_kw = {"save_path": saver.save_dir, "fig_name": name} if saver else {}
        DataViewer(**viewer_kw).display_data(
            data_list,
            structure=(1, n_cols),
            coord=None,
            data_names=data_names,
            cmaps=[cmap] * n_cols,
            v_ranges=v_ranges,
            figsize=None,
            normalize=False,
            yscale="linear",
        )


def plot_diagnostics(
    binned_decay: Any, all_fitset: Any, names: Any, mask: Any, saver: Any | None = None
) -> tuple[Any, ...]:
    """Pixel diagnostic overlays for all fitting results (log and linear scale).

    Returns
    -------
    fig_log, fig_lin : Figure
    """
    fig_log = plot_pixel_diagnostic(
        binned_decay,
        all_fitset,
        names,
        mask=mask,
        t=None,
        yscale="log",
        raw_style="line",
    )
    fig_lin = plot_pixel_diagnostic(
        binned_decay,
        all_fitset,
        names,
        mask=mask,
        t=None,
        yscale="linear",
        raw_style="line",
    )
    if saver:
        saver.save_plot("fit_log_diagnostics", fig=fig_log, close=False)
        saver.save_plot("fit_linear_diagnostics", fig=fig_lin, close=False)
    return fig_log, fig_lin


def plot_pixel_evidence(
    binned_decay: Any,
    binned_irf: Any,
    all_fitset: Any,
    all_datasets: Any,
    names: Any,
    mask: Any,
    saver: Any | None = None,
    num: int = 0,
) -> None:
    """Single-pixel fit evidence plot for a randomly selected valid pixel.

    Parameters
    ----------
    num : int   index into all_fitset / all_datasets to display (default 0)
    """
    x, y = random_true_pixel(mask)
    label = names[num]
    TRs = all_fitset[num]
    maps = all_datasets[num]
    irf_norm = Normalization(binned_irf).norm_scale(binned_decay)

    viewer_kw = (
        {"save_path": saver.save_dir, "fig_name": f"fit_evidence_{label}"}
        if saver
        else {}
    )
    DataViewer(**viewer_kw).plot_fli_px(
        data_list=[binned_decay, irf_norm, TRs["fit_map"], TRs["residual_map"]],
        pixel=(x, y),
        mode=[0, 1, 2],
        mode2=[0],
        names=["decay", "irf", "fit"],
    )
    MessageDisplay().get_pixel_summary(data_maps=maps, px=(x, y))


def plot_statistical_comparison(
    all_datasets: Any,
    names: Any,
    map_keys: Any,
    mask: Any,
    saver: Any | None = None,
    graph_type: str = "box",
    colors_list: Any | None = None,
    test_type: str = "none",
    per_key_thresholds: Any | None = None,
    percentile_clip: tuple[int, ...] = (1, 99),
) -> Any:
    """Comparative statistical plot per parameter key (box / violin / KDE / ...).

    One figure is produced per key so that each parameter is filtered by its
    own physically valid range (e.g. alpha ∈ [0,1] vs tau ∈ [0,5 ns]).
    Thresholds fall back to DEFAULT_KEY_THRESHOLDS when not overridden.

    Parameters
    ----------
    all_datasets        : list[dict]
    names               : list[str]
    map_keys            : list[str]   e.g. ['tau_map'] or
                                      ['alpha1_map', 'tau1_map', 'tau2_map']
    mask                : np.ndarray  (H, W) bool
    graph_type          : str         'box', 'violin', 'swarm', 'overlay',
                                      'raincloud', or 'kde'
    test_type           : str         'none', 'paired', or 'welch'
    colors_list         : list        per-source colour hex strings
    per_key_thresholds  : dict or None
                          Override thresholds per key, e.g.
                          ``{'tau_map': (0, 3), 'alpha1_map': (0, 1)}``.
                          Keys not listed fall back to DEFAULT_KEY_THRESHOLDS.
    percentile_clip     : tuple or None   (low%, high%) applied to every key;
                          pass None to disable

    Returns
    -------
    figs : dict[str, Figure]   keyed by map_key
    """
    if colors_list is None:
        colors_list = _DEFAULT_COLORS

    masking = mask.ravel()
    figs = {}

    for map_key in map_keys:
        threshold = _resolve_threshold(map_key, per_key_thresholds)
        ops = {
            "mask": masking,
            "remove_nan": True,
            "remove_zero": True,
            "threshold": threshold,
        }
        if percentile_clip is not None:
            ops["percentile_clip"] = percentile_clip

        painter = Plotter(
            *all_datasets,
            values=[map_key],
            style_config=colors_list,
            source_names=names,
            operations=ops,
        )
        fig = painter.make_plot(
            title=f"Multi-method comparison — {map_key}  "
            f"(threshold {threshold}, clip {percentile_clip})",
            graph_type=graph_type,
            point_type="strip",
            show_mean=True,
            show_median=True,
            show_significance=True,
            test_type=test_type,
            correction=False,
        )
        figs[map_key] = fig
        if saver:
            saver.save_plot(f"comparative_{map_key}", fig=fig, close=False)
            saver.log(f"Comparative analysis saved — {map_key}  threshold={threshold}")

    return figs


def plot_2d_analysis(
    all_datasets: Any,
    names: Any,
    map_keys: Any,
    mask: Any,
    per_key_thresholds: Any | None = None,
    saver: Any | None = None,
    cmap: str = "jet",
) -> None:
    """2D subplot analysis (map + histogram + violin + boxplot + KDE + qq + CDF)
    per parameter map, for every fitting result.

    Thresholds are resolved per key via DEFAULT_KEY_THRESHOLDS so that alpha
    maps are automatically clipped to [0, 1] and tau maps to [0, 5] unless
    overridden.

    Parameters
    ----------
    all_datasets        : list[dict]
    names               : list[str]
    map_keys            : list[str]   e.g. ['tau_map'] or
                                      ['alpha1_map', 'tau1_map', 'tau2_map']
    mask                : np.ndarray  (H, W) bool
    per_key_thresholds  : dict or None
                          Override per key, e.g. ``{'tau2_map': (0, 3)}``.
                          Keys not listed fall back to DEFAULT_KEY_THRESHOLDS.
    saver               : DataSaver or None
    cmap                : str   colormap for the spatial map panels
    """
    plot_types = ["map", "histogram", "violinplot", "boxplot", "KDE", "qq", "cdF"]

    for map_key in map_keys:
        threshold = _resolve_threshold(map_key, per_key_thresholds)
        datasets_to_plot = [ds[map_key] for ds in all_datasets]
        base_op = {
            "remove_nan": True,
            "remove_zero": True,
            "threshold": threshold,
            "mask": mask,
        }
        operations = [base_op for _ in datasets_to_plot]
        fig = plot_2d_subplots(
            *datasets_to_plot,
            plot_types=plot_types,
            titles=names[: len(datasets_to_plot)],
            operations=operations,
            figsize=(20, 3 * len(names)),
            cmap=cmap,
        )
        if saver:
            saver.save_plot(f"{map_key}", fig=fig, close=False)
            saver.log(f"Detailed analysis of {map_key} saved  threshold={threshold}")


def run_mono_bi_classifier(
    all_datasets: Any,
    names: Any,
    mask: Any,
    alpha_upper: float = 0.95,
    alpha_lower: float = 0.05,
    tau_tol: float = 0.01,
    scatter_keys: Any | None = None,
    saver: Any | None = None,
) -> tuple[Any, ...]:
    """
    Run mono bi classifier.

    Parameters
    ----------
    all_datasets : Any
        Collection of fitted datasets to classify, compare, or summarize.
    names : Any
        Dataset names used in summaries and plots.
    mask : Any
        Boolean or labeled mask selecting pixels for the operation.
    alpha_upper : float
        Upper alpha-fraction threshold used by the classifier.
    alpha_lower : float
        Lower alpha-fraction threshold used by the classifier.
    tau_tol : float
        Lifetime tolerance used by the mono/bi classifier.
    scatter_keys : Any | None
        Parameter keys used in classifier scatter plots.
    saver : Any | None
        Optional saver used to persist messages or figures.

    Returns
    -------
    tuple[Any, ...]
        Tuple containing classifier outputs and mono/bi-exponential labels.
    """
    if scatter_keys is None:
        scatter_keys = ["tau1_map", "tau2_map"]

    clf = MonoBiClassifier(
        mask,
        names=names,
        alpha_upper=alpha_upper,
        alpha_lower=alpha_lower,
        tau_tol=tau_tol,
    )
    classes = clf.classify(all_datasets, display=True)

    clf.agreement(metric="jaccard")
    clf.agreement(metric="count")

    for key in scatter_keys:
        clf.param_scatter_matrix(key, cls="mono")
        clf.param_scatter_matrix(key, cls="bi")

    df = clf.agreed_param_table(cls="mono")

    if saver:
        saver.log(
            f"MonoBiClassifier run: alpha_upper={alpha_upper}, "
            f"alpha_lower={alpha_lower}, tau_tol={tau_tol}, "
            f"scatter_keys={scatter_keys}"
        )

    return clf, classes, df
