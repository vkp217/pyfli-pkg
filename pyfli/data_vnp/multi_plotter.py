"""
Class map
─────────
PlotConfig          @dataclass – visual + statistical defaults (shared)
DataProcessor       2-D spatial cleaning (mask / threshold / NaN)
SourceLoader        Multi-source dict / npz / ndarray ingestion
PlotKit             Static axis-level draw primitives
SubplotVisualizer   Grid of spatial maps + 1-D distribution plots
Plotter             Multi-source comparison orchestrator
  DLModelComparator   W / KL / Energy metrics + plot
plot_2d_subplots()  Backward-compatible module function
"""

from pyfli import logging

from dataclasses import dataclass, field, replace as dc_replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import (
    gaussian_kde,
    probplot,
    wasserstein_distance,
    energy_distance,
    entropy,
)


# ─────────────────────────────────────────────────────────────────────────────
# PlotConfig  –  single source of truth for every default
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class PlotConfig:
    """
    Collect shared plotting and statistical defaults for comparison figures. Pass one
    configuration object to map, histogram, KDE, violin, box, CDF, QQ, scatter, and
    clustered comparison plots for consistent styling.

    Parameters
    ----------
    figsize : Tuple[int, int]
        Figure size passed to Matplotlib.
    cmap : str
        Matplotlib colormap used for image and map rendering.
    bins : int
        Histogram bin specification.
    colors : List[str]
        Color sequence used for plotted groups.
    imshow_source : str
        Data source used for image panels.
    shared_colorbar : bool
        If ``True``, draw one colorbar shared by comparable image panels.
    annotate_stats : bool
        If ``True``, annotate plots with summary statistics.
    scatter_pair : Optional[Tuple[int, int]]
        Pair of variables to compare in a scatter plot.
    qq_reference : str
        Reference distribution used for QQ plots.
    point_type : str
        Marker style used for plotted points.
    show_mean : bool
        If ``True``, draw the group mean on distribution plots.
    show_median : bool
        If ``True``, draw the group median on distribution plots.
    test_type : str
        Statistical test to apply when comparing groups.
    correction : bool
        Multiple-comparison correction method.
    """

    # ── visual ────────────────────────────────────────────────────────────────
    figsize: Tuple[int, int] = (14, 8)
    cmap: str = "viridis"
    bins: int = 100
    colors: List[str] = field(
        default_factory=lambda: [
            "#3498db",
            "#e74c3c",
            "#2ecc71",
            "#f1c40f",
            "#9b59b6",
            "#2433bb",
        ]
    )
    imshow_source: str = "processed"  # "raw" | "processed"
    shared_colorbar: bool = False
    annotate_stats: bool = True
    scatter_pair: Optional[Tuple[int, int]] = None
    qq_reference: str = "norm"
    # ── comparison ────────────────────────────────────────────────────────────
    point_type: str = "strip"  # "strip" | "swarm"
    show_mean: bool = True
    show_median: bool = True
    # ── stats testing ─────────────────────────────────────────────────────────
    test_type: str = "welch"  # "welch"|"paired"|"none"
    correction: bool = False

    def color(self, i: int) -> str:
        """Safely cycle through colors by index."""
        return self.colors[i % len(self.colors)]


# ─────────────────────────────────────────────────────────────────────────────
# DataProcessor  –  2-D spatial cleaning
# ─────────────────────────────────────────────────────────────────────────────


class DataProcessor:
    """
    Apply declarative preprocessing operations to two-dimensional arrays. Operations
    include mask handling, thresholding, finite-value filtering, and simple summary
    statistics used by plotting classes.
    """

    MIN_SAMPLES: int = 5

    @staticmethod
    def process(
        data: np.ndarray,
        operations: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (processed_map, valid_1d)."""
        data = np.array(data, dtype=float)
        ops = operations or {}
        dm = data.copy()

        if "mask" in ops:
            mask = np.asarray(ops["mask"], dtype=bool)
            if mask.shape != dm.shape:
                if mask.size == dm.size:
                    mask = mask.reshape(dm.shape)
                else:
                    raise ValueError(
                        f"mask has {mask.size} elements but data has {dm.size}"
                    )
            dm = np.where(mask, dm, np.nan)
        if ops.get("remove_nan", False):
            dm[~np.isfinite(dm)] = np.nan
        if ops.get("remove_zero", False):
            dm[dm == 0] = np.nan
        if "threshold" in ops:
            tmin, tmax = ops["threshold"]
            if tmin is not None:
                dm[dm < tmin] = np.nan
            if tmax is not None:
                dm[dm > tmax] = np.nan
        if "percentile_clip" in ops:
            pmin, pmax = ops["percentile_clip"]
            fin = dm[np.isfinite(dm)]
            if fin.size:
                lo, hi = np.percentile(fin, pmin), np.percentile(fin, pmax)
                dm[(dm < lo) | (dm > hi)] = np.nan
        if "custom" in ops:
            dm = ops["custom"](dm)

        valid = dm[np.isfinite(dm) & ~np.isnan(dm)].ravel()
        return dm, valid[np.isfinite(valid)]

    @classmethod
    def is_valid(cls, valid: np.ndarray, min_samples: Optional[int] = None) -> bool:
        """
        Return whether valid.

        Parameters
        ----------
        valid : np.ndarray
            Finite one-dimensional sample values used by a plot or statistic.
        min_samples : Optional[int]
            Minimum number of finite samples required for a group.

        Returns
        -------
        bool
            Boolean result computed by is valid.
        """
        n = min_samples or cls.MIN_SAMPLES
        v = np.asarray(valid)
        v = v[np.isfinite(v)]
        return len(v) >= n and np.nanstd(v) != 0

    @staticmethod
    def stats(valid: np.ndarray) -> Dict[str, float]:
        """
        Compute summary statistics for valid sample values.

        Parameters
        ----------
        valid : np.ndarray
            Finite one-dimensional sample values used by a plot or statistic.

        Returns
        -------
        Dict[str, float]
            Object produced by stats.
        """
        if not len(valid):
            return {}
        return dict(
            mean=float(np.mean(valid)),
            std=float(np.std(valid)),
            median=float(np.median(valid)),
            min=float(np.min(valid)),
            max=float(np.max(valid)),
            n=int(len(valid)),
        )


# ─────────────────────────────────────────────────────────────────────────────
# SourceLoader  –  multi-source dict / npz / ndarray ingestion
# ─────────────────────────────────────────────────────────────────────────────


class SourceLoader:
    """
    Normalize heterogeneous plot inputs into a consistent source dictionary. It accepts
    direct arrays, dictionaries, or named value collections so downstream plotters can
    compare multiple data sources uniformly.

    Parameters
    ----------
    *args : Any
        Additional positional values accepted by the object.
    values : np.ndarray | None
        Explicit values to load as plotting sources.
    source_names : np.ndarray | None
        Names assigned to plotted or compared data sources.
    """

    def __init__(
        self,
        *args: Any,
        values: np.ndarray | None = None,
        source_names: np.ndarray | None = None,
    ) -> None:
        self.raw_sources = args
        self.values = values
        self.source_names = source_names or [
            f"Source {i + 1}" for i in range(len(args))
        ]
        self.labels = self._infer_labels()

    def _infer_labels(self) -> List[str]:
        """
        Run the infer labels routine.

        Returns
        -------
        List[str]
            Object produced by infer labels.
        """
        if self.values:
            return list(self.values)
        seen: Dict[str, None] = {}
        for src in self.raw_sources:
            if hasattr(src, "files"):  # npz
                for k in src.files:
                    seen[k] = None
            elif isinstance(src, dict):
                for k in src.keys():
                    seen[k] = None
        return list(seen)

    @staticmethod
    def _extract(source: str, key: str) -> np.ndarray:
        """
        Run the extract routine.

        Parameters
        ----------
        source : str
            Source label recorded with the loaded dataset.
        key : str
            Dictionary key or parameter-map name to extract.

        Returns
        -------
        np.ndarray
            Finite values extracted for plotting or statistics.
        """
        try:
            return np.asanyarray(source[key]).astype(float).flatten()
        except (KeyError, ValueError, TypeError):
            return np.array([])

    def load(self) -> Dict[str, List[np.ndarray]]:
        """Return {label: [arr_per_source]}."""
        groups: Dict[str, List[np.ndarray]] = {k: [] for k in self.labels}
        for src in self.raw_sources:
            if isinstance(src, np.ndarray):
                for i, key in enumerate(self.labels):
                    if src.ndim == 2 and i < src.shape[1]:
                        arr = src[:, i].astype(float)
                    elif src.ndim == 1 and i == 0:
                        arr = src.astype(float)
                    else:
                        continue
                    arr = arr[np.isfinite(arr)]
                    if arr.size:
                        groups[key].append(arr)
            else:
                for key in self.labels:
                    arr = self._extract(src, key)
                    if arr.size:
                        groups[key].append(arr)
        return groups


# ─────────────────────────────────────────────────────────────────────────────
# PlotKit  –  standalone static draw primitives
# ─────────────────────────────────────────────────────────────────────────────


class PlotKit:
    """
    Provide stateless axis-level plotting primitives. The methods draw maps, histograms,
    KDEs, violin and box plots, CDFs, QQ plots, scatters, raincloud plots, and metric
    bars on caller-provided axes.
    """

    # ── map / imshow ─────────────────────────────────────────────────────────
    @staticmethod
    def map(
        ax: Axes,
        data_map: np.ndarray,
        *,
        config: Any | None = None,
        title: str = "",
        vmin: np.ndarray | None = None,
        vmax: np.ndarray | None = None,
        fig: Any | None = None,
        add_colorbar: bool = True,
        **kw: Any,
    ) -> None:
        """
        Draw a two-dimensional parameter map.

        Parameters
        ----------
        ax : Axes
            Matplotlib axes object on which the plot is drawn.
        data_map : np.ndarray
            Parameter or mask map processed by the routine.
        config : Any | None
            Plotting, fitting, or simulation configuration object.
        title : str
            Title displayed on the generated plot.
        vmin : np.ndarray | None
            Lower color-limit value.
        vmax : np.ndarray | None
            Upper color-limit value.
        fig : Any | None
            Matplotlib figure object to update or save.
        add_colorbar : bool
            Whether to add a colorbar to the generated plot.
        **kw : Any
            Additional keyword options forwarded to the underlying implementation.

        Returns
        -------
        None
            No object is returned; the function perform map.
        """
        cfg = config or PlotConfig()
        im = ax.imshow(data_map, cmap=cfg.cmap, vmin=vmin, vmax=vmax, **kw)
        if add_colorbar and fig is not None:
            fig.colorbar(im, ax=ax)
        ax.set_title(f"{title} Map".strip())

    # ── histogram ────────────────────────────────────────────────────────────
    @staticmethod
    def histogram(
        ax: Axes,
        valid: np.ndarray,
        *,
        config: Any | None = None,
        title: str = "",
        **kw: Any,
    ) -> None:
        """
        Draw a histogram for valid sample values.

        Parameters
        ----------
        ax : Axes
            Matplotlib axes object on which the plot is drawn.
        valid : np.ndarray
            Finite one-dimensional sample values used by a plot or statistic.
        config : Any | None
            Plotting, fitting, or simulation configuration object.
        title : str
            Title displayed on the generated plot.
        **kw : Any
            Additional keyword options forwarded to the underlying implementation.

        Returns
        -------
        None
            No object is returned; the function perform histogram.
        """
        ax.hist(valid, bins=(config or PlotConfig()).bins, **kw)
        ax.set_title(f"{title} Histogram".strip())

    # ── log histogram ─────────────────────────────────────────────────────────
    @staticmethod
    def log_histogram(
        ax: Axes,
        valid: np.ndarray,
        *,
        config: Any | None = None,
        title: str = "",
        **kw: Any,
    ) -> None:
        """
        Draw a logarithmic histogram for valid sample values.

        Parameters
        ----------
        ax : Axes
            Matplotlib axes object on which the plot is drawn.
        valid : np.ndarray
            Finite one-dimensional sample values used by a plot or statistic.
        config : Any | None
            Plotting, fitting, or simulation configuration object.
        title : str
            Title displayed on the generated plot.
        **kw : Any
            Additional keyword options forwarded to the underlying implementation.

        Returns
        -------
        None
            No object is returned; the function perform log histogram.
        """
        ax.hist(valid[valid > 0], bins=(config or PlotConfig()).bins, log=True, **kw)
        ax.set_title(f"{title} Log Histogram".strip())

    # ── KDE ──────────────────────────────────────────────────────────────────
    @staticmethod
    def kde(
        ax: Axes,
        valid: np.ndarray,
        *,
        config: Any | None = None,
        title: str = "",
        color: str | None = None,
        label: str | None = None,
        fill: bool = False,
        alpha: float = 0.35,
        n_points: int = 1000,
        **kw: Any,
    ) -> None:
        """
        Draw a kernel-density estimate for valid sample values.

        Parameters
        ----------
        ax : Axes
            Matplotlib axes object on which the plot is drawn.
        valid : np.ndarray
            Finite one-dimensional sample values used by a plot or statistic.
        config : Any | None
            Plotting, fitting, or simulation configuration object.
        title : str
            Title displayed on the generated plot.
        color : str | None
            Matplotlib color used for drawing the plot element.
        label : str | None
            Display label assigned to the data or plot element.
        fill : bool
            Whether to fill the KDE area under the curve.
        alpha : float
            Regularization strength, fraction value, or significance threshold used by the
        routine.
        n_points : int
            Number of points sampled for a curve or density.
        **kw : Any
            Additional keyword options forwarded to the underlying implementation.

        Returns
        -------
        None
            No object is returned; the function perform kde.
        """
        if len(valid) > 1:
            kf = gaussian_kde(valid)
            x = np.linspace(valid.min(), valid.max(), n_points)
            y = kf(x)
            if fill:
                ax.fill_between(x, y, alpha=alpha, color=color)
                ax.plot(x, y, color=color, linewidth=1, label=label)
            else:
                ax.plot(x, y, color=color, label=label, **kw)
        ax.set_title(f"{title} KDE".strip())

    # ── violin ────────────────────────────────────────────────────────────────
    @staticmethod
    def violinplot(
        ax: Axes,
        valid: np.ndarray,
        *,
        config: Any | None = None,
        title: str = "",
        **kw: Any,
    ) -> None:
        """
        Draw a violin plot for valid sample values.

        Parameters
        ----------
        ax : Axes
            Matplotlib axes object on which the plot is drawn.
        valid : np.ndarray
            Finite one-dimensional sample values used by a plot or statistic.
        config : Any | None
            Plotting, fitting, or simulation configuration object.
        title : str
            Title displayed on the generated plot.
        **kw : Any
            Additional keyword options forwarded to the underlying implementation.

        Returns
        -------
        None
            No object is returned; the function perform violinplot.
        """
        if DataProcessor.is_valid(valid):
            ax.violinplot(valid, showmeans=True, showmedians=True, **kw)
        else:
            ax.text(
                0.5,
                0.5,
                "Insufficient data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        ax.set_title(f"{title} Violin".strip())

    # ── boxplot ───────────────────────────────────────────────────────────────
    @staticmethod
    def boxplot(
        ax: Axes,
        valid: np.ndarray,
        *,
        config: Any | None = None,
        title: str = "",
        **kw: Any,
    ) -> None:
        """
        Draw a box plot for valid sample values.

        Parameters
        ----------
        ax : Axes
            Matplotlib axes object on which the plot is drawn.
        valid : np.ndarray
            Finite one-dimensional sample values used by a plot or statistic.
        config : Any | None
            Plotting, fitting, or simulation configuration object.
        title : str
            Title displayed on the generated plot.
        **kw : Any
            Additional keyword options forwarded to the underlying implementation.

        Returns
        -------
        None
            No object is returned; the function perform boxplot.
        """
        if DataProcessor.is_valid(valid):
            ax.boxplot(valid, orientation="vertical", **kw)
        else:
            ax.text(
                0.5,
                0.5,
                "Insufficient data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        ax.set_title(f"{title} Boxplot".strip())

    # ── CDF ───────────────────────────────────────────────────────────────────
    @staticmethod
    def cdf(
        ax: Axes,
        valid: np.ndarray,
        *,
        config: Any | None = None,
        title: str = "",
        color: str | None = None,
        label: str | None = None,
        **kw: Any,
    ) -> None:
        """
        Draw an empirical cumulative distribution plot.

        Parameters
        ----------
        ax : Axes
            Matplotlib axes object on which the plot is drawn.
        valid : np.ndarray
            Finite one-dimensional sample values used by a plot or statistic.
        config : Any | None
            Plotting, fitting, or simulation configuration object.
        title : str
            Title displayed on the generated plot.
        color : str | None
            Matplotlib color used for drawing the plot element.
        label : str | None
            Display label assigned to the data or plot element.
        **kw : Any
            Additional keyword options forwarded to the underlying implementation.

        Returns
        -------
        None
            No object is returned; the function perform CDF.
        """
        s = np.sort(valid)
        ax.plot(s, np.arange(len(s)) / len(s), color=color, label=label, **kw)
        ax.set_ylabel("Cumulative probability")
        ax.set_title(f"{title} CDF".strip())

    # ── QQ ────────────────────────────────────────────────────────────────────
    @staticmethod
    def qq(
        ax: Axes,
        valid: np.ndarray,
        *,
        config: Any | None = None,
        title: str = "",
        **kw: Any,
    ) -> None:
        """
        Draw a quantile-quantile diagnostic plot.

        Parameters
        ----------
        ax : Axes
            Matplotlib axes object on which the plot is drawn.
        valid : np.ndarray
            Finite one-dimensional sample values used by a plot or statistic.
        config : Any | None
            Plotting, fitting, or simulation configuration object.
        title : str
            Title displayed on the generated plot.
        **kw : Any
            Additional keyword options forwarded to the underlying implementation.

        Returns
        -------
        None
            No object is returned; the function perform QQ.
        """
        probplot(valid, dist=(config or PlotConfig()).qq_reference, plot=ax)
        ax.set_title(f"{title} QQ Plot".strip())

    # ── scatter ───────────────────────────────────────────────────────────────
    @staticmethod
    def scatter(
        ax: Axes,
        x: np.ndarray,
        y: np.ndarray,
        *,
        config: Any | None = None,
        title: str = "",
        **kw: Any,
    ) -> None:
        """
        Draw a scatter plot for paired arrays.

        Parameters
        ----------
        ax : Axes
            Matplotlib axes object on which the plot is drawn.
        x : np.ndarray
            Input array, coordinate, or signal being transformed.
        y : np.ndarray
            Observed signal, target data, or coordinate array.
        config : Any | None
            Plotting, fitting, or simulation configuration object.
        title : str
            Title displayed on the generated plot.
        **kw : Any
            Additional keyword options forwarded to the underlying implementation.

        Returns
        -------
        None
            No object is returned; the function perform scatter.
        """
        ax.scatter(x, y, **kw)
        ax.set_title(f"{title} Scatter".strip())

    # ── raincloud  (half violin + box + jittered strip) ───────────────────────
    @staticmethod
    def raincloud(
        ax: Axes,
        valid: np.ndarray,
        *,
        config: Any | None = None,
        title: str = "",
        color: str | None = None,
        position: int = 0,
        width: float = 0.4,
        **kw: Any,
    ) -> None:
        """Half-violin + embedded box + jittered strip at a given x position."""
        if not DataProcessor.is_valid(valid):
            ax.text(
                0.5,
                0.5,
                "Insufficient data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"{title} Raincloud".strip())
            return
        v = ax.violinplot(valid, positions=[position], widths=width, showextrema=False)
        body = v["bodies"][0]
        verts = body.get_paths()[0].vertices
        verts[:, 0] = np.clip(verts[:, 0], np.mean(verts[:, 0]), np.inf)
        body.set_facecolor(color)
        body.set_alpha(0.4)
        body.set_edgecolor("black")
        body.set_linewidth(0.8)

        ax.boxplot(
            valid,
            positions=[position],
            widths=width * 0.35,
            patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor=color, alpha=0.7),
            medianprops=dict(color="black"),
        )

        jitter = np.random.normal(position - width * 0.55, width * 0.06, len(valid))
        ax.scatter(jitter, valid, s=8, color=color, alpha=0.5)
        if title:
            ax.set_title(f"{title} Raincloud")

    # ── distribution metrics bar ──────────────────────────────────────────────
    @staticmethod
    def metrics_bar(
        ax: Axes,
        metrics: List[Dict],
        *,
        config: Any | None = None,
        title: str = "Distribution Metrics",
        **kw: Any,
    ) -> None:
        """Grouped bar chart of Wasserstein / Energy / KL per key × model."""
        cfg = config or PlotConfig()
        names = ["Wasserstein", "Energy", "KL"]
        n, w = len(metrics), 0.25
        x = np.arange(n)
        for mi, mname in enumerate(names):
            ax.bar(
                x + mi * w,
                [m[mname] for m in metrics],
                w,
                label=mname,
                color=cfg.color(mi),
                **kw,
            )
        ax.set_xticks(x + w)
        ax.set_xticklabels(
            [
                f"{m['Key']}\n{m.get('ModelName', 'M' + str(m['Model']))}"
                for m in metrics
            ],
            rotation=30,
            ha="right",
            fontsize=8,
        )
        ax.legend(frameon=False)
        ax.set_title(title)

    # ── name → method dispatcher ──────────────────────────────────────────────
    _NAME_MAP: Dict[str, str] = {
        "map": "map",
        "imshow": "map",
        "hist": "histogram",
        "histogram": "histogram",
        "log_hist": "log_histogram",
        "loghist": "log_histogram",
        "log_histogram": "log_histogram",
        "kde": "kde",
        "violin": "violinplot",
        "violinplot": "violinplot",
        "box": "boxplot",
        "boxplot": "boxplot",
        "cdf": "cdf",
        "qq": "qq",
        "qqplot": "qq",
        "scatter": "scatter",
        "raincloud": "raincloud",
        "metrics_bar": "metrics_bar",
    }

    @classmethod
    def get_method(cls, name: str) -> Callable:
        """
        Return method.

        Parameters
        ----------
        name : str
            Dataset, experiment, figure, or output name.

        Returns
        -------
        Callable
            Object produced by get method.
        """
        key = name.strip().lower()
        canonical = cls._NAME_MAP.get(key)
        if canonical is None:
            raise ValueError(
                f"Unknown plot type '{name}'. Available: {sorted(cls._NAME_MAP)}"
            )
        return getattr(cls, canonical)


# ─────────────────────────────────────────────────────────────────────────────
# SubplotVisualizer  –  spatial grid orchestrator
# ─────────────────────────────────────────────────────────────────────────────


class SubplotVisualizer:
    """
    Build compact subplot grids for spatial maps and one-dimensional distributions. It
    is useful when comparing several operations or plot types over a shared set of
    arrays.

    Parameters
    ----------
    config : Optional[PlotConfig]
        Plotting or processing configuration object.
    **kw : Any
        Additional keyword arguments forwarded to the underlying implementation.
    """

    def __init__(self, config: Optional[PlotConfig] = None, **kw: Any) -> None:
        if config is not None:
            self.config = config
        else:
            fields = PlotConfig.__dataclass_fields__
            self.config = PlotConfig(**{k: v for k, v in kw.items() if k in fields})

    def plot(
        self,
        *data_arrays: Any,
        plot_types: Sequence[str] = ("map", "histogram", "violinplot", "boxplot"),
        titles: np.ndarray | None = None,
        operations: np.ndarray | None = None,
        fig: Any | None = None,
        axes: Any | None = None,
    ) -> Figure:
        """
        Run the plot routine.

        Parameters
        ----------
        *data_arrays : Any
            Arrays used to compute shared plot ranges.
        plot_types : Sequence[str]
            Plot families requested by the caller.
        titles : np.ndarray | None
            Subplot titles displayed by the visualizer.
        operations : np.ndarray | None
            Processing operations applied before plotting or fitting.
        fig : Any | None
            Matplotlib figure object to update or save.
        axes : Any | None
            Matplotlib axes collection used for drawing subplots.

        Returns
        -------
        Figure
            Matplotlib figure generated by plot.
        """
        n = len(data_arrays)
        titles = titles or [f"Data {i + 1}" for i in range(n)]
        operations = operations or [{} for _ in range(n)]

        if fig is None or axes is None:
            fig, axes = plt.subplots(
                n, len(plot_types), figsize=self.config.figsize, squeeze=False
            )

        vmin, vmax = self._global_range(data_arrays, operations)

        for row, (data, title, ops) in enumerate(zip(data_arrays, titles, operations)):
            dm, valid = DataProcessor.process(data, ops)
            raw = np.array(data, dtype=float)
            display = raw if self.config.imshow_source == "raw" else dm

            for col, ptype in enumerate(plot_types):
                self._render(
                    axes[row, col],
                    ptype,
                    display,
                    valid,
                    title,
                    fig,
                    vmin,
                    vmax,
                    row,
                    data_arrays,
                    operations,
                )

        if self.config.shared_colorbar:
            self._shared_cbar(fig, axes, plot_types)

        fig.tight_layout()
        return fig

    def _render(
        self,
        ax: Any,
        ptype: np.ndarray,
        data_map: np.ndarray,
        valid: np.ndarray,
        title: str,
        fig: Any,
        vmin: np.ndarray,
        vmax: np.ndarray,
        row: np.ndarray,
        data_arrays: np.ndarray,
        operations: np.ndarray,
    ) -> None:
        """
        Run the render routine.

        Parameters
        ----------
        ax : Any
            Matplotlib axes object on which the plot is drawn.
        ptype : np.ndarray
            Plot type selected for a rendered subplot.
        data_map : np.ndarray
            Parameter or mask map processed by the routine.
        valid : np.ndarray
            Finite one-dimensional sample values used by a plot or statistic.
        title : str
            Title displayed on the generated plot.
        fig : Any
            Matplotlib figure object to update or save.
        vmin : np.ndarray
            Lower color-limit value.
        vmax : np.ndarray
            Upper color-limit value.
        row : np.ndarray
            Subplot row index used during rendering.
        data_arrays : np.ndarray
            Arrays used to compute shared plot ranges.
        operations : np.ndarray
            Processing operations applied before plotting or fitting.

        Returns
        -------
        None
            No object is returned; the function perform render.
        """
        key = ptype.strip().lower()
        if key in ("map", "imshow"):
            PlotKit.map(
                ax,
                data_map,
                config=self.config,
                title=title,
                vmin=vmin,
                vmax=vmax,
                fig=fig,
                add_colorbar=not self.config.shared_colorbar,
            )
        elif key == "scatter":
            pair = self.config.scatter_pair
            if pair is None:
                ax.text(
                    0.5,
                    0.5,
                    "scatter_pair not set",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            else:
                i, j = pair
                _, vi = DataProcessor.process(data_arrays[i], operations[i])
                _, vj = DataProcessor.process(data_arrays[j], operations[j])
                m = min(len(vi), len(vj))
                PlotKit.scatter(ax, vi[:m], vj[:m], config=self.config, title=title)
        else:
            PlotKit.get_method(key)(ax, valid, config=self.config, title=title)

        if self.config.annotate_stats and len(valid):
            st = DataProcessor.stats(valid)
            if st:
                ax.text(
                    0.97,
                    0.97,
                    f"μ={st['mean']:.3g}  σ={st['std']:.3g}\nn={st['n']}",
                    transform=ax.transAxes,
                    fontsize=7,
                    va="top",
                    ha="right",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.6),
                )

    def _global_range(
        self, data_arrays: np.ndarray, operations: np.ndarray
    ) -> tuple[Any, ...]:
        """
        Run the global range routine.

        Parameters
        ----------
        data_arrays : np.ndarray
            Arrays used to compute shared plot ranges.
        operations : np.ndarray
            Processing operations applied before plotting or fitting.

        Returns
        -------
        tuple[Any, ...]
            Tuple containing global finite value range across selected datasets.
        """
        if not self.config.shared_colorbar:
            return None, None
        all_v = []
        for d, o in zip(data_arrays, operations):
            _, v = DataProcessor.process(d, o)
            if len(v):
                all_v.append(v)
        if not all_v:
            return None, None
        m = np.concatenate(all_v)
        return float(m.min()), float(m.max())

    @staticmethod
    def _shared_cbar(fig: Any, axes: Any, plot_types: np.ndarray) -> None:
        """
        Run the shared cbar routine.

        Parameters
        ----------
        fig : Any
            Matplotlib figure object to update or save.
        axes : Any
            Matplotlib axes collection used for drawing subplots.
        plot_types : np.ndarray
            Plot families requested by the caller.

        Returns
        -------
        None
            No object is returned; the function perform shared cbar.
        """
        map_cols = [
            c
            for c, pt in enumerate(plot_types)
            if pt.strip().lower() in ("map", "imshow")
        ]
        if not map_cols:
            return
        imgs = [ax.images[0] for ax in axes[:, map_cols].ravel() if ax.images]
        if imgs:
            fig.colorbar(imgs[0], ax=axes[:, map_cols].ravel().tolist())


# ─────────────────────────────────────────────────────────────────────────────
# Plotter  –  multi-source comparison orchestrator
# ─────────────────────────────────────────────────────────────────────────────


class Plotter:
    """
    Run the plotter routine.
    class cleans data, applies processing operations, dispatches plot types, annotates
    significance, and exports underlying data.

    Parameters
    ----------
    *args : Any
        Additional positional values accepted by the object.
    values : np.ndarray | None
        Explicit values to load as plotting sources.
    style_config : np.ndarray | None
        Plot configuration object controlling colors, layout, and statistics.
    source_names : np.ndarray | None
        Names assigned to plotted or compared data sources.
    operations : np.ndarray | None
        List of plotting or analysis operations to execute.
    """

    def __init__(
        self,
        *args: Any,
        values: np.ndarray | None = None,
        style_config: np.ndarray | None = None,
        source_names: np.ndarray | None = None,
        operations: np.ndarray | None = None,
    ) -> None:
        self.raw_data = args
        self.source_names = source_names or [
            f"Source {i + 1}" for i in range(len(args))
        ]
        self.stats_results: List[Dict] = []
        self.current_fig: Optional[Figure] = None

        # operations: None | dict | list[dict] (one per source)
        self._operations = operations

        # ── resolve config ────────────────────────────────────────────────────
        if isinstance(style_config, PlotConfig):
            self.config = style_config
        else:
            colors = (
                list(style_config.values())
                if isinstance(style_config, dict)
                else style_config
                if isinstance(style_config, list)
                else ["#3498db", "#e74c3c", "#2ecc71", "#f1c40f", "#9b59b6"]
            )
            self.config = PlotConfig(colors=colors)

        self._loader = SourceLoader(
            *args, values=values, source_names=self.source_names
        )
        self.labels = self._loader.labels

        # kept for backward compatibility with any code that calls _clean_data
        self.values = values
        self._clean_data()

    def _get_clean_array(self, data_source: Any, key: str) -> np.ndarray:
        """Backward-compatible extraction helper."""
        try:
            val = data_source[key]
            return np.asanyarray(val).astype(float).flatten()
        except (KeyError, ValueError, TypeError):
            return np.array([])

    def _clean_data(self) -> None:
        """Backward-compatible label inference (no-op: SourceLoader already did it)."""
        if self.values:
            self.labels = list(self.values)
        elif not self.labels:
            all_keys = []
            for data in self.raw_data:
                if hasattr(data, "files"):
                    all_keys.extend(data.files)
                elif isinstance(data, dict):
                    all_keys.extend(data.keys())
            self.labels = list(dict.fromkeys(all_keys))
            self._loader.labels = self.labels

    def _apply_processing(
        self,
        groups: Dict[str, List[np.ndarray]],
    ) -> Dict[str, List[np.ndarray]]:
        """Run DataProcessor on every array in *groups*.

        self._operations may be:
          None        – no-op; returns groups unchanged
          dict        – same ops applied to every source's array
          list[dict]  – one ops dict per source (index-matched)
        """
        if not self._operations:
            return groups
        processed: Dict[str, List[np.ndarray]] = {k: [] for k in groups}
        for key, arrs in groups.items():
            for i, arr in enumerate(arrs):
                ops = (
                    self._operations[i]
                    if isinstance(self._operations, list)
                    else self._operations
                )
                _, valid = DataProcessor.process(arr, ops)
                if len(valid):
                    processed[key].append(valid)
        return processed

    # ── main plotting method ──────────────────────────────────────────────────
    def make_plot(
        self,
        title: str = "Data Analysis",
        graph_type: str = "box",
        show_significance: bool = True,
        # legacy positional-style kwargs kept for backward compatibility
        point_type: np.ndarray | None = None,
        show_mean: np.ndarray | None = None,
        show_median: np.ndarray | None = None,
        test_type: np.ndarray | None = None,
        correction: np.ndarray | None = None,
        **config_overrides: Any,
    ) -> Any:
        """Render a multi-source comparison plot.

        Legacy parameters (point_type, show_mean, show_median, test_type,
        correction) are accepted directly as well as via config_overrides
        so existing call-sites continue to work unchanged.
        """
        # merge legacy explicit kwargs into config_overrides
        legacy = {
            k: v
            for k, v in [
                ("point_type", point_type),
                ("show_mean", show_mean),
                ("show_median", show_median),
                ("test_type", test_type),
                ("correction", correction),
            ]
            if v is not None
        }
        config_overrides = {**legacy, **config_overrides}

        unknown = {
            k for k in config_overrides if k not in PlotConfig.__dataclass_fields__
        }
        if unknown:
            raise TypeError(
                f"make_plot() got unexpected keyword arguments: {sorted(unknown)}"
            )

        cfg = (
            dc_replace(self.config, **config_overrides)
            if config_overrides
            else self.config
        )

        self.stats_results = []
        groups = self._apply_processing(self._loader.load())
        n_sources = len(self.raw_data)

        # ── dispatch graph types handled with per-key subplots ────────────────
        if graph_type == "kde":
            return self._plot_kde(groups, n_sources, title, cfg)
        if graph_type == "cdf":
            return self._plot_cdf(groups, n_sources, title, cfg)
        if graph_type == "qq":
            return self._plot_qq(groups, n_sources, title, cfg)

        # ── single-axes graph types ───────────────────────────────────────────
        fig, ax = plt.subplots(figsize=cfg.figsize)
        width = 0.6 / n_sources
        x_centers = np.arange(len(self.labels))

        if graph_type in ("box", "swarm", "overlay"):
            self._plot_box_family(
                ax, groups, n_sources, width, x_centers, graph_type, cfg
            )
        elif graph_type in ("violin", "raincloud"):
            self._plot_violin_family(
                ax, groups, n_sources, width, x_centers, graph_type, cfg
            )

        ax.set_xticks(x_centers)
        ax.set_xticklabels(self.labels)

        if show_significance and cfg.test_type.lower() != "none" and n_sources >= 2:
            self._annotate_significance(ax, groups, n_sources, x_centers, width, cfg)
        self._add_legend(ax, n_sources, cfg)
        ax.set_title(title)
        plt.tight_layout()
        self.current_fig = fig
        return fig

    # ── per-type rendering helpers ────────────────────────────────────────────
    def _plot_kde(
        self, groups: np.ndarray, n_sources: int, title: str, cfg: Any
    ) -> np.ndarray:
        """
        Plot kde.

        Parameters
        ----------
        groups : np.ndarray
            Grouped valid samples plotted by the comparison routine.
        n_sources : int
            Number of samples, components, gates, or iterations used by the routine.
        title : str
            Title displayed on the generated plot.
        cfg : Any
            Configuration object or keyword dictionary used by the algorithm.

        Returns
        -------
        np.ndarray
            Matplotlib figure or axes containing the kernel-density estimate.
        """
        n_keys = len(self.labels)
        fig, axes = plt.subplots(
            n_keys, 1, figsize=(cfg.figsize[0], 4 * n_keys), sharex=False
        )
        if n_keys == 1:
            axes = [axes]
        for idx, key in enumerate(self.labels):
            ax = axes[idx]
            for i in range(n_sources):
                if len(groups[key]) > i:
                    PlotKit.kde(
                        ax,
                        groups[key][i],
                        color=cfg.color(i),
                        label=self.source_names[i],
                        fill=True,
                        alpha=0.35,
                    )
            ax.set_title(key)
            ax.legend(frameon=False, fontsize=8)
        plt.suptitle(title)
        plt.tight_layout()
        plt.subplots_adjust(right=0.82)
        self.current_fig = fig
        return fig

    def _plot_cdf(
        self, groups: np.ndarray, n_sources: int, title: str, cfg: Any
    ) -> np.ndarray:
        """
        Plot cdf.

        Parameters
        ----------
        groups : np.ndarray
            Grouped valid samples plotted by the comparison routine.
        n_sources : int
            Number of samples, components, gates, or iterations used by the routine.
        title : str
            Title displayed on the generated plot.
        cfg : Any
            Configuration object or keyword dictionary used by the algorithm.

        Returns
        -------
        np.ndarray
            Matplotlib figure or axes containing the cumulative distribution plot.
        """
        n_keys = len(self.labels)
        fig, axes = plt.subplots(1, n_keys, figsize=cfg.figsize, squeeze=False)
        for idx, key in enumerate(self.labels):
            ax = axes[0, idx]
            for i in range(n_sources):
                if len(groups[key]) > i:
                    PlotKit.cdf(
                        ax,
                        groups[key][i],
                        color=cfg.color(i),
                        label=self.source_names[i],
                        title=key,
                    )
            ax.legend(frameon=False, fontsize=8)
        plt.suptitle(title)
        plt.tight_layout()
        self.current_fig = fig
        return fig

    def _plot_qq(
        self, groups: np.ndarray, n_sources: int, title: str, cfg: Any
    ) -> np.ndarray:
        """
        Plot qq.

        Parameters
        ----------
        groups : np.ndarray
            Grouped valid samples plotted by the comparison routine.
        n_sources : int
            Number of samples, components, gates, or iterations used by the routine.
        title : str
            Title displayed on the generated plot.
        cfg : Any
            Configuration object or keyword dictionary used by the algorithm.

        Returns
        -------
        np.ndarray
            Matplotlib figure or axes containing the quantile-quantile plot.
        """
        n_keys = len(self.labels)
        fig, axes = plt.subplots(
            n_keys, n_sources, figsize=(4 * n_sources, 4 * n_keys), squeeze=False
        )
        for row, key in enumerate(self.labels):
            for col in range(n_sources):
                if len(groups[key]) > col:
                    PlotKit.qq(
                        axes[row, col],
                        groups[key][col],
                        config=cfg,
                        title=f"{key} / {self.source_names[col]}",
                    )
        plt.suptitle(title)
        plt.tight_layout()
        self.current_fig = fig
        return fig

    def _plot_box_family(
        self,
        ax: Any,
        groups: np.ndarray,
        n_sources: int,
        width: float,
        x_centers: np.ndarray,
        graph_type: np.ndarray,
        cfg: Any,
    ) -> None:
        """
        Plot box family.

        Parameters
        ----------
        ax : Any
            Matplotlib axes object on which the plot is drawn.
        groups : np.ndarray
            Grouped valid samples plotted by the comparison routine.
        n_sources : int
            Number of samples, components, gates, or iterations used by the routine.
        width : float
            Gate width used by the gate matrix.
        x_centers : np.ndarray
            X positions used to place grouped box or violin plots.
        graph_type : np.ndarray
            Mode or type selector used by the routine.
        cfg : Any
            Configuration object or keyword dictionary used by the algorithm.

        Returns
        -------
        None
            No object is returned; the function plot box family.
        """
        for src in range(n_sources):
            color = cfg.color(src)
            positions, data_list = [], []
            for idx, key in enumerate(self.labels):
                if len(groups[key]) > src:
                    offset = (src - (n_sources - 1) / 2) * width * 1.2
                    positions.append(x_centers[idx] + offset)
                    data_list.append(groups[key][src])
            if not data_list:
                continue

            if graph_type in ("box", "overlay"):
                ax.boxplot(
                    data_list,
                    positions=positions,
                    widths=width * 0.9,
                    patch_artist=True,
                    showfliers=False,
                    boxprops=dict(facecolor=color, alpha=0.5),
                    medianprops=dict(color="black"),
                )

            if graph_type in ("swarm", "overlay"):
                for pos, arr in zip(positions, data_list):
                    samp = (
                        arr
                        if len(arr) < 250
                        else np.random.choice(arr, 250, replace=False)
                    )
                    if cfg.point_type == "swarm":
                        sns.swarmplot(
                            x=np.repeat(pos, len(samp)),
                            y=samp,
                            ax=ax,
                            color=color,
                            size=4,
                            edgecolor="black",
                            linewidth=0.4,
                        )
                    else:
                        ax.scatter(
                            np.random.normal(pos, width * 0.08, len(samp)),
                            samp,
                            s=12,
                            color=color,
                            alpha=0.6,
                        )

            for pos, arr in zip(positions, data_list):
                if cfg.show_mean:
                    ax.scatter(
                        pos,
                        np.nanmean(arr),
                        color="white",
                        edgecolor="black",
                        s=40,
                        zorder=5,
                    )
                if cfg.show_median:
                    ax.hlines(
                        np.nanmedian(arr),
                        pos - width / 3,
                        pos + width / 3,
                        color="black",
                        lw=2,
                    )

    def _plot_violin_family(
        self,
        ax: Any,
        groups: np.ndarray,
        n_sources: int,
        width: float,
        x_centers: np.ndarray,
        graph_type: np.ndarray,
        cfg: Any,
    ) -> None:
        """
        Plot violin family.

        Parameters
        ----------
        ax : Any
            Matplotlib axes object on which the plot is drawn.
        groups : np.ndarray
            Grouped valid samples plotted by the comparison routine.
        n_sources : int
            Number of samples, components, gates, or iterations used by the routine.
        width : float
            Gate width used by the gate matrix.
        x_centers : np.ndarray
            X positions used to place grouped box or violin plots.
        graph_type : np.ndarray
            Mode or type selector used by the routine.
        cfg : Any
            Configuration object or keyword dictionary used by the algorithm.

        Returns
        -------
        None
            No object is returned; the function plot violin family.
        """
        for src in range(n_sources):
            color = cfg.color(src)
            for idx, key in enumerate(self.labels):
                if len(groups[key]) <= src:
                    continue
                arr = groups[key][src]
                offset = (src - (n_sources - 1) / 2) * width * 1.2
                pos = x_centers[idx] + offset

                if graph_type == "violin":
                    v = ax.violinplot(
                        arr,
                        positions=[pos],
                        widths=width,
                        showmeans=False,
                        showmedians=False,
                        showextrema=True,
                    )
                    for part in ("cbars", "cmins", "cmaxes", "cmedians"):
                        if part in v:
                            v[part].set_edgecolor("black")
                            v[part].set_linewidth(1.2)
                    for body in v["bodies"]:
                        body.set_facecolor(color)
                        body.set_alpha(0.6)
                        body.set_edgecolor("black")
                        body.set_linewidth(1.2)

                elif graph_type == "raincloud":
                    PlotKit.raincloud(
                        ax, arr, config=cfg, color=color, position=pos, width=width
                    )

                if cfg.show_mean:
                    ax.scatter(
                        pos,
                        np.nanmean(arr),
                        color="white",
                        edgecolor="black",
                        s=40,
                        zorder=6,
                    )
                if cfg.show_median:
                    ax.hlines(
                        np.nanmedian(arr),
                        pos - width / 3,
                        pos + width / 3,
                        color="black",
                        lw=2,
                        zorder=6,
                    )

    def _annotate_significance(
        self,
        ax: Any,
        groups: np.ndarray,
        n_sources: int,
        x_centers: np.ndarray,
        width: float,
        cfg: Any,
    ) -> None:
        """
        Run the annotate significance routine.

        Parameters
        ----------
        ax : Any
            Matplotlib axes object on which the plot is drawn.
        groups : np.ndarray
            Grouped valid samples plotted by the comparison routine.
        n_sources : int
            Number of samples, components, gates, or iterations used by the routine.
        x_centers : np.ndarray
            X positions used to place grouped box or violin plots.
        width : float
            Gate width used by the gate matrix.
        cfg : Any
            Configuration object or keyword dictionary used by the algorithm.

        Returns
        -------
        None
            No object is returned; the function perform annotate significance.
        """
        p_val_text = []
        num_comps = len(self.labels) * (n_sources - 1) if cfg.correction else 1
        for idx, key in enumerate(self.labels):
            if len(groups[key]) < 2:
                continue
            s1 = groups[key][0]
            ymin, ymax = ax.get_ylim()
            star_y = ymax - 0.05 * (ymax - ymin)
            for src in range(1, n_sources):
                s2 = groups[key][src]
                if cfg.test_type.lower() == "paired":
                    m = min(len(s1), len(s2))
                    _, p = stats.ttest_rel(s1[:m], s2[:m])
                else:
                    _, p = stats.ttest_ind(s1, s2, equal_var=False)
                adj_p = min(1.0, p * num_comps)
                sig = (
                    "***"
                    if adj_p < 0.001
                    else "**"
                    if adj_p < 0.01
                    else "*"
                    if adj_p < 0.05
                    else "NS"
                )
                offset = (src - (n_sources - 1) / 2) * width * 1.2
                ax.text(
                    x_centers[idx] + offset,
                    star_y,
                    sig,
                    ha="center",
                    fontsize=11,
                    fontweight="bold",
                )
                p_val_text.append(f"{key} vs S{src + 1}: {adj_p:.2e}")
                self.stats_results.append({"Key": key, "Source": src + 1, "P": adj_p})
        if p_val_text:
            ax.text(
                1.02,
                0.4,
                "P-Values:\n" + "\n".join(p_val_text),
                transform=ax.transAxes,
                fontsize=8,
                bbox=dict(
                    boxstyle="round", facecolor="none", edgecolor="none", alpha=0.4
                ),
            )

    def _add_legend(self, ax: Any, n_sources: int, cfg: Any) -> None:
        """
        Add legend.

        Parameters
        ----------
        ax : Any
            Matplotlib axes object on which the plot is drawn.
        n_sources : int
            Number of samples, components, gates, or iterations used by the routine.
        cfg : Any
            Configuration object or keyword dictionary used by the algorithm.

        Returns
        -------
        None
            No object is returned; the function add legend.
        """
        elements = [
            Line2D(
                [0],
                [0],
                marker="s",
                color=cfg.color(i),
                label=self.source_names[i],
                markersize=10,
                linestyle="None",
            )
            for i in range(n_sources)
        ]
        ax.legend(
            handles=elements, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False
        )

    # ── cluster comparison ────────────────────────────────────────────────────
    def make_cluster_plot(
        self,
        multi_cluster_mask: np.ndarray,
        cluster_names: Optional[List[str]] = None,
        title: str = "Cluster Analysis",
        graph_type: str = "box",
        show_significance: bool = True,
        point_type: np.ndarray | None = None,
        show_mean: np.ndarray | None = None,
        show_median: np.ndarray | None = None,
        test_type: np.ndarray | None = None,
        correction: np.ndarray | None = None,
        **config_overrides: Any,
    ) -> Figure:
        """Per-cluster breakdown of make_plot.

        For every key in ``self.labels`` one subplot is drawn; within each
        subplot the **x-axis represents cluster IDs** and grouped
        boxes/violins/etc. represent **data sources** (color-coded by source).

        Parameters
        ----------
        multi_cluster_mask : 2-D int array (H, W)
            0 = background (ignored); 1, 2, 3 … = cluster IDs.
        cluster_names : list[str], optional
            Display labels for each cluster. Auto-generated when *None*.
        All remaining parameters are identical to ``make_plot``.

        Supported graph_type values: ``"box"``, ``"overlay"``, ``"swarm"``,
        ``"violin"``, ``"raincloud"``.
        """
        # ── resolve config (same merge logic as make_plot) ───────────────────
        legacy = {
            k: v
            for k, v in [
                ("point_type", point_type),
                ("show_mean", show_mean),
                ("show_median", show_median),
                ("test_type", test_type),
                ("correction", correction),
            ]
            if v is not None
        }
        config_overrides = {**legacy, **config_overrides}

        unknown = {
            k for k in config_overrides if k not in PlotConfig.__dataclass_fields__
        }
        if unknown:
            raise TypeError(
                f"make_cluster_plot() got unexpected keyword arguments: "
                f"{sorted(unknown)}"
            )

        cfg = (
            dc_replace(self.config, **config_overrides)
            if config_overrides
            else self.config
        )

        self.stats_results = []

        # ── cluster IDs ──────────────────────────────────────────────────────
        mask_arr = np.asarray(multi_cluster_mask)
        unique_ids = sorted(int(c) for c in np.unique(mask_arr) if c != 0)
        if not unique_ids:
            raise ValueError("multi_cluster_mask contains no non-zero cluster IDs.")

        if cluster_names is None:
            cluster_names = [f"Cluster {cid}" for cid in unique_ids]
        if len(cluster_names) != len(unique_ids):
            raise ValueError(
                "cluster_names length must match the number of cluster IDs."
            )

        n_clusters = len(unique_ids)
        n_sources = len(self.raw_data)
        n_keys = len(self.labels)

        # ── extract per-cluster pixel arrays ─────────────────────────────────
        # groups[key][cluster_idx] = [arr_src0, arr_src1, ...]
        groups: Dict[str, List[List[np.ndarray]]] = {}
        for key in self.labels:
            groups[key] = [[] for _ in unique_ids]
            for src_idx, src in enumerate(self.raw_data):
                try:
                    raw = np.asanyarray(src[key]).astype(float)
                except (KeyError, TypeError):
                    for ci in range(n_clusters):
                        groups[key][ci].append(np.array([]))
                    continue

                # per-source operations: strip 'mask' (cluster mask takes over)
                if isinstance(self._operations, list):
                    ops_raw = (
                        self._operations[src_idx]
                        if src_idx < len(self._operations)
                        else {}
                    )
                else:
                    ops_raw = self._operations or {}
                ops = {k: v for k, v in ops_raw.items() if k != "mask"}

                for ci, cid in enumerate(unique_ids):
                    cluster_px = mask_arr == cid
                    vals = (
                        raw[cluster_px]
                        if raw.shape == mask_arr.shape
                        else raw.flatten()
                    )
                    _, valid = DataProcessor.process(vals.reshape(-1), ops)
                    groups[key][ci].append(valid)

        # ── layout: one subplot per key ───────────────────────────────────────
        col_w = max(cfg.figsize[0] / max(n_keys, 1), 4)
        fig, axes = plt.subplots(
            1,
            n_keys,
            figsize=(col_w * n_keys, cfg.figsize[1]),
            squeeze=False,
        )
        axes = axes[0]  # shape (n_keys,)

        x_centers = np.arange(n_clusters, dtype=float)
        width = 0.6 / max(n_sources, 1)

        for key_idx, key in enumerate(self.labels):
            ax = axes[key_idx]

            for src_idx in range(n_sources):
                color = cfg.color(src_idx)
                positions = []
                data_list = []

                for ci in range(n_clusters):
                    src_arrs = groups[key][ci]
                    arr = src_arrs[src_idx] if src_idx < len(src_arrs) else np.array([])
                    if len(arr):
                        offset = (src_idx - (n_sources - 1) / 2) * width * 1.2
                        positions.append(x_centers[ci] + offset)
                        data_list.append(arr)

                if not data_list:
                    continue

                if graph_type in ("box", "overlay"):
                    ax.boxplot(
                        data_list,
                        positions=positions,
                        widths=width * 0.9,
                        patch_artist=True,
                        showfliers=False,
                        boxprops=dict(facecolor=color, alpha=0.5),
                        medianprops=dict(color="black"),
                    )

                if graph_type in ("swarm", "overlay"):
                    for pos, arr in zip(positions, data_list):
                        samp = (
                            arr
                            if len(arr) < 250
                            else np.random.choice(arr, 250, replace=False)
                        )
                        if cfg.point_type == "swarm":
                            sns.swarmplot(
                                x=np.repeat(pos, len(samp)),
                                y=samp,
                                ax=ax,
                                color=color,
                                size=4,
                                edgecolor="black",
                                linewidth=0.4,
                            )
                        else:
                            ax.scatter(
                                np.random.normal(pos, width * 0.08, len(samp)),
                                samp,
                                s=12,
                                color=color,
                                alpha=0.6,
                            )

                if graph_type == "violin":
                    for pos, arr in zip(positions, data_list):
                        if DataProcessor.is_valid(arr):
                            v = ax.violinplot(
                                arr,
                                positions=[pos],
                                widths=width,
                                showmeans=False,
                                showmedians=False,
                                showextrema=True,
                            )
                            for part in ("cbars", "cmins", "cmaxes"):
                                if part in v:
                                    v[part].set_edgecolor("black")
                                    v[part].set_linewidth(1.2)
                            for body in v["bodies"]:
                                body.set_facecolor(color)
                                body.set_alpha(0.6)
                                body.set_edgecolor("black")
                                body.set_linewidth(1.2)

                if graph_type == "raincloud":
                    for pos, arr in zip(positions, data_list):
                        PlotKit.raincloud(
                            ax, arr, config=cfg, color=color, position=pos, width=width
                        )

                for pos, arr in zip(positions, data_list):
                    if cfg.show_mean:
                        ax.scatter(
                            pos,
                            np.nanmean(arr),
                            color="white",
                            edgecolor="black",
                            s=40,
                            zorder=5,
                        )
                    if cfg.show_median:
                        ax.hlines(
                            np.nanmedian(arr),
                            pos - width / 3,
                            pos + width / 3,
                            color="black",
                            lw=2,
                        )

            ax.set_xticks(x_centers)
            ax.set_xticklabels(cluster_names, rotation=20, ha="right")
            ax.set_title(key)

            if show_significance and cfg.test_type.lower() != "none" and n_sources >= 2:
                self._annotate_cluster_significance(
                    ax, groups[key], n_sources, x_centers, width, cfg
                )

        self._add_legend(axes[-1], n_sources, cfg)
        fig.suptitle(title, fontsize=13, fontweight="bold")
        plt.tight_layout()
        self.current_fig = fig
        return fig

    def _annotate_cluster_significance(
        self,
        ax: Any,
        key_groups: np.ndarray,
        n_sources: int,
        x_centers: np.ndarray,
        width: float,
        cfg: Any,
    ) -> None:
        """Significance stars between sources at each cluster position."""
        n_clusters = len(key_groups)
        num_comps = n_clusters * (n_sources - 1) if cfg.correction else 1
        ymin, ymax = ax.get_ylim()
        star_y = ymax - 0.05 * (ymax - ymin)

        for ci in range(n_clusters):
            if not key_groups[ci]:
                continue
            s1 = key_groups[ci][0]
            for src in range(1, n_sources):
                if src >= len(key_groups[ci]):
                    continue
                s2 = key_groups[ci][src]
                if len(s1) < 3 or len(s2) < 3:
                    continue
                if cfg.test_type.lower() == "paired":
                    m = min(len(s1), len(s2))
                    _, p = stats.ttest_rel(s1[:m], s2[:m])
                else:
                    _, p = stats.ttest_ind(s1, s2, equal_var=False)
                adj_p = min(1.0, p * num_comps)
                sig = (
                    "***"
                    if adj_p < 0.001
                    else "**"
                    if adj_p < 0.01
                    else "*"
                    if adj_p < 0.05
                    else "NS"
                )
                offset = (src - (n_sources - 1) / 2) * width * 1.2
                ax.text(
                    x_centers[ci] + offset,
                    star_y,
                    sig,
                    ha="center",
                    fontsize=9,
                    fontweight="bold",
                )
                self.stats_results.append(
                    {"Cluster": ci, "Source": src + 1, "P": adj_p}
                )

    # ── export ─────────────────────────────────────────────────────────────────
    def export_data(
        self,
        save_pdf: bool = False,
        save_png: bool = False,
        save_csv: bool = False,
        filename: str = "results",
        dpi: int = 150,
    ) -> None:
        """
        Export data.

        Parameters
        ----------
        save_pdf : bool
            If ``True``, save the generated figures to a PDF file.
        save_png : bool
            Whether to export the figure as PNG.
        save_csv : bool
            Whether to export comparison data as CSV.
        filename : str
            File name used for saving or loading results.
        dpi : int
            Resolution used when saving a figure.

        Returns
        -------
        None
            No object is returned; the function export data.
        """
        if self.current_fig is not None:
            if save_pdf:
                self.current_fig.savefig(
                    f"{filename}.pdf", format="pdf", bbox_inches="tight"
                )
            if save_png:
                self.current_fig.savefig(
                    f"{filename}.png", dpi=dpi, bbox_inches="tight"
                )
        if save_csv and self.stats_results:
            pd.DataFrame(self.stats_results).to_csv(f"{filename}.csv", index=False)


# ─────────────────────────────────────────────────────────────────────────────
# DLModelComparator  –  subclasses Plotter, enhanced
# ─────────────────────────────────────────────────────────────────────────────


class DLModelComparator(Plotter):
    """
    Extend the general plotter with distribution-distance metrics for model evaluation.
    It computes Wasserstein, KL, and energy-style summaries and can annotate those
    metrics on comparison figures.
    """

    def compute_distribution_metrics(self) -> List[Dict]:
        """
        Compute distribution metrics.

        Returns
        -------
        List[Dict]
            Object produced by compute distribution metrics.
        """
        groups = self._apply_processing(self._loader.load())
        results = []
        for key in self.labels:
            if not groups[key]:
                continue
            gt = groups[key][0]
            for i in range(1, len(self.raw_data)):
                if len(groups[key]) <= i:
                    continue
                model = groups[key][i]
                m_len = min(len(gt), len(model))
                g, m = gt[:m_len], model[:m_len]

                w = wasserstein_distance(g, m)
                e = energy_distance(g, m)
                h_gt, bins = np.histogram(g, bins=50, density=True)
                h_m, _ = np.histogram(m, bins=bins, density=True)
                kl = entropy(h_gt + 1e-10, h_m + 1e-10)

                results.append(
                    {
                        "Key": key,
                        "Model": i,
                        "ModelName": self.source_names[i],
                        "Wasserstein": w,
                        "Energy": e,
                        "KL": kl,
                    }
                )
        return results

    def annotate_distribution_metrics(self, ax: Axes) -> None:
        """
        Run the annotate distribution metrics routine.

        Parameters
        ----------
        ax : Axes
            Matplotlib axes object on which the plot is drawn.

        Returns
        -------
        None
            No object is returned; the function perform annotate distribution metrics.
        """
        metrics = self.compute_distribution_metrics()
        lines = [
            f"{m['Key']} / {m['ModelName']}: "
            f"W={m['Wasserstein']:.3f}, "
            f"E={m['Energy']:.3f}, "
            f"KL={m['KL']:.3f}"
            for m in metrics
        ]
        ax.text(
            1.02,
            0.12,
            "Distribution Metrics\n" + "\n".join(lines),
            transform=ax.transAxes,
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="none", edgecolor="gray", alpha=0.4),
        )

    def plot_metrics(self, title: str = "Distribution Metrics") -> Optional[Figure]:
        """Standalone bar chart of W / Energy / KL for all key × model pairs."""
        metrics = self.compute_distribution_metrics()
        if not metrics:
            logging.info("No metrics – verify data sources.")
            return None
        fig, ax = plt.subplots(figsize=self.config.figsize)
        PlotKit.metrics_bar(ax, metrics, config=self.config, title=title)
        plt.tight_layout()
        self.current_fig = fig
        return fig

    def make_plot(
        self,
        title: str = "DL Model Comparison",
        graph_type: str = "box",
        show_significance: bool = True,
        show_metrics: bool = True,
        **config_overrides: Any,
    ) -> np.ndarray:
        """Override: adds optional distribution-metrics annotation block."""
        fig = super().make_plot(
            title=title,
            graph_type=graph_type,
            show_significance=show_significance,
            **config_overrides,
        )
        if (
            show_metrics
            and self.current_fig is not None
            and graph_type not in ("kde", "cdf", "qq")
        ):
            self.annotate_distribution_metrics(self.current_fig.axes[0])
        return fig


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compatible module-level function
# ─────────────────────────────────────────────────────────────────────────────


def plot_2d_subplots(
    *data_arrays: Any,
    plot_types: tuple[str, ...] = ("map", "histogram", "violinplot", "boxplot"),
    titles: np.ndarray | None = None,
    operations: np.ndarray | None = None,
    figsize: tuple[int, ...] = (18, 8),
    cmap: str = "viridis",
    bins: int = 100,
    imshow_source: str = "processed",
    shared_colorbar: bool = False,
    annotate_stats: bool = True,
    scatter_pair: np.ndarray | None = None,
    qq_reference: str = "norm",
) -> Figure:
    """Drop-in replacement – all original keyword arguments preserved."""
    cfg = PlotConfig(
        figsize=figsize,
        cmap=cmap,
        bins=bins,
        imshow_source=imshow_source,
        shared_colorbar=shared_colorbar,
        annotate_stats=annotate_stats,
        scatter_pair=scatter_pair,
        qq_reference=qq_reference,
    )
    fig = SubplotVisualizer(config=cfg).plot(
        *data_arrays,
        plot_types=plot_types,
        titles=titles,
        operations=operations,
    )
    return fig
