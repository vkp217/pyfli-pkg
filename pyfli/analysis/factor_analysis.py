"""
factor_analysis.py
-------------------
FactorAnalysis: bins pixel-wise FLIM results by a factor and compares how
each fitting method's estimated parameters vary across that factor.

Key idea
--------
`decay`, `irf`, and `mask` are the RAW / SHARED inputs that feed every
fitting method in `all_datasets` / `all_fitset` -- they are the same arrays
regardless of which method (fbi, cpu_nlsf, cpu_mle, ...) produced a given
result. Because of that, any factor computed directly from decay/irf (e.g.
`total_photons = decay.sum(time_axis)`) is *guaranteed* to be pixel-for-pixel
identical across methods -- giving a true, common x-axis to compare methods
against. This is different from (and safer than) binning by a method's own
*estimated* `photon_count_map`, which can differ in scale/definition between
methods (that was the source of the earlier one-to-one mismatch).

Typical usage
-------------
    fa = FactorAnalysis(
        decay=binned_decay,          # shape (H, W, T) -- or wherever your time axis is
        irf=binned_irf,
        mask=b_bool_mask,            # shared 2D boolean mask
        all_datasets=all_datasets,   # list[dict] of per-method parameter maps
        all_fitset=all_fitset,       # list[dict] of per-method fit results, each with
                                      # 'fit_map' (reconstructed decay) and 'residual_map'
                                      # (decay - fit_map), same (H, W, T) shape as decay
        method_names=list(experiments.values()),
        time_axis=-1,
    )

    # link total_photons (from the shared decay) to estimated parameter maps
    df, edges = fa.analyze(factor_key='total_photons', target_keys=['tau1_map', 'tau2_map', 'chi2_map'])
    fig, axes = fa.plot(df)

    # link total_photons to fit-quality computed from fit_map/residual_map vs the raw decay
    df2, edges2 = fa.analyze(factor_key='total_photons', target_source='fitset',
                              target_keys=['residual_chi2', 'fit_total_photons'])
    fig2, axes2 = fa.plot(df2)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class FactorAnalysis:
    """
    Parameters
    ----------
    decay : ndarray
        Raw (binned) decay histogram, shape (..., T) with the time axis given
        by `time_axis` (default: last axis). All non-time axes together form
        the pixel grid, e.g. (H, W, T).
    irf : ndarray or None
        Instrument response function, shared across all methods. Same time
        convention as `decay`. Can be per-pixel (matches decay's pixel shape
        + time axis) or a smaller/1D IRF -- both are accepted and simply
        stored as-is; only used directly by default factors/targets when its
        shape matches `decay`.
    mask : ndarray or None
        Boolean mask shaped like decay's pixel grid, shared across all
        methods. True = keep pixel. None disables masking.
    all_datasets : list[dict]
        Per-method dictionaries of estimated parameter maps
        (e.g. {'tau1_map': ..., 'chi2_map': ..., ...}), one dict per method,
        same order as `method_names`.
    all_fitset : list[dict]
        Per-method fit-result dicts. Each dict must contain at least:
          - 'fit_map'      : reconstructed/predicted decay, same pixel grid +
                             time axis convention as `decay`, e.g. (H, W, T).
          - 'residual_map' : decay - fit_map (or the method's own residual
                             definition), same shape as 'fit_map'.
        Extra keys (e.g. 'sdf_map', 'convolved_map') are allowed and may
        differ across methods -- only 'fit_map'/'residual_map' are assumed to
        exist for every method. Use list_fitset_keys(i) / get_fitset_array(i, key)
        to work with any extra, method-specific keys.
        Same order as `method_names`.
    method_names : list[str]
        Label for each method, same order/length as all_datasets/all_fitset.
    time_axis : int
        Axis of `decay` (and irf/fit_map/residual_map, when per-pixel) that
        indexes time bins. Default -1 (last axis).
    """

    def __init__(
        self,
        decay,
        irf,
        mask,
        all_datasets,
        all_fitset,
        method_names,
        time_axis=-1,
        sns_style="whitegrid",
        sns_palette="colorblind",
    ):
        self.decay = np.asarray(decay)
        self.irf = np.asarray(irf) if irf is not None else None
        self.time_axis = time_axis % self.decay.ndim
        self.pixel_shape = tuple(
            s for ax, s in enumerate(self.decay.shape) if ax != self.time_axis
        )

        self.mask = None
        if mask is not None:
            m = np.asarray(mask).astype(bool)
            if m.shape != self.pixel_shape:
                raise ValueError(
                    f"mask shape {m.shape} does not match decay's pixel shape {self.pixel_shape}"
                )
            self.mask = m

        n = len(all_datasets)
        if not (len(all_fitset) == n == len(method_names)):
            raise ValueError(
                "all_datasets, all_fitset, and method_names must be the same length"
            )
        self.all_datasets = list(all_datasets)
        self.all_fitset = list(all_fitset)
        self.method_names = list(method_names)

        # all_fitset[i] is a dict per method (e.g. {'fit_map':..., 'residual_map':...,
        # possibly 'sdf_map'/'convolved_map' too -- but those aren't guaranteed across
        # every method, so only 'fit_map' and 'residual_map' are required/used here).
        self._fitset_required_keys = ("fit_map", "residual_map")
        for i, fs in enumerate(self.all_fitset):
            if not isinstance(fs, dict):
                raise ValueError(
                    f"all_fitset[{i}] ({method_names[i]}) must be a dict with keys "
                    f"{self._fitset_required_keys}, got {type(fs)}."
                )
            missing = [k for k in self._fitset_required_keys if k not in fs]
            if missing:
                raise ValueError(
                    f"all_fitset[{i}] ({method_names[i]}) is missing required key(s) "
                    f"{missing}; has keys {list(fs.keys())}."
                )
            for key in self._fitset_required_keys:
                arr = np.asarray(fs[key])
                if arr.ndim == 0:
                    raise ValueError(
                        f"all_fitset[{i}]['{key}'] ({method_names[i]}) is 0-dimensional "
                        f"(dtype={arr.dtype}, value={arr!r}); expected an ndarray shaped "
                        f"like decay's pixel grid + time axis, e.g. "
                        f"{self.pixel_shape + (self.decay.shape[self.time_axis],)}."
                    )
                arr_time_axis = self.time_axis % arr.ndim
                arr_pixel_shape = tuple(
                    s for ax, s in enumerate(arr.shape) if ax != arr_time_axis
                )
                if arr_pixel_shape != self.pixel_shape:
                    raise ValueError(
                        f"all_fitset[{i}]['{key}'] ({method_names[i]}) pixel shape "
                        f"{arr_pixel_shape} does not match decay's pixel shape {self.pixel_shape}"
                    )

        # factor maps derived from the shared decay/irf (name -> 2D array)
        self._factors = {}
        self._register_default_factors()

        # fitset-derived target functions (name -> fn(decay, fitset_dict, irf) -> 2D array)
        self._fitset_target_fns = {}
        self._register_default_fitset_targets()

        # seaborn styling, shared everywhere so line plots, legends, and spatial
        # panel titles all use the SAME color per method
        sns.set_theme(style=sns_style)
        colors = sns.color_palette(sns_palette, n_colors=len(self.method_names))
        self.palette = {m: c for m, c in zip(self.method_names, colors)}

    # ------------------------------------------------------------------ #
    # factor maps (x-axis) -- derived from the shared decay/irf
    # ------------------------------------------------------------------ #
    def _register_default_factors(self):
        self.add_factor("total_photons", self.decay.sum(axis=self.time_axis))
        self.add_factor("peak_counts", self.decay.max(axis=self.time_axis))
        if self.irf is not None and np.asarray(self.irf).shape == self.decay.shape:
            self.add_factor(
                "total_irf_photons", np.asarray(self.irf).sum(axis=self.time_axis)
            )

    def add_factor(self, name, array):
        """Register a pre-computed pixel-grid-shaped map as a reusable factor."""
        array = np.asarray(array)
        if array.shape != self.pixel_shape:
            raise ValueError(
                f"factor '{name}' shape {array.shape} != pixel shape {self.pixel_shape}"
            )
        self._factors[name] = array
        return self

    def add_factor_fn(self, name, fn):
        """Register a factor computed as fn(decay, irf) -> pixel-grid-shaped map."""
        return self.add_factor(name, fn(self.decay, self.irf))

    def list_factors(self):
        return sorted(self._factors.keys())

    def get_factor_map(self, factor_key):
        """
        Resolve a factor map by name.
          1. Shared factors derived from decay/irf (e.g. 'total_photons') --
             identical across all methods, so `shared=True` is returned.
          2. A key present in every all_datasets[i] dict (a per-method
             estimated map). Use with caution: these are NOT guaranteed to be
             on a common scale/definition across methods, so `shared=False`.

        Returns
        -------
        maps : list[ndarray]   one map per method (same order as method_names)
        shared : bool          True if the SAME array object is used for every method
        """
        if factor_key in self._factors:
            return [self._factors[factor_key]] * len(self.method_names), True
        if all(factor_key in d for d in self.all_datasets):
            return [np.asarray(d[factor_key]) for d in self.all_datasets], False
        raise KeyError(
            f"'{factor_key}' not found as a shared factor ({self.list_factors()}) "
            f"nor as a key present in every all_datasets entry."
        )

    def factor_values(self, factor_key, method_index=None):
        """
        Convenience accessor returning a single 2D factor map.
        - If factor_key is shared (e.g. 'total_photons'), returns that one map
          directly (no need to pick a method_index).
        - If factor_key is a per-method map (e.g. 'fret_efficiency_map'),
          you must pass method_index to select which method's version to use.
        """
        maps, is_shared = self.get_factor_map(factor_key)
        if is_shared:
            return maps[0]
        if method_index is None:
            raise ValueError(
                f"'{factor_key}' is a per-method factor (values differ by method); "
                f"pass method_index=0..{len(self.method_names) - 1} "
                f"(methods: {self.method_names})."
            )
        return maps[method_index]

    def _register_default_fitset_targets(self):
        eps = 1e-8
        self._fitset_target_fns["fit_total_photons"] = lambda decay, fs, irf: (
            np.asarray(fs["fit_map"]).sum(axis=self.time_axis)
        )
        self._fitset_target_fns["residual_sum"] = lambda decay, fs, irf: np.asarray(
            fs["residual_map"]
        ).sum(axis=self.time_axis)
        self._fitset_target_fns["residual_chi2"] = lambda decay, fs, irf: (
            (np.asarray(fs["residual_map"]) ** 2) / (np.asarray(fs["fit_map"]) + eps)
        ).sum(axis=self.time_axis)
        self._fitset_target_fns["residual_abs_mean"] = lambda decay, fs, irf: np.abs(
            np.asarray(fs["residual_map"])
        ).mean(axis=self.time_axis)

    def register_fitset_target(self, name, fn):
        """
        Add a derived target computed from a method's fitset dict, e.g. a
        custom fit-quality metric.
        fn(decay, fitset_dict, irf) -> pixel-grid-shaped map, where
        fitset_dict is one entry of all_fitset (has 'fit_map', 'residual_map',
        and possibly extra method-specific keys -- check with
        list_fitset_keys(method_index) before relying on anything beyond the
        two required keys).
        """
        self._fitset_target_fns[name] = fn
        return self

    def list_fitset_targets(self):
        return sorted(self._fitset_target_fns.keys())

    def list_fitset_keys(self, method_index):
        """Keys actually present in all_fitset[method_index] for this method."""
        return sorted(self.all_fitset[method_index].keys())

    def get_fitset_array(self, method_index, key="fit_map"):
        """
        Raw (H, W, T)-shaped array for one method's fitset entry, e.g.
        fa.get_fitset_array(0, 'fit_map'). Only 'fit_map' and 'residual_map'
        are guaranteed present for every method; other keys (e.g. 'sdf_map',
        'convolved_map') may only exist for some methods -- check with
        list_fitset_keys(method_index) first.
        """
        fs = self.all_fitset[method_index]
        if key not in fs:
            raise KeyError(
                f"'{key}' not in all_fitset[{method_index}] ({self.method_names[method_index]}); "
                f"available keys: {self.list_fitset_keys(method_index)}"
            )
        return np.asarray(fs[key])

    def _get_fitset_target_maps(self, target_key):
        if target_key not in self._fitset_target_fns:
            raise KeyError(
                f"'{target_key}' is not a registered fitset target "
                f"({self.list_fitset_targets()}). "
                "Use register_fitset_target(name, fn) to add one."
            )
        fn = self._fitset_target_fns[target_key]
        return [fn(self.decay, fs, self.irf) for fs in self.all_fitset]

    def selection_mask(self, factor_key, value_range, method_index=None):
        """
        Boolean pixel-grid mask(s) selecting pixels whose `factor_key` value
        falls within value_range=(low, high) (inclusive), AND passes the
        shared mask.

        Returns a single 2D array if factor_key is shared (or method_index is
        given), otherwise a list of 2D arrays (one per method) since a
        per-method factor like 'fret_efficiency_map' selects different
        pixels per method.
        """
        factor_maps, factor_is_shared = self.get_factor_map(factor_key)
        base_mask = (
            self.mask
            if self.mask is not None
            else np.ones(self.pixel_shape, dtype=bool)
        )
        lo, hi = value_range

        def _sel(fv):
            return (fv >= lo) & (fv <= hi) & np.isfinite(fv) & base_mask

        if method_index is not None:
            return _sel(factor_maps[method_index])
        if factor_is_shared:
            return _sel(factor_maps[0])
        return [_sel(fv) for fv in factor_maps]

    # ------------------------------------------------------------------ #
    # shared compact-number formatting (used by every plotting method)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _engineering_exponent(values):
        """
        Pick a common power-of-10 exponent (multiple of 3, e.g. 0, 3, 6, ...)
        so that dividing `values` by 10**exponent brings the largest magnitude
        into a compact ~1-3 digit range. Returns 0 if values are already small
        (< 1000), meaning no scaling is needed.
        """
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return 0
        max_abs = np.max(np.abs(values))
        if max_abs < 1000:
            return 0
        return int(np.floor(np.log10(max_abs) / 3) * 3)

    @staticmethod
    def _fmt_bin_range(lo, hi, exponent):
        """Format a [lo, hi) bin edge pair, scaled by 10**exponent, as a short label string."""
        scale = 10.0**exponent
        return f"[{lo / scale:.3g}, {hi / scale:.3g}]"

    @staticmethod
    def _exponent_suffix(exponent):
        return f"  (Ã10^{exponent})" if exponent else ""

    @staticmethod
    def _apply_compact_ticks(ax, axis="both"):
        """
        Apply matplotlib's native compact/scientific tick formatting to a
        NUMERIC axis: large numbers collapse to short tick labels plus one
        shared 'Ã10^n' offset text in the corner, instead of each tick
        repeating a long number. Silently no-ops on axes that don't support it
        (e.g. categorical axes -- those are handled via _fmt_bin_range instead).
        """
        try:
            ax.ticklabel_format(
                axis=axis, style="sci", scilimits=(-3, 3), useMathText=True
            )
        except (AttributeError, ValueError):
            pass

    @staticmethod
    def _apply_compact_colorbar(cbar):
        """Same compact formatting as _apply_compact_ticks, applied to a colorbar."""
        try:
            cbar.ax.ticklabel_format(
                axis="y", style="sci", scilimits=(-3, 3), useMathText=True
            )
        except (AttributeError, ValueError):
            pass

    @staticmethod
    def _maybe_save(saver, name, default_name, fig):
        """Save `fig` via `saver.save_plot(...)` when a saver is provided,
        matching the DataSaver.save_plot(name, fig=fig, close=False)
        convention used elsewhere in the package (e.g. FittingComparator)."""
        if saver is not None:
            saver.save_plot(name or default_name, fig=fig, close=False)

    def _overlay_panel(self, ax, full_map, sel_mask, cmap, vmin, vmax, bg_cmap="gray"):
        """
        Draw `full_map` as a grayscale structural background (so the image's
        overall shape/content stays visible everywhere), then overlay ONLY the
        pixels where sel_mask is True in color (cmap, vmin/vmax) on top --
        everything else is transparent, letting the grayscale background show
        through. Returns the foreground image (for colorbars).
        """
        full_map = np.asarray(full_map)
        finite = np.isfinite(full_map)

        bg = np.ma.masked_where(~finite, full_map)
        bg_cmap_obj = plt.get_cmap(bg_cmap).copy()
        bg_cmap_obj.set_bad(alpha=0)
        ax.imshow(bg, cmap=bg_cmap_obj)

        fg_cmap_obj = plt.get_cmap(cmap).copy()
        fg_cmap_obj.set_bad(alpha=0)  # fully transparent outside the selection
        fg = np.ma.masked_where(~(sel_mask & finite), full_map)
        im = ax.imshow(fg, cmap=fg_cmap_obj, vmin=vmin, vmax=vmax)
        return im

    # ------------------------------------------------------------------ #
    # spatial ("which pixels") plotting
    # ------------------------------------------------------------------ #
    def plot_spatial_selection(
        self,
        factor_key,
        value_range,
        ncols=3,
        figsize=None,
        cmap="viridis",
        bg_cmap="gray",
        saver=None,
        name=None,
    ):
        """
        Spatial map(s) of `factor_key` showing WHERE pixels fall inside
        value_range=(low, high). The full image is drawn in grayscale (`bg_cmap`)
        so the overall structure stays visible everywhere; ONLY the selected
        (in-range) pixels are drawn in color (`cmap`), on top.

        cmap    : colormap for the selected (in-range) pixels -- any matplotlib
                  colormap name, e.g. 'viridis', 'plasma', 'magma', 'jet'.
        bg_cmap : colormap for the grayscale structural background, default 'gray'.

        Produces ONE panel if factor_key is shared (same selection for every
        method), or one panel PER METHOD if it's an individual per-dataset
        factor (e.g. 'fret_efficiency_map'), since the selected pixels can
        differ by method.

        saver : DataSaver-like object or None
            If provided, the figure is saved via
            ``saver.save_plot(name or default, fig=fig, close=False)``.
        name : str or None
            Explicit save name; defaults to ``f"spatial_selection_{factor_key}"``.
        """
        factor_maps, factor_is_shared = self.get_factor_map(factor_key)
        lo, hi = value_range

        if factor_is_shared:
            panels = [("all methods", factor_maps[0], None)]
        else:
            panels = [
                (self.method_names[i], factor_maps[i], i)
                for i in range(len(self.method_names))
            ]

        n = len(panels)
        ncols = min(ncols, n)
        nrows = int(np.ceil(n / ncols))
        if figsize is None:
            figsize = (4 * ncols, 3.7 * nrows)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        axes_flat = axes.ravel()

        for ax, (label, fv, midx) in zip(axes_flat, panels):
            sel = self.selection_mask(factor_key, value_range, method_index=midx)
            im = self._overlay_panel(
                ax, fv, sel, cmap=cmap, vmin=lo, vmax=hi, bg_cmap=bg_cmap
            )
            title_color = (
                self.palette.get(label, "black") if midx is not None else "black"
            )
            ax.set_title(
                f"{str(label).lstrip('_')}\nn={int(sel.sum())} px",
                color=title_color,
                fontweight="bold",
            )
            ax.axis("off")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=factor_key)
            self._apply_compact_colorbar(cbar)

        for ax in axes_flat[n:]:
            ax.axis("off")

        fig.suptitle(
            f"{factor_key} in [{lo:.3g}, {hi:.3g}]  (grayscale = image structure, color = in range)",
            y=1.02,
        )
        fig.tight_layout()
        self._maybe_save(saver, name, f"spatial_selection_{factor_key}", fig)
        return fig, axes

    def plot_range_selection_grid(
        self,
        factor_key,
        value_range,
        target_keys,
        target_source="datasets",
        figsize=None,
        cmap="viridis",
        bg_cmap="gray",
        shared_scale=True,
        saver=None,
        name=None,
    ):
        """
        Combined grid: top row shows WHICH pixels are selected by `factor_key`
        in value_range=(low, high), one column per method; each subsequent row
        shows one target parameter map (e.g. 'tau1_map'), restricted to that
        same selection. Every panel draws the full map in grayscale first (so
        the image's structure is always visible), then overlays ONLY the
        in-range pixels in color on top -- so you can visually compare how
        methods estimate a parameter for pixels drawn from a specific factor
        range (e.g. a photon-count band), without losing spatial context.

        cmap : colormap for the colored (in-range) overlay. Either:
               - a single string applied to every row, e.g. 'viridis', or
               - a list with one colormap PER ROW, ordered as
                 [factor_key_row, target_keys[0]_row, target_keys[1]_row, ...],
                 e.g. cmap=['jet', 'plasma', 'plasma'] for
                 factor_key='total_photons', target_keys=['tau1_map', 'tau2_map'].
                 Must have length 1 + len(target_keys).
        bg_cmap : colormap for the grayscale structural background, default 'gray'.
        shared_scale=True uses one colorbar range per target row (pooled
        across methods' selected pixels) so panels are visually comparable;
        set False to let each panel auto-scale to its own selected pixels.

        saver : DataSaver-like object or None
            If provided, the figure is saved via
            ``saver.save_plot(name or default, fig=fig, close=False)``.
        name : str or None
            Explicit save name; defaults to ``f"range_selection_grid_{factor_key}"``.
        """
        factor_maps, factor_is_shared = self.get_factor_map(factor_key)
        lo, hi = value_range
        n_methods = len(self.method_names)

        row_labels = [factor_key] + list(target_keys)
        if isinstance(cmap, str):
            cmaps = [cmap] * len(row_labels)
        else:
            cmaps = list(cmap)
            if len(cmaps) != len(row_labels):
                raise ValueError(
                    f"cmap list must have length {len(row_labels)} "
                    f"(1 for '{factor_key}' + {len(target_keys)} for target_keys "
                    f"{target_keys}), got {len(cmaps)}: {cmaps}"
                )

        if target_source == "datasets":

            def get_map(i, key):
                d = self.all_datasets[i]
                return np.asarray(d[key]) if key in d else None
        elif target_source == "fitset":
            cache = {key: self._get_fitset_target_maps(key) for key in target_keys}

            def get_map(i, key):
                return cache[key][i]
        else:
            raise ValueError("target_source must be 'datasets' or 'fitset'")

        sel_masks = [
            self.selection_mask(factor_key, value_range, method_index=i)
            for i in range(n_methods)
        ]

        n_rows = 1 + len(target_keys)
        if figsize is None:
            figsize = (3.6 * n_methods, 3.3 * n_rows)
        fig, axes = plt.subplots(n_rows, n_methods, figsize=figsize, squeeze=False)

        # row 0: factor selection itself
        for i, method in enumerate(self.method_names):
            ax = axes[0, i]
            im = self._overlay_panel(
                ax,
                factor_maps[i],
                sel_masks[i],
                cmap=cmaps[0],
                vmin=lo,
                vmax=hi,
                bg_cmap=bg_cmap,
            )
            ax.set_title(
                f"{method.lstrip('_')}\nn={int(sel_masks[i].sum())} px",
                color=self.palette.get(method, "black"),
                fontweight="bold",
                fontsize=9,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_ylabel(factor_key, fontsize=9)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            self._apply_compact_colorbar(cbar)

        # subsequent rows: each target parameter, restricted to the selection
        for r, key in enumerate(target_keys, start=1):
            maps_row = [get_map(i, key) for i in range(n_methods)]
            row_cmap = cmaps[r]
            vmin = vmax = None
            if shared_scale:
                vals = [
                    m[sel_masks[i]][np.isfinite(m[sel_masks[i]])]
                    for i, m in enumerate(maps_row)
                    if m is not None
                ]
                vals = np.concatenate(vals) if vals else np.array([])
                if vals.size:
                    vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))

            for i, method in enumerate(self.method_names):
                ax = axes[r, i]
                m = maps_row[i]
                if m is None:
                    ax.axis("off")
                    continue
                if not shared_scale:
                    sv = m[sel_masks[i]]
                    sv = sv[np.isfinite(sv)]
                    vmin, vmax = (
                        (float(sv.min()), float(sv.max())) if sv.size else (None, None)
                    )
                im = self._overlay_panel(
                    ax,
                    m,
                    sel_masks[i],
                    cmap=row_cmap,
                    vmin=vmin,
                    vmax=vmax,
                    bg_cmap=bg_cmap,
                )
                if i == 0:
                    ax.set_ylabel(key, fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                self._apply_compact_colorbar(cbar)

        fig.suptitle(
            f"Pixels selected by {factor_key} in [{lo:.3g}, {hi:.3g}]  "
            "(grayscale = image structure, color = in range)",
            y=1.01,
        )
        fig.tight_layout()
        self._maybe_save(saver, name, f"range_selection_grid_{factor_key}", fig)
        return fig, axes

    def analyze(
        self,
        factor_key="total_photons",
        target_keys=None,
        target_source="datasets",  # 'datasets' or 'fitset'
        n_bins=10,
        bin_mode="quantile",  # 'quantile' or 'linear'
        bin_scope="auto",  # 'auto', 'pooled', or 'per_method'
        exclude_keys=("convergence_map", "pixel_health_map"),
    ):
        """
        Bin pixels by `factor_key` and compute per-bin mean/median/std/sem/count
        of every target parameter, per method. The shared `mask` (if given) is
        applied for every method.

        factor_key can be:
          - a SHARED factor derived from decay/irf (e.g. 'total_photons') --
            identical values for every method, since decay/irf are common
            raw inputs. Example: fa.analyze(factor_key='total_photons', ...).
          - an INDIVIDUAL per-method map already present in all_datasets
            (e.g. 'fret_efficiency_map') -- each method computed its own
            version, so values/scales can legitimately differ by method.
            Example: fa.analyze(factor_key='fret_efficiency_map',
                                 target_keys=['tau1_map', 'tau2_map', 'tau_mean_map'])

        target_source='datasets' pulls named maps from all_datasets[i][key]
                                  (default target_keys = every key present).
        target_source='fitset'   pulls derived maps computed from all_fitset's
                                  'fit_map'/'residual_map' vs. the shared decay
                                  (default target_keys = all registered fitset
                                  targets; see list_fitset_targets() /
                                  register_fitset_target()).

        bin_scope controls how bin edges are computed:
          - 'pooled'     : ONE set of edges from all methods' values pooled
                           together. Correct when factor_key is shared (same
                           array for every method) since edges then trivially
                           agree with per-method edges too.
          - 'per_method' : EACH method gets its own edges, computed only from
                           its own factor values. Use this when factor_key is
                           an individual per-method map that may sit on a
                           different scale/range per method (e.g.
                           'fret_efficiency_map') -- pooling in that case can
                           distort bins the way it did for mismatched
                           photon-count scales. bin_rank (0..n_bins-1) is then
                           the fair way to compare methods at "corresponding"
                           positions along each one's own distribution; the
                           actual bin_center/bin_low/bin_high values may still
                           differ by method.
          - 'auto'       : 'pooled' if factor_key resolves to a shared factor,
                           'per_method' otherwise. This is a sensible default,
                           not a strict rule -- override explicitly if needed.

        Returns
        -------
        df : pandas.DataFrame  (long-form: method, bin_rank, bin_center, bin_low,
             bin_high, parameter, mean, median, std, sem, count)
        bin_edges : np.ndarray or dict[str, np.ndarray]
             A single edges array if bin_scope ended up 'pooled', otherwise a
             dict of {method_name: edges} for 'per_method'.
        """
        factor_maps, factor_is_shared = self.get_factor_map(factor_key)
        mask = (
            self.mask
            if self.mask is not None
            else np.ones(self.pixel_shape, dtype=bool)
        )
        mask_flat = mask.ravel()

        if bin_scope == "auto":
            bin_scope = "pooled" if factor_is_shared else "per_method"
        if bin_scope not in ("pooled", "per_method"):
            raise ValueError("bin_scope must be 'auto', 'pooled', or 'per_method'")

        if target_source == "datasets":
            if target_keys is None:
                first_keys = set(self.all_datasets[0].keys())
                skip = {factor_key} | set(exclude_keys)
                target_keys = sorted(k for k in first_keys if k not in skip)
            target_map_lookup = {
                key: [
                    np.asarray(d[key]) if key in d else None for d in self.all_datasets
                ]
                for key in target_keys
            }
        elif target_source == "fitset":
            if target_keys is None:
                target_keys = self.list_fitset_targets()
            target_map_lookup = {
                key: self._get_fitset_target_maps(key) for key in target_keys
            }
        else:
            raise ValueError("target_source must be 'datasets' or 'fitset'")

        # ---- per-method valid (finite + masked) factor values ----
        fv_keep_list = []
        keep_list = []
        for i in range(len(self.method_names)):
            fv = factor_maps[i].ravel().astype(float)
            keep = np.isfinite(fv) & mask_flat
            keep_list.append(keep)
            fv_keep_list.append(fv[keep])

        def _compute_edges(values):
            if values.size == 0:
                raise ValueError(
                    f"No valid pixels found for factor_key='{factor_key}'."
                )
            if bin_mode == "quantile":
                e = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
            elif bin_mode == "linear":
                e = np.linspace(values.min(), values.max(), n_bins + 1)
            else:
                raise ValueError("bin_mode must be 'quantile' or 'linear'")
            if len(e) < 2:
                raise ValueError(
                    "Not enough distinct factor values to form bins; try fewer n_bins."
                )
            return e

        if bin_scope == "pooled":
            pooled_edges = _compute_edges(np.concatenate(fv_keep_list))
            edges_per_method = [pooled_edges] * len(self.method_names)
            returned_edges = pooled_edges
        else:  # per_method
            edges_per_method = [_compute_edges(fv) for fv in fv_keep_list]
            returned_edges = dict(zip(self.method_names, edges_per_method))

        rows = []
        for i, method in enumerate(self.method_names):
            edges = edges_per_method[i]
            bin_centers = 0.5 * (edges[:-1] + edges[1:])
            keep = keep_list[i]
            fv_keep = fv_keep_list[i]
            bin_idx = pd.cut(fv_keep, bins=edges, labels=False, include_lowest=True)

            for key, maps_list in target_map_lookup.items():
                mp = maps_list[i]
                if mp is None:
                    continue
                pv = np.asarray(mp).ravel().astype(float)[keep]
                valid_p = np.isfinite(pv)
                tmp = pd.DataFrame({"bin_rank": bin_idx[valid_p], "value": pv[valid_p]})
                grouped = tmp.groupby("bin_rank")["value"].agg(
                    ["mean", "median", "std", "count"]
                )
                for b in range(len(bin_centers)):
                    if b in grouped.index:
                        r = grouped.loc[b]
                        n_ = r["count"]
                        rows.append(
                            {
                                "method": method,
                                "bin_rank": b,
                                "bin_center": bin_centers[b],
                                "bin_low": edges[b],
                                "bin_high": edges[b + 1],
                                "parameter": key,
                                "mean": r["mean"],
                                "median": r["median"],
                                "std": r["std"],
                                "sem": r["std"] / np.sqrt(n_) if n_ > 0 else np.nan,
                                "count": int(n_),
                            }
                        )

        df = pd.DataFrame(rows)
        df.attrs["factor_key"] = factor_key
        df.attrs["factor_is_shared"] = factor_is_shared
        df.attrs["target_source"] = target_source
        df.attrs["bin_scope"] = bin_scope
        return df, returned_edges

    # ------------------------------------------------------------------ #
    # plotting
    # ------------------------------------------------------------------ #
    def _target_map_getter(self, target_source, target_keys):
        if target_source == "datasets":

            def get_map(i, key):
                d = self.all_datasets[i]
                return np.asarray(d[key]) if key in d else None
        elif target_source == "fitset":
            cache = {key: self._get_fitset_target_maps(key) for key in target_keys}

            def get_map(i, key):
                return cache[key][i]
        else:
            raise ValueError("target_source must be 'datasets' or 'fitset'")
        return get_map

    def _reconstruct_edges(self, df, method):
        """Rebuild a method's bin edges from the bin_low/bin_high columns already in df."""
        sub = (
            df[df["method"] == method][["bin_rank", "bin_low", "bin_high"]]
            .drop_duplicates()
            .sort_values("bin_rank")
        )
        if sub.empty:
            return None
        return np.append(sub["bin_low"].values, sub["bin_high"].values[-1])

    def _raw_binned_long(self, df, target_keys, exponent=None):
        """
        Raw per-pixel values re-associated with the SAME bins already computed
        in df. Bin labels use a single shared power-of-10 exponent (computed
        across ALL methods' edges, or passed explicitly) so large numbers
        collapse to short labels with one common 'Ã10^n' scale factor instead
        of each label repeating a long number.

        Returns (raw_df, exponent).
        """
        factor_key = df.attrs.get("factor_key")
        target_source = df.attrs.get("target_source", "datasets")
        factor_maps, _ = self.get_factor_map(factor_key)
        mask = (
            self.mask
            if self.mask is not None
            else np.ones(self.pixel_shape, dtype=bool)
        )
        mask_flat = mask.ravel()
        get_map = self._target_map_getter(target_source, target_keys)

        if exponent is None:
            all_edges = np.concatenate(
                [df["bin_low"].to_numpy(), df["bin_high"].to_numpy()]
            )
            exponent = self._engineering_exponent(all_edges)

        frames = []
        for i, method in enumerate(self.method_names):
            edges = self._reconstruct_edges(df, method)
            if edges is None or len(edges) < 2:
                continue
            fv = factor_maps[i].ravel().astype(float)
            keep = np.isfinite(fv) & mask_flat
            fv_keep = fv[keep]
            bin_idx = pd.cut(fv_keep, bins=edges, labels=False, include_lowest=True)
            valid_bin = np.isfinite(bin_idx)
            bin_idx_int = np.where(valid_bin, bin_idx, -1).astype(int)
            label_lut = np.array(
                [
                    self._fmt_bin_range(edges[b], edges[b + 1], exponent)
                    for b in range(len(edges) - 1)
                ],
                dtype=object,
            )
            bin_labels = np.where(
                valid_bin, label_lut[np.clip(bin_idx_int, 0, None)], None
            )

            for key in target_keys:
                mp = get_map(i, key)
                if mp is None:
                    continue
                pv = np.asarray(mp).ravel().astype(float)[keep]
                valid = valid_bin & np.isfinite(pv)
                frames.append(
                    pd.DataFrame(
                        {
                            "method": method,
                            "parameter": key,
                            "bin_rank": bin_idx_int[valid],
                            "bin_label": bin_labels[valid],
                            "value": pv[valid],
                        }
                    )
                )
        raw_df = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(
                columns=["method", "parameter", "bin_rank", "bin_label", "value"]
            )
        )
        return raw_df, exponent

    def _raw_unbinned_long(self, df, target_keys, max_points=None, seed=0):
        """Raw per-pixel (factor_value, target_value) pairs, no binning -- for scatter plots."""
        factor_key = df.attrs.get("factor_key")
        target_source = df.attrs.get("target_source", "datasets")
        factor_maps, _ = self.get_factor_map(factor_key)
        mask = (
            self.mask
            if self.mask is not None
            else np.ones(self.pixel_shape, dtype=bool)
        )
        mask_flat = mask.ravel()
        get_map = self._target_map_getter(target_source, target_keys)
        rng = np.random.default_rng(seed)

        frames = []
        for i, method in enumerate(self.method_names):
            fv = factor_maps[i].ravel().astype(float)
            keep = np.isfinite(fv) & mask_flat
            fv_keep = fv[keep]
            for key in target_keys:
                mp = get_map(i, key)
                if mp is None:
                    continue
                pv = np.asarray(mp).ravel().astype(float)[keep]
                valid = np.isfinite(pv)
                x, y = fv_keep[valid], pv[valid]
                if max_points is not None and len(x) > max_points:
                    idx = rng.choice(len(x), size=max_points, replace=False)
                    x, y = x[idx], y[idx]
                frames.append(
                    pd.DataFrame({"method": method, "parameter": key, "x": x, "y": y})
                )
        return (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=["method", "parameter", "x", "y"])
        )

    def plot(
        self,
        df,
        target_keys=None,
        kind="line",
        stat="mean",
        error="sem",
        ncols=3,
        figsize=None,
        logx=False,
        x_axis="auto",
        max_scatter_points=3000,
        scatter_alpha=0.35,
        scatter_size=10,
        box_showfliers=False,
        saver=None,
        name=None,
    ):
        """
        Grid of plots (one subplot per parameter, one series per method).

        kind : 'line', 'box', or 'scatter'
            'line'    (default) -- aggregated per-bin `stat` ('mean' or
                       'median') with a shaded `error` band ('sem', 'std', or
                       None). Uses the pre-aggregated values already in `df`.
            'box'     -- boxplot of the RAW per-pixel values within each
                       factor bin, one box per (bin, method) -- shows spread
                       and outliers instead of a single summary stat.
                       `box_showfliers` controls whether outlier points are drawn.
            'scatter' -- raw per-pixel scatter of factor value (x, unbinned)
                       vs target value (y), one color per method -- shows the
                       actual relationship with no binning at all. Subsampled
                       to `max_scatter_points` per (method, parameter) for
                       plotting speed; control point look via `scatter_alpha`
                       and `scatter_size`.

        x_axis (only used for kind='line') : 'auto', 'bin_center', or 'bin_rank'
            'bin_center' plots each method's line at its own actual factor
            values -- meaningful when bin_scope='pooled' (edges are identical
            across methods anyway).
            'bin_rank' plots each method's line at its bin index (0..n_bins-1)
            instead -- use this when bin_scope='per_method', since each
            method's bin_center values live on that method's own scale and
            aren't directly comparable at the same x position otherwise.
            'auto' picks 'bin_center' if df.attrs['bin_scope']=='pooled',
            else 'bin_rank'.

        saver : DataSaver-like object or None
            If provided, the figure is saved via
            ``saver.save_plot(name or default, fig=fig, close=False)``.
        name : str or None
            Explicit save name; defaults to ``f"factor_analysis_{factor_key}_{kind}"``.
        """
        if kind not in ("line", "box", "scatter"):
            raise ValueError("kind must be 'line', 'box', or 'scatter'")

        factor_key = df.attrs.get("factor_key", "factor")
        if target_keys is None:
            target_keys = sorted(df["parameter"].unique())

        n = len(target_keys)
        ncols = min(ncols, n)
        nrows = int(np.ceil(n / ncols))
        if figsize is None:
            figsize = (5 * ncols, 4 * nrows)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        axes_flat = axes.ravel()
        methods = df["method"].unique()

        if kind == "line":
            bin_scope = df.attrs.get("bin_scope", "pooled")
            if x_axis == "auto":
                x_axis = "bin_center" if bin_scope == "pooled" else "bin_rank"
            if x_axis not in ("bin_center", "bin_rank"):
                raise ValueError("x_axis must be 'auto', 'bin_center', or 'bin_rank'")

            for ax, param in zip(axes_flat, target_keys):
                sub = df[df["parameter"] == param]
                plotted = False
                for method in methods:
                    m = sub[sub["method"] == method].sort_values(x_axis)
                    if m.empty:
                        continue
                    x = m[x_axis]
                    color = self.palette.get(method)
                    ax.plot(
                        x,
                        m[stat],
                        marker="o",
                        linewidth=2,
                        markersize=5,
                        color=color,
                        label=str(method).lstrip("_"),
                    )
                    if error is not None and error in m.columns:
                        lo = m[stat] - m[error]
                        hi = m[stat] + m[error]
                        ax.fill_between(x, lo, hi, alpha=0.2, color=color, linewidth=0)
                    plotted = True
                ax.set_title(param, fontweight="bold")
                ax.set_xlabel(
                    factor_key if x_axis == "bin_center" else f"{factor_key} (bin rank)"
                )
                ax.set_ylabel(stat)
                if x_axis == "bin_center":
                    if logx:
                        ax.set_xscale("log")
                    else:
                        self._apply_compact_ticks(ax, axis="x")
                self._apply_compact_ticks(ax, axis="y")
                if plotted:
                    ax.legend(fontsize=8, frameon=False)
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "no data",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=9,
                        color="gray",
                    )
                sns.despine(ax=ax)

        elif kind == "box":
            raw_df, exponent = self._raw_binned_long(df, target_keys)
            suffix = self._exponent_suffix(exponent)
            palette = {m.lstrip("_"): c for m, c in self.palette.items()}
            for ax, param in zip(axes_flat, target_keys):
                sub = raw_df[raw_df["parameter"] == param]
                if sub.empty:
                    ax.text(
                        0.5,
                        0.5,
                        "no data",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=9,
                        color="gray",
                    )
                    continue
                order = (
                    sub[["bin_rank", "bin_label"]]
                    .drop_duplicates()
                    .sort_values("bin_rank")["bin_label"]
                    .tolist()
                )
                plot_df = sub.copy()
                plot_df["method"] = plot_df["method"].str.lstrip("_")
                sns.boxplot(
                    data=plot_df,
                    x="bin_label",
                    y="value",
                    hue="method",
                    order=order,
                    palette=palette,
                    ax=ax,
                    showfliers=box_showfliers,
                )
                ax.set_title(param, fontweight="bold")
                ax.set_xlabel(f"{factor_key} bin{suffix}")
                ax.set_ylabel(param)
                self._apply_compact_ticks(ax, axis="y")
                plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
                ax.legend(fontsize=8, frameon=False)
                sns.despine(ax=ax)

        elif kind == "scatter":
            raw_df = self._raw_unbinned_long(
                df, target_keys, max_points=max_scatter_points
            )
            for ax, param in zip(axes_flat, target_keys):
                sub = raw_df[raw_df["parameter"] == param]
                plotted = False
                for method in methods:
                    m = sub[sub["method"] == method]
                    if m.empty:
                        continue
                    ax.scatter(
                        m["x"],
                        m["y"],
                        s=scatter_size,
                        alpha=scatter_alpha,
                        color=self.palette.get(method),
                        label=str(method).lstrip("_"),
                        edgecolors="none",
                    )
                    plotted = True
                ax.set_title(param, fontweight="bold")
                ax.set_xlabel(factor_key)
                ax.set_ylabel(param)
                if logx:
                    ax.set_xscale("log")
                else:
                    self._apply_compact_ticks(ax, axis="x")
                self._apply_compact_ticks(ax, axis="y")
                if plotted:
                    ax.legend(fontsize=8, frameon=False)
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "no data",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=9,
                        color="gray",
                    )
                sns.despine(ax=ax)

        for ax in axes_flat[n:]:
            ax.axis("off")

        fig.tight_layout()
        self._maybe_save(saver, name, f"factor_analysis_{factor_key}_{kind}", fig)
        return fig, axes

    def plot_bin_distribution(
        self,
        factor_key,
        target_key,
        target_source="datasets",
        n_bins=6,
        bin_mode="quantile",
        bin_scope="auto",
        kind="violin",
        figsize=None,
        saver=None,
        name=None,
    ):
        """
        Seaborn violin/boxen plot of the RAW per-pixel distribution of
        `target_key`, split by factor bin (x-axis) and method (hue) --
        complements plot()'s aggregated mean/sem lines by showing the actual
        spread/shape of each bin's pixel values, not just a summary stat.

        kind : 'violin' or 'boxen'

        saver : DataSaver-like object or None
            If provided, the figure is saved via
            ``saver.save_plot(name or default, fig=fig, close=False)``.
        name : str or None
            Explicit save name; defaults to
            ``f"bin_distribution_{factor_key}_{target_key}"``.
        """
        factor_maps, factor_is_shared = self.get_factor_map(factor_key)
        mask = (
            self.mask
            if self.mask is not None
            else np.ones(self.pixel_shape, dtype=bool)
        )
        mask_flat = mask.ravel()

        if bin_scope == "auto":
            bin_scope = "pooled" if factor_is_shared else "per_method"

        if target_source == "datasets":
            target_maps = [
                np.asarray(d[target_key]) if target_key in d else None
                for d in self.all_datasets
            ]
        elif target_source == "fitset":
            target_maps = self._get_fitset_target_maps(target_key)
        else:
            raise ValueError("target_source must be 'datasets' or 'fitset'")

        fv_keep_list, keep_list = [], []
        for i in range(len(self.method_names)):
            fv = factor_maps[i].ravel().astype(float)
            keep = np.isfinite(fv) & mask_flat
            keep_list.append(keep)
            fv_keep_list.append(fv[keep])

        def _edges(values):
            if bin_mode == "quantile":
                e = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
            else:
                e = np.linspace(values.min(), values.max(), n_bins + 1)
            return e

        if bin_scope == "pooled":
            edges = _edges(np.concatenate(fv_keep_list))
            edges_per_method = [edges] * len(self.method_names)
        else:
            edges_per_method = [_edges(fv) for fv in fv_keep_list]

        rows = []
        all_edges_flat = np.concatenate(edges_per_method)
        exponent = self._engineering_exponent(all_edges_flat)
        for i, method in enumerate(self.method_names):
            if target_maps[i] is None:
                continue
            edges = edges_per_method[i]
            keep = keep_list[i]
            fv_keep = fv_keep_list[i]
            bin_idx = pd.cut(fv_keep, bins=edges, labels=False, include_lowest=True)
            pv = target_maps[i].ravel().astype(float)[keep]
            valid = np.isfinite(pv) & np.isfinite(bin_idx.astype(float))
            for b, v in zip(bin_idx[valid], pv[valid]):
                lo_e, hi_e = edges[int(b)], edges[int(b) + 1]
                rows.append(
                    {
                        "method": method.lstrip("_"),
                        "bin_rank": int(b),
                        "bin_label": self._fmt_bin_range(lo_e, hi_e, exponent),
                        "value": v,
                    }
                )

        raw_df = pd.DataFrame(rows)
        if raw_df.empty:
            raise ValueError(
                "No valid pixels to plot -- check factor_key/target_key/mask."
            )
        # order bin labels by rank so the x-axis reads low -> high
        order = (
            raw_df[["bin_rank", "bin_label"]]
            .drop_duplicates()
            .sort_values("bin_rank")["bin_label"]
            .tolist()
        )

        fig, ax = plt.subplots(figsize=figsize or (max(8, 1.6 * n_bins), 5))
        palette = {m.lstrip("_"): c for m, c in self.palette.items()}
        plot_fn = sns.violinplot if kind == "violin" else sns.boxenplot
        kwargs = dict(inner="quartile", cut=0) if kind == "violin" else {}
        plot_fn(
            data=raw_df,
            x="bin_label",
            y="value",
            hue="method",
            order=order,
            palette=palette,
            ax=ax,
            **kwargs,
        )
        suffix = self._exponent_suffix(exponent)
        per_method_note = " (per-method edges)" if bin_scope == "per_method" else ""
        ax.set_xlabel(f"{factor_key} bin{per_method_note}{suffix}")
        ax.set_ylabel(target_key)
        ax.set_title(
            f"{target_key} distribution by {factor_key} bin", fontweight="bold"
        )
        self._apply_compact_ticks(ax, axis="y")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        sns.despine(ax=ax)
        fig.tight_layout()
        self._maybe_save(
            saver, name, f"bin_distribution_{factor_key}_{target_key}", fig
        )
        return fig, ax, raw_df

    def analyze_and_plot(
        self,
        factor_key="total_photons",
        target_keys=None,
        target_source="datasets",
        n_bins=10,
        bin_mode="quantile",
        bin_scope="auto",
        kind="line",
        stat="mean",
        error="sem",
        ncols=3,
        figsize=None,
        logx=False,
        x_axis="auto",
        saver=None,
        name=None,
    ):
        """Convenience wrapper: analyze() then plot() in one call."""
        df, edges = self.analyze(
            factor_key=factor_key,
            target_keys=target_keys,
            target_source=target_source,
            n_bins=n_bins,
            bin_mode=bin_mode,
            bin_scope=bin_scope,
        )
        fig, axes = self.plot(
            df,
            target_keys=target_keys,
            kind=kind,
            stat=stat,
            error=error,
            ncols=ncols,
            figsize=figsize,
            logx=logx,
            x_axis=x_axis,
            saver=saver,
            name=name,
        )
        return df, edges, fig, axes


if __name__ == "__main__":
    # ---- minimal smoke test with synthetic data ----
    rng = np.random.default_rng(0)
    H, W, T = 40, 40, 256
    t = np.linspace(0, 12, T)

    irf = np.exp(-0.5 * ((t - 1.0) / 0.15) ** 2)
    irf /= irf.sum()

    def make_decay(tau, amp_scale):
        amp = rng.gamma(2.0, amp_scale, size=(H, W))
        decay = amp[..., None] * np.exp(-t[None, None, :] / tau)
        decay = np.array(
            [
                np.convolve(decay[i, j], irf, mode="same")
                for i in range(H)
                for j in range(W)
            ]
        ).reshape(H, W, T)
        decay = rng.poisson(np.clip(decay, 0, None)).astype(float)
        return decay

    decay = make_decay(tau=2.2, amp_scale=150.0)
    mask = (
        decay.sum(axis=-1) > 50
    )  # shared boolean mask, e.g. thresholded on total photons

    def fake_tr(decay, bias, noise):
        fit_map = np.clip(
            decay * (1 + bias) + rng.normal(0, noise, decay.shape), 0, None
        )
        residual_map = decay - fit_map
        return {"fit_map": fit_map, "residual_map": residual_map}

    def fake_dataset(bias, fret_scale):
        total_photons = decay.sum(axis=-1)
        tau1 = (
            2.2
            + bias
            + rng.normal(
                0, 0.05 / np.sqrt(np.clip(total_photons, 10, None) / 100), (H, W)
            )
        )
        tau2 = (
            4.0
            + bias
            + rng.normal(
                0, 0.08 / np.sqrt(np.clip(total_photons, 10, None) / 100), (H, W)
            )
        )
        tau_mean = 0.5 * (tau1 + tau2)
        chi2 = 1.0 + rng.normal(0, 0.05, (H, W))
        # simulate a per-method factor map on a DIFFERENT scale per method,
        # like fret_efficiency_map might be if methods disagree on range
        fret = np.clip(rng.beta(2, 5, (H, W)) * fret_scale, 0, None)
        return {
            "tau1_map": tau1,
            "tau2_map": tau2,
            "tau_mean_map": tau_mean,
            "chi2_map": chi2,
            "fret_efficiency_map": fret,
        }

    all_datasets = [
        fake_dataset(0.0, 1.0),
        fake_dataset(0.03, 0.6),
        fake_dataset(-0.02, 1.3),
    ]
    all_fitset = [
        fake_tr(decay, 0.0, 2.0),
        fake_tr(decay, 0.05, 3.0),
        fake_tr(decay, -0.03, 1.5),
    ]
    all_fitset[0]["sdf_map"] = (
        decay * 0.9
    )  # extra, method-specific key (mirrors real data)
    method_names = ["_fbi_fit_bi", "_cpu_nlsf_bi", "_cpu_mle_bi"]

    fa = FactorAnalysis(
        decay=decay,
        irf=None,
        mask=mask,
        all_datasets=all_datasets,
        all_fitset=all_fitset,
        method_names=method_names,
        time_axis=-1,
    )

    print("factors:", fa.list_factors())
    print("fitset targets:", fa.list_fitset_targets())
    print("fitset keys per method:", [fa.list_fitset_keys(i) for i in range(3)])

    df, edges = fa.analyze(
        factor_key="total_photons", target_keys=["tau1_map", "chi2_map"]
    )
    print(df.head())
    fig, axes = fa.plot(df)
    fig.savefig("/tmp/factor_analysis_class_smoketest_datasets.png", dpi=100)

    df2, edges2 = fa.analyze(
        factor_key="total_photons",
        target_source="fitset",
        target_keys=["residual_chi2", "fit_total_photons"],
    )
    print(df2.head())
    fig2, axes2 = fa.plot(df2)
    fig2.savefig("/tmp/factor_analysis_class_smoketest_fitset.png", dpi=100)

    # individual per-method factor (different scale per method) -> auto picks bin_scope='per_method'
    df3, edges3 = fa.analyze(
        factor_key="fret_efficiency_map",
        target_keys=["tau1_map", "tau2_map", "tau_mean_map"],
    )
    print("bin_scope used:", df3.attrs["bin_scope"])
    print("per-method edges:", {k: (v.min(), v.max()) for k, v in edges3.items()})
    fig3, axes3 = fa.plot(
        df3
    )  # x_axis='auto' -> bin_rank, since bin_scope='per_method'
    fig3.savefig("/tmp/factor_analysis_class_smoketest_fret.png", dpi=100)

    # spatial: which pixels fall in a chosen total_photons range (shared factor -> 1 panel)
    lo, hi = np.quantile(fa.factor_values("total_photons")[mask], [0.6, 0.9])
    fig4, axes4 = fa.plot_spatial_selection("total_photons", (lo, hi))
    fig4.savefig("/tmp/factor_analysis_spatial_shared.png", dpi=100)

    # spatial: which pixels fall in a chosen fret_efficiency_map range (per-method -> 3 panels)
    lo2, hi2 = 0.1, 0.3
    fig5, axes5 = fa.plot_spatial_selection("fret_efficiency_map", (lo2, hi2))
    fig5.savefig("/tmp/factor_analysis_spatial_fret.png", dpi=100)

    # combined grid: selection row + estimate maps restricted to that selection
    fig6, axes6 = fa.plot_range_selection_grid(
        "total_photons",
        (lo, hi),
        target_keys=["tau1_map", "tau2_map"],
        cmap=["jet", "plasma", "plasma"],  # one colormap per row: [factor, tau1, tau2]
    )
    fig6.savefig("/tmp/factor_analysis_range_grid.png", dpi=100)

    # raw per-pixel distribution comparison (violin) across bins/methods
    fig7, ax7, raw_df = fa.plot_bin_distribution("total_photons", "tau1_map", n_bins=5)
    fig7.savefig("/tmp/factor_analysis_violin.png", dpi=100)

    # plot() kind options: line (mean/median), box, scatter
    df8, edges8 = fa.analyze(
        factor_key="total_photons",
        target_source="fitset",
        target_keys=["residual_chi2", "fit_total_photons"],
    )
    fig8a, _ = fa.plot(df8, kind="line", stat="mean")
    fig8a.savefig("/tmp/factor_analysis_kind_line_mean.png", dpi=100)
    fig8b, _ = fa.plot(df8, kind="line", stat="median")
    fig8b.savefig("/tmp/factor_analysis_kind_line_median.png", dpi=100)
    fig8c, _ = fa.plot(df8, kind="box")
    fig8c.savefig("/tmp/factor_analysis_kind_box.png", dpi=100)
    fig8d, _ = fa.plot(df8, kind="scatter", max_scatter_points=200)
    fig8d.savefig("/tmp/factor_analysis_kind_scatter.png", dpi=100)

    print("Smoke test OK.")
