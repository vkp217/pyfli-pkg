from .colorProcess import Colorprocess
from .mdataViz import DataViewer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


class MonoBiClassifier:
    """
    Per-dataset mono- vs bi-exponential pixel classification, plus cross-method
    agreement and parameter-correlation analysis.

    A pixel is classified 'mono' when one component overwhelmingly dominates:
      - alpha1 > alpha_upper  (component 1 carries most of the signal)
      - alpha1 < alpha_lower  (component 2 carries most of the signal)
      - |tau1 - tau2| <= tau_tol  (both lifetimes are indistinguishable)
    Otherwise the pixel is 'bi'.  All maps are restricted to the ROI
    given by `b_bool_mask`.

    Parameters
    ----------
    alpha_upper : float   upper saturation threshold (default 0.95)
    alpha_lower : float   lower saturation threshold (default 0.05).
                          Use a small value (e.g. 0.05) so that only pixels
                          where component 2 is truly dominant are called mono.
                          Using 0.5 would label 55 % of the alpha space mono.
    tau_tol     : float   lifetime coincidence tolerance in ns (default 0.01).
                          Replaces exact float equality (tau1 == tau2 is almost
                          never True for fitted values).

    Workflow
    --------
        clf = MonoBiClassifier(b_bool_mask, names=names, coord=px)
        clf.classify(all_datasets)                  # populates clf.results
        clf.agreement(metric="jaccard")             # method-vs-method overlap
        clf.param_scatter_matrix("tau1_map", cls="bi")
        df = clf.agreed_param_table(cls="mono")

    Per-dataset result dict keys:
        {'name', 'mono', 'bi', 'combined', 'mono_mask', 'mono_frac', 'bi_frac'}
    """

    CMAP_NAMES = ("jet", "Spectral", "Spectral_r")
    PALETTE = ["#5DADE2", "#EC7063", "#58D68D", "#F4D03F", "#AF7AC5",
               "#EB984E", "#48C9B0", "#52BE80", "#AAB7B8", "#F1948A",
               "#BB8FCE", "#7FB3D5", "#76D7C4"]

    def __init__(self, b_bool_mask, names=None,
                 alpha_upper=0.95, alpha_lower=0.05,
                 tau_tol=0.01,
                 coord=None, figsize=None):
        self.roi         = np.asarray(b_bool_mask).astype(int)
        self.n_roi       = int(self.roi.sum())
        self.names       = names
        self.alpha_upper = alpha_upper
        self.alpha_lower = alpha_lower
        self.tau_tol     = tau_tol
        self.coord       = coord
        self.figsize     = figsize

        self.cmaps        = [Colorprocess().lowest_zero(n) for n in self.CMAP_NAMES]
        self.results      = []       # per-dataset result dicts
        self.all_datasets = None     # stored by classify() for the analysis methods

    # ════════════════════════════ classification ═══════════════════════════
    def classify_one(self, res, name="Dataset"):
        # Lifetime coincidence: use tolerance instead of exact float equality.
        # tau1 == tau2 almost never holds for fitted floats even when both
        # optimisers converge to the same value (e.g. 1.2000000001 != 1.2).
        tau_coincide = np.abs(
            np.asarray(res['tau1_map'], dtype=float) -
            np.asarray(res['tau2_map'], dtype=float)
        ) <= self.tau_tol

        mono_mask = ((res['alpha1_map'] > self.alpha_upper) |
                     (res['alpha1_map'] < self.alpha_lower) |
                     tau_coincide)

        # Apply ROI — mono_mask is kept ROI-restricted for consistency
        roi_bool  = self.roi.astype(bool)
        mono_mask = mono_mask & roi_bool

        mono = mono_mask.astype(int)
        bi   = (~mono_mask & roi_bool).astype(int)
        combined = mono * 1 + bi * 2   # 0=outside, 1=mono, 2=bi

        mono_frac = float(mono.sum() / self.n_roi) if self.n_roi else np.nan
        bi_frac   = float(bi.sum()   / self.n_roi) if self.n_roi else np.nan

        return {'name': name, 'mono': mono, 'bi': bi, 'combined': combined,
                'mono_mask': mono_mask,          # ROI-restricted boolean
                'mono_frac': mono_frac, 'bi_frac': bi_frac}

    def display_one(self, result):
        name = result['name']
        DataViewer().display_data(
            [result['combined'], result['mono'], result['bi']],
            structure=(1, 3), coord=self.coord,
            data_names=[f'{name} combined', f'{name} mono', f'{name} bi'],
            cmaps=self.cmaps, v_ranges=None, figsize=self.figsize,
            normalize=False, yscale='linear')

    def classify(self, all_datasets, names=None, display=True):
        """Classify every dataset; store self.results and self.all_datasets."""
        self.all_datasets = list(all_datasets)
        base = names or self.names or []
        self.names = [base[i] if i < len(base) else f"Dataset {i + 1}"
                      for i in range(len(self.all_datasets))]

        self.results = []
        for i, res in enumerate(self.all_datasets):
            r = self.classify_one(res, self.names[i])
            print(f"{self.names[i]:<18s}  mono: {r['mono_frac']:6.1%}   "
                  f"bi: {r['bi_frac']:6.1%}")
            if display:
                self.display_one(r)
            self.results.append(r)
        return self.results

    def summary(self):
        for r in self.results:
            print(f"{r['name']:<18s}  mono: {r['mono_frac']:6.1%}   "
                  f"bi: {r['bi_frac']:6.1%}")
        return self.results

    # ════════════════════════ cross-method analysis ════════════════════════
    def _require_classified(self):
        if not self.results or self.all_datasets is None:
            raise RuntimeError("Call .classify(all_datasets) before running "
                               "agreement / correlation analysis.")

    def agreement(self, metric="jaccard", classes_to_show=("mono", "bi"),
                  cmap="viridis", figsize=None, show=True):
        """
        Pairwise agreement between methods on the mono/bi classification.

        metric : 'count'    raw # pixels both methods call this class
                 'jaccard'  |A∩B| / |A∪B|   (symmetric, 0..1)
                 'fraction' |A∩B| / |A|     (asymmetric, read by row)

        Returns (dict {class: NxN matrix}, fig).
        """
        self._require_classified()
        classes, names = self.results, self.names
        N = len(classes)
        out = {}
        fig, axarr = plt.subplots(1, len(classes_to_show),
                                  figsize=figsize or (5.5 * len(classes_to_show), 5),
                                  squeeze=False)
        for c_idx, cls in enumerate(classes_to_show):
            masks = [np.asarray(classes[i][cls]).astype(bool) for i in range(N)]
            M = np.zeros((N, N))
            for i in range(N):
                ni = masks[i].sum()
                for j in range(N):
                    inter = np.sum(masks[i] & masks[j])
                    if metric == "count":
                        M[i, j] = inter
                    elif metric == "fraction":
                        M[i, j] = inter / ni if ni else 0.0
                    else:  # jaccard
                        union = np.sum(masks[i] | masks[j])
                        M[i, j] = inter / union if union else 0.0
            out[cls] = M

            ax = axarr[0, c_idx]
            vmax = M.max() if metric == "count" else 1.0
            im = ax.imshow(M, cmap=cmap, vmin=0, vmax=vmax)
            ax.set_xticks(range(N)); ax.set_yticks(range(N))
            ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(names, fontsize=8)
            fmt = "{:.0f}" if metric == "count" else "{:.2f}"
            for i in range(N):
                for j in range(N):
                    shade = M[i, j] / vmax if vmax else 0
                    ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center",
                            fontsize=7, color="white" if shade < 0.5 else "black")
            ax.set_title(f"'{cls}' agreement ({metric})")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        if show:
            plt.show()
        return out, fig

    def param_scatter_matrix(self, param="tau1_map", cls="mono",
                             agree="pairwise", max_points=3000, colors=None,
                             figsize=None, rng=None, show=True):
        """
        Cross-method correlation of ONE parameter over the pixels of ONE class.

        diagonal (i, i)  histogram of `param` for method i over its `cls` pixels
        off-diag (i, j)  scatter of method_i (y) vs method_j (x), 1:1 line + r

        agree : 'pairwise' (cell uses pixels both i & j call cls)
                'all'      (every cell uses pixels ALL methods agree on)
        """
        self._require_classified()
        classes, all_datasets, names = self.results, self.all_datasets, self.names
        rng    = np.random.default_rng() if rng is None else rng
        colors = colors or self.PALETTE
        N      = len(all_datasets)
        cmask  = [np.asarray(classes[i][cls]).astype(bool) for i in range(N)]
        pdata  = [np.asarray(all_datasets[i][param], dtype=float) for i in range(N)]
        common = np.logical_and.reduce(cmask) if agree == "all" else None

        fig, axes = plt.subplots(N, N, figsize=figsize or (2.3 * N, 2.3 * N),
                                 squeeze=False)
        for i in range(N):
            for j in range(N):
                ax = axes[i, j]
                if i == j:
                    sel = common if agree == "all" else cmask[i]
                    v = pdata[i][sel]; v = v[np.isfinite(v)]
                    if v.size:
                        ax.hist(v, bins=40, color=colors[i % len(colors)], alpha=0.85)
                    ax.set_facecolor("#f5f5f5")
                else:
                    sel = common if agree == "all" else (cmask[i] & cmask[j])
                    yv, xv = pdata[i][sel], pdata[j][sel]
                    m = np.isfinite(xv) & np.isfinite(yv)
                    xv, yv = xv[m], yv[m]
                    if xv.size:
                        if xv.size > max_points:
                            k = rng.choice(xv.size, max_points, replace=False)
                            xs, ys = xv[k], yv[k]
                        else:
                            xs, ys = xv, yv
                        ax.scatter(xs, ys, s=4, alpha=0.25, color="steelblue",
                                   edgecolors="none")
                        lo = float(min(xv.min(), yv.min()))
                        hi = float(max(xv.max(), yv.max()))
                        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
                        r = np.corrcoef(xv, yv)[0, 1] if xv.size > 2 else np.nan
                        ax.text(0.05, 0.95, f"r={r:.2f}\nn={xv.size}",
                                transform=ax.transAxes, fontsize=7, va="top",
                                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                          ec="none", alpha=0.7))
                ax.tick_params(labelsize=6)
                if i == 0:     ax.set_title(names[j], fontsize=8)
                if j == 0:     ax.set_ylabel(names[i], fontsize=8)
                if i == N - 1: ax.set_xlabel(names[j], fontsize=7)
        fig.suptitle(f"Cross-method correlation of {param}  —  class '{cls}'  "
                     f"(agree='{agree}')", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        if show:
            plt.show()
        return fig

    def agreed_param_table(self, cls="mono",
                           params=("alpha1_map", "tau1_map", "tau2_map")):
        """
        Long-form table of parameter values at pixels where ALL methods agree on
        `cls`. One row per (pixel, method) -> groupby('method').describe().

        Parameters missing from a dataset (e.g. Phasor has only 'tau_map',
        mono-exp datasets lack 'alpha1_map'/'tau2_map') are filled with NaN
        instead of raising KeyError.
        """
        self._require_classified()
        classes, all_datasets, names = self.results, self.all_datasets, self.names
        N = len(all_datasets)
        common = np.logical_and.reduce(
            [np.asarray(classes[i][cls]).astype(bool) for i in range(N)])
        n_common = int(common.sum())
        rows, cols = np.nonzero(common)
        frames = []
        for i in range(N):
            rec = {"method": names[i], "row": rows, "col": cols}
            for p in params:
                if p in all_datasets[i]:
                    rec[p] = np.asarray(all_datasets[i][p], dtype=float)[common]
                else:
                    rec[p] = np.full(n_common, np.nan)
            frames.append(pd.DataFrame(rec))
        return pd.concat(frames, ignore_index=True)


# ── backward-compatible functional wrapper ──────────────────────────────────
def classify_mono_bi(all_datasets, b_bool_mask, names=None,
                     alpha_upper=0.95, alpha_lower=0.05,
                     tau_tol=0.01,
                     coord=None, display=True, figsize=None):
    """Drop-in replacement for the original function (delegates to the class)."""
    clf = MonoBiClassifier(b_bool_mask, names=names,
                           alpha_upper=alpha_upper, alpha_lower=alpha_lower,
                           tau_tol=tau_tol,
                           coord=coord, figsize=figsize)
    clf.classify(all_datasets, display=display)
    return clf.results


# ── parameter correlation analysis (no mono/bi classification required) ──────
class ParamCorrelationMatrix:
    """
    Cross-method parameter correlation analysis over a shared ROI mask.

    Works directly on a list of parameter-map dicts (the same format as
    MonoBiClassifier's all_datasets); no classify() step is required.

    Parameter arrays may be:
      - 2-D (H, W)    : single scalar per pixel  → plotted as a plain point.
      - 3-D (H, W, N) : distribution per pixel   → plotted as mean (circle)
                         with ± std error bars.

    Parameters
    ----------
    all_datasets : list of dict
        Each dict maps parameter name → np.ndarray, either (H, W) or (H, W, N).
    bool_mask    : np.ndarray (H, W) bool
        Pixels included in every analysis.
    names        : list of str, optional
        Display labels for each dataset.  Defaults to 'Dataset 1', …

    Methods
    -------
    scatter_matrix(param, agree, max_points, colors, figsize, rng, show)
        N × N scatter / histogram grid for ONE parameter across ALL datasets.
    pairwise_scatter(idx_a, idx_b, params, max_points, colors, figsize, rng, show)
        Multi-parameter scatter grid comparing exactly TWO datasets.
    """

    PALETTE = ["#5DADE2", "#EC7063", "#58D68D", "#F4D03F", "#AF7AC5",
               "#EB984E", "#48C9B0", "#52BE80", "#AAB7B8", "#F1948A",
               "#BB8FCE", "#7FB3D5", "#76D7C4"]

    def __init__(self, all_datasets, bool_mask, names=None):
        self.all_datasets = list(all_datasets)
        self.roi          = np.asarray(bool_mask).astype(bool)
        N                 = len(self.all_datasets)
        self.names        = (list(names) if names is not None
                             else [f"Dataset {i + 1}" for i in range(N)])
        if len(self.names) != N:
            raise ValueError(
                f"names length ({len(self.names)}) must match "
                f"all_datasets length ({N})")

    # ── internal helpers ──────────────────────────────────────────────────────
    def _masked_stats(self, dataset_idx, param):
        """
        Return (mean, std, is_dist) for one parameter over the ROI.

        2-D (H, W)    → scalar per pixel : is_dist=False, std=None.
        3-D (H, W, N) → distribution    : is_dist=True,
                          mean and std computed along the last axis.
        Missing key   → NaN array, is_dist=False.
        """
        ds = self.all_datasets[dataset_idx]
        n  = int(self.roi.sum())
        if param not in ds:
            return np.full(n, np.nan), None, False
        arr = np.asarray(ds[param], dtype=float)
        if arr.ndim == 3:                          # (H, W, N) — distribution
            roi_vals = arr[self.roi]               # (n_pixels, N)
            return roi_vals.mean(axis=-1), roi_vals.std(axis=-1), True
        return arr[self.roi].ravel(), None, False  # scalar

    def _resolve_idx(self, ref):
        """Accept an int index or a dataset name string."""
        if isinstance(ref, str):
            return self.names.index(ref)
        return int(ref)

    @staticmethod
    def _subsample(rng, max_points, *arrays):
        """Randomly subsample all arrays to at most max_points rows."""
        n = arrays[0].size
        if n <= max_points:
            return arrays
        k = rng.choice(n, max_points, replace=False)
        return tuple(a[k] if a is not None else None for a in arrays)

    @staticmethod
    def _plot_points(ax, xs, ys, xe, ye, is_dist, color, alpha, ms):
        """Draw scatter (scalar) or mean±std error bars (distribution)."""
        if is_dist:
            ax.errorbar(xs, ys, xerr=xe, yerr=ye,
                        fmt='o', ms=ms, alpha=alpha, color=color,
                        elinewidth=0.6, capsize=2.0, ecolor=color, zorder=2)
        else:
            ax.scatter(xs, ys, s=ms ** 2, alpha=alpha,
                       color=color, edgecolors="none", zorder=2)

    # ── Method 1: N × N scatter matrix for one parameter ─────────────────────
    def scatter_matrix(self, param="tau1_map",
                       agree="pairwise", max_points=3000,
                       colors=None, figsize=None, rng=None, show=True):
        """
        N × N cross-method scatter matrix for a single parameter.

        Diagonal    : histogram of pixel means within the ROI.
        Off-diagonal: dataset_i (y) vs dataset_j (x) with identity line + r.
                      Scalar data → scatter points.
                      Distribution data → mean circle with ± std error bars.

        Parameters
        ----------
        param      : str   Key present in the dataset dicts.
        agree      : str   'pairwise' — each cell uses pixels finite in both
                           datasets.  'all' — restricted to pixels finite in
                           ALL datasets simultaneously.
        max_points : int   Cap on plotted points (random sub-sample).
        colors     : list  Per-dataset colours (cycles PALETTE by default).
        figsize    : tuple Figure size; auto-scaled to N if None.
        rng        : np.random.Generator  For reproducible sub-sampling.
        show       : bool  Call plt.show() when True.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        N      = len(self.all_datasets)
        rng    = np.random.default_rng() if rng is None else rng
        colors = colors or self.PALETTE

        # pre-fetch (mean, std, is_dist) for every dataset
        stats  = [self._masked_stats(i, param) for i in range(N)]
        pmeans = [s[0] for s in stats]

        # pixels finite in ALL datasets (used when agree=='all')
        all_finite = np.ones(pmeans[0].shape, dtype=bool)
        for v in pmeans:
            all_finite &= np.isfinite(v)

        fig, axes = plt.subplots(N, N,
                                 figsize=figsize or (2.3 * N, 2.3 * N),
                                 squeeze=False)

        for i in range(N):
            for j in range(N):
                ax = axes[i, j]

                if i == j:                              # ── diagonal: histogram ──
                    sel = all_finite if agree == "all" else np.isfinite(pmeans[i])
                    v   = pmeans[i][sel]
                    if v.size:
                        ax.hist(v, bins=40,
                                color=colors[i % len(colors)], alpha=0.85)
                    ax.set_facecolor("#f5f5f5")

                else:                                   # ── off-diagonal: scatter ──
                    ymean, ystd, y_is_dist = stats[i]
                    xmean, xstd, x_is_dist = stats[j]
                    sel = (all_finite if agree == "all"
                           else np.isfinite(xmean) & np.isfinite(ymean))
                    xm, ym = xmean[sel], ymean[sel]
                    xs_e = xstd[sel] if xstd is not None else None
                    ys_e = ystd[sel] if ystd is not None else None
                    is_dist = x_is_dist or y_is_dist

                    if xm.size:
                        xm, ym, xs_e, ys_e = self._subsample(
                            rng, max_points, xm, ym, xs_e, ys_e)
                        self._plot_points(ax, xm, ym, xs_e, ys_e,
                                          is_dist, "steelblue", 0.35, 3)
                        lo = float(min(xm.min(), ym.min()))
                        hi = float(max(xm.max(), ym.max()))
                        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
                        r = (np.corrcoef(xm, ym)[0, 1]
                             if xm.size > 2 else np.nan)
                        ax.text(0.05, 0.95,
                                f"r={r:.2f}\nn={xm.size}",
                                transform=ax.transAxes, fontsize=7, va="top",
                                bbox=dict(boxstyle="round,pad=0.2",
                                          fc="white", ec="none", alpha=0.7))

                ax.tick_params(labelsize=6)
                if i == 0:      ax.set_title(self.names[j], fontsize=8)
                if j == 0:      ax.set_ylabel(self.names[i], fontsize=8)
                if i == N - 1:  ax.set_xlabel(self.names[j], fontsize=7)

        fig.suptitle(
            f"Cross-method correlation — {param}  (agree='{agree}')",
            fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        if show:
            plt.show()
        return fig

    # ── Method 2: multi-parameter scatter between any two datasets ────────────
    def pairwise_scatter(self, idx_a, idx_b,
                         params=("tau1_map", "tau2_map", "alpha1_map"),
                         max_points=3000, colors=None,
                         figsize=None, rng=None, show=True):
        """
        Scatter plots for multiple parameters between exactly two datasets.

        Scalar parameter (H, W)    → plain scatter point per pixel.
        Distribution parameter (H, W, N) → mean circle with ± std error bars.

        Parameters
        ----------
        idx_a, idx_b : int or str
            Dataset for y-axis (idx_a) and x-axis (idx_b).
            Accepts integer position or name string.
        params       : sequence of str
            Parameter keys to compare; missing keys produce blank panels.
        max_points   : int   Cap on plotted points (random sub-sample).
        colors       : list  One colour per parameter panel (cycles PALETTE).
        figsize      : tuple Figure size; auto-sized to number of params if None.
        rng          : np.random.Generator  For reproducible sub-sampling.
        show         : bool  Call plt.show() when True.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        ia, ib   = self._resolve_idx(idx_a), self._resolve_idx(idx_b)
        name_a   = self.names[ia]
        name_b   = self.names[ib]
        rng      = np.random.default_rng() if rng is None else rng
        colors   = colors or self.PALETTE
        n_params = len(params)
        cols     = min(n_params, 4)
        rows     = math.ceil(n_params / cols)

        fig, axes = plt.subplots(rows, cols,
                                 figsize=figsize or (4.5 * cols, 4.0 * rows),
                                 squeeze=False)
        axes = axes.flatten()

        for p_idx, param in enumerate(params):
            ax                       = axes[p_idx]
            col                      = colors[p_idx % len(colors)]
            xmean, xstd, x_is_dist  = self._masked_stats(ib, param)
            ymean, ystd, y_is_dist  = self._masked_stats(ia, param)
            sel = np.isfinite(xmean) & np.isfinite(ymean)
            xm, ym = xmean[sel], ymean[sel]
            xs_e = xstd[sel] if xstd is not None else None
            ys_e = ystd[sel] if ystd is not None else None
            is_dist = x_is_dist or y_is_dist

            if xm.size:
                xm, ym, xs_e, ys_e = self._subsample(
                    rng, max_points, xm, ym, xs_e, ys_e)
                self._plot_points(ax, xm, ym, xs_e, ys_e,
                                  is_dist, col, 0.45, 4)
                lo = float(min(xm.min(), ym.min()))
                hi = float(max(xm.max(), ym.max()))
                ax.plot([lo, hi], [lo, hi], "k--", lw=1.0)
                r = np.corrcoef(xm, ym)[0, 1] if xm.size > 2 else np.nan
                ax.text(0.05, 0.95,
                        f"r = {r:.3f}\nn = {xm.size}",
                        transform=ax.transAxes, fontsize=9, va="top",
                        bbox=dict(boxstyle="round,pad=0.25",
                                  fc="white", ec="none", alpha=0.8))
            else:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", color="gray", fontsize=10)

            ax.set_title(param, fontsize=10)
            ax.set_xlabel(name_b, fontsize=9)
            ax.set_ylabel(name_a, fontsize=9)
            ax.tick_params(labelsize=7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        for p_idx in range(n_params, len(axes)):
            axes[p_idx].axis("off")

        fig.suptitle(
            f"Pairwise parameter correlation: {name_a}  vs  {name_b}",
            fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        if show:
            plt.show()
        return fig