from .colorProcess import Colorprocess
from .mdataViz import DataViewer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class MonoBiClassifier:
    """
    Per-dataset mono- vs bi-exponential pixel classification, plus cross-method
    agreement and parameter-correlation analysis.

    A pixel is 'mono' when its first-component fraction is saturated
    (alpha1 > alpha_upper or alpha1 < alpha_lower) or the two lifetimes coincide
    (tau1 == tau2); otherwise it is 'bi'. All maps are restricted to the ROI
    given by `b_bool_mask`.

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
                 alpha_upper=0.95, alpha_lower=0.5,
                 coord=None, figsize=None):
        self.roi         = np.asarray(b_bool_mask).astype(int)
        self.n_roi       = int(self.roi.sum())
        self.names       = names
        self.alpha_upper = alpha_upper
        self.alpha_lower = alpha_lower
        self.coord       = coord
        self.figsize     = figsize

        self.cmaps        = [Colorprocess().lowest_zero(n) for n in self.CMAP_NAMES]
        self.results      = []       # per-dataset result dicts
        self.all_datasets = None     # stored by classify() for the analysis methods

    # ════════════════════════════ classification ═══════════════════════════
    def classify_one(self, res, name="Dataset"):
        mono_mask = ((res['alpha1_map'] > self.alpha_upper) |
                     (res['alpha1_map'] < self.alpha_lower) |
                     (res['tau1_map'] == res['tau2_map']))

        mono = np.where(mono_mask, 1, 0) * self.roi
        bi   = np.where(~mono_mask, 1, 0) * self.roi
        combined = mono * 1 + bi * 2

        mono_frac = float(mono.sum() / self.n_roi) if self.n_roi else np.nan
        bi_frac   = float(bi.sum()   / self.n_roi) if self.n_roi else np.nan

        return {'name': name, 'mono': mono, 'bi': bi, 'combined': combined,
                'mono_mask': mono_mask,
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
        """
        self._require_classified()
        classes, all_datasets, names = self.results, self.all_datasets, self.names
        N = len(all_datasets)
        common = np.logical_and.reduce(
            [np.asarray(classes[i][cls]).astype(bool) for i in range(N)])
        rows, cols = np.nonzero(common)
        frames = []
        for i in range(N):
            rec = {"method": names[i], "row": rows, "col": cols}
            for p in params:
                rec[p] = np.asarray(all_datasets[i][p], dtype=float)[common]
            frames.append(pd.DataFrame(rec))
        return pd.concat(frames, ignore_index=True)


# ── backward-compatible functional wrapper ──────────────────────────────────
def classify_mono_bi(all_datasets, b_bool_mask, names=None,
                     alpha_upper=0.95, alpha_lower=0.5,
                     coord=None, display=True, figsize=None):
    """Drop-in replacement for the original function (delegates to the class)."""
    clf = MonoBiClassifier(b_bool_mask, names=names,
                           alpha_upper=alpha_upper, alpha_lower=alpha_lower,
                           coord=coord, figsize=figsize)
    clf.classify(all_datasets, display=display)
    return clf.results