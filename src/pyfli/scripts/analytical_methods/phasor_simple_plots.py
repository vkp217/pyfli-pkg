"""Plotting and color-mapping mixin for phasor-based FLI analysis.

Contains all matplotlib visualization methods (phasor diagrams, lifetime
maps, colored overlays and per-pixel decay fit plots) used by
`PhasorAnalyzer`. Intended to be mixed into `PhasorAnalyzer` rather than
instantiated directly, since its methods rely on attributes such as
``self.eps``, ``self.frequency``, ``self.device`` and helper methods like
``self.lifetime_to_phasor`` / ``self.compute_lifetime`` defined on that
class.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from .phasor_simple_utils import (
    _TAU_MARKS_NS,
    _universal_circle_xy,
    _draw_lifetime_ticks,
    _style_phasor_ax,
)


class PhasorPlotsMixin:
    """Mixin providing plotting and color-mapping methods for `PhasorAnalyzer`.

    All methods assume they are mixed into a class (namely
    `PhasorAnalyzer`) that defines ``self.eps``, ``self.frequency``,
    ``self.device``, ``self.time_axis_ns`` and phasor/lifetime conversion
    helpers.
    """

    def phasor_colormap(self, G, S, intensity=None, colormap="viridis"):
        """Map phasor magnitude to RGB colors, optionally scaled by intensity.

        Uses the first-harmonic G/S maps (or the maps directly if 2D) to
        compute the phasor magnitude ``sqrt(G**2 + S**2)``, min-max
        normalizes it, and looks it up in ``colormap``. If ``intensity``
        is given, the resulting colors are scaled by the min-max
        normalized intensity so low-intensity pixels appear dimmer.

        Args:
            G: G phasor map, shape (H, W) or (n_harmonics, H, W).
            S: S phasor map, shape (H, W) or (n_harmonics, H, W).
            intensity: Optional intensity image, shape (H, W), used to
                scale the brightness of the returned colors.
            colormap: Name of a matplotlib colormap to sample.

        Returns:
            np.ndarray: RGB image of shape (H, W, 3) with values in [0, 1].
        """
        G_col = G[0] if G.ndim == 3 else G
        S_col = S[0] if S.ndim == 3 else S
        phasor_val = np.sqrt(G_col ** 2 + S_col ** 2)
        p_min, p_max = np.nanmin(phasor_val), np.nanmax(phasor_val)
        phasor_val = (phasor_val - p_min) / (p_max - p_min + self.eps)
        colors = plt.colormaps[colormap](phasor_val)[:, :, :3]
        if intensity is not None:
            denom     = intensity.max() - intensity.min() + self.eps
            int_norm  = (intensity - intensity.min()) / denom
            colors    = colors * int_norm[:, :, np.newaxis]
        return colors

    def phasor_radial_color(self, G, S, colormap="viridis",
                            norm_color=False, half_circle=True):
        """Color pixels by their angle and radius relative to the universal circle.

        Computes each pixel's angle ``phi`` and radial distance ``r``
        relative to the universal-circle center (0.5, 0), maps ``phi`` to
        a hue via ``colormap`` and uses ``r`` to control value/brightness:
        pixels inside the circle are dimmed toward the center, pixels
        outside are desaturated (blended toward white) the farther out
        they are. NaN phasor values are mapped to black.

        Args:
            G: G phasor map, shape (H, W) or (n_harmonics, H, W) (first
                harmonic is used if 3D).
            S: S phasor map, same shape convention as ``G``.
            colormap: Name of a matplotlib colormap used for the hue.
            norm_color: If True, normalize the angle range to the data's
                own [min, max] before color lookup; otherwise use a fixed
                angular range (0 to pi for half circle, else -pi to pi).
            half_circle: Selects the fixed angular range used when
                ``norm_color`` is False.

        Returns:
            np.ndarray: RGB image of shape (H, W, 3) with values in [0, 1].
        """

        G_col = G[0] if G.ndim == 3 else np.asarray(G)
        S_col = S[0] if S.ndim == 3 else np.asarray(S)

        # Angle and radius relative to universal circle centre (0.5, 0)
        dG  = G_col - 0.5
        dS  = S_col
        phi = np.arctan2(dS, dG)
        r   = np.sqrt(dG ** 2 + dS ** 2)

        # Normalise angle to [0, 1] for colormap lookup
        if norm_color:
            phi_min = np.nanmin(phi)
            phi_max = np.nanmax(phi)
            if phi_max <= phi_min:
                phi_max = phi_min + 1.0
            H = np.clip((phi - phi_min) / (phi_max - phi_min + self.eps), 0, 1)
        else:
            phi_lo = 0.0 if half_circle else -np.pi
            phi_hi = np.pi
            H = np.clip((phi - phi_lo) / (phi_hi - phi_lo + self.eps), 0, 1)

        # Base colour from named colormap
        base_rgb = plt.colormaps[colormap](H)[..., :3]

        norm_r = r / 0.5
        V      = np.clip(norm_r, 0, 1)

        # Inside/on circle: dim proportionally toward centre
        colors = base_rgb * V[..., np.newaxis]

        # Outside circle: restore brightness, blend toward white (desaturate)
        outside = norm_r > 1.0
        if outside.any():
            sat = np.clip(1.0 / (norm_r[outside] + self.eps), 0, 1)
            colors[outside] = (base_rgb[outside] * sat[:, np.newaxis]
                               + (1.0 - sat[:, np.newaxis]))

        nan_mask = np.isnan(phi) | np.isnan(r)
        if nan_mask.any():
            colors[nan_mask] = 0.0

        return colors

    
    def plot_phasor_diagram(self, G, S, mask=None, colors=None,
                            hexbin_color=None, ax=None, figsize=(8, 3),
                            half_circle=True, title="Phasor Diagram",
                            xlim=(-0.1, 1.1), ylim=(0.0, 0.6)):
        """Draw a G/S phasor scatter/hexbin diagram with the universal circle.

        Plots the universal (semi)circle, then the pixel-wise (G, S)
        points either as a hexbin density plot (default, or when
        ``hexbin_color`` is set and ``colors`` is None), as a scatter
        colored by a named colormap (``colors`` is a string), or as a
        scatter with explicit per-point RGB colors (``colors`` is an
        array). NaN and masked-out points are excluded. Reference
        lifetime ticks are drawn on top via ``_draw_lifetime_ticks``.

        Args:
            G: G phasor map, shape (H, W) or (n_harmonics, H, W) (first
                harmonic used if 3D).
            S: S phasor map, same shape convention as ``G``.
            mask: Optional boolean mask selecting which pixels to plot.
            colors: Either None (use hexbin), a colormap name (scatter
                colored by G value), or an (H, W, 3) / flattened RGB array
                (scatter with explicit colors).
            hexbin_color: Colormap name for the hexbin density plot; used
                only when ``colors`` is None.
            ax: Existing matplotlib axis to draw on; a new figure/axis is
                created if None.
            figsize: Figure size used when ``ax`` is None.
            half_circle: If True, draw only the upper half of the
                universal circle.
            title: Axis title.
            xlim: G-axis limits.
            ylim: S-axis limits.

        Returns:
            matplotlib.figure.Figure: The figure the diagram was drawn on.
        """
        created_fig = ax is None
        if created_fig:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        ug, us = _universal_circle_xy(half_circle=half_circle)
        ax.plot(ug, us, "k--")

        G_2d   = G[0] if (np.ndim(G) == 3) else np.asarray(G)
        S_2d   = S[0] if (np.ndim(S) == 3) else np.asarray(S)
        g_flat = np.ravel(G_2d)
        s_flat = np.ravel(S_2d)

        if mask is not None:
            mask_flat = np.ravel(mask).astype(bool)  # uint8 masks must be bool; int array would integer-index otherwise
            g_plot = g_flat[mask_flat]
            s_plot = s_flat[mask_flat]
        else:
            g_plot = g_flat
            s_plot = s_flat

        valid  = ~np.isnan(g_plot) & ~np.isnan(s_plot)
        g_plot = g_plot[valid]
        s_plot = s_plot[valid]

        if colors is None:
            cmap_to_use = hexbin_color if hexbin_color is not None else 'autumn'
            hb = ax.hexbin(g_plot, s_plot, gridsize=100, cmap=cmap_to_use, mincnt=1)
            fig.colorbar(hb, ax=ax).set_label("Pixel Count")
        else:
            if isinstance(colors, str):
                c_vals = g_plot
                path   = ax.scatter(g_plot, s_plot, cmap=colors, c=c_vals, s=8, marker="o")
                fig.colorbar(path, ax=ax).set_label("Phasor G Value")
            else:
                c_flat = np.reshape(colors, (-1, 3))
                if mask is not None:
                    c_plot = c_flat[mask_flat][valid]
                else:
                    c_plot = c_flat[valid]
                ax.scatter(g_plot, s_plot, c=c_plot, s=8, marker="o")

        G_mark, S_mark = self.lifetime_to_phasor(_TAU_MARKS_NS, self.frequency)
        _draw_lifetime_ticks(ax, G_mark, S_mark,
                             color="black", lw=4, fontsize=10, show_units=True)
        _style_phasor_ax(ax, title=title,
                         xlim=xlim, ylim=ylim,
                         half_circle=half_circle)

        if created_fig:
            plt.tight_layout()
        return fig

    
    def plot_map(self, image, scales=[0, 2], title="", ax=None, figsize=(8, 6)):
        """Display a 2D image (e.g. a lifetime map) with a colorbar.

        Clips ``image`` to ``scales`` and renders it with the "viridis"
        colormap.

        Args:
            image: 2D array to display.
            scales: Two-element [vmin, vmax] used to clip the image before
                display.
            title: Axis title.
            ax: Existing matplotlib axis to draw on; a new figure/axis is
                created if None.
            figsize: Figure size used when ``ax`` is None.

        Returns:
            matplotlib.figure.Figure: The figure the image was drawn on.
        """
        created_fig = ax is None
        if created_fig:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        im = ax.imshow(np.clip(image, scales[0], scales[1]),
                       origin="upper", cmap="viridis")
        fig.colorbar(im, ax=ax).set_label("Lifetime (ns)")
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(False)
        if created_fig:
            plt.tight_layout()
        return fig

    def plot_phasor_overlay(self, decay, G, S, colormap="viridis",
                            ax=None, figsize=(8, 8)):
        """Overlay phasor-derived colors on the intensity image.

        Combines the intensity image (from `generate_intensity_image`,
        min-max normalized) with the phasor colormap (from
        `phasor_colormap`) by multiplying them channel-wise, so brighter
        pixels show more saturated phasor color.

        Args:
            decay: Time-resolved decay stack, shape (H, W, T), used to
                derive the intensity image.
            G: G phasor map passed to `phasor_colormap`.
            S: S phasor map passed to `phasor_colormap`.
            colormap: Name of the matplotlib colormap used for phasor
                colors.
            ax: Existing matplotlib axis to draw on; a new figure/axis is
                created if None.
            figsize: Figure size used when ``ax`` is None.

        Returns:
            matplotlib.figure.Figure: The figure the overlay was drawn on.
        """
        created_fig = ax is None
        if created_fig:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        intensity_img  = self.generate_intensity_image(decay)
        phasor_colors  = self.phasor_colormap(G, S, colormap=colormap)
        int_norm       = (intensity_img - intensity_img.min()) / \
                         (intensity_img.max() - intensity_img.min() + self.eps)
        overlay = np.stack([int_norm] * 3, axis=2) * phasor_colors
        ax.imshow(overlay, origin="upper")
        ax.set_title("Intensity + Phasor Color Overlay")
        ax.axis("off")
        if created_fig:
            plt.tight_layout()
        return fig

    def plot_pure_phasor_map(self, G, S, decay, noise_removed=True,
                             colormap="viridis", ax=None, figsize=(4, 4)):
        """Show phasor colors alone, optionally masking low-intensity pixels.

        Computes phasor colors via `phasor_colormap` and, if
        ``noise_removed`` is True, zeroes out pixels whose normalized
        intensity (from `generate_intensity_image`) is below 0.1 so noisy
        background pixels are suppressed.

        Args:
            G: G phasor map passed to `phasor_colormap`.
            S: S phasor map passed to `phasor_colormap`.
            decay: Time-resolved decay stack, shape (H, W, T), used to
                derive the intensity mask when ``noise_removed`` is True.
            noise_removed: If True, mask out low-intensity pixels.
            colormap: Name of the matplotlib colormap used for phasor
                colors.
            ax: Existing matplotlib axis to draw on; a new figure/axis is
                created if None.
            figsize: Figure size used when ``ax`` is None.

        Returns:
            matplotlib.figure.Figure: The figure the map was drawn on.
        """
        phasor_colors = self.phasor_colormap(G, S, colormap=colormap)
        if phasor_colors.shape[-1] == 4:
            phasor_colors = phasor_colors[..., :3]

        if noise_removed:
            intensity_img = self.generate_intensity_image(decay)
            i_min, i_max  = intensity_img.min(), intensity_img.max()
            denom         = (i_max - i_min) if (i_max - i_min) != 0 else 1
            int_norm      = (intensity_img - i_min) / denom
            final_mask    = int_norm > 0.1
        else:
            final_mask = np.ones(phasor_colors.shape[:2], dtype=bool)

        pure_overlay = np.zeros_like(phasor_colors)
        pure_overlay[final_mask] = phasor_colors[final_mask]

        created_fig = ax is None
        if created_fig:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        ax.imshow(pure_overlay, origin="upper")
        ax.set_title(f"Pure Phasor Map (Noise Removed: {noise_removed})")
        ax.axis("off")
        if created_fig:
            plt.tight_layout()
        return fig

    def plot_overlay_subplots(self, decay, G, S, mask=None,
                              colormaps=["jet", "jet", "viridis", "jet"],
                              noise_removed=True, figsize=(15, 10),
                              half_circle=True,
                              xlim=(-0.1, 1.1), ylim=(0.0, 0.6),
                              bg_color="black", transpose=False):
        """Render a 2x3 grid of intensity, lifetime and phasor diagnostic panels.

        Builds a figure with: intensity image, radial-colored phasor
        projection, lifetime map (`compute_lifetime`), intensity-weighted
        phasor overlay, a colored phasor scatter, and a phasor hexbin
        density plot. Pixels are restricted to ``mask`` if given, else to
        an intensity-derived active mask when ``noise_removed`` is True.

        Args:
            decay: Time-resolved decay stack, shape (H, W, T).
            G: G phasor map, shape (H, W) or (n_harmonics, H, W).
            S: S phasor map, same shape convention as ``G``.
            mask: Optional boolean mask of active pixels; overrides the
                intensity-derived mask when given.
            colormaps: Four colormap names/specs used for
                [intensity panel, lifetime panel, radial-color hue,
                hexbin density panel].
            noise_removed: If True (and ``mask`` is None), derive an
                active-pixel mask from normalized intensity > 0.1.
            figsize: Overall figure size.
            half_circle: If True, draw only the upper half of the
                universal circle in the phasor panels.
            xlim: G-axis limits for the phasor scatter/hexbin panels.
            ylim: S-axis limits for the phasor scatter/hexbin panels.
            bg_color: Background color ("black" or any other value,
                treated as white) used for masked-out pixels.
            transpose: If True, swap the first two axes of each 2D image
                before display.

        Returns:
            matplotlib.figure.Figure: The assembled figure.
        """
        _t = (lambda a: np.swapaxes(a, 0, 1)) if transpose else (lambda a: a)

        G_2d          = G[0] if G.ndim == 3 else G
        S_2d          = S[0] if S.ndim == 3 else S
        intensity_img = self.generate_intensity_image(decay)

        phasor_colors_raw = self.phasor_radial_color(G_2d, S_2d,
                                                      colormap=colormaps[2],
                                                      half_circle=half_circle)

        if mask is None:
            if noise_removed:
                int_norm    = (intensity_img - intensity_img.min()) / \
                              (intensity_img.max() - intensity_img.min() + self.eps)
                active_mask = int_norm > 0.1
            else:
                active_mask = np.ones(G_2d.shape, dtype=bool)
        else:
            active_mask = mask.astype(bool)

        bg_val = 0.0 if bg_color == "black" else 1.0

        def _resolve_cmap(spec):
            cmap = plt.colormaps[spec] if isinstance(spec, str) else spec
            cmap = cmap.copy()
            cmap.set_bad(bg_color)
            return cmap

        cmap1 = _resolve_cmap(colormaps[0])
        cmap3 = _resolve_cmap(colormaps[1])

        fig = plt.figure(figsize=figsize)
        gs  = gridspec.GridSpec(2, 3, figure=fig)

        # (0,0) Intensity — inactive pixels → NaN → bg_color
        ax1           = fig.add_subplot(gs[0, 0])
        int_masked    = np.where(active_mask, intensity_img.astype(float), np.nan)
        im1           = ax1.imshow(_t(int_masked), origin="upper", cmap=cmap1)
        ax1.set_title("Intensity")
        ax1.axis("off")
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # (0,1) Phasor colour projection — inactive pixels → bg_val
        ax2          = fig.add_subplot(gs[0, 1])
        pure_overlay = np.full_like(phasor_colors_raw, bg_val)
        pure_overlay[active_mask] = phasor_colors_raw[active_mask]
        ax2.imshow(_t(pure_overlay), origin="upper")
        ax2.set_title("Phasor Color Projections")
        ax2.axis("off")

        # (1,0) Lifetime map — inactive pixels → NaN → bg_color
        ax3        = fig.add_subplot(gs[1, 0])
        tau_map_ns = np.clip(self.compute_lifetime(G_2d, S_2d), 0, None)
        tau_masked = np.where(active_mask, tau_map_ns, np.nan)
        im3        = ax3.imshow(_t(tau_masked), origin="upper", cmap=cmap3)
        ax3.set_title("Lifetime Map (ns)")
        ax3.axis("off")
        fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04).set_label("ns")

        # (1,1) Intensity-weighted overlay — inactive pixels → bg_val
        ax4          = fig.add_subplot(gs[1, 1])
        int_norm_3d  = (intensity_img - intensity_img.min()) / \
                       (intensity_img.max() - intensity_img.min() + self.eps)
        weighted_overlay = np.full_like(phasor_colors_raw, bg_val)
        weighted_overlay[active_mask] = (
            np.stack([int_norm_3d] * 3, axis=2) * phasor_colors_raw
        )[active_mask]
        ax4.imshow(_t(weighted_overlay), origin="upper")
        ax4.set_title("Intensity-weighted Overlay")
        ax4.axis("off")

        mask_flat = np.ravel(active_mask)
        g_plot    = np.ravel(G_2d)[mask_flat]
        s_plot    = np.ravel(S_2d)[mask_flat]
        c_plot    = np.reshape(phasor_colors_raw, (-1, 3))[mask_flat]
        valid     = ~np.isnan(g_plot) & ~np.isnan(s_plot)
        g_v, s_v, c_v = g_plot[valid], s_plot[valid], c_plot[valid]

        ug, us = _universal_circle_xy(half_circle=half_circle)
        try:
            G_mark, S_mark = self.lifetime_to_phasor(_TAU_MARKS_NS, self.frequency)
        except Exception:
            G_mark = S_mark = None

        # scatter color by phasor_radial_color
        ax5 = fig.add_subplot(gs[0, 2])
        ax5.plot(ug, us, "k--", alpha=0.8, zorder=1)
        if len(g_v):
            ax5.scatter(g_v, s_v, c=c_v, s=2, alpha=0.6, edgecolors='none', zorder=2)
        if G_mark is not None:
            _draw_lifetime_ticks(ax5, G_mark, S_mark, color="black", lw=2, fontsize=9)
        _style_phasor_ax(ax5, title="Phasor (colour)",
                         xlim=xlim, ylim=ylim, half_circle=half_circle)

        # ax6 (1,2): phasor hexbin density plot
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.plot(ug, us, "k--", alpha=0.8, zorder=1)
        if len(g_v):
            hb = ax6.hexbin(g_v, s_v, gridsize=100, cmap=colormaps[3], mincnt=1, zorder=2)
            fig.colorbar(hb, ax=ax6, fraction=0.046, pad=0.04).set_label("Pixel count")
        if G_mark is not None:
            _draw_lifetime_ticks(ax6, G_mark, S_mark, color="black", lw=2, fontsize=9)
        _style_phasor_ax(ax6, title="Phasor (density)",
                         xlim=xlim, ylim=ylim, half_circle=half_circle)

        plt.tight_layout()
        return fig

    # ── pixel-level decay fits ────────────────────────────────────────────────

    def plot_pixel_fit(self, irf, decay, reconstructed_decay, x, y,
                       log_scale=True, ax=None, figsize=(10, 6)):
        """Plot IRF, raw decay and reconstructed fit for a single pixel.

        Extracts the (x, y) trace from ``irf`` (if per-pixel), ``decay``
        and ``reconstructed_decay``, normalizes each trace by its own max
        (via a GPU/CPU tensor on ``self.device``), and plots them together
        against ``self.time_axis_ns``.

        Args:
            irf: Instrument response function, shape (H, W, T) for a
                per-pixel IRF or (T,) for a single shared IRF.
            decay: Raw decay stack, shape (H, W, T).
            reconstructed_decay: Fitted/reconstructed decay stack, shape
                (H, W, T).
            x: Pixel column index.
            y: Pixel row index.
            log_scale: If True, use a log y-axis (clipped to [1e-3, 1.2]).
            ax: Existing matplotlib axis to draw on; a new figure/axis is
                created if None.
            figsize: Figure size used when ``ax`` is None.

        Returns:
            matplotlib.figure.Figure: The figure the traces were drawn on.
        """
        irf_trace = irf[y, x, :] if irf.ndim == 3 else np.asarray(irf)
        raw_trace = decay[y, x, :]
        fit_trace = reconstructed_decay[y, x, :]

        traces_np = np.stack([irf_trace, raw_trace, fit_trace], axis=0).astype(np.float32)
        traces_t  = torch.tensor(traces_np, device=self.device)
        maxvals   = traces_t.amax(dim=1, keepdim=True).clamp(min=self.eps)
        norm_t    = (traces_t / maxvals).cpu().numpy()

        irf_norm, raw_norm, fit_norm = norm_t[0], norm_t[1], norm_t[2]

        created_fig = ax is None
        if created_fig:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        ax.plot(self.time_axis_ns, irf_norm,
                "k--", alpha=0.5, label="IRF (Normalized)")
        ax.plot(self.time_axis_ns, raw_norm,
                "ro", markersize=4, alpha=0.6, label=f"Raw Decay (Pixel {x},{y})")
        ax.plot(self.time_axis_ns, fit_norm,
                "b-", lw=2, label="Reconstructed Fit")

        if log_scale:
            ax.set_yscale("log")
            ax.set_ylim(1e-3, 1.2)
            ax.set_ylabel("Normalized Intensity (Log Scale)")
        else:
            ax.set_ylabel("Normalized Intensity (Linear Scale)")

        ax.set_xlabel("Time (ns)")
        ax.set_title(f"Decay Analysis at Pixel (X: {x}, Y: {y})  "
                     f"[device: {self.device}]")
        ax.legend()
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        if created_fig:
            plt.tight_layout()
        return fig

    def plot_pixel_fit_single_exp(self, irf, decay, tau_ns, x, y,
                                  log_scale=True, ax=None, figsize=(10, 6)):
        """Plot IRF, raw decay and a single-exponential fit for one pixel.

        Builds a mono-exponential model ``exp(-t/tau_ns)``, convolves it
        with the (normalized) IRF via `_convolve_batch`, and plots the
        result alongside the normalized IRF and raw decay traces for
        pixel (x, y).

        Args:
            irf: Instrument response function, shape (H, W, T) for a
                per-pixel IRF or (T,) for a single shared IRF.
            decay: Raw decay stack, shape (H, W, T).
            tau_ns: Lifetime in nanoseconds; either a scalar, or a
                (H, W)+ array/tensor from which the value at (y, x) is
                taken.
            x: Pixel column index.
            y: Pixel row index.
            log_scale: If True, use a log y-axis (clipped to [1e-3, 1.2]).
            ax: Existing matplotlib axis to draw on; a new figure/axis is
                created if None.
            figsize: Figure size used when ``ax`` is None.

        Returns:
            matplotlib.figure.Figure: The figure the traces were drawn on.
        """
        if isinstance(tau_ns, (torch.Tensor, np.ndarray)):
            if tau_ns.ndim >= 2:
                tau_val = tau_ns[y, x]
            else:
                tau_val = tau_ns
            if torch.is_tensor(tau_val):
                tau_val = tau_val.item()
        else:
            tau_val = tau_ns

        t_ns_t  = torch.tensor(self.t_s_np * 1e9, dtype=torch.float32,
                                device=self.device)
        model_t = torch.exp(-t_ns_t / tau_val).unsqueeze(0)

        irf_trace_np = irf[y, x, :] if irf.ndim == 3 else np.asarray(irf)
        irf_trace_t  = torch.tensor(
            irf_trace_np.astype(np.float32), device=self.device
        ).unsqueeze(0)

        irf_norm_t  = irf_trace_t / irf_trace_t.sum(dim=1, keepdim=True).clamp(min=self.eps)
        fit_t       = self._convolve_batch(model_t, irf_norm_t)
        fit_trace_np = fit_t.squeeze(0).cpu().numpy()

        raw_trace_np = decay[y, x, :]

        traces_np = np.stack(
            [irf_trace_np, raw_trace_np, fit_trace_np], axis=0
        ).astype(np.float32)
        traces_t = torch.tensor(traces_np, device=self.device)
        maxvals  = traces_t.amax(dim=1, keepdim=True).clamp(min=self.eps)
        norm_t   = (traces_t / maxvals).cpu().numpy()

        irf_norm, raw_norm, fit_norm = norm_t[0], norm_t[1], norm_t[2]

        created_fig = ax is None
        if created_fig:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        ax.plot(self.time_axis_ns, irf_norm,
                "k--", alpha=0.5, label="IRF (Normalized)")
        ax.plot(self.time_axis_ns, raw_norm,
                "ro", markersize=4, alpha=0.6, label=f"Raw Decay (Pixel {x},{y})")
        ax.plot(self.time_axis_ns, fit_norm,
                "b-", lw=2, label=f"Single-Exp Fit  τ = {tau_val} ns")

        if log_scale:
            ax.set_yscale("log")
            ax.set_ylim(1e-3, 1.2)
            ax.set_ylabel("Normalized Intensity (Log Scale)")
        else:
            ax.set_ylabel("Normalized Intensity (Linear Scale)")

        ax.set_xlabel("Time (ns)")
        ax.set_title(
            f"Single-Exponential Decay at Pixel (X: {x}, Y: {y})  "
            f"τ = {tau_val} ns  [device: {self.device}]"
        )
        ax.legend()
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        if created_fig:
            plt.tight_layout()
        return fig

    # ── multi-harmonic & traceable ────────────────────────────────────────────

    def plot_phasor_harmonics(self, G, S, harmonics=(1, 2, 3, 4), mask=None,
                              colors=None, hexbin_color=None, figsize=(22, 5),
                              axes=None, half_circle=True,
                              xlim=(-0.1, 1.1), ylim=(0.0, 0.6)):
        """Draw one phasor diagram panel per requested harmonic.

        For each harmonic index in ``harmonics``, plots the corresponding
        (G, S) slice (falling back to the first harmonic/2D map if the
        requested index is unavailable) as a hexbin or scatter, with
        harmonic-specific reference lifetime ticks.

        Args:
            G: G phasor map, shape (H, W) or (n_harmonics, H, W).
            S: S phasor map, same shape convention as ``G``.
            harmonics: Sequence of 1-based harmonic indices to plot, one
                panel each.
            mask: Optional boolean mask selecting which pixels to plot.
            colors: Either None (use hexbin), a colormap name (scatter
                colored by G value), or an RGB array (optionally with a
                leading harmonic axis) for explicit per-point colors.
            hexbin_color: Colormap name for the hexbin density plot; used
                only when ``colors`` is None.
            figsize: Overall figure size used when ``axes`` is None.
            axes: Existing sequence of matplotlib axes (one per harmonic)
                to draw on; a new figure/axes is created if None.
            half_circle: If True, draw only the upper half of the
                universal circle.
            xlim: G-axis limits.
            ylim: S-axis limits.

        Returns:
            matplotlib.figure.Figure: The figure containing all harmonic
            panels.
        """
        G = np.asarray(G)
        S = np.asarray(S)
        n_panels = len(harmonics)

        created_fig = axes is None
        if created_fig:
            fig, axes = plt.subplots(1, n_panels, figsize=figsize)
            if n_panels == 1:
                axes = [axes]
        else:
            fig = axes[0].get_figure()
        mask_flat = np.ravel(mask).astype(bool) if mask is not None else None

        for ax, k in zip(axes, harmonics):
            if G.ndim == 3 and k <= G.shape[0]:
                g_panel = G[k - 1]
                s_panel = S[k - 1]
            else:
                g_panel = G[0] if G.ndim == 3 else G
                s_panel = S[0] if S.ndim == 3 else S

            ug, us = _universal_circle_xy(half_circle=half_circle)
            ax.plot(ug, us, "k--", lw=1.2)

            g_flat = np.ravel(g_panel)
            s_flat = np.ravel(s_panel)

            if mask_flat is not None:
                g_plot = g_flat[mask_flat]
                s_plot = s_flat[mask_flat]
            else:
                g_plot = g_flat
                s_plot = s_flat

            valid  = ~np.isnan(g_plot) & ~np.isnan(s_plot)
            g_plot = g_plot[valid]
            s_plot = s_plot[valid]

            if colors is None:
                cmap_to_use = hexbin_color if hexbin_color is not None else 'jet'
                hb = ax.hexbin(g_plot, s_plot, gridsize=100, cmap=cmap_to_use, mincnt=1)
                fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04).set_label("Pixel Count")
            else:
                if isinstance(colors, str):
                    path = ax.scatter(g_plot, s_plot, cmap=colors, c=g_plot, s=8, marker="o")
                    fig.colorbar(path, ax=ax, fraction=0.046, pad=0.04).set_label("Phasor G Value")
                else:
                    # Bug fix 2: if colors has a harmonic axis (n, H, W, 3) use slice k-1;
                    # otherwise treat as a single (H, W, 3) array shared across all harmonics
                    colors_arr = np.asarray(colors)
                    if colors_arr.ndim == 4 and k - 1 < colors_arr.shape[0]:
                        c_panel = colors_arr[k - 1]
                    else:
                        c_panel = colors_arr
                    c_flat = np.reshape(c_panel, (-1, 3))
                    if mask_flat is not None:
                        c_plot = c_flat[mask_flat][valid]
                    else:
                        c_plot = c_flat[valid]
                    ax.scatter(g_plot, s_plot, c=c_plot, s=8, marker="o")

            G_mark, S_mark = self.lifetime_to_phasor(_TAU_MARKS_NS, k * self.frequency)
            _draw_lifetime_ticks(ax, G_mark, S_mark,
                                 color="black", lw=2, fontsize=7,
                                 show_units=(k == harmonics[0]),
                                 tick_length=0.03, text_offset=0.05)
            _style_phasor_ax(ax,
                             title=f"Harmonic {k} ($\omega_{{{k}}}$)",
                             xlim=xlim, ylim=ylim,
                             half_circle=half_circle)

        if created_fig:
            fig.suptitle("Phasor Diagram — Multiple Harmonics",
                         fontsize=12, fontweight="bold", y=1.01)
            plt.tight_layout()
        return fig

    def plot_traceable_analysis(self, G, S, decay, mask=None,
                                colormap="viridis", figsize=(14, 6),
                                axes=None, half_circle=True,
                                xlim=(-0.1, 1.1), ylim=(0.0, 0.6)):
        """Draw a two-panel view: radial-colored phasor map plus scatter distribution.

        The left panel shows the phasor-radial-colored projection
        (`phasor_radial_color`) restricted to ``mask``, with a colorbar
        approximating the lifetime range spanned by the first-quadrant
        phase angles. The right panel shows a scatter of (G, S) points
        colored the same way, over the universal circle with reference
        lifetime ticks.

        Args:
            G: G phasor map, shape (H, W) or (n_harmonics, H, W) (first
                harmonic used if 3D).
            S: S phasor map, same shape convention as ``G``.
            decay: Unused directly by this method; present for interface
                consistency with related plotting methods.
            mask: Optional boolean mask of active pixels; if None, all
                pixels are considered active.
            colormap: Name of the matplotlib colormap used for the
                lifetime colorbar in the left panel.
            figsize: Overall figure size used when ``axes`` is None.
            axes: Existing pair of matplotlib axes to draw on; a new
                figure/axes is created if None.
            half_circle: If True, draw only the upper half of the
                universal circle.
            xlim: G-axis limits for the scatter panel.
            ylim: S-axis limits for the scatter panel.

        Returns:
            matplotlib.figure.Figure: The figure containing both panels.
        """
        G_2d = G[0] if G.ndim == 3 else G
        S_2d = S[0] if S.ndim == 3 else S

        phasor_colors_raw = self.phasor_radial_color(G_2d, S_2d, half_circle=half_circle)

        if mask is None:
            active_mask = np.ones(G_2d.shape, dtype=bool)
        else:
            active_mask = mask.astype(bool)

        created_fig = axes is None
        if created_fig:
            fig, axes = plt.subplots(1, 2, figsize=figsize)
        else:
            fig = axes[0].get_figure()

        pure_overlay = np.zeros_like(phasor_colors_raw)
        pure_overlay[active_mask] = phasor_colors_raw[active_mask]
        axes[0].imshow(pure_overlay, origin="upper")
        axes[0].set_title("Phasor Color Projections")
        axes[0].axis("off")

        phi         = np.arctan2(S_2d, G_2d)
        first_q     = phi[(G_2d > 0) & (S_2d > 0)]
        phi_min_val = float(np.nanmin(first_q)) if first_q.size > 0 else 0.0
        phi_max_val = float(np.nanmax(first_q)) if first_q.size > 0 else 0.5
        omega       = 2 * np.pi * self.frequency
        tau_min     = np.tan(np.clip(phi_min_val, 1e-6, np.pi / 2 - 0.05)) / omega * 1e9
        tau_max     = np.tan(np.clip(phi_max_val, 1e-6, np.pi / 2 - 0.05)) / omega * 1e9

        sm = ScalarMappable(cmap=plt.colormaps[colormap],
                            norm=Normalize(vmin=tau_min, vmax=tau_max))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes[0], fraction=0.046, pad=0.04)
        cbar.set_label("Lifetime (ns)")

        ug, us = _universal_circle_xy(half_circle=half_circle)
        axes[1].plot(ug, us, "k--", alpha=0.8, zorder=1)

        mask_flat  = np.ravel(active_mask)
        g_plot     = np.ravel(G_2d)[mask_flat]
        s_plot     = np.ravel(S_2d)[mask_flat]
        c_plot     = np.reshape(phasor_colors_raw, (-1, 3))[mask_flat]
        valid_data = ~np.isnan(g_plot) & ~np.isnan(s_plot)
        g_plot, s_plot, c_plot = g_plot[valid_data], s_plot[valid_data], c_plot[valid_data]

        if len(g_plot) > 0:
            axes[1].scatter(g_plot, s_plot, c=c_plot,
                            s=10, alpha=0.8, edgecolors='none', zorder=2)
        else:
            axes[1].text(0.5, 0.25, "No valid data points", ha='center', color='red')

        try:
            G_mark, S_mark = self.lifetime_to_phasor(_TAU_MARKS_NS, self.frequency)
            _draw_lifetime_ticks(axes[1], G_mark, S_mark, color="black", lw=2, fontsize=9)
        except Exception:
            pass

        _style_phasor_ax(axes[1], title="Phasor Distribution",
                         xlim=xlim, ylim=ylim,
                         half_circle=half_circle)

        if created_fig:
            plt.tight_layout()
        return fig
