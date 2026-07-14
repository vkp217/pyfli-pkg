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
    # Plotting and color-mapping methods for PhasorAnalyzer.
  
    def phasor_colormap(self, G, S, intensity=None, colormap="viridis"):
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
