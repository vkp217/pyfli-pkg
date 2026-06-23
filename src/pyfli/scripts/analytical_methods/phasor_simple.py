import numpy as np
import torch
import matplotlib.pyplot as plt
import h5py
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

_TAU_MARKS_NS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                           0.9, 1, 1.5, 2, 3, 5, 7, 10])
_UNIVERSAL_CIRCLE_CENTER = (0.5, 0.0)
_UNIVERSAL_CIRCLE_RADIUS = 0.5

def _universal_circle_xy(n_points: int = 500, half_circle: bool = False):
    theta = np.linspace(0, np.pi if half_circle else 2 * np.pi, n_points)
    cx, cy = _UNIVERSAL_CIRCLE_CENTER
    r = _UNIVERSAL_CIRCLE_RADIUS
    return cx + r * np.cos(theta), cy + r * np.sin(theta)

def _draw_lifetime_ticks(ax, G_mark, S_mark,
                         tick_length: float = 0.02,
                         text_offset: float = 0.035,
                         color: str = "black",
                         lw: float = 2,
                         fontsize: int = 7,
                         show_units: bool = False):
    cx, cy = _UNIVERSAL_CIRCLE_CENTER
    for tau, Gm, Sm in zip(_TAU_MARKS_NS, G_mark, S_mark):
        normal = np.array([Gm - cx, Sm - cy])
        norm = np.linalg.norm(normal)
        if norm == 0:
            continue
        normal /= norm

        tick_start = np.array([Gm, Sm]) - tick_length * normal / 2
        tick_end = np.array([Gm, Sm]) + tick_length * normal / 2
        ax.plot([tick_start[0], tick_end[0]], [tick_start[1], tick_end[1]],
                color=color, lw=lw)

        label = f"{tau:.1f} ns" if show_units else f"{tau:.1f}"
        text_pos = tick_end + text_offset * normal
        ax.text(text_pos[0], text_pos[1], label,
                color=color, fontsize=fontsize, ha="center")

def _style_phasor_ax(ax, title: str = "Phasor Diagram",
                     xlim=(-0.1, 1.1), ylim=(-0.6, 0.6), half_circle: bool = False):
    if half_circle:
        ylim = (0, ylim[1])
    ax.set_xlabel("G")
    ax.set_ylabel("S")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.axhline(0, color="black", lw=1)
    ax.axvline(0, color="black", lw=1)
    ax.tick_params(direction="in", length=6, width=1)

class PhasorAnalyzer:

    def __init__(self, frequency_hz, time_axis_ns, n_harmonics=1, device=None):
        self.frequency = float(frequency_hz)
        self.time_axis_ns = np.asarray(time_axis_ns)
        self.n_harmonics = int(n_harmonics)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.omega = 2 * np.pi * self.frequency
        self.t_s_np = self.time_axis_ns * 1e-9
        self.t_s_torch = torch.tensor(self.t_s_np, dtype=torch.float32,
                                      device=self.device)
        self.eps = 1e-12

    def _phasor_numpy(self, decay):
        decay = np.asarray(decay, dtype=np.float64)
        *spatial, T = decay.shape
        decay_flat = decay.reshape(-1, T)

        I_sum = np.clip(np.sum(decay_flat, axis=1), self.eps, None)
        G_all, S_all = [], []

        for k in range(1, self.n_harmonics + 1):
            omega_k = k * self.omega
            cos_k = np.cos(omega_k * self.t_s_np)
            sin_k = np.sin(omega_k * self.t_s_np)
            G_all.append((np.sum(decay_flat * cos_k, axis=1) / I_sum).reshape(spatial))
            S_all.append((np.sum(decay_flat * sin_k, axis=1) / I_sum).reshape(spatial))

        return np.stack(G_all), np.stack(S_all)

    def _phasor_torch(self, decay):
        decay_t = torch.tensor(np.asarray(decay), dtype=torch.float32,
                               device=self.device)
        *spatial, T = decay_t.shape
        decay_flat = decay_t.reshape(-1, T)

        I_sum = torch.clamp(torch.sum(decay_flat, dim=1), min=self.eps)
        G_all, S_all = [], []

        for k in range(1, self.n_harmonics + 1):
            omega_k = k * self.omega
            cos_k = torch.cos(omega_k * self.t_s_torch)
            sin_k = torch.sin(omega_k * self.t_s_torch)
            G_all.append((torch.sum(decay_flat * cos_k, dim=1) / I_sum).reshape(spatial))
            S_all.append((torch.sum(decay_flat * sin_k, dim=1) / I_sum).reshape(spatial))

        return torch.stack(G_all), torch.stack(S_all)

    def create_phasor_cpu(self, decay):
        return self._phasor_numpy(decay)

    def create_phasor_gpu(self, decay):
        G, S = self._phasor_torch(decay)
        return G.cpu().numpy(), S.cpu().numpy()

    def calibrate(self, G, S, irf):
        G = np.asarray(G)
        S = np.asarray(S)
        irf = np.asarray(irf)
        if irf.ndim == 3:
            irf = irf.mean(axis=(0, 1))

        denom = np.clip(np.sum(irf), self.eps, None)
        G_irf, S_irf = [], []
        for k in range(1, self.n_harmonics + 1):
            omega_k = k * self.omega
            G_irf.append(np.sum(irf * np.cos(omega_k * self.t_s_np)) / denom)
            S_irf.append(np.sum(irf * np.sin(omega_k * self.t_s_np)) / denom)

        G_irf = np.array(G_irf)
        S_irf = np.array(S_irf)

        P = G + 1j * S
        P_irf = G_irf[:, None, None] + 1j * S_irf[:, None, None]
        P_irf_abs_sq = np.clip(G_irf[:, None, None] ** 2 + S_irf[:, None, None] ** 2, self.eps, None)
        P_true = P * np.conj(P_irf) / P_irf_abs_sq

        return np.real(P_true), np.imag(P_true)

    def calibrate_pixelwise(self, G, S, irf):       
        G   = np.asarray(G,   dtype=np.float32)
        S   = np.asarray(S,   dtype=np.float32)
        irf = np.asarray(irf, dtype=np.float32)

        H, W, T = irf.shape
        K       = self.n_harmonics

        irf_flat = torch.tensor(irf.reshape(-1, T), device=self.device)
        I_sum    = irf_flat.sum(dim=1, keepdim=True).clamp(min=self.eps)
        irf_norm = irf_flat / I_sum

        t_s = self.t_s_torch

        Gc_list, Sc_list = [], []

        for k in range(1, K + 1):
            omega_k = k * self.omega

            cos_k = torch.cos(torch.tensor(omega_k, dtype=torch.float32,
                                           device=self.device) * t_s)    # (T,)
            sin_k = torch.sin(torch.tensor(omega_k, dtype=torch.float32,
                                           device=self.device) * t_s)    # (T,)

            G_irf_flat = (irf_norm * cos_k).sum(dim=1)
            S_irf_flat = (irf_norm * sin_k).sum(dim=1)

            G_irf = G_irf_flat.reshape(H, W)
            S_irf = S_irf_flat.reshape(H, W)

            G_meas = torch.tensor(G[k - 1], device=self.device)
            S_meas = torch.tensor(S[k - 1], device=self.device)

            denom = (G_irf ** 2 + S_irf ** 2).clamp(min=self.eps)

            Gc_k = (G_meas * G_irf + S_meas * S_irf) / denom
            Sc_k = (S_meas * G_irf - G_meas * S_irf) / denom

            Gc_list.append(Gc_k.cpu().numpy())
            Sc_list.append(Sc_k.cpu().numpy())

        return np.stack(Gc_list), np.stack(Sc_list)

    def lifetime_to_phasor(self, tau_ns, frequency_hz):
        tau_s = np.asarray(tau_ns) * 1e-9
        omega = 2 * np.pi * frequency_hz
        denom = 1 + (omega * tau_s) ** 2
        return 1 / denom, (omega * tau_s) / denom

    def compute_lifetime(self, G, S):
        G = np.asarray(G, dtype=np.float64)
        S = np.asarray(S, dtype=np.float64)
        return np.where(np.abs(G) > 1e-4, S / (G * self.omega) * 1e9, np.nan)

    def compute_modulation_lifetime(self, G, S):
        G = np.asarray(G, dtype=np.float64)
        S = np.asarray(S, dtype=np.float64)
        M_sq = np.clip(G ** 2 + S ** 2, self.eps, 1.0 - self.eps)
        return np.sqrt(1.0 / M_sq - 1.0) / self.omega * 1e9

    def compute_fractions(self, G, S, tau1_ns, tau2_ns, mask=None, hexbin_color=None, plot_graph=True, ax=None, half_circle=False):
        g1, s1 = self.lifetime_to_phasor(tau1_ns, self.frequency)
        g2, s2 = self.lifetime_to_phasor(tau2_ns, self.frequency)
        if plot_graph:
            created_fig = ax is None
            if created_fig:
                fig, ax = plt.subplots(figsize=(8, 6))
            else:
                fig = ax.get_figure()
            self.plot_phasor_diagram(G, S, colors=None, mask=mask, hexbin_color="jet_r", ax=ax, half_circle=half_circle)
            ax.plot([g1, g2], [s1, s2], color="#2C0F02", linestyle="--", lw=2, zorder=10)
            ax.plot(g1, s1, "o", color="#E5D16E", markersize=8, label="...", zorder=11)
            ax.plot(g2, s2, "o", color="#363D45", markersize=8, label="...", zorder=11)
            ax.legend(loc="upper right")
            if created_fig:
                plt.tight_layout()

        line_vec_g = g1 - g2
        line_vec_s = s1 - s2
        line_mag_sq = line_vec_g ** 2 + line_vec_s ** 2 + self.eps

        A1 = np.clip(
            ((G - g2) * line_vec_g + (S - s2) * line_vec_s) / line_mag_sq,
            0, 1
        )
        return A1, 1 - A1

    def _convolve_batch(self, signal, kernel):

        N, T  = signal.shape
        L     = 2 * T - 1
        nfft  = 1 << (L - 1).bit_length()

        S   = torch.fft.rfft(signal, n=nfft, dim=1)
        K   = torch.fft.rfft(kernel,  n=nfft, dim=1)
        out = torch.fft.irfft(S * K,  n=nfft, dim=1)
        return out[:, :T]

    def _build_model_decay(self, A1, A2, tau1_ns, tau2_ns):

        t_ns = torch.tensor(self.t_s_np * 1e9, dtype=torch.float32,
                            device=self.device)              # (T,)
        a1   = torch.tensor(A1.ravel(), dtype=torch.float32,
                            device=self.device).unsqueeze(1) # (N, 1)
        a2   = torch.tensor(A2.ravel(), dtype=torch.float32,
                            device=self.device).unsqueeze(1) # (N, 1)
        return a1 * torch.exp(-t_ns / tau1_ns) + a2 * torch.exp(-t_ns / tau2_ns)

    def _normalize_irf(self, irf):
        irf_flat = np.asarray(irf, dtype=np.float32).reshape(-1, irf.shape[2])
        irf_t    = torch.tensor(irf_flat, dtype=torch.float32, device=self.device)
        norms    = irf_t.sum(dim=1, keepdim=True).clamp(min=self.eps)
        return irf_t / norms

    def analyze_biexponential_and_reconstruct(self, G, S, irf,
                                               tau1_ns=None, tau2_ns=None,
                                               plot=True, axes=None):

        if tau1_ns is None or tau2_ns is None:
            return None
        A1, A2 = self.compute_fractions(G, S, tau1_ns, tau2_ns, plot_graph=False)
        tau_map_ns = self.compute_lifetime(G, S)

        if plot:
            created_fig = axes is None
            if created_fig:
                fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            else:
                fig = axes[0].get_figure()
            im1 = axes[0].imshow(A1, origin="upper", cmap="viridis")
            axes[0].set_title(f"A1 Map (Fraction of {tau1_ns} ns)")
            fig.colorbar(im1, ax=axes[0])

            im2 = axes[1].imshow(A2, origin="upper", cmap="plasma")
            axes[1].set_title(f"A2 Map (Fraction of {tau2_ns} ns)")
            fig.colorbar(im2, ax=axes[1])

            im3 = axes[2].imshow(np.clip(tau_map_ns, 0, 5), origin="upper", cmap="magma")
            axes[2].set_title("Phase Lifetime Map (ns)")
            fig.colorbar(im3, ax=axes[2])

            for ax in axes:
                ax.axis("off")
            if created_fig:
                plt.tight_layout()

        H, W = A1.shape
        T = irf.shape[2]

        model_t = self._build_model_decay(A1, A2, tau1_ns, tau2_ns)
        irf_t   = self._normalize_irf(irf)
        recon_t = self._convolve_batch(model_t, irf_t)
        reconstructed_decay = recon_t.cpu().numpy().reshape(H, W, T)

        return reconstructed_decay

    def generate_intensity_image(self, decay):
        return np.sum(decay, axis=2)

    def phasor_colormap(self, G, S, intensity=None, colormap="viridis"):
        G_col = G[0] if G.ndim == 3 else G
        S_col = S[0] if S.ndim == 3 else S
        phasor_val = np.sqrt(G_col ** 2 + S_col ** 2)
        p_min, p_max = phasor_val.min(), phasor_val.max()
        phasor_val = (phasor_val - p_min) / (p_max - p_min + self.eps)
        colors = plt.get_cmap(colormap)(phasor_val)[:, :, :3]
        if intensity is not None:
            denom = intensity.max() - intensity.min() + self.eps
            int_norm = (intensity - intensity.min()) / denom
            colors = colors * int_norm[:, :, np.newaxis]
        return colors

    def plot_phasor_diagram(self, G, S, mask=None, colors=None, hexbin_color=None, ax=None, figsize=(8, 6), half_circle=False, title="Phasor Diagram"):
        created_fig = ax is None
        if created_fig:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        ug, us = _universal_circle_xy(half_circle=half_circle)
        ax.plot(ug, us, "k--")
        G_2d = G[0] if (np.ndim(G) == 3) else np.asarray(G)
        S_2d = S[0] if (np.ndim(S) == 3) else np.asarray(S)
        g_flat = np.ravel(G_2d)
        s_flat = np.ravel(S_2d)
        if mask is not None:
            mask_flat = np.ravel(mask).astype(bool)  # uint8 masks must be bool; int array would integer-index instead
            g_plot = g_flat[mask_flat]
            s_plot = s_flat[mask_flat]
        else:
            g_plot = g_flat
            s_plot = s_flat
        valid = ~np.isnan(g_plot) & ~np.isnan(s_plot)
        g_plot = g_plot[valid]
        s_plot = s_plot[valid]
        if colors is None:
            cmap_to_use = hexbin_color if hexbin_color is not None else 'autumn'
            hb = ax.hexbin(g_plot, s_plot, gridsize=100, cmap=cmap_to_use, mincnt=1)
            fig.colorbar(hb, ax=ax).set_label("Pixel Count")
        else:
            if isinstance(colors, str):
                c_vals = g_plot
                path = ax.scatter(g_plot, s_plot, cmap=colors, c=c_vals, s=8, marker="o")
                fig.colorbar(path, ax=ax).set_label("Phasor G Value")
            else:
                c_flat = np.reshape(colors, (-1, 3))
                if mask is not None:
                    c_plot = c_flat[mask_flat][valid]
                else:
                    c_plot = c_flat[valid]
                ax.scatter(g_plot, s_plot, c=c_plot, s=8, marker="o")

        G_mark, S_mark = self.lifetime_to_phasor(_TAU_MARKS_NS, self.frequency)
        _draw_lifetime_ticks(ax, G_mark, S_mark, color="black", lw=4, fontsize=10, show_units=True)

        _style_phasor_ax(ax, title=title,
                        xlim=(-0.1, 1.1), ylim=(-0.6, 0.6), half_circle=half_circle)

        if created_fig:
            plt.tight_layout()
        return fig

    def plot_map(self, image, scales=[0, 2], title="", ax=None, figsize=(8, 6)):
        created_fig = ax is None
        if created_fig:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        im = ax.imshow(np.clip(image, scales[0], scales[1]), origin="upper", cmap="viridis")
        fig.colorbar(im, ax=ax).set_label("Lifetime (ns)")
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(False)
        if created_fig:
            plt.tight_layout()
        return fig

    def plot_phasor_overlay(self, decay, G, S, colormap="viridis", ax=None, figsize=(8, 8)):
        created_fig = ax is None
        if created_fig:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        intensity_img = self.generate_intensity_image(decay)
        phasor_colors = self.phasor_colormap(G, S, colormap=colormap)
        int_norm = (intensity_img - intensity_img.min()) / \
                   (intensity_img.max() - intensity_img.min() + self.eps)
        overlay = np.stack([int_norm] * 3, axis=2) * phasor_colors
        ax.imshow(overlay, origin="upper")
        ax.set_title("Intensity + Phasor Color Overlay")
        ax.axis("off")
        if created_fig:
            plt.tight_layout()
        return fig

    def plot_pure_phasor_map(self, G, S, decay,
                             noise_removed=True,
                             colormap="viridis",
                             ax=None,
                             figsize=(4, 4)):
        phasor_colors = self.phasor_colormap(G, S, colormap=colormap)
        if phasor_colors.shape[-1] == 4:
            phasor_colors = phasor_colors[..., :3]

        if noise_removed:
            final_mask = np.ones(G.shape, dtype=bool)
        else:
            intensity_img = self.generate_intensity_image(decay)
            
            i_min, i_max = intensity_img.min(), intensity_img.max()
            denom = (i_max - i_min) if (i_max - i_min) != 0 else 1
            int_norm = (intensity_img - i_min) / denom
            
            final_mask = int_norm > 0.1

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
                          colormaps=["jet", "jet"],
                          hexbin_color='jet',
                          noise_removed=True, figsize=(15, 10), half_circle=False):
        G_2d = G[0] if G.ndim == 3 else G
        S_2d = S[0] if S.ndim == 3 else S
        intensity_img = self.generate_intensity_image(decay)
        
        phasor_colors_raw = self.phasor_radial_color(G_2d, S_2d, colormap=colormaps[0])
        if phasor_colors_raw.shape[-1] == 4:
            phasor_colors_raw = phasor_colors_raw[..., :3]

        if mask is None:
            if noise_removed:
                int_norm = (intensity_img - intensity_img.min()) / (intensity_img.max() - intensity_img.min() + self.eps)
                active_mask = int_norm > 0.1
            else:
                active_mask = np.ones(G_2d.shape, dtype=bool)
        else:
            active_mask = mask.astype(bool)

        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 3, figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(intensity_img, origin="upper", cmap=colormaps[0])
        ax1.set_title("Intensity")
        ax1.axis("off")
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        ax2 = fig.add_subplot(gs[0, 1])
        pure_overlay = np.zeros_like(phasor_colors_raw)
        pure_overlay[active_mask] = phasor_colors_raw[active_mask]
        ax2.imshow(pure_overlay, origin="upper")
        ax2.set_title("Phasor Color Projections")
        ax2.axis("off")

        ax3 = fig.add_subplot(gs[1, 0])
        tau_map_ns = np.clip(self.compute_lifetime(G_2d, S_2d), 0, None)
        im3 = ax3.imshow(tau_map_ns, origin="upper", cmap=colormaps[1])
        ax3.set_title("Lifetime Map (ns)")
        ax3.axis("off")
        fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04).set_label("ns")

        ax4 = fig.add_subplot(gs[1, 1])
        int_norm_3d = (intensity_img - intensity_img.min()) / (intensity_img.max() - intensity_img.min() + self.eps)
        weighted_overlay = np.stack([int_norm_3d] * 3, axis=2) * phasor_colors_raw
        ax4.imshow(weighted_overlay, origin="upper")
        ax4.set_title("Intensity-weighted Overlay")
        ax4.axis("off")

        ax5 = fig.add_subplot(gs[:, 2])
        ug, us = _universal_circle_xy(half_circle=half_circle)
        ax5.plot(ug, us, "k--", alpha=0.8, zorder=1)
        
        mask_flat = np.ravel(active_mask)
        g_plot = np.ravel(G_2d)[mask_flat]
        s_plot = np.ravel(S_2d)[mask_flat]
        c_plot = np.reshape(phasor_colors_raw, (-1, 3))[mask_flat]
        
        valid = ~np.isnan(g_plot) & ~np.isnan(s_plot)
        if np.any(valid):
            ax5.scatter(g_plot[valid], s_plot[valid], c=c_plot[valid], s=2, alpha=0.6, edgecolors='none', zorder=2)
        
        try:
            G_mark, S_mark = self.lifetime_to_phasor(_TAU_MARKS_NS, self.frequency)
            _draw_lifetime_ticks(ax5, G_mark, S_mark, color="black", lw=2, fontsize=9)
        except:
            pass

        _style_phasor_ax(ax5, title="Phasor Distribution", xlim=(-0.1, 1.1), ylim=(-0.6, 0.6), half_circle=half_circle)
        
        plt.tight_layout()
        return fig

    def plot_pixel_fit(self, irf, decay, reconstructed_decay, x, y,
                       log_scale=True, ax=None, figsize=(10, 6)):

        irf_trace = irf[y, x, :] if irf.ndim == 3 else np.asarray(irf)
        raw_trace  = decay[y, x, :]
        fit_trace  = reconstructed_decay[y, x, :]

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

        T = decay.shape[2]

        t_ns_t = torch.tensor(self.t_s_np * 1e9, dtype=torch.float32,
                               device=self.device)                      # (T,)
        model_t = torch.exp(-t_ns_t / tau_val).unsqueeze(0)

        irf_trace_np = irf[y, x, :] if irf.ndim == 3 else np.asarray(irf)
        irf_trace_t  = torch.tensor(
            irf_trace_np.astype(np.float32), device=self.device
        ).unsqueeze(0)                                                  # (1, T)

        irf_norm_t = irf_trace_t / irf_trace_t.sum(dim=1, keepdim=True).clamp(
            min=self.eps
        )
        fit_t = self._convolve_batch(model_t, irf_norm_t)
        fit_trace_np = fit_t.squeeze(0).cpu().numpy()

        raw_trace_np = decay[y, x, :]

        traces_np = np.stack(
            [irf_trace_np, raw_trace_np, fit_trace_np], axis=0
        ).astype(np.float32)                                            # (3, T)
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

    def plot_phasor_harmonics(self, G, S, harmonics=(1, 2, 3, 4), mask=None,
                          colors=None, hexbin_color=None, figsize=(22, 5), axes=None, half_circle=False):
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

        mask_flat = np.ravel(mask) if mask is not None else None

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

            if colors is None:
                cmap_to_use = hexbin_color if hexbin_color is not None else 'jet'
                hb = ax.hexbin(g_plot, s_plot, gridsize=100, cmap=cmap_to_use, mincnt=1)
                fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04).set_label("Pixel Count")
            
            else:
                if isinstance(colors, str):
                    c_vals = g_plot 
                    path = ax.scatter(g_plot, s_plot, cmap=colors, c=c_vals, s=8, marker="o")
                    fig.colorbar(path, ax=ax, fraction=0.046, pad=0.04).set_label("Phasor G Value")
                else:
                    c_flat = np.reshape(colors, (-1, 3))
                    c_plot = c_flat[mask_flat] if mask_flat is not None else c_flat
                    ax.scatter(g_plot, s_plot, c=c_plot, s=8, marker="o")

            G_mark, S_mark = self.lifetime_to_phasor(_TAU_MARKS_NS, k * self.frequency)
            _draw_lifetime_ticks(ax, G_mark, S_mark,
                                color="black", lw=2, fontsize=7,
                                show_units=(k == harmonics[0]),
                                tick_length=0.03, text_offset=0.05)

            _style_phasor_ax(ax,
                            title=f"Harmonic {k} ($\omega_{{{k}}}$)",
                            xlim=(-0.3, 1.3),
                            ylim=(-0.85, 0.85), half_circle=half_circle)
                                
        if created_fig:
            fig.suptitle("Phasor Diagram — Multiple Harmonics",
                         fontsize=12, fontweight="bold", y=1.01)
            plt.tight_layout()
        return fig

    def save_phasors_hdf5(self, Gc, Sc, tau_phasor, save_file):
        try:
            with h5py.File(save_file, 'w') as hf:
                hf.create_dataset('Gc', data=Gc, compression="gzip", chunks=True)
                hf.create_dataset('Sc', data=Sc, compression="gzip", chunks=True)
                hf.create_dataset('tau_phasor', data=tau_phasor, compression="gzip", chunks=True)
                
                hf.attrs['n_harmonics'] = Gc.shape[0]
                hf.attrs['resolution'] = f"{Gc.shape[1]}x{Gc.shape[2]}"
                
            print(f"Successfully saved data to {save_file}")
            
        except Exception as e:
            print(f"An error occurred while saving: {e}")

    def phasor_radial_color(self, G, S, intensity=None, colormap="viridis"):
        G_col = G[0] if G.ndim == 3 else G
        S_col = S[0] if S.ndim == 3 else S        
        phi = np.arctan2(S_col, G_col)        
        phi_min = phi.min()
        phi_max = phi.max()
        if phi_max <= phi_min:
            phi_max = phi_min + 1.0
        phi_norm = (phi - phi_min) / (phi_max - phi_min + self.eps)
        phi_norm = np.clip(phi_norm, 0, 1)
        colors = plt.get_cmap(colormap)(phi_norm)[:, :, :3]
        return colors

    def plot_traceable_analysis(self, G, S, decay, mask=None, colormap="viridis", figsize=(14, 6), axes=None, half_circle=False):
        G_2d = G[0] if G.ndim == 3 else G
        S_2d = S[0] if S.ndim == 3 else S
        
        phasor_colors_raw = self.phasor_radial_color(G_2d, S_2d, colormap=colormap)
        
        if phasor_colors_raw.shape[-1] == 4:
            phasor_colors_raw = phasor_colors_raw[..., :3]

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
        im_map = axes[0].imshow(pure_overlay, origin="upper")
        axes[0].set_title("Phasor Color Projections")
        axes[0].axis("off")
        phi = np.arctan2(S_2d, G_2d)
        first_q = phi[(G_2d > 0) & (S_2d > 0)]
        phi_min_val = float(first_q.min()) if first_q.size > 0 else 0.0
        phi_max_val = float(first_q.max()) if first_q.size > 0 else 0.5
        omega = 2 * np.pi * self.frequency
        tau_min = np.tan(np.clip(phi_min_val, 1e-6, np.pi / 2 - 0.05)) / omega * 1e9
        tau_max = np.tan(np.clip(phi_max_val, 1e-6, np.pi / 2 - 0.05)) / omega * 1e9

        sm = ScalarMappable(cmap=plt.get_cmap(colormap), norm=Normalize(vmin=tau_min, vmax=tau_max))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes[0], fraction=0.046, pad=0.04)
        cbar.set_label("Lifetime (ns)")

        ug, us = _universal_circle_xy(half_circle=half_circle)
        axes[1].plot(ug, us, "k--", alpha=0.8, zorder=1)
        mask_flat = np.ravel(active_mask)
        g_plot = np.ravel(G_2d)[mask_flat]
        s_plot = np.ravel(S_2d)[mask_flat]
        c_plot = np.reshape(phasor_colors_raw, (-1, 3))[mask_flat]
        valid_data = ~np.isnan(g_plot) & ~np.isnan(s_plot)
        g_plot, s_plot, c_plot = g_plot[valid_data], s_plot[valid_data], c_plot[valid_data]
        if len(g_plot) > 0:
            axes[1].scatter(g_plot, s_plot, c=c_plot, s=10, alpha=0.8, edgecolors='none', zorder=2)
        else:
            axes[1].text(0.5, 0.25, "No valid data points", ha='center', color='red')
        try:
            G_mark, S_mark = self.lifetime_to_phasor(_TAU_MARKS_NS, self.frequency)
            _draw_lifetime_ticks(axes[1], G_mark, S_mark, color="black", lw=2, fontsize=9)
        except:
            pass
        _style_phasor_ax(axes[1], title="Phasor Distribution",
                         xlim=(-0.1, 1.1), ylim=(-0.6, 0.6), half_circle=half_circle)
        if created_fig:
            plt.tight_layout()
        return fig
