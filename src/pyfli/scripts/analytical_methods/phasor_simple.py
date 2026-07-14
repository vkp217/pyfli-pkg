import numpy as np
import torch
import matplotlib.pyplot as plt
import h5py

from .phasor_simple_plots import PhasorPlotsMixin


class PhasorAnalyzer(PhasorPlotsMixin):
    """Phasor-based FLI analysis.

    Plotting methods live in PhasorPlotsMixin (phasor_simple_plots.py).
    Shared helper functions and constants live in phasor_simple_utils.py.
    """

    def __init__(self, frequency_hz, time_axis_ns, n_harmonics=1, device=None):
        self.frequency    = float(frequency_hz)
        self.time_axis_ns = np.asarray(time_axis_ns)
        self.n_harmonics  = int(n_harmonics)
        self.device       = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.omega        = 2 * np.pi * self.frequency
        self.t_s_np       = self.time_axis_ns * 1e-9
        self.t_s_torch    = torch.tensor(self.t_s_np, dtype=torch.float32,
                                         device=self.device)
        self.eps = 1e-12

    # ── phasor computation ────────────────────────────────────────────────────

    def _phasor_numpy(self, decay):
        decay      = np.asarray(decay, dtype=np.float64)
        *spatial, T = decay.shape
        decay_flat = decay.reshape(-1, T)

        I_sum  = np.clip(np.sum(decay_flat, axis=1), self.eps, None)
        G_all, S_all = [], []

        for k in range(1, self.n_harmonics + 1):
            omega_k = k * self.omega
            cos_k   = np.cos(omega_k * self.t_s_np)
            sin_k   = np.sin(omega_k * self.t_s_np)
            G_all.append((np.sum(decay_flat * cos_k, axis=1) / I_sum).reshape(spatial))
            S_all.append((np.sum(decay_flat * sin_k, axis=1) / I_sum).reshape(spatial))

        return np.stack(G_all), np.stack(S_all)

    def _phasor_torch(self, decay):
        decay_t     = torch.tensor(np.asarray(decay), dtype=torch.float32,
                                   device=self.device)
        *spatial, T = decay_t.shape
        decay_flat  = decay_t.reshape(-1, T)

        I_sum  = torch.clamp(torch.sum(decay_flat, dim=1), min=self.eps)
        G_all, S_all = [], []

        for k in range(1, self.n_harmonics + 1):
            omega_k = k * self.omega
            cos_k   = torch.cos(omega_k * self.t_s_torch)
            sin_k   = torch.sin(omega_k * self.t_s_torch)
            G_all.append((torch.sum(decay_flat * cos_k, dim=1) / I_sum).reshape(spatial))
            S_all.append((torch.sum(decay_flat * sin_k, dim=1) / I_sum).reshape(spatial))

        return torch.stack(G_all), torch.stack(S_all)

    def create_phasor_cpu(self, decay):
        return self._phasor_numpy(decay)

    def create_phasor_gpu(self, decay):
        G, S = self._phasor_torch(decay)
        return G.cpu().numpy(), S.cpu().numpy()

    # ── calibration ───────────────────────────────────────────────────────────

    def calibrate(self, G, S, irf):
        G   = np.asarray(G)
        S   = np.asarray(S)
        irf = np.asarray(irf)
        if irf.ndim == 3:
            irf = irf.mean(axis=(0, 1))

        denom  = np.clip(np.sum(irf), self.eps, None)
        G_irf, S_irf = [], []
        for k in range(1, self.n_harmonics + 1):
            omega_k = k * self.omega
            G_irf.append(np.sum(irf * np.cos(omega_k * self.t_s_np)) / denom)
            S_irf.append(np.sum(irf * np.sin(omega_k * self.t_s_np)) / denom)

        G_irf = np.array(G_irf)
        S_irf = np.array(S_irf)

        P            = G + 1j * S
        P_irf        = G_irf[:, None, None] + 1j * S_irf[:, None, None]
        P_irf_abs_sq = np.clip(G_irf[:, None, None] ** 2 + S_irf[:, None, None] ** 2,
                               self.eps, None)
        P_true = P * np.conj(P_irf) / P_irf_abs_sq

        return np.real(P_true), np.imag(P_true)

    def calibrate_pixelwise(self, G, S, irf):
        G   = np.asarray(G,   dtype=np.float32)
        S   = np.asarray(S,   dtype=np.float32)
        irf = np.asarray(irf, dtype=np.float32)

        H, W, T  = irf.shape
        K        = self.n_harmonics
        irf_flat = torch.tensor(irf.reshape(-1, T), device=self.device)
        I_sum    = irf_flat.sum(dim=1, keepdim=True).clamp(min=self.eps)
        irf_norm = irf_flat / I_sum
        t_s      = self.t_s_torch

        Gc_list, Sc_list = [], []

        for k in range(1, K + 1):
            omega_k = k * self.omega
            cos_k   = torch.cos(torch.tensor(omega_k, dtype=torch.float32,
                                             device=self.device) * t_s)
            sin_k   = torch.sin(torch.tensor(omega_k, dtype=torch.float32,
                                             device=self.device) * t_s)

            G_irf_flat = (irf_norm * cos_k).sum(dim=1)
            S_irf_flat = (irf_norm * sin_k).sum(dim=1)
            G_irf      = G_irf_flat.reshape(H, W)
            S_irf      = S_irf_flat.reshape(H, W)

            G_meas = torch.tensor(G[k - 1], device=self.device)
            S_meas = torch.tensor(S[k - 1], device=self.device)
            denom  = (G_irf ** 2 + S_irf ** 2).clamp(min=self.eps)

            Gc_k = (G_meas * G_irf + S_meas * S_irf) / denom
            Sc_k = (S_meas * G_irf - G_meas * S_irf) / denom

            Gc_list.append(Gc_k.cpu().numpy())
            Sc_list.append(Sc_k.cpu().numpy())

        return np.stack(Gc_list), np.stack(Sc_list)

    # ── lifetime conversion ───────────────────────────────────────────────────

    def lifetime_to_phasor(self, tau_ns, frequency_hz):
        tau_s = np.asarray(tau_ns) * 1e-9
        omega = 2 * np.pi * frequency_hz
        denom = 1 + (omega * tau_s) ** 2
        return 1 / denom, (omega * tau_s) / denom

    def compute_lifetime(self, G, S):
        G = np.asarray(G, dtype=np.float64)
        S = np.asarray(S, dtype=np.float64)
        safe_denom = np.where(np.abs(G) > 1e-4, G * self.omega, np.inf)
        return np.where(np.abs(G) > 1e-4, S / safe_denom * 1e9, np.nan)

    def compute_modulation_lifetime(self, G, S):
        G    = np.asarray(G, dtype=np.float64)
        S    = np.asarray(S, dtype=np.float64)
        M_sq = np.clip(G ** 2 + S ** 2, self.eps, 1.0 - self.eps)
        return np.sqrt(1.0 / M_sq - 1.0) / self.omega * 1e9

    # ── two-component analysis ────────────────────────────────────────────────

    def compute_fractions(self, G, S, tau1_ns, tau2_ns, mask=None,
                          hexbin_color=None, plot_graph=True, ax=None,
                          half_circle=False):
        g1, s1 = self.lifetime_to_phasor(tau1_ns, self.frequency)
        g2, s2 = self.lifetime_to_phasor(tau2_ns, self.frequency)

        if plot_graph:
            created_fig = ax is None
            if created_fig:
                fig, ax = plt.subplots(figsize=(8, 6))
            else:
                fig = ax.get_figure()
            self.plot_phasor_diagram(G, S, colors=None, mask=mask,
                                     hexbin_color="jet_r", ax=ax,
                                     half_circle=half_circle)
            ax.plot([g1, g2], [s1, s2], color="#2C0F02", linestyle="--", lw=2, zorder=10)
            ax.plot(g1, s1, "o", color="#E5D16E", markersize=8, label="...", zorder=11)
            ax.plot(g2, s2, "o", color="#363D45", markersize=8, label="...", zorder=11)
            ax.legend(loc="upper right")
            if created_fig:
                plt.tight_layout()

        line_vec_g  = g1 - g2
        line_vec_s  = s1 - s2
        line_mag_sq = line_vec_g ** 2 + line_vec_s ** 2 + self.eps

        A1 = np.clip(
            ((G - g2) * line_vec_g + (S - s2) * line_vec_s) / line_mag_sq,
            0, 1
        )
        return A1, 1 - A1

    # ── biexponential reconstruction ──────────────────────────────────────────

    def _convolve_batch(self, signal, kernel):
        N, T  = signal.shape
        L     = 2 * T - 1
        nfft  = 1 << (L - 1).bit_length()
        S_fft = torch.fft.rfft(signal, n=nfft, dim=1)
        K_fft = torch.fft.rfft(kernel,  n=nfft, dim=1)
        out   = torch.fft.irfft(S_fft * K_fft, n=nfft, dim=1)
        return out[:, :T]

    def _build_model_decay(self, A1, A2, tau1_ns, tau2_ns):
        t_ns = torch.tensor(self.t_s_np * 1e9, dtype=torch.float32,
                            device=self.device)
        a1   = torch.tensor(A1.ravel(), dtype=torch.float32,
                            device=self.device).unsqueeze(1)
        a2   = torch.tensor(A2.ravel(), dtype=torch.float32,
                            device=self.device).unsqueeze(1)
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

        A1, A2       = self.compute_fractions(G, S, tau1_ns, tau2_ns, plot_graph=False)
        tau_map_ns   = self.compute_lifetime(G, S)

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
        T    = irf.shape[2]

        model_t              = self._build_model_decay(A1, A2, tau1_ns, tau2_ns)
        irf_t                = self._normalize_irf(irf)
        recon_t              = self._convolve_batch(model_t, irf_t)
        reconstructed_decay  = recon_t.cpu().numpy().reshape(H, W, T)

        return reconstructed_decay

    # ── utilities ─────────────────────────────────────────────────────────────

    def generate_intensity_image(self, decay):
        return np.sum(decay, axis=2)

    def save_phasors_hdf5(self, Gc, Sc, tau_phasor, save_file):
        try:
            with h5py.File(save_file, 'w') as hf:
                hf.create_dataset('Gc',         data=Gc,         compression="gzip", chunks=True)
                hf.create_dataset('Sc',         data=Sc,         compression="gzip", chunks=True)
                hf.create_dataset('tau_phasor', data=tau_phasor, compression="gzip", chunks=True)
                hf.attrs['n_harmonics'] = Gc.shape[0]
                hf.attrs['resolution']  = f"{Gc.shape[1]}x{Gc.shape[2]}"
            print(f"Successfully saved data to {save_file}")
        except Exception as e:
            print(f"An error occurred while saving: {e}")
