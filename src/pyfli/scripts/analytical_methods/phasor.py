"""
phasor_analyzer.py
==================
Fluorescence Lifetime Imaging Microscopy (FLIM) Phasor Analysis API.

Public API
----------
PhasorAnalyzer(frequency_hz, time_axis_ns, n_harmonics=1, device=None)
    .create_phasor_cpu(decay)               -> (G, S)
    .create_phasor_gpu(decay)               -> (G, S)
    .calibrate(G, S, irf)                   -> (Gc, Sc)
        [Mean-IRF calibration — single scalar correction per harmonic]
    .calibrate_pixelwise(G, S, irf)         -> (Gc, Sc)
        [Pixel-wise calibration — independent complex division at every (i,j);
         GPU/CPU vectorised, no Python pixel loop.
         Maps each pixel's IRF phasor to (1,0) so spatial gate delays and
         sensor non-uniformities are removed independently per pixel.]
    .compute_lifetime(S, G)                 -> tau_map_ns
    .lifetime_to_phasor(tau_ns, frequency_hz) -> (G, S)
    .compute_fractions(G, S, tau1_ns, tau2_ns, plot_graph=True) -> (A1, A2)
    .analyze_biexponential_and_reconstruct(G, S, irf, tau1_ns, tau2_ns, plot=True) -> reconstructed_decay
        [GPU/CPU FFT-vectorised — no Python pixel loop; matches scipy.signal.convolve exactly]
    .generate_intensity_image(decay)        -> intensity_img
    .phasor_colormap(G, S, intensity=None, colormap="jet") -> colors

Visualization helpers
---------------------
    .plot_phasor_diagram(G, S, colors=None, ax=None, figsize=(8, 8))
    .plot_map(image, title="")
    .plot_phasor_overlay(decay, G, S, colormap="viridis", figsize=(8, 8))
    .plot_overlay_subplots(decay, G, S, colormap="viridis", figsize=(20, 8))
    .plot_pixel_fit(irf, decay, reconstructed_decay, x, y, log_scale=True)
        [GPU-parallel batch normalisation of all three traces]
    .plot_pixel_fit_single_exp(irf, decay, tau_ns, x, y, log_scale=True)
        [Single-exponential model: A*exp(-t/tau) convolved with IRF, GPU-parallel normalisation]
    .plot_phasor_harmonics(G, S, harmonics=(1,2,3,4), colors=None, figsize=(20,5))
        [Multi-harmonic phasor grid; each panel shows its own universal semicircle
         and lifetime ticks re-evaluated at k·ω.]
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import h5py


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_TAU_MARKS_NS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                           0.9, 1, 1.5, 2, 3, 5, 7, 10])
_UNIVERSAL_CIRCLE_CENTER = (0.5, 0.0)
_UNIVERSAL_CIRCLE_RADIUS = 0.5


# ---------------------------------------------------------------------------
# Helper – universal semicircle geometry
# ---------------------------------------------------------------------------
def _universal_circle_xy(n_points: int = 500):
    theta = np.linspace(0, 2 * np.pi, n_points)
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
    """Draw lifetime tick marks and labels on a phasor axis."""
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
                     xlim=(-0.1, 1.1), ylim=(-0.6, 0.6)):
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


# ---------------------------------------------------------------------------
# PhasorAnalyzer
# ---------------------------------------------------------------------------
class PhasorAnalyzer:
    """
    Phasor-based FLIM analysis tool.

    Parameters
    ----------
    frequency_hz : float
        Laser repetition frequency in Hz (e.g. 80e6).
    time_axis_ns : array-like
        Time axis of the TCSPC histogram in nanoseconds.
    n_harmonics : int, optional
        Number of harmonics to compute (default 1).
    device : str or None, optional
        Torch device ("cuda", "cpu", or None for auto-detect).
    """

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

    # ------------------------------------------------------------------
    # Core phasor computation
    # ------------------------------------------------------------------
    def _phasor_numpy(self, decay):
        """Compute phasor (G, S) stacks using NumPy (CPU)."""
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
        """Compute phasor (G, S) stacks using PyTorch (GPU-capable)."""
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

    # ------------------------------------------------------------------
    # Public phasor creation
    # ------------------------------------------------------------------
    def create_phasor_cpu(self, decay):
        """
        Compute phasor coordinates on CPU.

        Parameters
        ----------
        decay : ndarray, shape (..., T)

        Returns
        -------
        G, S : ndarray, shape (n_harmonics, ...)
        """
        return self._phasor_numpy(decay)

    def create_phasor_gpu(self, decay):
        """
        Compute phasor coordinates on GPU (falls back to CPU if unavailable).

        Parameters
        ----------
        decay : ndarray, shape (..., T)

        Returns
        -------
        G, S : ndarray, shape (n_harmonics, ...)
        """
        G, S = self._phasor_torch(decay)
        return G.cpu().numpy(), S.cpu().numpy()

    # ------------------------------------------------------------------
    # IRF calibration
    # ------------------------------------------------------------------
    def calibrate(self, G, S, irf):
        """
        Calibrate phasor coordinates against an instrument response function.

        Parameters
        ----------
        G, S : ndarray, shape (n_harmonics, H, W)
        irf  : ndarray, shape (H, W, T) or (T,)

        Returns
        -------
        Gc, Sc : ndarray, shape (n_harmonics, H, W)
            IRF-corrected phasor coordinates.
        """
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
        P_true = P / (P_irf + self.eps)

        return np.real(P_true), np.imag(P_true)

    # ------------------------------------------------------------------
    # Pixel-wise IRF calibration
    # ------------------------------------------------------------------
    def calibrate_pixelwise(self, G, S, irf):
        """
        Pixel-wise phasor calibration for spatially non-uniform IRFs.

        In wide-field / camera-based FLIM systems the IRF varies across the
        sensor due to gate propagation delays, optical path differences, and
        pixel-level electronics.  A single mean-IRF correction (``calibrate``)
        introduces spatial artefacts because it assumes every pixel shares the
        same phase delay and modulus.  This method removes those artefacts by
        treating every pixel as its own independent detector.

        Mathematical formulation
        ------------------------
        For harmonic k and pixel (i, j) the measured phasor is the *product*
        (in the Fourier / complex sense) of the true fluorescence phasor and
        the local IRF phasor:

            P_meas[k, i, j]  =  P_true[k, i, j]  ×  P_irf[k, i, j]

        where the complex phasors are

            P_meas  = G_meas  + j · S_meas
            P_irf   = G_irf   + j · S_irf
            P_true  = G_true  + j · S_true          (what we want)

        Solving for the true phasor by complex division:

            P_true[k, i, j]  =  P_meas[k, i, j]  /  P_irf[k, i, j]

        This division simultaneously:
          • *de-rotates* the phase  (φ_meas − φ_irf → 0 for a delta-function sample)
          • *rescales* the modulus  (|P_meas| / |P_irf| → 1 on the semicircle)

        so that a sample with an infinitely short lifetime maps to (1, 0), and
        all other lifetimes land on the universal semicircle as expected.

        The IRF phasor at pixel (i, j) for harmonic k is computed as the
        normalised DFT cosine and sine projections of the local IRF histogram:

            G_irf[k, i, j]  =  Σ_t  irf[i,j,t] · cos(k·ω·t)  /  Σ_t irf[i,j,t]
            S_irf[k, i, j]  =  Σ_t  irf[i,j,t] · sin(k·ω·t)  /  Σ_t irf[i,j,t]

        Everything is fully vectorised on ``self.device`` (GPU when available):
        no Python loop over pixels.  The complex division is performed in
        PyTorch complex arithmetic for numerical stability; the denominator is
        regularised by ``self.eps`` to guard against dark / zero-count pixels.

        Parameters
        ----------
        G : ndarray, shape (n_harmonics, H, W)
            Uncalibrated phasor G (cosine) component.
        S : ndarray, shape (n_harmonics, H, W)
            Uncalibrated phasor S (sine) component.
        irf : ndarray, shape (H, W, T)
            Per-pixel IRF histograms.  Must cover the same spatial region as
            the decay data used to compute G and S.

        Returns
        -------
        Gc : ndarray, shape (n_harmonics, H, W)
            Pixel-wise calibrated G component.
        Sc : ndarray, shape (n_harmonics, H, W)
            Pixel-wise calibrated S component.

        Notes
        -----
        * If a pixel has zero total IRF counts (dark pixel) the denominator
          is clamped to ``self.eps`` so the output is numerically safe (the
          result will be large but finite).  Consider masking such pixels with
          an intensity threshold before downstream analysis.
        * The method respects ``self.n_harmonics``; G and S must have been
          computed with the same setting.
        * For a spatially *uniform* IRF the result is identical to
          ``calibrate`` (up to floating-point rounding).
        """
        G   = np.asarray(G,   dtype=np.float32)   # (K, H, W)
        S   = np.asarray(S,   dtype=np.float32)
        irf = np.asarray(irf, dtype=np.float32)   # (H, W, T)

        H, W, T = irf.shape
        K       = self.n_harmonics

        # ---- Move IRF to device and L1-normalise per pixel ----------------
        # irf_t  : (H*W, T)
        irf_flat = torch.tensor(irf.reshape(-1, T), device=self.device)  # (N, T)
        I_sum    = irf_flat.sum(dim=1, keepdim=True).clamp(min=self.eps) # (N, 1)
        irf_norm = irf_flat / I_sum                                       # (N, T)

        # time axis on device
        t_s = self.t_s_torch                                              # (T,)

        Gc_list, Sc_list = [], []

        for k in range(1, K + 1):
            omega_k = k * self.omega

            # ---- Per-pixel IRF phasor at harmonic k ----------------------
            # cos_k, sin_k : (T,)  broadcast over (N, T)
            cos_k = torch.cos(torch.tensor(omega_k, dtype=torch.float32,
                                           device=self.device) * t_s)    # (T,)
            sin_k = torch.sin(torch.tensor(omega_k, dtype=torch.float32,
                                           device=self.device) * t_s)    # (T,)

            G_irf_flat = (irf_norm * cos_k).sum(dim=1)                   # (N,)
            S_irf_flat = (irf_norm * sin_k).sum(dim=1)                   # (N,)

            # Reshape to (H, W)
            G_irf = G_irf_flat.reshape(H, W)
            S_irf = S_irf_flat.reshape(H, W)

            # ---- Measured phasor for this harmonic -----------------------
            G_meas = torch.tensor(G[k - 1], device=self.device)          # (H, W)
            S_meas = torch.tensor(S[k - 1], device=self.device)          # (H, W)

            # ---- Complex division  P_true = P_meas / P_irf ---------------
            #
            #   Re(P_true)  =  (G_meas·G_irf + S_meas·S_irf) / |P_irf|²
            #   Im(P_true)  =  (S_meas·G_irf − G_meas·S_irf) / |P_irf|²
            #
            # This is equivalent to multiplying P_meas by the conjugate of
            # P_irf and dividing by |P_irf|², which is the standard formula
            # for complex division expressed in real arithmetic — no complex
            # dtype needed, avoids potential CUDA complex precision issues.

            denom = (G_irf ** 2 + S_irf ** 2).clamp(min=self.eps)        # |P_irf|²

            Gc_k = (G_meas * G_irf + S_meas * S_irf) / denom            # (H, W)
            Sc_k = (S_meas * G_irf - G_meas * S_irf) / denom            # (H, W)

            Gc_list.append(Gc_k.cpu().numpy())
            Sc_list.append(Sc_k.cpu().numpy())

        return np.stack(Gc_list), np.stack(Sc_list)                      # (K, H, W)

    # ------------------------------------------------------------------
    # Lifetime utilities
    # ------------------------------------------------------------------
    def lifetime_to_phasor(self, tau_ns, frequency_hz):
        """
        Convert fluorescence lifetime(s) to phasor coordinates.
        Parameters
        ----------
        tau_ns : float or array-like
            Lifetime(s) in nanoseconds.
        frequency_hz : float
            Repetition frequency in Hz.
        Returns
        -------
        G, S : float or ndarray
        """
        tau_s = np.asarray(tau_ns) * 1e-9
        omega = 2 * np.pi * frequency_hz
        denom = 1 + (omega * tau_s) ** 2
        return 1 / denom, (omega * tau_s) / denom

    def compute_lifetime(self, S, G):
        """
        Compute per-pixel phase lifetime from phasor coordinates.
        Parameters
        ----------
        S, G : ndarray

        Returns
        -------
        tau_map_ns : ndarray
            Phase lifetime map in nanoseconds.
        """
        return (1 / self.omega) * (S / (G + self.eps)) * 1e9

    # ------------------------------------------------------------------
    # Fraction decomposition
    # ------------------------------------------------------------------
    def compute_fractions(self, G, S, tau1_ns, tau2_ns, plot_graph=True):
        """
        Compute fractional contributions of two lifetime components.

        The fraction A1 is determined by projecting each pixel's phasor
        coordinate onto the line segment connecting the two reference
        phasors (g1,s1) and (g2,s2).  The projection is a scalar
        parameter α defined by:

            phasor ≈ α·(g1,s1) + (1-α)·(g2,s2)

        Solving for α via dot-product projection along the mixing line:

            α = [(G-g2)·(g1-g2) + (S-s2)·(s1-s2)] / |(g1-g2, s1-s2)|²

        α=1 ⟹ pixel lies at the tau1 endpoint (pure tau1 component).
        α=0 ⟹ pixel lies at the tau2 endpoint (pure tau2 component).
        Values are clipped to [0,1] to handle noise-driven excursions
        off the mixing line.

        Parameters
        ----------
        G, S : ndarray
            Phasor coordinates (first harmonic, shape H×W).
        tau1_ns, tau2_ns : float
            Reference lifetimes in nanoseconds.
        plot_graph : bool, optional
            If True, display the phasor diagram with the mixing line.

        Returns
        -------
        A1, A2 : ndarray
            Fractional maps for tau1 and tau2.  A1 + A2 = 1 everywhere
            after clipping.
        """
        g1, s1 = self.lifetime_to_phasor(tau1_ns, self.frequency)
        g2, s2 = self.lifetime_to_phasor(tau2_ns, self.frequency)

        if plot_graph:
            fig, ax = plt.subplots(figsize=(8, 8))
            self.plot_phasor_diagram(G, S, colors=None, ax=ax)
            ax.plot([g1, g2], [s1, s2], color="red", linestyle="--", lw=2, zorder=10)
            ax.plot(g1, s1, "ro", markersize=12, label=f"tau1: {tau1_ns} ns", zorder=11)
            ax.plot(g2, s2, "go", markersize=12, label=f"tau2: {tau2_ns} ns", zorder=11)
            ax.legend(loc="upper right")
            plt.show()

        # Vector along the mixing line from tau2 endpoint toward tau1 endpoint
        line_vec_g = g1 - g2
        line_vec_s = s1 - s2
        line_mag_sq = line_vec_g ** 2 + line_vec_s ** 2 + self.eps

        # Project (G - g2, S - s2) onto the mixing-line direction
        A1 = np.clip(
            ((G - g2) * line_vec_g + (S - s2) * line_vec_s) / line_mag_sq,
            0, 1
        )
        return A1, 1 - A1

    # ------------------------------------------------------------------
    # GPU-parallel convolution engine + biexponential reconstruction
    # ------------------------------------------------------------------
    def _convolve_batch(self, signal, kernel):
        """
        Batch 1-D convolution via FFT — numerically identical to::

            scipy.signal.convolve(signal[i], kernel[i], mode='full')[:T]

        for every pixel i simultaneously.

        Uses ``torch.fft.rfft`` which runs on GPU when available and falls
        back to multi-threaded CPU automatically — no code change needed.
        Complexity is O(N · T · log T) vs O(N · T²) for direct convolution.

        Parameters
        ----------
        signal : Tensor, shape (N, T)  on self.device
        kernel : Tensor, shape (N, T)  on self.device

        Returns
        -------
        out : Tensor, shape (N, T)  on self.device

        Why not ``F.conv1d`` with groups?
        ----------------------------------
        ``torch.nn.functional.conv1d`` computes *cross-correlation*, not
        convolution.  Without explicitly flipping the kernel it produces
        wrong results, and even with flipping the 'full'-mode index
        alignment is non-trivial.  FFT avoids both issues entirely.
        """
        N, T  = signal.shape
        L     = 2 * T - 1
        nfft  = 1 << (L - 1).bit_length()          # next power-of-2 ≥ L

        S   = torch.fft.rfft(signal, n=nfft, dim=1)  # (N, nfft//2+1) complex
        K   = torch.fft.rfft(kernel,  n=nfft, dim=1)
        out = torch.fft.irfft(S * K,  n=nfft, dim=1) # (N, nfft)  real
        return out[:, :T]                              # truncate → (N, T)

    def _build_model_decay(self, A1, A2, tau1_ns, tau2_ns):
        """
        Build biexponential model decay on the compute device.

        Parameters
        ----------
        A1, A2 : ndarray, shape (H, W)
        tau1_ns, tau2_ns : float

        Returns
        -------
        model : Tensor, shape (N, T)  on self.device
        """
        t_ns = torch.tensor(self.t_s_np * 1e9, dtype=torch.float32,
                            device=self.device)              # (T,)
        a1   = torch.tensor(A1.ravel(), dtype=torch.float32,
                            device=self.device).unsqueeze(1) # (N, 1)
        a2   = torch.tensor(A2.ravel(), dtype=torch.float32,
                            device=self.device).unsqueeze(1) # (N, 1)
        return a1 * torch.exp(-t_ns / tau1_ns) + a2 * torch.exp(-t_ns / tau2_ns)

    def _normalize_irf(self, irf):
        """
        Flatten and L1-normalise the per-pixel IRF on the compute device.

        Parameters
        ----------
        irf : ndarray, shape (H, W, T)

        Returns
        -------
        Tensor, shape (N, T)  on self.device
        """
        irf_flat = np.asarray(irf, dtype=np.float32).reshape(-1, irf.shape[2])
        irf_t    = torch.tensor(irf_flat, dtype=torch.float32, device=self.device)
        norms    = irf_t.sum(dim=1, keepdim=True).clamp(min=self.eps)
        return irf_t / norms

    def analyze_biexponential_and_reconstruct(self, G, S, irf,
                                               tau1_ns=None, tau2_ns=None,
                                               plot=True):
        """
        Reconstruct per-pixel decays from a biexponential model convolved with the IRF.
        The convolution is fully vectorised and runs on GPU when available,
        falling back to multi-threaded CPU automatically.
        Parameters
        ----------
        G, S : ndarray, shape (H, W)
            Calibrated phasor coordinates (first harmonic).
        irf : ndarray, shape (H, W, T)
        tau1_ns, tau2_ns : float
            Biexponential lifetime components in nanoseconds.
        plot : bool, optional
            If True, display A1, A2, and lifetime maps.
        Returns
        -------
        reconstructed_decay : ndarray, shape (H, W, T), or None
            IRF-convolved model decay.  Returns None if tau1_ns or tau2_ns
            is not provided.
        """
        if tau1_ns is None or tau2_ns is None:
            return None
        A1, A2 = self.compute_fractions(G, S, tau1_ns, tau2_ns, plot_graph=False)
        tau_map_ns = self.compute_lifetime(S, G)

        if plot:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            im1 = axes[0].imshow(A1, origin="lower", cmap="viridis")
            axes[0].set_title(f"A1 Map (Fraction of {tau1_ns} ns)")
            plt.colorbar(im1, ax=axes[0])

            im2 = axes[1].imshow(A2, origin="lower", cmap="plasma")
            axes[1].set_title(f"A2 Map (Fraction of {tau2_ns} ns)")
            plt.colorbar(im2, ax=axes[1])

            im3 = axes[2].imshow(np.clip(tau_map_ns, 0, 5), origin="lower", cmap="magma")
            axes[2].set_title("Phase Lifetime Map (ns)")
            plt.colorbar(im3, ax=axes[2])

            for ax in axes:
                ax.axis("off")
            plt.tight_layout()
            plt.show()

        H, W = A1.shape
        T = irf.shape[2]

        # ---- GPU/CPU-vectorised FFT convolution (no Python pixel loop) ----
        model_t = self._build_model_decay(A1, A2, tau1_ns, tau2_ns)  # (N, T)
        irf_t   = self._normalize_irf(irf)                            # (N, T)
        recon_t = self._convolve_batch(model_t, irf_t)                # (N, T)
        reconstructed_decay = recon_t.cpu().numpy().reshape(H, W, T)

        return reconstructed_decay

    # ------------------------------------------------------------------
    # Image utilities
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    def plot_phasor_diagram(self, G, S, colors=None, hexbin_color = None ,ax=None, figsize=(8, 6)):
        created_fig = ax is None
        if created_fig:
            fig, ax = plt.subplots(figsize=figsize)

        # Universal circle
        ug, us = _universal_circle_xy()
        ax.plot(ug, us, "k--")

        # Scatter / density
        g_flat, s_flat = np.ravel(G), np.ravel(S)
        if colors is None:
            if hexbin_color is None:
                hb = ax.hexbin(g_flat, s_flat, gridsize=200, cmap='autumn', mincnt=1)
            else:
                hb = ax.hexbin(g_flat, s_flat, gridsize=200, cmap=hexbin_color, mincnt=1)
            if created_fig:
                fig.colorbar(hb, ax=ax).set_label("Pixel Count")
        else:
            # If colors is a string, treat it as a colormap name
            if isinstance(colors, str):
                ax.scatter(g_flat, s_flat, cmap=colors, c=g_flat, s=8, marker="o") 
            else:
                # If it's an array of RGB values, keep your reshape logic
                ax.scatter(g_flat, s_flat, c=np.reshape(colors, (-1, 3)), s=8, marker="o")

        # Lifetime ticks
        G_mark, S_mark = self.lifetime_to_phasor(_TAU_MARKS_NS, self.frequency)
        _draw_lifetime_ticks(ax, G_mark, S_mark,
                             color="black", lw=4, fontsize=10, show_units=True)

        _style_phasor_ax(ax, title="IRF-Calibrated Phasor Diagram",
                         xlim=(-0.1, 1.1), ylim=(-0.6, 0.6))

        if created_fig:
            plt.tight_layout()
            plt.show()

    def plot_map(self, image, scales = [0, 5], title=""):
        """
        Display a lifetime map clipped to [0, 5] ns.

        Parameters
        ----------
        image : ndarray, shape (H, W)
        title : str
        """
        plt.figure(figsize=(8, 6))
        plt.imshow(np.clip(image, scales[0], scales[1]), origin="lower", cmap="viridis")
        plt.colorbar().set_label("Lifetime (ns)")
        plt.title(title)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(False)
        plt.show()

    def plot_phasor_overlay(self, decay, G, S, colormap="viridis", figsize=(8, 8)):
        """
        Display phasor-coloured intensity overlay.

        Parameters
        ----------
        decay : ndarray, shape (H, W, T)
        G, S : ndarray
        colormap : str
        figsize : tuple
        """
        intensity_img = self.generate_intensity_image(decay)
        phasor_colors = self.phasor_colormap(G, S, colormap=colormap)
        int_norm = (intensity_img - intensity_img.min()) / \
                   (intensity_img.max() - intensity_img.min() + self.eps)
        overlay = np.stack([int_norm] * 3, axis=2) * phasor_colors

        plt.figure(figsize=figsize)
        plt.imshow(overlay, origin="lower")
        plt.title("Intensity + Phasor Color Overlay")
        plt.axis("off")
        plt.show()

    def plot_overlay_subplots(self, decay, G, S, colormap="viridis", figsize=(20, 8)):
        """
        Four-panel figure: intensity, lifetime map, colour overlay, phasor scatter.

        Parameters
        ----------
        decay : ndarray, shape (H, W, T)
        G, S : ndarray, shape (H, W)  (first harmonic)
        colormap : str
        figsize : tuple
        """
        intensity_img = self.generate_intensity_image(decay)
        phasor_colors = self.phasor_colormap(G, S, colormap=colormap)
        tau_map_ns = self.compute_lifetime(S, G)

        int_norm = (intensity_img - intensity_img.min()) / \
                   (intensity_img.max() - intensity_img.min() + self.eps)
        overlay = np.stack([int_norm] * 3, axis=2) * phasor_colors

        fig, axes = plt.subplots(1, 4, figsize=figsize)

        im0 = axes[0].imshow(intensity_img, origin="lower", cmap="gray")
        axes[0].set_title("Grayscale Intensity")
        axes[0].axis("off")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(np.clip(tau_map_ns, 0, 5), origin="lower", cmap="jet")
        axes[1].set_title("Lifetime Map (ns)")
        axes[1].axis("off")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04).set_label("ns")

        im2 = axes[2].imshow(overlay, origin="lower")
        axes[2].set_title("Intensity + Phasor Color")
        axes[2].axis("off")
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        axes[3].scatter(np.ravel(G), np.ravel(S),
                        c=phasor_colors.reshape(-1, 3), s=5)
        ug, us = _universal_circle_xy()
        axes[3].plot(ug, us, "k--")
        G_mark, S_mark = self.lifetime_to_phasor(_TAU_MARKS_NS, self.frequency)
        _draw_lifetime_ticks(axes[3], G_mark, S_mark,
                             color="blue", lw=2, fontsize=7, show_units=False)
        _style_phasor_ax(axes[3], title="Phasor Scatter",
                         xlim=(-0.1, 1.1), ylim=(-0.1, 0.6))

        plt.tight_layout()
        plt.show()

    def plot_pixel_fit(self, irf, decay, reconstructed_decay, x, y,
                       log_scale=True):
        """
        Plot raw decay, IRF, and reconstructed fit for a single pixel.

        Trace extraction and normalisation run on GPU when available,
        falling back to CPU automatically (self.device is respected).

        Parameters
        ----------
        irf : ndarray, shape (H, W, T) or (T,)
        decay : ndarray, shape (H, W, T)
        reconstructed_decay : ndarray, shape (H, W, T)
        x, y : int
            Pixel coordinates (column, row).
        log_scale : bool, optional
            If True, use a logarithmic y-axis (default True).
        """
        # ---- stack all three traces and normalise in one GPU kernel -------
        irf_trace = irf[y, x, :] if irf.ndim == 3 else np.asarray(irf)
        raw_trace  = decay[y, x, :]
        fit_trace  = reconstructed_decay[y, x, :]

        traces_np = np.stack([irf_trace, raw_trace, fit_trace], axis=0).astype(np.float32)
        traces_t  = torch.tensor(traces_np, device=self.device)                    # (3, T)
        maxvals   = traces_t.amax(dim=1, keepdim=True).clamp(min=self.eps)         # (3, 1)
        norm_t    = (traces_t / maxvals).cpu().numpy()                             # (3, T)

        irf_norm, raw_norm, fit_norm = norm_t[0], norm_t[1], norm_t[2]
        # -------------------------------------------------------------------

        plt.figure(figsize=(10, 6))
        plt.plot(self.time_axis_ns, irf_norm,
                 "k--", alpha=0.5, label="IRF (Normalized)")
        plt.plot(self.time_axis_ns, raw_norm,
                 "ro", markersize=4, alpha=0.6, label=f"Raw Decay (Pixel {x},{y})")
        plt.plot(self.time_axis_ns, fit_norm,
                 "b-", lw=2, label="Reconstructed Fit")

        if log_scale:
            plt.yscale("log")
            plt.ylim(1e-3, 1.2)
            plt.ylabel("Normalized Intensity (Log Scale)")
        else:
            plt.ylabel("Normalized Intensity (Linear Scale)")

        plt.xlabel("Time (ns)")
        plt.title(f"Decay Analysis at Pixel (X: {x}, Y: {y})  "
                  f"[device: {self.device}]")
        plt.legend()
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.show()

    def plot_pixel_fit_single_exp(self, irf, decay, tau_ns, x, y,
                                  log_scale=True):
        """
        Plot raw decay, IRF, and a single-exponential model fit for one pixel.

        The model is I(t) = exp(-t / tau_ns), convolved with the pixel's
        IRF via GPU/CPU-vectorised FFT (same engine as the biexponential
        path).  The amplitude is set to 1.0 because both curves are
        peak-normalised before plotting, so only the shape matters.

        All three traces are peak-normalised in a single GPU kernel,
        identical to the approach used in ``plot_pixel_fit``.

        Parameters
        ----------
        irf : ndarray, shape (H, W, T) or (T,)
            Instrument response function.  If 3-D, the trace at (y, x)
            is used; if 1-D, the same IRF is applied to all pixels.
        decay : ndarray, shape (H, W, T)
            Measured TCSPC decay data.
        tau_ns : float
            User-supplied fluorescence lifetime in nanoseconds.
            This is the *fixed* decay constant for the single-exponential
            model:  I(t) = exp(-t / tau_ns).
        x, y : int
            Pixel coordinates (column x, row y).
        log_scale : bool, optional
            If True, use a logarithmic y-axis (default True).

        Notes
        -----
        * The model decay is built on ``self.device`` (GPU if available).
        * Convolution length is T samples (``mode='full'`` truncated),
          matching the time axis exactly — no resampling needed.
        * Peak-normalisation is performed as a single batched GPU op
          on all three traces simultaneously.
        """
        T = decay.shape[2]

        # ---- Build single-exponential model on device --------------------
        # I(t) = exp(-t / tau_ns),  shape (1, T) for _convolve_batch
        t_ns_t = torch.tensor(self.t_s_np * 1e9, dtype=torch.float32,
                               device=self.device)                      # (T,)
        model_t = torch.exp(-t_ns_t / tau_ns).unsqueeze(0)             # (1, T)

        # ---- Normalise and convolve with pixel IRF -----------------------
        irf_trace_np = irf[y, x, :] if irf.ndim == 3 else np.asarray(irf)
        irf_trace_t  = torch.tensor(
            irf_trace_np.astype(np.float32), device=self.device
        ).unsqueeze(0)                                                  # (1, T)

        # L1-normalise the IRF so convolution preserves decay amplitude
        irf_norm_t = irf_trace_t / irf_trace_t.sum(dim=1, keepdim=True).clamp(
            min=self.eps
        )
        fit_t = self._convolve_batch(model_t, irf_norm_t)              # (1, T)
        fit_trace_np = fit_t.squeeze(0).cpu().numpy()                  # (T,)

        # ---- Extract raw decay trace -------------------------------------
        raw_trace_np = decay[y, x, :]

        # ---- Peak-normalise all three traces in one GPU kernel -----------
        traces_np = np.stack(
            [irf_trace_np, raw_trace_np, fit_trace_np], axis=0
        ).astype(np.float32)                                            # (3, T)
        traces_t  = torch.tensor(traces_np, device=self.device)
        maxvals   = traces_t.amax(dim=1, keepdim=True).clamp(min=self.eps)
        norm_t    = (traces_t / maxvals).cpu().numpy()

        irf_norm, raw_norm, fit_norm = norm_t[0], norm_t[1], norm_t[2]
        # -------------------------------------------------------------------

        plt.figure(figsize=(10, 6))
        plt.plot(self.time_axis_ns, irf_norm,
                 "k--", alpha=0.5, label="IRF (Normalized)")
        plt.plot(self.time_axis_ns, raw_norm,
                 "ro", markersize=4, alpha=0.6, label=f"Raw Decay (Pixel {x},{y})")
        plt.plot(self.time_axis_ns, fit_norm,
                 "b-", lw=2, label=f"Single-Exp Fit  τ = {tau_ns} ns")

        if log_scale:
            plt.yscale("log")
            plt.ylim(1e-3, 1.2)
            plt.ylabel("Normalized Intensity (Log Scale)")
        else:
            plt.ylabel("Normalized Intensity (Linear Scale)")

        plt.xlabel("Time (ns)")
        plt.title(
            f"Single-Exponential Decay at Pixel (X: {x}, Y: {y})  "
            f"τ = {tau_ns} ns  [device: {self.device}]"
        )
        plt.legend()
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.show()

    # ------------------------------------------------------------------
    # Multi-harmonic phasor visualization
    # ------------------------------------------------------------------
    def plot_phasor_harmonics(self, G, S, harmonics=(1, 2, 3, 4),
                              colors=None, figsize=(22, 5)):
        """
        Display one phasor diagram per harmonic in a side-by-side grid.

        Each panel is a fully independent phasor plot drawn at harmonic k,
        complete with its own universal semicircle and lifetime ticks
        re-evaluated at  ω_k = k · ω_eff.

        Mathematical background
        -----------------------
        For a single-exponential fluorophore with lifetime τ the k-th
        harmonic phasor coordinates are:

            G_k(τ) =        1        /  (1 + (k·ω·τ)²)
            S_k(τ) = (k·ω·τ)        /  (1 + (k·ω·τ)²)

        These still satisfy  G_k² + S_k² = G_k  (the universal semicircle),
        but the lifetime ticks are *compressed* toward (0, 0) at higher k
        because the denominator grows as (k·ω·τ)².  Concretely:

          • Harmonic 1 : good lifetime contrast for τ ≈ 1–10 ns
          • Harmonic 2 : good contrast for τ ≈ 0.5–5 ns
          • Harmonic 3 : good contrast for τ ≈ 0.3–3 ns
          • Harmonic 4 : good contrast for τ ≈ 0.25–2 ns

        Inspecting all harmonics simultaneously reveals multi-component
        mixtures and serves as an internal consistency check.

        Parameters
        ----------
        G : ndarray, shape (n_harmonics, H, W)  or  (H, W)
            Phasor G (cosine) stack.  If 2-D, the same map is displayed in
            every panel (tick positions still change per harmonic).
        S : ndarray, shape (n_harmonics, H, W)  or  (H, W)
            Phasor S (sine) stack, same convention as G.
        harmonics : tuple of int, optional
            Harmonic indices to display.  Default (1, 2, 3, 4).
            Each k must satisfy  1 ≤ k ≤ n_harmonics  when G/S are 3-D.
        colors : ndarray or None, optional
            RGB array, shape (H, W, 3).  When provided a coloured scatter
            is drawn instead of the default hexbin density map.
        figsize : tuple, optional
            Total figure size.  Default (22, 5) suits 4 side-by-side panels.

        Notes
        -----
        * The method visualises whatever G/S arrays are passed in; it does
          not recompute phasors.  Ensure the arrays were computed with
          ``n_harmonics ≥ max(harmonics)``.
        * Lifetime ticks at harmonic k use  ω_k = k · ω_eff  so their
          positions are consistent with the phasor data at that harmonic.
        """
        G = np.asarray(G)
        S = np.asarray(S)
        n_panels = len(harmonics)

        fig, axes = plt.subplots(1, n_panels, figsize=figsize)
        if n_panels == 1:
            axes = [axes]

        for ax, k in zip(axes, harmonics):
            # ---- Select the correct harmonic slice ----------------------
            if G.ndim == 3 and k <= G.shape[0]:
                g_panel = G[k - 1]          # (H, W)
                s_panel = S[k - 1]
            else:
                # Fallback: use first (or only) available slice
                g_panel = G[0] if G.ndim == 3 else G
                s_panel = S[0] if S.ndim == 3 else S

            # ---- Universal semicircle -----------------------------------
            ug, us = _universal_circle_xy()
            ax.plot(ug, us, "k--", lw=1.2)

            # ---- Scatter / density -------------------------------------
            g_flat = np.ravel(g_panel)
            s_flat = np.ravel(s_panel)

            if colors is None:
                hb = ax.hexbin(g_flat, s_flat, gridsize=150,
                               cmap="jet", mincnt=1)
                fig.colorbar(hb, ax=ax, fraction=0.046,
                             pad=0.04).set_label("Pixel count")
            else:
                c_flat = np.reshape(colors, (-1, 3))
                ax.scatter(g_flat, s_flat, c=c_flat, s=6, marker="o")

            # ---- Lifetime ticks at harmonic k --------------------------
            # At harmonic k the angular frequency seen by the phasor is
            # k·ω, so we pass  k·frequency  as the frequency
            # argument to lifetime_to_phasor — that function internally
            # computes  ω = 2π·f, yielding  ω_k = k·2π·f_eff  as needed.
            G_mark, S_mark = self.lifetime_to_phasor(
                _TAU_MARKS_NS, k * self.frequency
            )
            _draw_lifetime_ticks(ax, G_mark, S_mark,
                                 color="black", lw=2, fontsize=8,
                                 show_units=(k == harmonics[0]),
                                 tick_length=0.022, text_offset=0.038)

            # ---- Axes style --------------------------------------------
            _style_phasor_ax(ax,
                             title=f"Harmonic {k}   (ω₍ₖ₎ = {k}·ω₀)",
                             xlim=(-0.1, 1.1), ylim=(-0.15, 0.65))

        fig.suptitle("Phasor Diagram — Multiple Harmonics",
                     fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()
        plt.show()

    def save_phasors_hdf5(self, Gc, Sc, tau_phasor, save_file):
        try:
            with h5py.File(save_file, 'w') as hf:
                # Create datasets with compression to save disk space
                hf.create_dataset('Gc', data=Gc, compression="gzip", chunks=True)
                hf.create_dataset('Sc', data=Sc, compression="gzip", chunks=True)
                hf.create_dataset('tau_phasor', data=tau_phasor, compression="gzip", chunks=True)
                
                # Optional: Add metadata/attributes for context
                hf.attrs['n_harmonics'] = Gc.shape[0]
                hf.attrs['resolution'] = f"{Gc.shape[1]}x{Gc.shape[2]}"
                
            print(f"Successfully saved data to {save_file}")
            
        except Exception as e:
            print(f"An error occurred while saving: {e}")
