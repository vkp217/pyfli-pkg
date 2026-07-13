"""
FLI (Fluorescence Lifetime Imaging) Simulator
==============================================
HardSimulator    -> uses generate_pixel_decay()  (analytical convolution + Poisson)
HardestSimulator  -> uses tcspc_pixel_decay()      (photon-by-photon TCSPC simulation)
"""

import numpy as np
from scipy.signal import fftconvolve
from scipy.optimize import curve_fit
from scipy.special import hyp2f1
from scipy.stats import truncnorm
from tqdm import tqdm
from PIL import Image
from .dataIO.dataoperations import DataOperations


# Maximum IRF gate shift used by _jitter()
_MAX_GATE_SHIFT = 3


class HardSimulator:
    """Per-pixel FLI simulator using analytical IRF convolution + Poisson noise.

    On each call, builds a fresh ``HeterogeneousFLISimulator`` configured
    with a randomly chosen tau2 prior and cycle count, and generates one
    pixel's decay via ``generate_pixel_decay`` (analytical bi-/mono-
    exponential decay convolved with the IRF, scaled by a sampled photon
    count, then Poisson-noised).
    """

    def __init__(self,
        irf_file_path='../data/raw/ICCD/paper_IRF700nm.mat',
        tau2=None,  # Can be: None (Random), (mu, sigma), or [(mu1, sig1), (mu2, sig2)]
        efficiency=(2, 5),
        f_fraction=(4, 5),
        photo_count=(5, 5),
        mono_fraction=0.2,
        bit=10,
        n_cycles=800_000
    ):
        """Loads the IRF and stores simulation hyperparameters.

        Args:
            irf_file_path: Path to a ``.mat`` IRF file, loaded via
                ``DataOperations.load_irf``.
            tau2: Controls the tau2 (donor lifetime) prior used per call:
                ``None`` draws a random ``(mu, sigma)`` pair each call
                (mu in [0.3, 3.0], sigma in [0.05, min(0.4, 0.4*mu)]); a
                single ``(mu, sigma)`` tuple fixes the prior; a list of
                ``(mu, sigma)`` tuples selects one uniformly at random
                per call.
            efficiency: ``(alpha, beta)`` beta prior for FRET efficiency
                ``E``, forwarded to ``HeterogeneousFLISimulator``.
            f_fraction: ``(alpha, beta)`` beta prior for amplitude
                fraction ``f``, forwarded to ``HeterogeneousFLISimulator``.
            photo_count: ``(alpha, beta)`` beta prior for photon count,
                forwarded to ``HeterogeneousFLISimulator``.
            mono_fraction: Probability that a pixel is sampled as
                mono-exponential.
            bit: ADC bit depth used to scale the sampled photon count.
            n_cycles: Upper bound on the number of excitation cycles; the
                actual cycle count is re-randomized on every call.
        """
        self.irf_data_full = DataOperations(irf_path = irf_file_path).load_irf()
        self.tau2_user = tau2 
        self.efficiency = efficiency
        self.f_fraction = f_fraction
        self.photo_count = photo_count
        self.mono_fraction = mono_fraction
        self.bit = bit
        self.n_cycles = n_cycles

    def _get_current_tau2(self):
        """Logic to determine which distribution to use for the current call."""
        if isinstance(self.tau2_user, list):
            idx = np.random.choice(len(self.tau2_user))
            return self.tau2_user[idx]
        if isinstance(self.tau2_user, tuple):
            return self.tau2_user
            
        mu = np.random.uniform(0.3, 3.0)
        max_sigma = max(0.06, min(0.4, mu * 0.4)) 
        sigma = np.random.uniform(0.05, max_sigma)        
        return (mu, sigma)

    def __call__(self):
        """Simulates one pixel and returns its parameters and decay features.

        Draws a random cycle count and tau2 prior, builds a new
        ``HeterogeneousFLISimulator``, generates the analytical decay
        (``generate_pixel_decay``), normalizes it, and computes FFT/phasor
        features.

        Returns:
            dict: The sampled parameter dict (``pars``) merged with
            ``tau1_l``, ``tau2_l``, ``a_l`` (amplitude fraction), ``tau_mean_l``
            (intensity-weighted mean lifetime), ``s_t`` (normalized decay,
            squeezed), ``fft`` (FFT/phasor harmonic features of the
            observed decay), and ``irf`` (the instrument response function
            used).
        """
        n_pixel_cycles = np.random.randint(1, self.n_cycles + 1)

        current_tau2 = self._get_current_tau2()
        self.fli_sim = HeterogeneousFLISimulator(
            self.irf_data_full,  
            efficiency=self.efficiency,
            f_fraction=self.f_fraction,
            tau2=current_tau2,
            photo_count=self.photo_count,
            mono_fraction=self.mono_fraction,
            bit=self.bit,
            n_cycles=n_pixel_cycles
        )

        decay, observed, scaled, pars, A, irf = self.fli_sim.generate_pixel_decay()
        decay_norm = self.fli_sim.pixel_wise_normalisation(observed)

        tau_mean = pars['tau1'] * pars['f'] + pars['tau2'] * (1 - pars['f'])

        return {**pars,
            "tau1_l": pars['tau1'],
            "tau2_l": pars['tau2'],
            "a_l": pars['f'],
            "tau_mean_l": tau_mean,
            "s_t": decay_norm.squeeze(),
            "fft": self.fli_sim.fft_features(observed.squeeze()),
            "irf": irf,
        }


class HardestSimulator:
    """
    Pixel simulator using photon-by-photon TCSPC Monte Carlo simulation
    (tcspc_pixel_decay).
    """

    def __init__(self,
        irf_file_path='../data/raw/SPCImage/SPCimage_IRF.txt',
        tau2=(1, 0.4),
        efficiency=(7, 5),
        f_fraction=(3, 5),
        photo_count=(1.1, 5),
        mono_fraction=0.2,
        bit=8, ):
        """Loads the IRF and stores simulation hyperparameters.

        Args:
            irf_file_path: Path to an IRF file (``.mat``, ``.txt``, etc.),
                loaded via ``DataOperations.load_irf``.
            tau2: ``(mu, sigma)`` prior for the fixed tau2 lifetime used at
                construction time (unlike ``HardSimulator``, this is not
                re-randomized per call here; ``__call__`` builds its own
                ``current_tau2`` instead).
            efficiency: ``(alpha, beta)`` beta prior for FRET efficiency
                ``E``, forwarded to ``HeterogeneousFLISimulator``.
            f_fraction: ``(alpha, beta)`` beta prior for amplitude
                fraction ``f``, forwarded to ``HeterogeneousFLISimulator``.
            photo_count: ``(alpha, beta)`` beta prior for photon count,
                forwarded to ``HeterogeneousFLISimulator``.
            mono_fraction: Probability that a pixel is sampled as
                mono-exponential.
            bit: ADC bit depth used to scale the sampled photon count.
        """
        self.irf_data_full = DataOperations(irf_path=irf_file_path).load_irf()
        self.efficiency = efficiency
        self.f_fraction = f_fraction
        self.photo_count = photo_count
        self.mono_fraction = mono_fraction
        self.bit = bit

    def __call__(self):
        """Simulates one pixel via photon-by-photon TCSPC Monte Carlo.

        Draws a random cycle count (up to 800,000) and a random tau2 mean
        in [0.2, 2.5] ns (fixed std 0.199999), builds a new
        ``HeterogeneousFLISimulator``, generates the photon histogram
        (``tcspc_pixel_decay``), normalizes it, and computes FFT/phasor
        features.

        Returns:
            dict: The sampled parameter dict (``pars``) merged with
            ``tau1_l``, ``tau2_l``, ``a_l`` (amplitude fraction), ``tau_mean_l``
            (intensity-weighted mean lifetime), ``s_t`` (normalized
            photon histogram, squeezed), ``fft`` (FFT/phasor harmonic
            features of the observed histogram), and ``irf`` (the
            instrument response function used).
        """
        n_cycles = np.random.randint(800_000)
        current_tau2 = (np.random.uniform(0.2, 2.5), 0.199999)
        self.fli_sim = HeterogeneousFLISimulator(
            self.irf_data_full,
            tau2=current_tau2,
            efficiency=self.efficiency,
            f_fraction=self.f_fraction,
            photo_count=self.photo_count,
            mono_fraction=self.mono_fraction,
            bit=self.bit,
            n_cycles=n_cycles,
        )
        decay, observed, pars, A, irf = self.fli_sim.tcspc_pixel_decay()
        decay_norm = self.fli_sim.pixel_wise_normalisation(observed)

        tau_mean = pars['tau1'] * pars['f'] + pars['tau2'] * (1 - pars['f'])

        return { **pars,
            "tau1_l": pars['tau1'],
            "tau2_l": pars['tau2'],
            "a_l": pars['f'],
            "tau_mean_l": tau_mean,
            "s_t": decay_norm.squeeze(),
            "fft": self.fli_sim.fft_features(observed.squeeze()),
            "irf": irf,
        }


# ============================================================================
#  Core simulator
# ============================================================================

class HeterogeneousFLISimulator:
    """Core single-pixel FLI physics model: sampling, decay, TCSPC, and phasor analysis.

    Models a fluorescence-lifetime pixel as a mixture of a "long" (tau2)
    and, for non-mono pixels, a "short" (tau1, derived from a sampled FRET
    efficiency ``E``) exponential decay component. Supports:

    * Analytical decay generation with IRF convolution and Poisson noise
      (``generate_pixel_decay``).
    * Photon-by-photon TCSPC Monte Carlo simulation with IRF timing jitter
      and pile-up rejection (``tcspc_pixel_decay``).
    * Post-processing utilities: baseline-corrected/normalized decay
      (``pixel_wise_normalisation``), FFT/phasor harmonic features
      (``fft_features``), analytical global parameters/phasor coordinates,
      curve-fitting-based lifetime recovery, and per-pixel Fisher
      information for parameter estimation.
    """

    def __init__(
        self,
        irf_full,
        tau2,                 # (mean_tau2, std_tau2)
        efficiency,               # Beta(alpha, beta) for FRET efficiency E
        f_fraction,               # Beta(alpha, beta) for amplitude fraction f
        photo_count=(1.5, 5),     # Beta(alpha, beta) scaled to photon count
        mono_fraction=0.1,        # probability of mono-exponential pixel
        bit=8,
        omega=0.08,               # angular frequency for phasor (rad/ns)
        n_cycles=800_000,
        norm_type='pdf_robust',
    ):
        """Resolves the IRF and stores sampling/normalization parameters.

        Args:
            irf_full: IRF data: a 1-D trace used directly, or a 3-D stack
                ``(H, W, T)`` from which a random pixel is selected (falling
                back to the center pixel if the randomly chosen ICCD pixel's
                total counts are below 5000).
            tau2: ``(mean_tau2, std_tau2)`` for the truncated-normal tau2
                (long-lifetime) prior.
            efficiency: ``(alpha, beta)`` beta prior for FRET efficiency
                ``E`` (non-mono pixels).
            f_fraction: ``(alpha, beta)`` beta prior for amplitude
                fraction ``f`` (non-mono pixels).
            photo_count: ``(alpha, beta)`` beta prior for photon count,
                scaled by ``2**bit - 1``.
            mono_fraction: Probability that a pixel is sampled as
                mono-exponential (see ``sample_local_parameters``).
            bit: ADC bit depth used to scale the sampled photon count.
            omega: Angular modulation frequency (rad/ns) used in
                ``analytical_phasor``.
            n_cycles: Number of laser excitation cycles simulated per
                pixel in ``tcspc_pixel_decay``.
            norm_type: Normalization mode used by
                ``pixel_wise_normalisation``: ``'None'``, ``'pdf'``,
                ``'min_max'``, or ``'pdf_robust'``.

        Raises:
            ValueError: If ``irf_full`` is neither 1-D nor 3-D.
        """
        # ---- IRF ----
        if irf_full.ndim == 3:
            x = np.random.randint(irf_full.shape[0])
            y = np.random.randint(irf_full.shape[1])
            if np.sum(irf_full[x, y, :]) < 5000: # this condition is for ICCD
                irf = irf_full[irf_full.shape[0] // 2, irf_full.shape[1] // 2, :]
            else:
                irf = irf_full[x, y, :]
        elif irf_full.ndim == 1:
            irf = irf_full
        else:
            raise ValueError(f'IRF must be 1-D or 3-D, got shape {irf_full.shape}')

        self.irf = np.nan_to_num(irf / irf.sum())

        # ---- Time axis ----
        n = self.irf.shape[0]
        self.t = np.linspace(0, 12.5, n)
        self.dt = self.t[1] - self.t[0]

        # ---- Distribution parameters ----
        self.tau2_mean, self.tau2_std = tau2
        self.alpha_E, self.beta_E = efficiency
        self.alpha_f, self.beta_f = f_fraction
        self.alpha_A, self.beta_A = photo_count

        self.mono_fraction = mono_fraction
        self.bit = bit
        self.omega = omega
        self.n_cycles = n_cycles
        self.T_rep = 12.5
        self.mu_per_cycle = 0.01
        self.norm_type = norm_type
        self.eps = 1e-4

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def pixel_wise_normalisation(self, decay_series):
        """Normalizes a decay/histogram trace according to ``self.norm_type``.

        Supported modes:

        * ``'None'`` — returns the input unchanged.
        * ``'pdf'`` — divides by the total sum (probability-density-like
          normalization); returns the input unchanged if the sum is 0.
        * ``'min_max'`` — rescales to ``[0, 1]`` via ``(x - min) / (max -
          min + eps)``.
        * ``'pdf_robust'`` — subtracts a baseline (median of the last
          ``self.baseline_pts``, default 20, samples), clips negative
          values to 0, then normalizes to sum to 1; falls back to the raw
          (baseline-subtracted-and-clipped) decay if the resulting total is
          non-finite or below ``eps``.

        Args:
            decay_series: Array-like decay or photon-count histogram to
                normalize.

        Returns:
            numpy.ndarray: The normalized decay, as ``float64``.

        Raises:
            ValueError: If ``self.norm_type`` is not one of the supported
                modes.
        """
        eps = 1e-12
        decay = np.asarray(decay_series, dtype=np.float64)

        if self.norm_type == 'None':
            return decay

        if self.norm_type == 'pdf':
            total = np.sum(decay)
            return decay / total if total > 0 else decay

        if self.norm_type == 'min_max':
            lo, hi = decay.min(), decay.max()
            return (decay - lo) / (hi - lo + eps)

        if self.norm_type == 'pdf_robust':
            baseline_pts = getattr(self, 'baseline_pts', 20)
            if baseline_pts > 0 and len(decay) >= baseline_pts:
                baseline = np.median(decay[-baseline_pts:])
            else:
                baseline = 0.0
            decay_bs = np.clip(decay - baseline, 0.0, None)
            total = np.sum(decay_bs)
            return decay_bs / total if np.isfinite(total) and total >= eps else decay

        raise ValueError(f'Unsupported norm_type: {self.norm_type!r}')

    # ------------------------------------------------------------------
    # FFT / phasor features
    # ------------------------------------------------------------------

    @staticmethod
    def fft_features(decay, n_harmonics=5):
        """Computes DC-normalized FFT (phasor) harmonic features of a decay.

        Takes the real FFT of the decay along its last axis, normalizes
        the first ``n_harmonics`` non-DC coefficients by the DC component
        (total intensity), and returns their real (``g``) and imaginary
        (``s``) parts concatenated — the same ``g``/``s`` quantities used
        in phasor-plot lifetime analysis.

        Args:
            decay: 1-D decay array, or a 2-D batch of decays (``(B, T)``).
            n_harmonics: Number of non-DC FFT harmonics to extract.

        Returns:
            numpy.ndarray: Shape ``(2 * n_harmonics,)`` for 1-D input, or
            ``(B, 2 * n_harmonics)`` for 2-D input, containing the
            harmonics' real parts followed by their imaginary parts.
        """
        decay = np.asarray(decay, dtype=np.float64)
        scalar = decay.ndim == 1
        if scalar:
            decay = decay[None, :]

        dc = np.clip(np.sum(decay, axis=1, keepdims=True), 1e-12, None)
        fft_vals = np.fft.rfft(decay, axis=1)
        coeffs = fft_vals[:, 1:n_harmonics + 1]

        g = np.real(coeffs) / dc
        s = np.imag(coeffs) / dc
        features = np.concatenate([g, s], axis=1)   # (B, 2*n_harmonics)

        return features[0] if scalar else features

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def _sample_tau2(self, lower_bound=0.01, upper_bound=5.0, size=1):
        a = (lower_bound - self.tau2_mean) / self.tau2_std
        b = (upper_bound - self.tau2_mean) / self.tau2_std
        return truncnorm.rvs(a, b, loc=self.tau2_mean, scale=self.tau2_std, size=size)

    def _safe_fraction(self, x):
        return np.clip(x, self.eps, 1.0 - self.eps)

    @staticmethod
    def _stretch_or_squeeze(samples, epsilon):
        """Map samples from [0,1] into [epsilon, 1-epsilon]."""
        return samples * (1.0 - 2.0 * epsilon) + epsilon

    # ------------------------------------------------------------------
    # Parameter sampling
    # ------------------------------------------------------------------

    def sample_local_parameters(self):
        """Samples per-pixel lifetime/amplitude parameters (mono or bi-exponential).

        Always samples tau2 from the truncated-normal prior. With
        probability ``mono_fraction``, samples a near-mono-exponential
        pixel: with 90% chance ``E=0`` (so ``tau1 == tau2``) and a
        dominant amplitude ``A1`` close to 1; otherwise ``E`` close to 1
        (so ``tau1`` close to 0) with a small ``A1``. Otherwise, samples a
        general bi-exponential pixel with ``E`` and ``f`` drawn from their
        respective beta priors and stretched away from 0/1 by a small
        margin (``E_min=0.1``, ``f_min=0.05``) via ``_stretch_or_squeeze``.

        Returns:
            dict: Parameter dict with keys ``mono`` (bool), ``E`` (FRET
            efficiency), ``f`` (amplitude fraction, steady-state-corrected
            for the mono branch), ``tau1``, ``tau2``, ``A1``, ``A2``.
        """
        tau2 = float(self._sample_tau2()[0])
        mono = np.random.rand() < self.mono_fraction

        if mono:
            T = 12.5
            eps = 1e-6
            rng = np.random.default_rng()
            if rng.random() < 0.9:
                E = 0.0
                A1 = rng.uniform(0.99, 1.0-eps)
            else:
                E = rng.uniform(0.99, 1.0)                
                A1 = rng.uniform(0+eps, 1-0.99)
            A2 = 1.0 - A1
            tau1 = tau2 * (1 - E)  # if E=0, tau1 == tau2; if E tends to 1, tau1 tends to 0 but not 
            exp_term1 = 1 - np.exp(-T / tau1)
            exp_term2 = 1 - np.exp(-T / tau2)
            f = (A1 * exp_term1) / (A1 * exp_term1 + A2 * exp_term2)
            # T = 12.5
            # rng = np.random.default_rng()         
            # A1 = rng.uniform(5e-2, 1.0 - 5e-2)
            # A2 = 1.0 - A1
            # if rng.random() < 0.9:
            #     E = 0.0
            # else:
            #     E = rng.uniform(0.99, 1.0)
            # tau1 = tau2 * (1 - E)  # if E=0, tau1 == tau2; if E tends to 1, tau1 tends to 0 but not 
            # exp_term1 = 1 - np.exp(-T / tau1)
            # exp_term2 = 1 - np.exp(-T / tau2)
            # f = (A1 * exp_term1) / (A1 * exp_term1 + A2 * exp_term2)
            return {
                "mono": True,
                "E": E,
                "f": f,
                "tau1": tau1,
                "tau2": tau2,
                "A1": A1,
                "A2": A2,
            }

        E_min = 0.1
        E = self._stretch_or_squeeze(
            round(np.random.beta(self.alpha_E, self.beta_E), 3), E_min)
        tau1 = tau2 * (1.0 - E)
        f_min = 0.05
        f = self._stretch_or_squeeze(
            round(np.random.beta(self.alpha_f, self.beta_f), 3), f_min)        
        return {
            "mono": False,
            "E": E,
            "f": f,
            "tau1": tau1,
            "tau2": tau2,
            "A1": f,
            "A2": 1.0 - f,
        }

    def sample_photon_count(self):
        """Samples a peak photon count from the beta photon-count prior.

        Returns:
            int: A photon count sampled from ``Beta(alpha_A, beta_A)`` and
            scaled to the ``[0, 2**bit - 1]`` ADC range.
        """
        return round(np.random.beta(self.alpha_A, self.beta_A) * (2 ** self.bit - 1))

    # IRF jitter
    @staticmethod
    def _jitter(decay):
        """Apply a random sub-bin shift to simulate timing jitter."""
        n = len(decay)
        r = np.random.rand()
        shift = np.random.randint(0, _MAX_GATE_SHIFT + 1)

        if r > 0.75 or shift == 0:
            return decay
        if r < 0.25:
            # shift right (delay)
            return np.concatenate([np.zeros(shift), decay[:n - shift]])
        # shift left (advance)
        return np.concatenate([decay[shift:], np.zeros(shift)])

    # Analytical convolution pixel decay  (used by HardSimulator)
    def generate_pixel_decay(self):
        """
        Bi-exponential decay convolved with IRF, then Poisson-sampled.

        Returns
        -------
        decay           : clean bi-exponential before convolution
        observed        : Poisson-noisy photon histogram
        scaled          : IRF-convolved decay scaled by photon count A
        pars            : sampled parameter dict
        A               : sampled photon count
        irf             : the instrument response function
        """
        pars = self.sample_local_parameters()
        A = self.sample_photon_count()
        t = self.t

        if pars["mono"]:
            decay = np.exp(-t / pars["tau1"])
        else:
            decay = pars["A1"] * np.exp(-t / pars["tau1"]) + pars["A2"] * np.exp(-t / pars["tau2"])
        decay_conv = fftconvolve(decay, self.irf.squeeze(), mode="full")[:len(decay)]
        decay_conv = self._jitter(decay_conv)

        # Scale to photon count; clip at 0 to keep Poisson valid
        scaled = np.clip(decay_conv * A, 0.0, None)
        observed = np.random.poisson(scaled).astype(np.float64)

        return decay, observed, scaled, pars, A, self.irf

    # Global analytical parameters / phasor / Fisher information
    def analytical_global_parameters(self):
        """Computes population-level (prior-mean) lifetime/fraction estimates.

        Derives expected tau1, tau2, and amplitude fraction directly from
        the configured prior hyperparameters (means of the ``f_fraction``
        and ``efficiency`` beta distributions and the tau2 prior mean),
        without drawing any random samples.

        Returns:
            dict: ``{"tau1_global": ..., "tau2_global": ..., "f_global":
            ...}``.
        """
        mu_f = self.alpha_f / (self.alpha_f + self.beta_f)
        tau2_global = self.tau2_mean
        tau1_global = tau2_global * (self.beta_E / (self.alpha_E + self.beta_E + 1))
        return {"tau1_global": tau1_global, "tau2_global": tau2_global, "f_global": mu_f}

    @staticmethod
    def biexponential(t, a1, tau1, a2, tau2):
        """Bi-exponential decay model function used for curve fitting.

        Args:
            t: Time points (array-like or scalar).
            a1: Amplitude of the first component.
            tau1: Lifetime of the first component.
            a2: Amplitude of the second component.
            tau2: Lifetime of the second component.

        Returns:
            Model evaluated at ``t``: ``a1 * exp(-t/tau1) + a2 * exp(-t/tau2)``.
        """
        return a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2)

    def recover_global_lifetime(self, decay):
        """Fits a bi-exponential model to a decay curve via least squares.

        Uses ``scipy.optimize.curve_fit`` with an initial guess derived
        from ``self.tau2_mean`` (``p0 = [0.4, 0.5*tau2_mean, 0.6,
        tau2_mean]``).

        Args:
            decay: Decay curve (same length as ``self.t``) to fit.

        Returns:
            dict: ``{"a1": ..., "tau1": ..., "a2": ..., "tau2": ...}``
            fitted parameters.

        Raises:
            RuntimeError: If the least-squares fit fails to converge
                within ``maxfev=20_000`` function evaluations.
        """
        tau2_init = self.tau2_mean
        p0 = [0.4, 0.5 * tau2_init, 0.6, tau2_init]
        popt, _ = curve_fit(self.biexponential, self.t, decay, p0=p0, maxfev=20_000)
        return {"a1": popt[0], "tau1": popt[1], "a2": popt[2], "tau2": popt[3]}

    def analytical_phasor(self):
        """Computes analytical population-averaged phasor coordinates.

        Analytically averages the single-exponential phasor point of the
        long-lifetime (tau2) component with the Gauss-hypergeometric-
        weighted phasor of the short-lifetime (tau1, via the ``E`` beta
        prior) component, mixed by the mean amplitude fraction ``f``.

        Returns:
            tuple: ``(g, s)`` analytical phasor coordinates.
        """
        mu_f = self.alpha_f / (self.alpha_f + self.beta_f)
        tau2 = self.tau2_mean
        c = self.omega * tau2

        g_long = 1.0 / (1.0 + c ** 2)
        s_long = c / (1.0 + c ** 2)

        g_short = hyp2f1(0.5, self.beta_E, self.alpha_E + self.beta_E, -(c ** 2))
        s_short = c * hyp2f1(1.5, self.beta_E, self.alpha_E + self.beta_E + 1, -(c ** 2))

        return (
            mu_f * g_short + (1 - mu_f) * g_long,
            mu_f * s_short + (1 - mu_f) * s_long,
        )

    def fisher_information(self):
        """Computes the Fisher information matrix for one sampled bi-exponential pixel.

        Samples a fresh set of local parameters and, under a Poisson
        photon-counting noise model, computes the Fisher information
        matrix for the parameters ``(tau1, tau2, f, E)`` from the
        analytic gradients of the expected intensity ``I(t)`` (using
        ``I ~ Poisson``, so the per-bin Fisher contribution is
        ``(dI/dtheta)^2 / I``).

        Returns:
            numpy.ndarray: A symmetric, non-negative ``4x4`` Fisher
            information matrix ordered ``(tau1, tau2, f, E)``. Returns an
            all-zero ``4x4`` matrix if the freshly sampled pixel happens
            to be mono-exponential.
        """
        pars = self.sample_local_parameters()
        if pars["mono"]:
            return np.zeros((4, 4))

        t = self.t
        tau1 = max(pars["tau1"], 0.01)
        tau2 = max(pars["tau2"], 0.01)
        f = self._safe_fraction(pars["f"])
        A = max(self.sample_photon_count(), 1)

        I = A * (f * np.exp(-t / tau1) + (1 - f) * np.exp(-t / tau2))
        I = np.clip(I, 1e-8, None)
        sqrt_I = np.sqrt(I)

        d_tau1 = f * np.exp(-t / tau1) * (t / tau1 ** 2) / sqrt_I
        d_tau2 = (1 - f) * np.exp(-t / tau2) * (t / tau2 ** 2) / sqrt_I
        d_f = (np.exp(-t / tau1) - np.exp(-t / tau2)) / sqrt_I
        d_E = f * np.exp(-t / tau1) * (t / tau1 ** 2) * tau2 / sqrt_I

        grads = np.vstack([d_tau1, d_tau2, d_f, d_E])
        F = grads @ grads.T
        return np.clip((F + F.T) / 2.0, 0, None)


## TCSPC photon-by-photon simulation  (used by HardestSimulator)
    def tcspc_pixel_decay(self):
        """
        Photon-by-photon TCSPC simulation with pile-up and IRF convolution.
        Returns:
        decay    : clean analytical decay (shape only)
        hist     : photon histogram (float64)
        pars     : sampled parameter dict
        A        : sampled photon count (unused for binning, informational)
        irf      : the instrument response function
        """
        pars = self.sample_local_parameters()
        A = self.sample_photon_count()
        t = self.t
        n_bins = self.irf.shape[0]
        dt = self.T_rep / n_bins
        if pars["mono"]:
            decay = np.exp(-t / max(pars["tau1"], 1e-6))
        else:
            decay = (
                pars["A1"] * np.exp(-t / pars["tau1"]) +
                pars["A2"] * np.exp(-t / pars["tau2"])
            )

        # Poisson photon counts per excitation cycle
        k_per_cycle = np.random.poisson(self.mu_per_cycle, size=self.n_cycles)
        total_photons = int(k_per_cycle.sum())

        hist = np.zeros(n_bins, dtype=np.float64)
        if total_photons == 0:
            return decay, hist, pars, A, self.irf
        # Sample emission times 
        if pars["mono"]:
            emission_times = np.random.exponential(scale=pars["tau1"], size=total_photons)
        else:
            comp1 = np.random.rand(total_photons) < pars["A1"]
            emission_times = np.empty(total_photons)
            n1 = comp1.sum()
            emission_times[comp1] = np.random.exponential(scale=pars["tau1"], size=n1)
            emission_times[~comp1] = np.random.exponential(scale=pars["tau2"], size=total_photons - n1)

        # ---- IRF timing jitter via inverse CDF ----
        irf_pdf = self.irf / (self.irf.sum() + 1e-12)
        irf_cdf = np.cumsum(irf_pdf)
        u = np.random.rand(total_photons)
        irf_shift = np.searchsorted(irf_cdf, u) * dt

        # ---- Arrival times (pile-up: only first photon within T_rep matters) ----
        arrival_times = emission_times + irf_shift
        arrival_times = arrival_times[arrival_times < self.T_rep]

        if arrival_times.size == 0:
            return decay, hist, pars, A, self.irf

        # ---- Histogram ----
        bins = (arrival_times / dt).astype(np.int32)
        bins = bins[bins < n_bins]
        hist = np.bincount(bins, minlength=n_bins).astype(np.float64)
        hist = self._jitter(hist)

        return decay, hist, pars, A, self.irf


#### Fluorescence Lifetime Image and Parameters Map Generator
class FLIImageGenerator:
    """
    Generates full FLI images with support for intensity masking and ROI-based 
    parameter variations using internalized HardSimulator logic.
    """
    def __init__(self, intensity_image_path=None, roi_mask_path=None, 
                 roi_params=None, image_shape=(32, 32), method='analytical'):
        """
        Parameters:
        -----------
        intensity_image_path : str, optional
            Path to a .png or .jpg to use as photon counts.
        roi_mask_path : str, optional
            Path to a grayscale/label image where 0, 1, 2... define different ROIs.
        roi_params : list of dict, optional
            A list of dictionaries containing simulator arguments for each ROI.
            Example: [{'tau2': (0.4, 0.2)}, {'tau2': (1.1, 0.3)}]
        image_shape : tuple
            Default shape if no intensity image is provided.
        method : str
            'analytical' or 'tcspc'.
        """
        self.method = method.lower()
        
        # 1. Load Intensity Image
        if intensity_image_path:
            img = Image.open(intensity_image_path).convert('L')
            self.intensity_mask = np.array(img).astype(np.float64)
            self.shape = self.intensity_mask.shape
            self.use_intensity_mask = True
        else:
            self.intensity_mask = None
            self.shape = image_shape
            self.use_intensity_mask = False

        # 2. Load ROI Mask
        if roi_mask_path:
            mask_img = Image.open(roi_mask_path).convert('L')
            self.roi_mask = np.array(mask_img).astype(np.int32)
            # Ensure mask matches intensity image dimensions
            if self.roi_mask.shape != self.shape:
                self.roi_mask = np.array(mask_img.resize((self.shape[1], self.shape[0]), Image.NEAREST))
        else:
            self.roi_mask = np.zeros(self.shape, dtype=np.int32)

        # 3. Initialize Internal Simulators per ROI
        self.roi_sims = {}
        unique_rois = np.unique(self.roi_mask)
        
        for idx, roi_val in enumerate(unique_rois):
            params = roi_params[idx] if (roi_params and idx < len(roi_params)) else {"tau2": None}
            self.roi_sims[roi_val] = HardSimulator(**params)

        sample_output = self.roi_sims[unique_rois[0]]()
        n_t = sample_output["s_t"].size

        # Storage
        self.decay_image = np.zeros((*self.shape, n_t))
        self.irf_image = np.zeros((*self.shape, n_t))
        self.photon_counts = np.zeros(self.shape)
        self.tau1_map = np.zeros(self.shape)
        self.tau2_map = np.zeros(self.shape)
        self.f_map = np.zeros(self.shape)
        self.E_map = np.zeros(self.shape)
        self.A1_map = np.zeros(self.shape)
        self.A2_map = np.zeros(self.shape)
        self.mono_map = np.zeros(self.shape, dtype=bool)
        self.tau_mean_map = np.zeros(self.shape)
        
        # Note: Added tau_mean_map to capture the calculated mean lifetime from the simulator

    def generate_image(self):
        """Generate 2D FLI image with ROI-specific simulators and tqdm progress."""
        total_pixels = self.shape[0] * self.shape[1]
        
        with tqdm(total=total_pixels, desc=f"Simulating ROI FLI ({self.method})") as pbar:
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    # Select the simulator assigned to this pixel's ROI
                    roi_val = self.roi_mask[i, j]
                    sim_wrapper = self.roi_sims[roi_val]
                    
                    # Generate pixel data using the __call__ method of HardSimulator
                    pixel_data = sim_wrapper()

                    observed = pixel_data["s_t"]
                    irf = pixel_data["irf"]
