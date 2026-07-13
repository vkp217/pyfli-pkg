# simulator/simulator_engine

"""Bi-exponential per-pixel FLI parameter/decay model engine.

Provides ``FLIEngine``, the core per-pixel model engine used by
``Macro_sim`` and ``TCSPC_sim``. Every pixel is modeled as a two-component
(FRET donor/acceptor-like) exponential mixture, with an optional
near-mono-exponential regime for a fraction of pixels (see
``sample_all_params``).
"""

import numpy as np

from .distributions import ParameterSampler
from .noise_models import NoiseEngine
from .sim_helper import irf_picker

class FLIEngine:
    """Samples per-pixel bi-exponential FLI parameters and generates decay curves.

    Every pixel is modeled with two lifetime components (``tau1``,
    ``tau2``) and their amplitude fractions (``A1``, ``A2``), related via a
    FRET efficiency ``E``. A configurable fraction of pixels is sampled in
    a near-mono-exponential regime (``E`` close to 0). Provides both an
    analytical (steady-state, pulse-repetition-corrected) decay model and a
    photon-by-photon TCSPC Monte Carlo simulator.
    """

    def __init__(self,
                 irf_full,
                 tau2=(1, 0.5),
                 efficiency = (5, 5),
                 A1_fraction = (5, 5),
                 photo_count = (1.2, 5),
                 mono_fraction = 0.2,
                 bit = 8,
                 n_cycles = 800_000,
                 dcr = 0.05,
                 laser_feq = 80,
                 seed = None,
                 **kwargs
                 ):
        """Initializes the engine and precomputes the IRF/time axis.

        Args:
            irf_full: IRF data (1-D trace or 3-D IRF stack); resolved to a
                single 1-D trace via ``irf_picker`` and stored normalized
                (sum to 1) on ``self.irf``.
            tau2: ``(mu, sigma)`` for the truncated-normal tau2 prior used
                by ``sample_all_params``.
            efficiency: ``(alpha, beta)`` for the FRET efficiency ``E``
                beta prior (non-mono pixels).
            A1_fraction: ``(alpha, beta)`` for the donor amplitude
                fraction ``A1`` beta prior (non-mono pixels).
            photo_count: ``(alpha, beta)`` for the Beta-distributed peak
                photon count / intensity.
            mono_fraction: Probability that a given pixel is sampled in
                the near-mono-exponential regime rather than the general
                bi-exponential regime.
            bit: ADC/counter bit depth, used to determine the max
                intensity/bin count.
            n_cycles: Maximum number of laser excitation cycles for TCSPC
                simulation.
            dcr: Dark count rate used by the noise pipeline.
            laser_feq: Laser repetition frequency in MHz; determines
                ``self.laser_period`` (in ns) and the time-bin width.
            seed: Optional seed for ``self.rng`` (``numpy.random.default_rng``).
            **kwargs: Additional entries merged into ``self.params_cfg``.

        Raises:
            ValueError: If the picked IRF sums to zero, a negative value,
                or a non-finite value.
        """
        irf = irf_picker(irf_full)
        # Timing and Normalization
        irf_sum = irf.sum()
        if not np.isfinite(irf_sum) or irf_sum <= 0:
            raise ValueError(f"Invalid IRF: sum={irf_sum}. IRF must be non-negative and non-zero.")
        self.irf = irf / irf_sum
        self.rng = np.random.default_rng(seed)
        self.laser_period = 1000 / laser_feq
        # N bins each of width dt covering [0, laser_period); endpoint-exclusive
        self.dt = self.laser_period / len(self.irf)
        self.t  = np.arange(len(self.irf)) * self.dt

        #  Parameters Storage
        self.params_cfg = {
            'tau2': tau2,
            'eff': efficiency,
            'A1': A1_fraction,
            'pc': photo_count,
            'mono': mono_fraction,
            'bit': bit,
            'cycles': n_cycles,
            'dcr': dcr,
            **kwargs
        }

    def sample_all_params(self):
        """Samples lifetime and fraction parameters for a single pixel."""
        t2 = ParameterSampler.truncated_normal(*self.params_cfg['tau2'])
        is_mono = self.rng.random() < self.params_cfg['mono']

        if is_mono:
            if self.rng.random() < 0.9:
                E, A1 = 0.0, self.rng.uniform(0.99, 0.9999)
            else:
                E, A1 = self.rng.uniform(0.99, 1.0), self.rng.uniform(0.0001, 0.01)
            t1 = t2 * (1 - E)
            # Steady-state correction for pulse repetition
            f = (A1 * (1-np.exp(-self.laser_period/t1))) / (A1*(1-np.exp(-self.laser_period/t1)) + (1-A1)*(1-np.exp(-self.laser_period/t2)))
            return {"mono": True, "E": E, "f": f, "tau1": t1, "tau2": t2, "A1": A1, "A2": 1.0 - A1}

        E  = ParameterSampler.sample_beta(*self.params_cfg['eff'], scale=0.9, offset=0.1, rng=self.rng)
        A1 = ParameterSampler.sample_beta(*self.params_cfg['A1'],  scale=0.9, offset=0.05, rng=self.rng)
        A2 = 1.0 - A1
        t1 = t2 * (1 - E)
        # Pulsed-repetition correction (same formula as mono mode)
        w1    = A1 * (1 - np.exp(-self.laser_period / t1))
        w2    = A2 * (1 - np.exp(-self.laser_period / t2))
        denom = w1 + w2
        f     = w1 / denom if denom > 0 else A1
        return {"mono": False, "E": E, "f": f, "tau1": t1, "tau2": t2, "A1": A1, "A2": A2}

    def get_analytical_decay(self, p):
        """Builds the clean (pre-IRF-convolution) bi-exponential decay curve.

        Sums the two lifetime components, each scaled by its amplitude
        fraction and a steady-state (pulse-repetition) correction factor
        ``1 / (1 - exp(-T / tau_i))``.

        Args:
            p: Parameter dict as returned by ``sample_all_params``, with
                keys ``A1``, ``A2``, ``tau1``, ``tau2``.

        Returns:
            numpy.ndarray: The clean decay curve sampled on ``self.t``.
        """
        T = self.laser_period
        # steady-state scaling factor per component
        scaling_factor1 = 1.0 / (1.0 - np.exp(-T / p["tau1"]))
        scaling_factor2 = 1.0 / (1.0 - np.exp(-T / p["tau2"]))
        return (p["A1"] * scaling_factor1 * np.exp(-self.t / p["tau1"]) +
                p["A2"] * scaling_factor2 * np.exp(-self.t / p["tau2"]))

    def simulate_tcspc(self, p, n_cycles, mu_per_cycle):
        """Photon-by-photon logic for TCSPC mode."""
        total_photons = self.rng.poisson(mu_per_cycle * n_cycles)
        if total_photons == 0: return np.zeros_like(self.t)

        # Emission (Inverse Transform Sampling)
        comp1 = self.rng.random(total_photons) < p["f"]
        times = np.empty(total_photons)
        times[comp1]  = self.rng.exponential(p["tau1"], size=comp1.sum())
        times[~comp1] = self.rng.exponential(p["tau2"], size=(~comp1).sum())

        # IRF Convolution (Sampling from the IRF distribution)
        irf_cdf = np.cumsum(self.irf)
        irf_cdf[-1] = 1.0  # pin to exactly 1.0 so searchsorted never returns len(irf)
        irf_shifts = np.searchsorted(irf_cdf, self.rng.random(total_photons)) * self.dt

        # Filter for pile-up and repetitive excitation window
        arrival_times = NoiseEngine.tcspc_pileup_filter(times + irf_shifts, self.laser_period)

        # Binning
        bins = (arrival_times / self.dt).astype(np.int32)
        hist = np.bincount(bins[bins < len(self.t)], minlength=len(self.t))

        return hist.astype(np.float64)
