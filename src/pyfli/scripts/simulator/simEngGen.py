# simulator/simulator_engine

"""Generalized per-pixel FLI parameter/decay model supporting mono- and bi-exponential kinetics.

Provides ``FLIModelSim``, the core per-pixel model engine used by
``ContinousEqSim`` and ``PhotonCountSim``. Unlike ``FLIEngine`` (which
always samples bi-exponential parameters), ``FLIModelSim`` randomly emits
either a pure mono-exponential or a bi-exponential (FRET-like) pixel on
each call, and can generate the corresponding analytical decay or a
photon-by-photon TCSPC histogram.
"""

import numpy as np

from .distributions import ParameterSampler
from .noise_models import NoiseEngine
from .sim_helper import irf_picker

class FLIModelSim:
    """Samples per-pixel FLI parameters and generates decay curves/TCSPC histograms.

    Supports both a pure single-exponential ("mono") pixel model and a
    two-component (bi-exponential, FRET-like donor/acceptor) pixel model,
    chosen randomly per call according to ``mono_fraction``. Provides both
    an analytical (steady-state, pulse-repetition-corrected) decay model
    and a photon-by-photon TCSPC Monte Carlo simulator.
    """

    def __init__(self,
                 irf_full,
                 tau2=(1, 0.5),
                 tau2_dist = 'normal',   # 'normal' -> truncated_normal | 'beta' -> sample_beta
                 tau2_beta_range = (4.8, 0.2),  # (scale, offset), only used when tau2_dist='beta'
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
        """Initializes the model engine and precomputes the IRF/time axis.

        Args:
            irf_full: IRF data (1-D trace or 3-D IRF stack); resolved to a
                single 1-D trace via ``irf_picker`` and stored normalized
                (sum to 1) on ``self.irf``.
            tau2: ``(mu, sigma)`` for the truncated-normal tau2 prior (used
                when ``tau2_dist='normal'``), or ``(alpha, beta)`` for the
                beta prior (used when ``tau2_dist='beta'``).
            tau2_dist: Distribution used to sample tau2: ``'normal'`` uses
                ``ParameterSampler.truncated_normal``, ``'beta'`` uses
                ``ParameterSampler.sample_beta``.
            tau2_beta_range: ``(scale, offset)`` applied to the beta-sampled
                tau2 when ``tau2_dist='beta'``.
            efficiency: ``(alpha, beta)`` for the FRET efficiency ``E``
                beta prior (bi-exponential pixels).
            A1_fraction: ``(alpha, beta)`` for the donor amplitude fraction
                ``A1`` beta prior (bi-exponential pixels).
            photo_count: ``(alpha, beta)`` for the Beta-distributed peak
                photon count / intensity.
            mono_fraction: Probability that a given pixel is sampled as
                pure mono-exponential rather than bi-exponential.
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
            'tau2_dist': tau2_dist,
            'tau2_beta_range': tau2_beta_range,
            'eff': efficiency,
            'A1': A1_fraction,
            'pc': photo_count,
            'mono': mono_fraction,
            'bit': bit,
            'cycles': n_cycles,
            'dcr': dcr,
            **kwargs
        }

    def _sample_tau2(self):
        """Draws tau2 from either a truncated normal or a beta prior, per tau2_dist."""
        if self.params_cfg['tau2_dist'] == 'beta':
            scale, offset = self.params_cfg['tau2_beta_range']
            t2 = ParameterSampler.sample_beta(*self.params_cfg['tau2'],
                                               scale=scale,
                                               offset=offset,
                                               rng=self.rng)
        else:
            t2 = ParameterSampler.truncated_normal(*self.params_cfg['tau2'])
        # Guard against a zero (or negative) lifetime, which would blow up any 1/tau2 term downstream
        return max(t2, 1e-3)

    def sample_mono_params(self):
        """Samples the lifetime parameter for a single pixel (pure single-exponential)."""
        t2 = self._sample_tau2()
        return {"mono": True, "tau": t2}

    def sample_bi_params(self):
        """Samples lifetime and fraction parameters for a single pixel."""
        t2 = self._sample_tau2()

        E  = ParameterSampler.sample_beta(*self.params_cfg['eff'], scale=0.998, offset=0.001, rng=self.rng)
        A1 = ParameterSampler.sample_beta(*self.params_cfg['A1'],  scale=0.998, offset=0.001, rng=self.rng)
        A2 = 1.0 - A1
        t1 = t2 * (1 - E)
        # Pulsed-repetition correction (same formula as mono mode)
        w1    = A1 * (1 - np.exp(-self.laser_period / t1))
        w2    = A2 * (1 - np.exp(-self.laser_period / t2))
        denom = w1 + w2
        f     = w1 / denom if denom > 0 else A1
        return {"mono": False, "E": E, "f": f, "tau1": t1, "tau2": t2, "A1": A1, "A2": A2}

    def sample_params(self):
        """Samples parameters for one pixel, choosing mono or bi-exponential.

        With probability ``params_cfg['mono']`` returns mono-exponential
        parameters (``sample_mono_params``); otherwise returns
        bi-exponential parameters (``sample_bi_params``).

        Returns:
            dict: A mono-exponential params dict (``mono=True``, ``tau``)
            or a bi-exponential params dict (``mono=False``, ``E``, ``f``,
            ``tau1``, ``tau2``, ``A1``, ``A2``).
        """
        if self.rng.random() < self.params_cfg['mono']:
            return self.sample_mono_params()
        return self.sample_bi_params()

    def get_model_analytical_decay(self, p):
        """Builds the clean (pre-IRF-convolution) analytical decay curve.

        For a mono-exponential pixel, returns a single steady-state-corrected
        exponential; for a bi-exponential pixel, sums each ``tau{i}``/``A{i}``
        component (as many as are present in ``p``) with the same
        steady-state (pulse-repetition) scaling factor
        ``1 / (1 - exp(-T / tau_i))``.

        Args:
            p: Parameter dict as returned by ``sample_params`` (or
                ``sample_mono_params``/``sample_bi_params``).

        Returns:
            numpy.ndarray: The clean decay curve sampled on ``self.t``.
        """
        T = self.laser_period
        if p.get("mono", False):
            tau = p["tau"]
            scaling_factor = 1.0 / (1.0 - np.exp(-T / tau))
            return scaling_factor * np.exp(-self.t / tau)

        decay = np.zeros_like(self.t)
        i = 1
        while f"tau{i}" in p:
            tau_i = p[f"tau{i}"]
            A_i = p[f"A{i}"]
            scaling_factor_i = 1.0 / (1.0 - np.exp(-T / tau_i))
            decay = decay + A_i * scaling_factor_i * np.exp(-self.t / tau_i)
            i += 1
        return decay

    def simulate_model_tcspc(self, p, n_cycles, mu_per_cycle):
        """Runs a photon-by-photon TCSPC Monte Carlo simulation for one pixel.

        Draws a Poisson-distributed total photon count, samples per-photon
        emission times by inverse-transform sampling from the pixel's
        exponential (mono) or mixture (bi-exponential, weighted by ``f``)
        model, adds a per-photon timing offset sampled from the IRF's CDF,
        filters out photons arriving beyond one laser period
        (``NoiseEngine.tcspc_pileup_filter``), and bins the survivors into
        a photon-count histogram.

        Args:
            p: Parameter dict as returned by ``sample_params`` (mono uses
                ``p["tau"]``; bi-exponential uses ``p["f"]``, ``p["tau1"]``,
                ``p["tau2"]``).
            n_cycles: Number of laser excitation cycles simulated.
            mu_per_cycle: Expected photon detection probability per cycle,
                used as the Poisson rate ``mu_per_cycle * n_cycles`` for
                the total photon count.

        Returns:
            numpy.ndarray: Photon-count histogram (``float64``) over the
            same time bins as ``self.t``. All zeros if zero photons are
            drawn or none survive the pile-up window.
        """
        total_photons = self.rng.poisson(mu_per_cycle * n_cycles)
        if total_photons == 0: return np.zeros_like(self.t)

        # Emission (Inverse Transform Sampling)
        times = np.empty(total_photons)
        if p.get("mono", False):
            times[:] = self.rng.exponential(p["tau"], size=total_photons)
        else:
            comp1 = self.rng.random(total_photons) < p["f"]
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
