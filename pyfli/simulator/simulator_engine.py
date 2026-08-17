# simulator/simulator_engine
"""
Drive the main FLI parameter sampler and TCSPC simulation engine.

This module belongs to :mod:`pyfli.simulator` and is part of PyFLI synthetic FLI/FLIM
data generation, hardware noise modeling, calibration, and validation tools. Public API
includes classes :class:`FLIEngine`.
"""

from typing import Any

import numpy as np

from .distributions import ParameterSampler
from .noise_models import NoiseEngine
from .sim_helper import irf_picker


class FLIEngine:
    """
    Run the fliengine routine.
    analytical decays, and simulates TCSPC counts from a shared IRF and configuration.

    Parameters
    ----------
    irf_full : np.ndarray
        Full instrument response function sampled over the decay window.
    tau2 : tuple[int, float]
        Long lifetime component.
    efficiency : tuple[int, ...]
        FRET transfer efficiency used to derive simulated lifetime components.
    A1_fraction : tuple[int, ...]
        Amplitude fraction assigned to the first exponential component.
    photo_count : tuple[float, int]
        Expected photon count used to scale the simulated decay.
    mono_fraction : float
        Fraction of pixels or events assigned to the mono-exponential component.
    bit : int
        Bit depth or quantization setting for simulated detector output.
    n_cycles : int
        Number of excitation cycles used when constructing the simulated decay.
    dcr : float
        Detector dark-count rate used by the noise model.
    laser_feq : int
        Laser repetition frequency used by the simulation.
    seed : int | None
        Seed for reproducible random sampling.
    **kwargs : Any
        Additional keyword arguments forwarded to the underlying implementation.
    """

    def __init__(
        self,
        irf_full: np.ndarray,
        tau2: tuple[int, float] = (1, 0.5),
        efficiency: tuple[int, ...] = (5, 5),
        A1_fraction: tuple[int, ...] = (5, 5),
        photo_count: tuple[float, int] = (1.2, 5),
        mono_fraction: float = 0.2,
        bit: int = 8,
        n_cycles: int = 800_000,
        dcr: float = 0.05,
        laser_feq: int = 80,
        seed: int | None = None,
        **kwargs: Any,
    ) -> None:

        irf = irf_picker(irf_full)
        # Timing and Normalization
        irf_sum = irf.sum()
        if not np.isfinite(irf_sum) or irf_sum <= 0:
            raise ValueError(
                f"Invalid IRF: sum={irf_sum}. IRF must be non-negative and non-zero."
            )
        self.irf = irf / irf_sum
        self.rng = np.random.default_rng(seed)
        self.laser_period = 1000 / laser_feq
        # N bins each of width dt covering [0, laser_period); endpoint-exclusive
        self.dt = self.laser_period / len(self.irf)
        self.t = np.arange(len(self.irf)) * self.dt

        #  Parameters Storage
        self.params_cfg = {
            "tau2": tau2,
            "eff": efficiency,
            "A1": A1_fraction,
            "pc": photo_count,
            "mono": mono_fraction,
            "bit": bit,
            "cycles": n_cycles,
            "dcr": dcr,
            **kwargs,
        }

    def sample_all_params(self) -> dict[Any, Any]:
        """Samples lifetime and fraction parameters for a single pixel."""
        t2 = ParameterSampler.truncated_normal(*self.params_cfg["tau2"])
        is_mono = self.rng.random() < self.params_cfg["mono"]

        if is_mono:
            if self.rng.random() < 0.9:
                E, A1 = 0.0, self.rng.uniform(0.99, 0.9999)
            else:
                E, A1 = self.rng.uniform(0.99, 1.0), self.rng.uniform(0.0001, 0.01)
            t1 = t2 * (1 - E)
            # Steady-state correction for pulse repetition
            f = (A1 * (1 - np.exp(-self.laser_period / t1))) / (
                A1 * (1 - np.exp(-self.laser_period / t1))
                + (1 - A1) * (1 - np.exp(-self.laser_period / t2))
            )
            return {
                "mono": True,
                "E": E,
                "f": f,
                "tau1": t1,
                "tau2": t2,
                "A1": A1,
                "A2": 1.0 - A1,
            }

        E = ParameterSampler.sample_beta(
            *self.params_cfg["eff"], scale=0.9, offset=0.1, rng=self.rng
        )
        A1 = ParameterSampler.sample_beta(
            *self.params_cfg["A1"], scale=0.9, offset=0.05, rng=self.rng
        )
        A2 = 1.0 - A1
        t1 = t2 * (1 - E)
        # Pulsed-repetition correction (same formula as mono mode)
        w1 = A1 * (1 - np.exp(-self.laser_period / t1))
        w2 = A2 * (1 - np.exp(-self.laser_period / t2))
        denom = w1 + w2
        f = w1 / denom if denom > 0 else A1
        return {
            "mono": False,
            "E": E,
            "f": f,
            "tau1": t1,
            "tau2": t2,
            "A1": A1,
            "A2": A2,
        }

    def get_analytical_decay(self, p: Any) -> Any:
        """
        Return analytical decay.

        Parameters
        ----------
        p : Any
            Detector parameter object or fitted parameter vector.

        Returns
        -------
        Any
            Object produced by get analytical decay.
        """
        T = self.laser_period
        # steady-state scaling factor per component
        scaling_factor1 = 1.0 / (1.0 - np.exp(-T / p["tau1"]))
        scaling_factor2 = 1.0 / (1.0 - np.exp(-T / p["tau2"]))
        return p["A1"] * scaling_factor1 * np.exp(-self.t / p["tau1"]) + p[
            "A2"
        ] * scaling_factor2 * np.exp(-self.t / p["tau2"])

    def simulate_tcspc(self, p: Any, n_cycles: int, mu_per_cycle: np.ndarray) -> Any:
        """Photon-by-photon logic for TCSPC mode."""
        total_photons = self.rng.poisson(mu_per_cycle * n_cycles)
        if total_photons == 0:
            return np.zeros_like(self.t)

        # Emission (Inverse Transform Sampling)
        comp1 = self.rng.random(total_photons) < p["f"]
        times = np.empty(total_photons)
        times[comp1] = self.rng.exponential(p["tau1"], size=comp1.sum())
        times[~comp1] = self.rng.exponential(p["tau2"], size=(~comp1).sum())

        # IRF Convolution (Sampling from the IRF distribution)
        irf_cdf = np.cumsum(self.irf)
        irf_cdf[-1] = 1.0  # pin to exactly 1.0 so searchsorted never returns len(irf)
        irf_shifts = np.searchsorted(irf_cdf, self.rng.random(total_photons)) * self.dt

        # Filter for pile-up and repetitive excitation window
        arrival_times = NoiseEngine.tcspc_pileup_filter(
            times + irf_shifts, self.laser_period
        )

        # Binning
        bins = (arrival_times / self.dt).astype(np.int32)
        hist = np.bincount(bins[bins < len(self.t)], minlength=len(self.t))

        return hist.astype(np.float64)
