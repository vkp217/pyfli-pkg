# pyfli/simulator/sim_engine_common.py

"""
Shared IRF/timing setup, cycles-range validation, and TCSPC photon-binning logic
for the sampling engines wrapped by the "combined" and "separate" simulator pairs.

This module belongs to :mod:`pyfli.simulator` and is part of PyFLI synthetic FLI/FLIM
data generation, hardware noise modeling, calibration, and validation tools.

:class:`~pyfli.simulator.combined.simulator_engine.FLIEngine` and
:class:`~pyfli.simulator.separate.model_simulator.FLIModelSimulator` both wrap a
single IRF into per-pixel decay/parameter sampling and TCSPC photon-histogram
simulation; they differ in how lifetime parameters are sampled and how many
exponential components the analytical decay supports. :class:`BaseFLIEngine`
owns the parts that are identical between them — IRF validation, timing setup,
``n_cycles`` range normalization, the steady-state pulse-repetition mixing
fraction, and the IRF-convolution/pile-up/binning tail of the TCSPC simulation —
so that logic is defined exactly once.
"""

from typing import Any

import numpy as np

from .noise_models import NoiseEngine
from .sim_helper import irf_picker


class BaseFLIEngine:
    """
    Shared setup and helpers for FLI sampling engines.

    Parameters
    ----------
    irf_full : np.ndarray
        Full instrument response function sampled over the decay window.
    laser_feq : int
        Laser repetition frequency used by the simulation.
    seed : int | None
        Seed for reproducible random sampling.
    """

    def __init__(
        self,
        irf_full: np.ndarray,
        laser_feq: int = 80,
        seed: int | None = None,
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

    @staticmethod
    def _normalize_cycles_range(n_cycles: int | tuple[int, int]) -> tuple[int, int]:
        """
        Normalizes ``n_cycles`` into a validated ``(low, high)`` range.

        A bare int is the upper bound, with the lower bound fixed at 1000.
        A ``(low, high)`` tuple sets both bounds explicitly. Both bounds must
        be >= 1000, and ``high`` must be >= ``low``.
        """
        if isinstance(n_cycles, (int, np.integer)):
            cycles_range = (1000, int(n_cycles))
        else:
            low_cycles, high_cycles = n_cycles
            cycles_range = (int(low_cycles), int(high_cycles))
        if cycles_range[0] < 1000 or cycles_range[1] < 1000:
            raise ValueError(f"n_cycles bounds {cycles_range} must both be >= 1000.")
        if cycles_range[1] < cycles_range[0]:
            raise ValueError(
                f"n_cycles upper bound ({cycles_range[1]}) must be >= "
                f"lower bound ({cycles_range[0]})."
            )
        return cycles_range

    @staticmethod
    def _steady_state_mix(
        A1: float, A2: float, tau1: float, tau2: float, laser_period: float
    ) -> float:
        """
        Steady-state pulse-repetition mixing fraction for a two-component decay:
        the amplitude-weighted, per-pulse-buildup share of component 1 among
        both components' emitted photons.
        """
        w1 = A1 * (1 - np.exp(-laser_period / tau1))
        w2 = A2 * (1 - np.exp(-laser_period / tau2))
        denom = w1 + w2
        return w1 / denom if denom > 0 else A1

    def _bin_tcspc_photons(self, times: np.ndarray) -> Any:
        """
        Convolves photon emission times with the IRF (via inverse-transform
        sampling of the IRF's CDF), applies the pile-up/repetitive-excitation
        window filter, and bins the result into a TCSPC histogram.
        """
        total_photons = len(times)
        irf_cdf = np.cumsum(self.irf)
        irf_cdf[-1] = 1.0  # pin to exactly 1.0 so searchsorted never returns len(irf)
        irf_shifts = np.searchsorted(irf_cdf, self.rng.random(total_photons)) * self.dt

        arrival_times = NoiseEngine.tcspc_pileup_filter(
            times + irf_shifts, self.laser_period
        )

        bins = (arrival_times / self.dt).astype(np.int32)
        hist = np.bincount(bins[bins < len(self.t)], minlength=len(self.t))

        return hist.astype(np.float64)
