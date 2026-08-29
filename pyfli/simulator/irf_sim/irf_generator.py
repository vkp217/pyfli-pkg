# simulator/irf_sim/irf_generator.py

"""
Generate synthetic instrument response function (IRF) traces for simulator and
alignment testing.

This module belongs to :mod:`pyfli.simulator.irf_sim` and is part of PyFLI synthetic
FLI/FLIM data generation, hardware noise modeling, calibration, and validation tools.
Public API includes classes :class:`IRFGenerator`.
"""

import numpy as np


class IRFGenerator:
    """
    Generate synthetic IRF traces for testing alignment, fitting, and simulation code
    without needing measured instrument data. Each method returns a 1D `(num_bins,)`
    trace, or a `(H, W, num_bins)` cube with the same trace broadcast to every pixel
    when `H` and `W` are both given.
    """

    @staticmethod
    def _broadcast(irf_1d: np.ndarray, H: int | None, W: int | None) -> np.ndarray:
        """
        Broadcasts a 1D IRF trace to a (H, W, num_bins) cube when H and W are both
        given, otherwise returns the 1D trace unchanged.
        """
        if H is None and W is None:
            return irf_1d
        if H is None or W is None:
            raise ValueError(
                "H and W must both be given for 3D output, or both omitted for 1D."
            )
        return np.broadcast_to(irf_1d, (H, W, len(irf_1d))).copy()

    @staticmethod
    def gaussianIRF(
        mu: float,
        T: float = 12.5,
        num_bins: int = 256,
        sigma: float = 0.1,
        H: int | None = None,
        W: int | None = None,
    ) -> np.ndarray:
        """
        Generates a Gaussian-shaped IRF: a narrow pulse centered at bin `mu`, peak
        normalized to 1.

        Parameters
        ----------
        mu : float
            Bin index of the pulse peak (fractional values are supported).
        T : float
            Laser period in nanoseconds.
        num_bins : int
            Number of time bins spanning one laser period.
        sigma : float
            Pulse spread, in nanoseconds. Converted to bins internally via
            `gate_delay = T / num_bins`.
        H, W : int | None
            If both given, the trace is broadcast to a (H, W, num_bins) cube.
            If both omitted, a 1D (num_bins,) trace is returned.
        """
        gate_delay = T / num_bins
        sigma_bins = sigma / gate_delay
        t = np.arange(num_bins)
        irf_1d = np.exp(-0.5 * ((t - mu) / sigma_bins) ** 2)
        return IRFGenerator._broadcast(irf_1d, H, W)

    @staticmethod
    def expdecay(
        mu: float,
        T: float = 12.5,
        num_bins: int = 256,
        tau: float = 0.1,
        H: int | None = None,
        W: int | None = None,
    ) -> np.ndarray:
        """
        Generates a single-exponential-decay IRF: a Dirac delta at bin `mu`
        convolved with a very fast causal exponential decay, peak normalized to 1.

        Parameters
        ----------
        mu : float
            Bin index of the delta impulse (rounded to the nearest bin).
        T : float
            Laser period in nanoseconds.
        num_bins : int
            Number of time bins spanning one laser period.
        tau : float
            Exponential decay time constant, in nanoseconds (0.1 ns or less for a
            "very fast" IRF-like decay).
        H, W : int | None
            If both given, the trace is broadcast to a (H, W, num_bins) cube.
            If both omitted, a 1D (num_bins,) trace is returned.
        """
        gate_delay = T / num_bins
        delta = np.zeros(num_bins)
        delta[round(mu)] = 1.0
        kernel = np.exp(-np.arange(num_bins) * gate_delay / tau)
        irf_1d = np.convolve(delta, kernel)[:num_bins]
        irf_1d = irf_1d / irf_1d.max()
        return IRFGenerator._broadcast(irf_1d, H, W)

    @staticmethod
    def gaussianExpIRF(
        mu: float,
        T: float = 12.5,
        num_bins: int = 256,
        sigma: float = 0.1,
        tau: float = 0.1,
        H: int | None = None,
        W: int | None = None,
    ) -> np.ndarray:
        """
        Generates an exponentially modified Gaussian (EMG) IRF: a Gaussian pulse
        (as in :meth:`gaussianIRF`) convolved with a fast causal exponential decay
        (as in :meth:`expdecay`), peak normalized to 1.

        This models a common real-detector IRF shape: a Gaussian core from optical
        and timing jitter, with an exponential tail from the detector's electronic
        response.

        Parameters
        ----------
        mu : float
            Bin index of the Gaussian core's peak.
        T : float
            Laser period in nanoseconds.
        num_bins : int
            Number of time bins spanning one laser period.
        sigma : float
            Gaussian core spread, in nanoseconds.
        tau : float
            Exponential tail time constant, in nanoseconds.
        H, W : int | None
            If both given, the trace is broadcast to a (H, W, num_bins) cube.
            If both omitted, a 1D (num_bins,) trace is returned.
        """
        gate_delay = T / num_bins
        sigma_bins = sigma / gate_delay
        t = np.arange(num_bins)
        gaussian = np.exp(-0.5 * ((t - mu) / sigma_bins) ** 2)
        kernel = np.exp(-np.arange(num_bins) * gate_delay / tau)
        irf_1d = np.convolve(gaussian, kernel)[:num_bins]
        irf_1d = irf_1d / irf_1d.max()
        return IRFGenerator._broadcast(irf_1d, H, W)
