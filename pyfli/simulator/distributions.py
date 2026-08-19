#  simulator/distributions.py

"""
Sample detector, noise, beta, and truncated-normal parameters for simulations.

This module belongs to :mod:`pyfli.simulator` and is part of PyFLI synthetic FLI/FLIM
data generation, hardware noise modeling, calibration, and validation tools. Public API
includes classes :class:`ParameterSampler`.
"""

from typing import Any

import numpy as np
from scipy.stats import truncnorm


class ParameterSampler:
    """
    Sample physically plausible detector and lifetime parameters for simulation
    workflows. Static methods cover quantum efficiency, noise parameters, beta draws,
    truncated normals, and interval stretching.
    """

    @staticmethod
    def sample_qe(sensor_type: str = "continuous", rng: Any | None = None) -> Any:
        """Samples QE based on sensor type."""
        _r = rng or np.random
        if sensor_type.upper() == "CONTINUOUS":
            return _r.uniform(0.15, 0.35)  # Typical continuous (ICCD) QE
        return _r.uniform(0.70, 0.90)  # Typical discrete (photon-counter) QE

    @staticmethod
    def sample_noise_params(
        bit_depth: int, sensor_type: str = "continuous", rng: Any | None = None
    ) -> dict[Any, Any]:
        """Centralized control for hardware noise levels."""
        _r = rng or np.random
        if sensor_type.upper() == "CONTINUOUS":
            # Read noise is a fixed electronic property (electrons RMS), independent of bit depth
            read_sigma = _r.uniform(1.0, 3.0)
            return {"read_sigma": read_sigma}
        return {
            "read_sigma": 0.0
        }  # discrete (photon-counter) sensors effectively have zero read noise

    @staticmethod
    def sample_beta(
        alpha: float,
        beta: float,
        scale: float = 1.0,
        offset: float = 0.0,
        rng: Any | None = None,
    ) -> Any:
        """Standard beta sampling with scale and offset."""
        _r = rng or np.random
        val = _r.beta(alpha, beta)
        return (val * scale) + offset

    @staticmethod
    def beta_sample(
        alpha: float,
        beta: float,
        scale: float = 1.0,
        clip_eps: float = 1e-4,
        rng: Any | None = None,
    ) -> np.ndarray:
        """Your specific photon-count beta sampling logic."""
        _r = rng or np.random
        val = _r.beta(alpha, beta)
        return np.clip(val * scale, clip_eps, scale - clip_eps)

    @staticmethod
    def truncated_normal(
        mu: float, sigma: float, lower: float = 0.01, upper: float = 5.0
    ) -> Any:
        """Fixed: Now takes mu and sigma as separate arguments."""
        a, b = (lower - mu) / sigma, (upper - mu) / sigma
        return truncnorm.rvs(a, b, loc=mu, scale=sigma)

    @staticmethod
    def stretch_squeeze(sample: Any, epsilon: Any) -> Any:
        """Maps [0,1] to [epsilon, 1-epsilon]."""
        return sample * (1.0 - 2.0 * epsilon) + epsilon
