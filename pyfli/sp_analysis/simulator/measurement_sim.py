# sp_analysis/simulator/measurement_sim.py

"""
Simulate patterned measurements with optional Gaussian and shot noise.

This module belongs to :mod:`pyfli.sp_analysis.simulator` and is part of PyFLI single-
pixel camera basis generation, acquisition simulation, and reconstruction solvers.
Public API includes classes :class:`MeasurementSimulator`.
"""

from typing import Any

import numpy as np


class MeasurementSimulator:
    """
    Simulate patterned single-pixel measurements from a scene and sensing basis. It can
    add Gaussian noise, shot noise, differential measurements, and simple SNR estimates.

    Parameters
    ----------
    noise_level : float
        Additive noise level used by the measurement simulator.
    shot_noise : bool
        Whether to add Poisson shot noise to simulated measurements.
    """

    def __init__(self, noise_level: float = 0.0, shot_noise: bool = False) -> None:
        self.noise_level = noise_level
        self.shot_noise = shot_noise

    def capture(self, scene: np.ndarray, patterns: np.ndarray) -> np.ndarray:
        """
        Simulates the physical projection: y = A * x
        """
        x = scene.flatten(order="C").astype(float)

        # Linear light integration
        measurements = np.dot(patterns, x)

        # Add Shot Noise (Poisson-like simulation)
        if self.shot_noise:
            # Scaled to keep it realistic; noise increases with intensity
            noise_scale = np.sqrt(np.abs(measurements)) * self.noise_level
            measurements += np.random.normal(0, 1, measurements.shape) * noise_scale

        # Add Electronic/Thermal Noise (Gaussian)
        elif self.noise_level > 0:
            noise = np.random.normal(0, self.noise_level, measurements.shape)
            measurements += noise

        return measurements

    def process_differential(self, measurements: np.ndarray) -> Any:
        """
        Implements y_diff = y_pos - y_neg.
        Matches the stacked output of BasisPatterns.generate_hadamard(differential=True).
        """
        # Ensure we have an even number of measurements for pairing
        n_pairs = len(measurements) // 2
        y_pos = measurements[:n_pairs]
        y_neg = measurements[n_pairs:]

        return y_pos - y_neg

    def simulate_fourier_acquisition(
        self, scene: np.ndarray, fourier_patterns: np.ndarray
    ) -> Any:
        """
        For grayscale fringes, we simulate the 'DC-centered' signal.
        In the lab, you often measure the average brightness of the room
        first and subtract it.
        """
        measurements = self.capture(scene, fourier_patterns)

        # Subtracting mean removes the global offset caused by the [0, 1]
        # normalization in pattern_gen.
        return measurements - np.mean(measurements)

    @staticmethod
    def get_snr(clean_signal: np.ndarray, noisy_signal: np.ndarray) -> Any:
        """
        Calculates SNR in decibels.
        Higher is better.
        """
        # Use clean signal for power calculation to avoid bias
        signal_power = np.mean(clean_signal**2)
        noise_power = np.mean((clean_signal - noisy_signal) ** 2)

        if noise_power == 0:
            return float("inf")

        return 10 * np.log10(signal_power / noise_power)
