# pyfli/simulator/irf_sim/tof_sim.py

"""
Convert a sample height/topology offset into an equivalent time-of-flight
shift, in units of IRF time bins, for the simulator workflow.

This module belongs to :mod:`pyfli.simulator.irf_sim` and is part of PyFLI synthetic
FLI/FLIM data generation, hardware noise modeling, calibration, and validation tools.
Public API includes classes :class:`ToFSim`.
"""

import numpy as np

SPEED_OF_LIGHT_MM_PER_NS = 299.792458  # c, in vacuum


class ToFSim:
    """
    Converts a physical height offset (mm) into the equivalent time-of-flight
    delay (ns) via the vacuum speed of light, then into an integer number of
    IRF time bins to ``np.roll`` an IRF trace by.

    A positive height is treated as being closer to the detector — a shorter
    one-way path length, so photons arrive faster (negative delay, negative
    roll); a negative height is farther away (positive delay, positive roll).

    Parameters
    ----------
    T : float
        Laser period, in nanoseconds (matches
        :class:`~pyfli.simulator.irf_sim.irf_generator.IRFGenerator`).
    num_bins : int
        Number of time bins spanning one laser period.
    """

    def __init__(self, T: float = 12.5, num_bins: int = 256):
        self.T = T
        self.num_bins = num_bins
        self.gate_delay = T / num_bins  # ns per bin, e.g. 12.5/256 =~ 48.8 ps

    def height_to_time(self, height_mm: float) -> float:
        """
        Converts a height offset (mm) to a time-of-flight delay (ns). Positive
        height (closer to the detector) yields a negative delay.
        """
        return -height_mm / SPEED_OF_LIGHT_MM_PER_NS

    def sample_shift(
        self, height_range_mm: tuple[float, float] = (-10.0, 10.0)
    ) -> tuple[int, float, float]:
        """
        Draws a height offset from ``height_range_mm``, converts it to a
        time-of-flight delay via :meth:`height_to_time`, and converts that
        delay to an integer number of ``gate_delay``-sized bins.

        Returns
        -------
        tuple[int, float, float]
            ``(roll_bins, height_mm, time_diff_ns)`` — the integer
            ``np.roll`` amount (positive or negative), the sampled height,
            and the time-of-flight delay it corresponds to.
        """
        height_mm = np.random.uniform(*height_range_mm)
        time_diff_ns = self.height_to_time(height_mm)
        roll_bins = int(round(time_diff_ns / self.gate_delay))
        return roll_bins, height_mm, time_diff_ns
