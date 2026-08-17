"""
Provide small analytical helper routines for acquisition frequency calculations.

This module belongs to :mod:`pyfli.analyticalWorkflow` and is part of PyFLI analytical
FLI reconstruction helpers. Public API includes classes :class:`AnalyticalHelpers`.
"""

from typing import Any


class AnalyticalHelpers:
    """
    Store acquisition timing parameters and derive the base and effective excitation
    frequencies used by analytical workflows.

    Parameters
    ----------
    laser_period : float
        Laser repetition period in nanoseconds.
    gate_delay : Any | None
        Delay between gated acquisition windows in nanoseconds.
    num_gate : int | None
        Number of acquisition gates used to compute the effective frequency.
    """

    def __init__(
        self,
        laser_period: float = 12.5,
        gate_delay: Any | None = None,
        num_gate: int | None = None,
    ) -> None:
        self.laser_period = laser_period
        self.gate_delay = gate_delay
        self.num_gate = num_gate

    def freq_computation(self) -> list[Any]:
        """
        Compute the base and effective acquisition frequencies.

        Returns
        -------
        list[Any]
            List containing the base excitation frequency and the effective gated frequency.
        """
        freq = 1000.0 / self.laser_period  # laser_period in ns; freq in Hz
        if self.gate_delay is None or self.num_gate is None:
            effective_freq = freq
        else:
            effective_freq = 1000.0 / (
                self.num_gate * self.gate_delay
            )  # frequency is computed in the MHz if the gate delays are in ns
        return [freq, effective_freq]
