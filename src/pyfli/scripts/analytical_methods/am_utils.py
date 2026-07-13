"""Small shared helper utilities for the analytical FLI methods.

Currently provides frequency-related computations used when converting
laser/gate timing parameters into modulation frequencies for phasor and
gated-detection based analyses.
"""

import numpy as np

class AnalyticalHelpers:
    """Helper for deriving modulation frequencies from acquisition timing.

    Wraps the laser repetition period and, optionally, the gate delay and
    number of gates used in gated-detection acquisitions, so the
    corresponding excitation and effective frequencies can be computed.
    """

    def __init__(self, laser_period = 12.5, gate_delay=None, num_gate = None):
        """Store acquisition timing parameters.

        Args:
            laser_period: Laser repetition period in nanoseconds.
            gate_delay: Delay between successive gates, in nanoseconds.
                If ``None`` (default), the effective frequency falls back
                to the laser frequency.
            num_gate: Number of gates acquired. If ``None`` (default), the
                effective frequency falls back to the laser frequency.
        """
        self.laser_period = laser_period
        self.gate_delay = gate_delay
        self.num_gate = num_gate

    def freq_computation(self):
        """Compute the laser (excitation) and effective modulation frequencies.

        The laser frequency is derived from ``laser_period`` (assumed in
        nanoseconds), giving a frequency in Hz. When both ``gate_delay``
        and ``num_gate`` are set, the effective frequency is instead
        derived from the total gate span (``num_gate * gate_delay``);
        otherwise it equals the laser frequency.

        Returns:
            list: Two-element list ``[freq, effective_freq]``, both in Hz.
        """
        freq = 1000.0/self.laser_period # laser_period in ns; freq in Hz
        if  self.gate_delay is None or self.num_gate is None:
            effective_freq = freq
        else:
            effective_freq = 1000.0/(self.num_gate*self.gate_delay) # frequency is computed in the MHz if the gate delays are in ns
        return [freq, effective_freq]
