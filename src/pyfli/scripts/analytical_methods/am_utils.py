import numpy as np

class AnalyticalHelpers:
    def __init__(self, laser_period = 12.5, gate_delay=None, num_gate = None):
        self.laser_period = laser_period
        self.gate_delay = gate_delay
        self.num_gate = num_gate

    def freq_computation(self):
        freq = 1000.0/self.laser_period # laser_period in ns; freq in Hz
        if  self.gate_delay is None or self.num_gate is None:
            effective_freq = freq
        else:
            effective_freq = 1000.0/(self.num_gate*self.gate_delay) # frequency is computed in the MHz if the gate delays are in ns
        return [freq, effective_freq]
