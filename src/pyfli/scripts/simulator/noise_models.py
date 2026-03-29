# simulator/noise_models.py

import numpy as np

class NoiseEngine:
    @staticmethod
    def apply_poisson(clean_signal):
        return np.random.poisson(np.clip(clean_signal, 0, None)).astype(np.float64)

    @staticmethod
    def apply_dcr(decay, dcr_level=0.5):
        """
        Simulates Dark Count Rate (thermal noise).
        dcr_level: average dark photons per bin per measurement.
        """
        dark_noise = np.random.poisson(dcr_level, size=decay.shape)
        return decay + dark_noise

    @staticmethod
    def apply_jitter(decay, max_shift=2):
        n = len(decay)
        shift = np.random.randint(-max_shift, max_shift + 1)
        if shift == 0: 
            return decay
        if shift > 0:
            return np.concatenate([np.zeros(shift), decay[:n - shift]])
        return np.concatenate([decay[-shift:], np.zeros(-shift)])

    @staticmethod
    def tcspc_pileup_filter(arrival_times, t_rep):
        # Simplistic pile-up: real systems might only take the first photon per cycle
        return arrival_times[arrival_times < t_rep]