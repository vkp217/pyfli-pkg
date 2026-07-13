# simulator/noise_models.py

"""Noise-model building blocks for simulated fluorescence-lifetime decays.

Contains a collection of static methods that inject detector- and
photon-statistics-related noise (Poisson shot noise, dark count rate,
electronic read noise, timing jitter, and TCSPC pile-up filtering) into a
clean simulated decay signal.
"""

import numpy as np

class NoiseEngine:
    """Static collection of noise-injection operators for decay signals.

    Each method takes a clean (or partially-noised) decay array and returns
    a new array with an additional noise source applied. Methods are
    stateless and can be composed in any order to build a noise pipeline.
    """

    @staticmethod
    def apply_poisson(clean_signal):
        """Applies Poisson shot noise to a clean signal.

        Args:
            clean_signal: Array-like clean (noise-free) signal, in photon-
                count-like units. Values are clipped to be non-negative
                before sampling since the Poisson distribution is undefined
                for negative rates.

        Returns:
            numpy.ndarray: Poisson-sampled signal with the same shape as
            ``clean_signal``, as ``float64``.
        """
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
    def apply_read_noise(decay, sigma_read=1.5):
        """
        Simulates electronic read noise (Gaussian).
        Common in ICCD sensors during CCD readout.
        """
        read_noise = np.random.normal(0, sigma_read, size=decay.shape)
        return decay + read_noise

    @staticmethod
    def apply_jitter(decay, max_shift=2):
        """Applies a random integer-bin timing shift to a decay array.

        Simulates trigger/timing jitter by shifting the whole decay array
        left or right by a random number of bins in ``[-max_shift, max_shift]``
        and zero-padding the vacated end.

        Args:
            decay: 1-D array-like decay signal to shift.
            max_shift: Maximum absolute number of bins to shift by (inclusive).

        Returns:
            numpy.ndarray: The shifted decay array, same length as ``decay``.
            If the randomly drawn shift is 0, the original array is returned
            unchanged.
        """
        n = len(decay)
        shift = np.random.randint(-max_shift, max_shift + 1)
        if shift == 0: 
            return decay
        if shift > 0:
            return np.concatenate([np.zeros(shift), decay[:n - shift]])
        return np.concatenate([decay[-shift:], np.zeros(-shift)])

    @staticmethod
    def tcspc_pileup_filter(arrival_times, t_rep):
        """Filters photon arrival times to simulate TCSPC pile-up rejection.

        Simplistic pile-up model: keeps only arrival times that fall within
        one laser repetition period. Real TCSPC hardware typically only
        registers the first photon per excitation cycle; this filter mimics
        that constraint by discarding times beyond the repetition window.

        Args:
            arrival_times: Array-like of absolute photon arrival times.
            t_rep: Laser repetition period (same time units as
                ``arrival_times``); arrival times greater than or equal to
                this are discarded.

        Returns:
            numpy.ndarray: Subset of ``arrival_times`` with values strictly
            less than ``t_rep``.
        """
        # Simplistic pile-up: real systems might only take the first photon per cycle
        return arrival_times[arrival_times < t_rep]