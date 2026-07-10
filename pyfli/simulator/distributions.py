#  simulator/distributions.py

import numpy as np
from scipy.stats import truncnorm

class ParameterSampler:
    @staticmethod
    def sample_qe(sensor_type='ICCD', rng=None):
        """Samples QE based on sensor type."""
        _r = rng or np.random
        if sensor_type.upper() == 'ICCD':
            return _r.uniform(0.15, 0.35) # Typical ICCD QE
        return _r.uniform(0.70, 0.90)     # Typical PHOTON_COUNTER QE

    @staticmethod
    def sample_noise_params(bit_depth, sensor_type='ICCD', rng=None):
        """Centralized control for hardware noise levels."""
        _r = rng or np.random
        if sensor_type.upper() == 'ICCD':
            # Read noise is a fixed electronic property (electrons RMS), independent of bit depth
            read_sigma = _r.uniform(1.0, 3.0)
            return {"read_sigma": read_sigma}
        return {"read_sigma": 0.0} # PHOTON_COUNTER sensors effectively have zero read noise

    @staticmethod
    def sample_beta(alpha, beta, scale=1.0, offset=0.0, rng=None):
        """Standard beta sampling with scale and offset."""
        _r = rng or np.random
        val = _r.beta(alpha, beta)
        return (val * scale) + offset

    @staticmethod
    def beta_sample(alpha, beta, scale=1.0, clip_eps=1e-4, rng=None):
        """Your specific photon-count beta sampling logic."""
        _r = rng or np.random
        val = _r.beta(alpha, beta)
        return np.clip(val * scale, clip_eps, scale - clip_eps)

    @staticmethod
    def truncated_normal(mu, sigma, lower=0.01, upper=5.0):
        """Fixed: Now takes mu and sigma as separate arguments."""
        a, b = (lower - mu) / sigma, (upper - mu) / sigma
        return truncnorm.rvs(a, b, loc=mu, scale=sigma)

    @staticmethod
    def stretch_squeeze(sample, epsilon):
        """Maps [0,1] to [epsilon, 1-epsilon]."""
        return sample * (1.0 - 2.0 * epsilon) + epsilon
