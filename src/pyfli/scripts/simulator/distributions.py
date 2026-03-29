#  simulator/distributions.py

import numpy as np
from scipy.stats import truncnorm

class ParameterSampler:
    @staticmethod
    def sample_qe(sensor_type='ICCD'):
        """Samples QE based on sensor type."""
        if sensor_type.upper() == 'ICCD':
            return np.random.uniform(0.15, 0.35) # Typical ICCD QE
        return np.random.uniform(0.70, 0.90)     # Typical SPAD QE
    
    @staticmethod
    def sample_noise_params(bit_depth, sensor_type='ICCD'):
        """Centralized control for hardware noise levels."""
        if sensor_type.upper() == 'ICCD':
            # Linear scaling for read noise and DCR
            read_sigma = np.random.uniform(1.0, 3.0) * (bit_depth / 8.0)
            return {"read_sigma": read_sigma}
        return {"read_sigma": 0.0} # SPADs effectively have zero read noise
    
    @staticmethod
    def sample_beta(alpha, beta, scale=1.0, offset=0.0):
        """Standard beta sampling with scale and offset."""
        val = np.random.beta(alpha, beta)
        return (val * scale) + offset

    @staticmethod
    def beta_sample(alpha, beta, scale=1.0, clip_eps=1e-4):
        """Your specific photon-count beta sampling logic."""
        val = np.random.beta(alpha, beta)
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