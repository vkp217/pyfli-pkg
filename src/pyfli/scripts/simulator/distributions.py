#  simulator/distributions.py

import numpy as np
from scipy.stats import truncnorm

class ParameterSampler:
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