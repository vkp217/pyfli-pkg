import numpy as np
from scipy.stats import truncnorm

class ParameterSampler:
    @staticmethod
    def beta_sample(alpha, beta, scale=1.0, clip_eps=1e-4):
        """Samples from a beta distribution and scales it."""
        val = np.random.beta(alpha, beta)
        return np.clip(val * scale, clip_eps, scale - clip_eps)

    @staticmethod
    def truncated_normal(mu, sigma, lower=0.01, upper=5.0):
        """Samples a lifetime value within physical bounds."""
        a, b = (lower - mu) / sigma, (upper - mu) / sigma
        return truncnorm.rvs(a, b, loc=mu, scale=sigma)

    @staticmethod
    def stretch_squeeze(sample, epsilon):
        """Maps [0,1] to [epsilon, 1-epsilon]."""
        return sample * (1.0 - 2.0 * epsilon) + epsilon