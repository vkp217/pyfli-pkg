import numpy as np
from scipy.signal import fftconvolve
from .distributions import ParameterSampler
from .noise_models import NoiseEngine

class FLIEngine:
    def __init__(self, irf, config):
        self.irf = irf / np.sum(irf)
        self.t = np.linspace(0, 12.5, len(irf))
        self.dt = self.t[1] - self.t[0]
        self.config = config

    def sample_params(self):
        """Generates bi-exponential parameters using the Sampler."""
        t2 = ParameterSampler.truncated_normal(self.config['tau2_mu'], self.config['tau2_sig'])
        e = ParameterSampler.beta_sample(*self.config['efficiency_beta'])
        f = ParameterSampler.beta_sample(*self.config['fraction_beta'])
        
        t1 = t2 * (1.0 - e)
        return {"tau1": t1, "tau2": t2, "f": f, "A1": f, "A2": 1.0 - f}

    def get_analytical_decay(self, p):
        return p["A1"] * np.exp(-self.t / p["tau1"]) + p["A2"] * np.exp(-self.t / p["tau2"])

    def simulate_tcspc(self, p, n_cycles, mu_per_cycle):
        """Photon-by-photon logic"""
        total_photons = np.random.poisson(mu_per_cycle * n_cycles)
        if total_photons == 0: return np.zeros_like(self.t)

        # Emission (Inverse Transform)
        comp1 = np.random.rand(total_photons) < p["f"]
        times = np.empty(total_photons)
        times[comp1] = np.random.exponential(p["tau1"], size=comp1.sum())
        times[~comp1] = np.random.exponential(p["tau2"], size=(~comp1).sum())

        # IRF Convolution (Random sampling from IRF shape)
        irf_cdf = np.cumsum(self.irf)
        irf_shifts = np.searchsorted(irf_cdf, np.random.rand(total_photons)) * self.dt

        arrival_times = NoiseEngine.tcspc_pileup_filter(times + irf_shifts, 12.5)
        
        # Binning
        bins = (arrival_times / self.dt).astype(np.int32)
        hist = np.bincount(bins[bins < len(self.t)], minlength=len(self.t))
        return NoiseEngine.apply_jitter(hist.astype(np.float64))