# simulator/simulator_engine
import numpy as np
from scipy.signal import fftconvolve
from .distributions import ParameterSampler
from .noise_models import NoiseEngine
from .sim_helper import irf_picker

class FLIEngine:
    def __init__(self, 
                 irf_full, 
                 tau2=(1, 0.5), 
                 efficiency = (5, 5), 
                 f_fraction = (5, 5), 
                 photo_count = (1.2, 5), 
                 mono_fraction = 0.2, 
                 bit = 8, 
                 n_cycles = 800_000, 
                 dcr = 0.05,
                 laser_feq = 80, 
                 **kwargs
                 ):
        
        irf = irf_picker(irf_full)
        # Timing and Normalization
        irf_sum = irf.sum()
        if not np.isfinite(irf_sum) or irf_sum <= 0:
            raise ValueError(f"Invalid IRF: sum={irf_sum}. IRF must be non-negative and non-zero.")
        self.irf = irf / irf_sum
        self.laser_period = 1000 / laser_feq
        # N bins each of width dt covering [0, laser_period); endpoint-exclusive
        self.dt = self.laser_period / len(self.irf)
        self.t  = np.arange(len(self.irf)) * self.dt

        #  Parameters Storage
        self.params_cfg = {
            'tau2': tau2, 
            'eff': efficiency, 
            'f': f_fraction, 
            'pc': photo_count, 
            'mono': mono_fraction, 
            'bit': bit, 
            'cycles': n_cycles, 
            'dcr': dcr,
            **kwargs 
        }

    def sample_all_params(self):
        """Samples lifetime and fraction parameters for a single pixel."""
        t2 = ParameterSampler.truncated_normal(*self.params_cfg['tau2'])
        is_mono = np.random.rand() < self.params_cfg['mono']
        
        if is_mono:
            rng = np.random.default_rng()
            if rng.random() < 0.9:
                E, A1 = 0.0, rng.uniform(0.99, 0.9999)
            else:
                E, A1 = rng.uniform(0.99, 1.0), rng.uniform(0.0001, 0.01)
            t1 = t2 * (1 - E)
            # Steady-state correction for pulse repetition
            f = (A1 * (1-np.exp(-self.laser_period/t1))) / (A1*(1-np.exp(-self.laser_period/t1)) + (1-A1)*(1-np.exp(-self.laser_period/t2)))
            return {"mono": True, "E": E, "f": f, "tau1": t1, "tau2": t2, "A1": A1, "A2": 1.0 - A1}
        
        E  = ParameterSampler.sample_beta(*self.params_cfg['eff'], scale=0.9, offset=0.1)
        A1 = ParameterSampler.sample_beta(*self.params_cfg['f'],   scale=0.9, offset=0.05)
        A2 = 1.0 - A1
        t1 = t2 * (1 - E)
        # Pulsed-repetition correction (same formula as mono mode)
        w1    = A1 * (1 - np.exp(-self.laser_period / t1))
        w2    = A2 * (1 - np.exp(-self.laser_period / t2))
        denom = w1 + w2
        f     = w1 / denom if denom > 0 else A1
        return {"mono": False, "E": E, "f": f, "tau1": t1, "tau2": t2, "A1": A1, "A2": A2}

    def get_analytical_decay(self, p):
        """Returns the clean multiexponential decay curve."""
        return p["A1"] * np.exp(-self.t / p["tau1"]) + p["A2"] * np.exp(-self.t / p["tau2"])

    def simulate_tcspc(self, p, n_cycles, mu_per_cycle):
        """Photon-by-photon logic for TCSPC mode."""
        total_photons = np.random.poisson(mu_per_cycle * n_cycles)
        if total_photons == 0: return np.zeros_like(self.t)

        # Emission (Inverse Transform Sampling)
        comp1 = np.random.rand(total_photons) < p["f"]
        times = np.empty(total_photons)
        times[comp1] = np.random.exponential(p["tau1"], size=comp1.sum())
        times[~comp1] = np.random.exponential(p["tau2"], size=(~comp1).sum())

        # IRF Convolution (Sampling from the IRF distribution)
        irf_cdf = np.cumsum(self.irf)
        irf_shifts = np.searchsorted(irf_cdf, np.random.rand(total_photons)) * self.dt

        # Filter for pile-up and repetitive excitation window
        arrival_times = NoiseEngine.tcspc_pileup_filter(times + irf_shifts, self.laser_period)
        
        # Binning
        bins = (arrival_times / self.dt).astype(np.int32)
        hist = np.bincount(bins[bins < len(self.t)], minlength=len(self.t))
        
        return hist.astype(np.float64)