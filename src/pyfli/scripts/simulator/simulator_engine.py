# simulator/simulator_engine
import numpy as np
from scipy.signal import fftconvolve
from .distributions import ParameterSampler
from .noise_models import NoiseEngine


class FLIEngine:
    def __init__(self, 
                 irf_full, 
                 tau2=(1, 0.5), 
                 efficiency = (5, 5), 
                 f_fraction = (5, 5), 
                 photo_count = (5, 5), 
                 mono_fraction = 0.2, 
                 bit = 10, 
                 n_cycles = 800_000, 
                 dcr = 0.05,
                 laser_feq = 80 # laser frequency in MHz
                 ):
        
        if irf_full.ndim == 3:
            x = np.random.randint(irf_full.shape[0])
            y = np.random.randint(irf_full.shape[1])
            if np.sum(irf_full[x, y, :]) < 5000: # this condition is for ICCD
                irf = irf_full[irf_full.shape[0] // 2, irf_full.shape[1] // 2, :]
            else:
                irf = irf_full[x, y, :]
        elif irf_full.ndim == 1:
            irf = irf_full
        else:
            raise ValueError(f'IRF must be 1-D or 3-D, got shape {irf_full.shape}')
        self.irf = np.nan_to_num(irf / irf.sum())
        self.laser_period = 1000/laser_feq
        self.t = np.linspace(0, self.laser_period, len(self.irf))
        self.params_cfg = {
            'tau2': tau2,           #tuple (mu, sigma)
            'eff': efficiency, 
            'f': f_fraction, 
            'pc': photo_count, 
            'mono': mono_fraction, 
            'bit': bit, 
            'cycles': n_cycles, 
            'dcr': dcr
        }

    def sample_all_params(self):
        t2 = ParameterSampler.truncated_normal(*self.params_cfg['tau2'])
        is_mono = np.random.rand() < self.params_cfg['mono']
        
        if is_mono:
            rng = np.random.default_rng()
            if rng.random() < 0.9:
                E, A1 = 0.0, rng.uniform(0.99, 0.9999)
            else:
                E, A1 = rng.uniform(0.99, 1.0), rng.uniform(0.0001, 0.01)
            t1 = t2 * (1 - E)
            f = (A1 * (1-np.exp(-self.laser_period/t1))) / (A1*(1-np.exp(-self.laser_period/t1)) + (1-A1)*(1-np.exp(-self.laser_period/t2)))
            return {"mono": True, 
                    "E": E, 
                    "f": f, 
                    "tau1": t1, 
                    "tau2": t2, 
                    "A1": A1, 
                    "A2": 1.0 - A1
                    }
        
        E = ParameterSampler.sample_beta(*self.params_cfg['eff'], scale=0.9, offset=0.1)
        f = ParameterSampler.sample_beta(*self.params_cfg['f'], scale=0.9, offset=0.05)
        return {"mono": False,
                 "E": E,
                 "f": f, 
                 "tau1": t2*(1-E), 
                 "tau2": t2, 
                 "A1": f, 
                 "A2": 1.0 - f
                 }

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