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
                 photo_count = (1.2, 5), 
                 mono_fraction = 0.2, 
                 bit = 8, 
                 n_cycles = 800_000, 
                 dcr = 0.05,
                 laser_feq = 80, 
                 **kwargs
                 ):
        
        # 1. IRF Selection Logic (Robust for 3D stacks)
        if irf_full.ndim == 3:
            max_attempts = 1000
            attempts = 0
            while True:
                x = np.random.randint(irf_full.shape[0])
                y = np.random.randint(irf_full.shape[1])
                pixel_data = irf_full[x, y, :]
                
                max_val = np.max(pixel_data)
                min_val = np.min(pixel_data)
                ratio = max_val / min_val if min_val > 0 else 0
                
                if max_val> 800 and ratio > 20: # Ensure a clear peak
                    irf = pixel_data
                    break
                
                attempts += 1
                if attempts >= max_attempts:
                    raise RuntimeError(f"Could not find a valid pixel after {max_attempts} attempts.")
                    
        elif irf_full.ndim == 1:
            irf = irf_full
        else:
            raise ValueError(f'IRF must be 1-D or 3-D, got shape {irf_full.shape}')

        # 2. Timing and Normalization
        self.irf = np.nan_to_num(irf / irf.sum())
        self.laser_period = 1000/laser_feq
        self.t = np.linspace(0, self.laser_period, len(self.irf))
        
        # CRITICAL: Compatibility with TCSPC logic
        self.dt = self.t[1] - self.t[0]

        # 3. Parameters Storage
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
        
        E = ParameterSampler.sample_beta(*self.params_cfg['eff'], scale=0.9, offset=0.1)
        f = ParameterSampler.sample_beta(*self.params_cfg['f'], scale=0.9, offset=0.05)
        return {"mono": False, "E": E, "f": f, "tau1": t2*(1-E), "tau2": t2, "A1": f, "A2": 1.0 - f}

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