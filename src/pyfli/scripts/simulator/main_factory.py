# simulator/main_factory.py
import numpy as np
from .simulator_engine import FLIEngine
from .noise_models import NoiseEngine
from .distributions import ParameterSampler

class Macro_sim:
    def __init__(self, irf_data, **cfg):
        self.engine = FLIEngine(irf_data, **cfg)

    def __call__(self):
        # 1. Sample Params
        p = self.engine.sample_all_params()
        
        # 2. Sample Photon Count (Beta(5,5) by default from distributions)
        alpha_pc, beta_pc = self.engine.params_cfg['pc']
        A = ParameterSampler.beta_sample(alpha_pc, beta_pc, scale=(2**self.engine.params_cfg['bit'] - 1))
        
        # 3. Generate Signal
        clean = self.engine.get_analytical_decay(p)
        # Analytical convolution
        fit = np.convolve(clean, self.engine.irf, mode='full')[:len(clean)] * A
        
        # 4. Apply Noise
        jittered = NoiseEngine.apply_jitter(fit)
        with_dcr = NoiseEngine.apply_dcr(jittered, self.engine.params_cfg['dcr'])
        observed = NoiseEngine.apply_poisson(with_dcr)
        
        return {
            "raw_data": {
                "decay": observed, 
                "irf": self.engine.irf
            },
            "results": {
                "maps": {
                    **p, 
                    "tau_mean": p['tau1']*p['f'] + p['tau2']*(1-p['f'])
                }
            },
            "TR_maps": {
                "fit_map": fit, 
                "residuals_map": observed - fit
            }
        }

class TCSPC_sim:
    def __init__(self, irf_data, **cfg):
        self.engine = FLIEngine(irf_data, **cfg)

    def __call__(self):
        p = self.engine.sample_all_params()
        n_cycles = np.random.randint(1, self.engine.params_cfg['cycles'] + 1)
        mu_per_cycle = 0.01 # Standard mu for TCSPC
        
        # 1. Generate noisy histogram using your engine's logic
        observed = self.engine.simulate_tcspc(p, n_cycles, mu_per_cycle)
        
        # 2. Generate noise-free fit for residuals
        total_photons_expected = mu_per_cycle * n_cycles
        clean = self.engine.get_analytical_decay(p)
        fit = np.convolve(clean, self.engine.irf, mode='full')[:len(clean)] * total_photons_expected
        
        return {
            "raw_data": {
                "decay": observed, 
                "irf": self.engine.irf
            },
            "results": {
                "maps": {
                    **p, 
                    "tau_mean": p['tau1']*p['f'] + p['tau2']*(1-p['f'])
                }
            },
            "TR_maps": {
                "fit_map": fit, 
                "residuals_map": observed - fit
            }
        }