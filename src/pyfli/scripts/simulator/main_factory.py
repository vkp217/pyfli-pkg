import numpy as np
from .simulator_engine import FLIEngine
from .noise_models import NoiseEngine
from .distributions import ParameterSampler

class Macro_sim:
    def __init__(self, irf_data, **cfg):
        # Extract noise flags before passing cfg to the engine
        self.use_jitter = cfg.get('jitter', True)
        self.use_dcr = cfg.get('dcr_on', True)
        self.use_poisson = cfg.get('poisson', True)
        
        self.engine = FLIEngine(irf_data, **cfg)

    def __call__(self):
        p = self.engine.sample_all_params()
        
        alpha_pc, beta_pc = self.engine.params_cfg['pc']
        A = ParameterSampler.beta_sample(alpha_pc, beta_pc, scale=(2**self.engine.params_cfg['bit'] - 1))
        
        clean = self.engine.get_analytical_decay(p)
        fit = np.convolve(clean, self.engine.irf, mode='full')[:len(clean)] * A
        
        # Sequential Noise Application
        obs = fit.copy()
        if self.use_jitter:
            obs = NoiseEngine.apply_jitter(obs)
        if self.use_dcr:
            obs = NoiseEngine.apply_dcr(obs, self.engine.params_cfg['dcr'])
        if self.use_poisson:
            obs = NoiseEngine.apply_poisson(obs)
            
        return {
            "raw_data": {"decay": obs, "irf": self.engine.irf},
            "results": {"maps": {**p, "tau_mean": p['tau1']*p['f'] + p['tau2']*(1-p['f'])}},
            "TR_maps": {"fit_map": fit, "residuals_map": obs - fit}
        }

class TCSPC_sim:
    def __init__(self, irf_data, **cfg):
        self.use_jitter = cfg.get('jitter', True)
        self.use_dcr = cfg.get('dcr_on', True)
        self.engine = FLIEngine(irf_data, **cfg)

    def __call__(self):
        p = self.engine.sample_all_params()
        n_cycles = np.random.randint(1, self.engine.params_cfg['cycles'] + 1)
        mu_per_cycle = 0.01 
        
        obs = self.engine.simulate_tcspc(p, n_cycles, mu_per_cycle)
        
        if self.use_jitter:
            obs = NoiseEngine.apply_jitter(obs)
        if self.use_dcr:
            obs = NoiseEngine.apply_dcr(obs, self.engine.params_cfg['dcr'])
        
        total_photons_expected = mu_per_cycle * n_cycles
        clean = self.engine.get_analytical_decay(p)
        fit = np.convolve(clean, self.engine.irf, mode='full')[:len(clean)] * total_photons_expected
        
        return {
            "raw_data": {"decay": obs, "irf": self.engine.irf},
            "results": {"maps": {**p, "tau_mean": p['tau1']*p['f'] + p['tau2']*(1-p['f'])}},
            "TR_maps": {"fit_map": fit, "residuals_map": obs - fit}
        }