from .simulator_engine import FLIEngine
from .distributions import ParameterSampler
from .noise_models import NoiseEngine
import numpy as np

class Macro_sim:
    def __init__(self, irf_data, **cfg):
        self.engine = FLIEngine(irf_data, cfg)
        self.bit_depth = cfg.get('bit', 10)
        self.dcr = cfg.get('dcr', 0.1)

    def __call__(self):
        p = self.engine.sample_params()
        # Scale by bit depth (intensity)
        photon_count = ParameterSampler.beta_sample(5, 5) * (2**self.bit_depth - 1)
        
        decay = self.engine.get_analytical_decay(p)
        conv = np.convolve(decay, self.engine.irf, mode='same')
        
        # Physics: Convolution -> Intensity Scaling -> DCR -> Poisson
        scaled = conv * photon_count
        with_dcr = NoiseEngine.apply_dcr(scaled, self.dcr)
        observed = NoiseEngine.apply_poisson(with_dcr)
        
        return {"s_t": observed, "irf": self.engine.irf, **p}


class TCSPC_sim:
    def __init__(self, irf_data, **cfg):
        self.engine = FLIEngine(irf_data, cfg)
        self.n_cycles = cfg.get('n_cycles', 800_000)
        self.dcr = cfg.get('dcr', 0.05)

    def __call__(self):
        p = self.engine.sample_params()
        # Full Monte Carlo arrival simulation
        photon_hist = self.engine.simulate_tcspc(p, self.n_cycles, 0.01)
        
        # DCR is applied to the histogram bins after photon counting
        observed = NoiseEngine.apply_dcr(photon_hist, self.dcr)
        return {"s_t": observed, "irf": self.engine.irf, **p}