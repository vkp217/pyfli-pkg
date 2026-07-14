import numpy as np
from scipy.signal import fftconvolve
from .simulator_engine import FLIEngine
from .noise_models import NoiseEngine
from .distributions import ParameterSampler

class Macro_sim:
    def __init__(self, irf_data, sensor_type='ICCD', **cfg):
        # Toggles
        self.use_jitter = cfg.get('jitter', True)
        self.use_dcr = cfg.get('dcr_on', True)
        self.use_poisson = cfg.get('poisson', True)
        self.use_qe = cfg.get('qe_on', True)
        self.use_read_noise = cfg.get('read_noise_on', True)
        self.use_rounding = cfg.get('round_on', True)
        self.use_clipping = cfg.get('clip_on', True)
        
        self.sensor_type = sensor_type.upper()
        self.engine = FLIEngine(irf_data, **cfg)

    def __call__(self):
        p = self.engine.sample_all_params()
        
        # 1. Determine target intensity (A) based on bit-depth
        alpha_pc, beta_pc = self.engine.params_cfg['pc']
        bit_depth = self.engine.params_cfg['bit']
        max_adc_val = (2**bit_depth) - 1
        
        # Sample peak intensity (A) from Beta distribution
        A = ParameterSampler.beta_sample(alpha_pc, beta_pc, scale=max_adc_val)
        
        # Generating Clean Analytical Convolution
        clean_decay = self.engine.get_analytical_decay(p)
        full_conv = fftconvolve(clean_decay, self.engine.irf, mode='full')[:len(clean_decay)]
        
        # INITIAL SCALING: Scale the convolved signal so its PEAK matches A
        if np.max(full_conv) > 0:
            scale_factor = A / np.max(full_conv)
            obs = full_conv * scale_factor
        else:
            obs = full_conv.copy()

        # Applying Modular Noise Pipeline
        if self.use_jitter:
            shift = np.random.randint(-2, 3)
            n = len(obs)
            if shift > 0:
                obs = np.concatenate([np.zeros(shift), obs[:n - shift]])
            elif shift < 0:
                obs = np.concatenate([obs[-shift:], np.zeros(-shift)])

        if self.use_qe:
            obs = obs * ParameterSampler.sample_qe(self.sensor_type)

        if self.use_dcr:
            bit_scaling = bit_depth / 8.0
            obs = NoiseEngine.apply_dcr(obs, self.engine.params_cfg['dcr'] * bit_scaling)

        if self.use_read_noise and self.sensor_type == 'ICCD':
            hw = ParameterSampler.sample_noise_params(bit_depth, self.sensor_type)
            obs = NoiseEngine.apply_read_noise(obs, hw['read_sigma'])

        if self.use_poisson:
            # Poisson must be applied to the intensity-scaled signal
            obs = NoiseEngine.apply_poisson(obs)

        # Final Quantization
        if self.use_rounding:
            obs = np.round(obs)
            
        if self.use_clipping:
            obs = np.clip(obs, 0, max_adc_val)

        # FINAL FIT SCALING: Scale the "Clean Fit" to match the "Observed Peak"
        # This ensures the residuals (obs - fit) reflect ONLY the noise/stochastics
        obs_peak = np.max(obs)
        if obs_peak > 0:
            fit_map = full_conv * (obs_peak / np.max(full_conv) if np.max(full_conv) > 0 else 1.0)
            if self.use_jitter:
                n = len(fit_map)
                if shift > 0:
                    fit_map = np.concatenate([np.zeros(shift), fit_map[:n - shift]])
                elif shift < 0:
                    fit_map = np.concatenate([fit_map[-shift:], np.zeros(-shift)])
        else:
            fit_map = np.zeros_like(obs)

        return {
            "raw_data": {"decay": obs, "irf": self.engine.irf},
            "results": {
                "maps": {
                    "tau1_map":            p['tau1'],
                    "tau2_map":            p['tau2'],
                    "alpha1_map":          p['f'],
                    "A1_map":              p['A1'],
                    "A2_map":              p['A2'],
                    "fret_efficiency_map": p['E'],
                    "tau_mean_map":        p['tau1']*p['f'] + p['tau2']*(1-p['f']),
                    "photon_count_map":    A,
                    "mono_map":            p['mono'],
                },
                "TR_maps": {"fit_map": fit_map, "residual_map": obs - fit_map}
            }
        }

class TCSPC_sim:
    def __init__(self, irf_data, sensor_type='PHOTON_COUNTER', **cfg):
        self.use_jitter = cfg.get('jitter', True)
        self.use_dcr = cfg.get('dcr_on', True)
        self.use_qe = cfg.get('qe_on', False)
        
        # TCSPC is inherently integer-based, but clipping simulates counter overflow
        self.use_clipping = cfg.get('clip_on', True)

        self.sensor_type = sensor_type.upper()
        # TCSPC counters are 16-bit by default
        cfg.setdefault('bit', 16)
        self.engine = FLIEngine(irf_data, **cfg)

    def __call__(self):
        p = self.engine.sample_all_params()
        n_cycles = np.random.randint(1, self.engine.params_cfg['cycles'] + 1)
        mu_per_cycle = 0.01 
        bit_depth = self.engine.params_cfg['bit']
        max_bin_count = (2**bit_depth) - 1

        effective_mu = mu_per_cycle * ParameterSampler.sample_qe(self.sensor_type) if self.use_qe else mu_per_cycle
        obs = self.engine.simulate_tcspc(p, n_cycles, effective_mu)
        
        # Fit Scaling
        total_photons_expected = effective_mu * n_cycles
        clean = self.engine.get_analytical_decay(p)
        fit_norm = fftconvolve(clean, self.engine.irf, mode='full')[:len(clean)]
        fit = fit_norm * (total_photons_expected / np.sum(fit_norm)) if np.sum(fit_norm) > 0 else fit_norm

        if self.use_jitter:
            shift = np.random.randint(-2, 3)
            n = len(obs)
            if shift > 0:
                obs = np.concatenate([np.zeros(shift), obs[:n - shift]])
                fit = np.concatenate([np.zeros(shift), fit[:n - shift]])
            elif shift < 0:
                obs = np.concatenate([obs[-shift:], np.zeros(-shift)])
                fit = np.concatenate([fit[-shift:], np.zeros(-shift)])

        if self.use_dcr:
            # DCR is a detector property (dark counts/bin/cycle) — independent of counter bit depth
            obs = NoiseEngine.apply_dcr(obs, self.engine.params_cfg['dcr'])

        if self.use_clipping:
            obs = np.clip(obs, 0, max_bin_count)
        
        # total_photons_captured = np.sum(obs)
        # if np.sum(fit) > 0:
        #     fit = fit * (total_photons_captured / np.sum(fit))

        return {
            "raw_data": {"decay": obs, "irf": self.engine.irf},
            "results": {
                "maps": {
                    "tau1_map":            p['tau1'],
                    "tau2_map":            p['tau2'],
                    "alpha1_map":          p['f'],
                    "A1_map":              p['A1'],
                    "A2_map":              p['A2'],
                    "fret_efficiency_map": p['E'],
                    "tau_mean_map":        p['tau1']*p['f'] + p['tau2']*(1-p['f']),
                    "photon_count_map":    total_photons_expected,
                    "mono_map":            p['mono'],
                },
                "TR_maps": {"fit_map": fit, "residual_map": obs - fit}
            }
        }