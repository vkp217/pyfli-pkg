# import numpy as np


# parameters = {'mono-exponential':{'tau' : None},
#                 'bi-exponential':{'tau1': None, 'tau2':None, 'A1':None}}


# class FliPhysics:
#     def __init__(self, t, parameters): # t is the time-vector
#         self.parameters = parameters
#         self.t = t

#     def mono_decay(self):
#         sample_decay = np.exp(-self.t/self.parameters['mono-exponential']['tau'])
#         return sample_decay

#     def bi_decay(self):
#         tau1 = self.parameters['bi-exponential']['tau1']
#         tau2 = self.parameters['bi-exponential']['tau2']
#         A1 = self.parameters['bi-exponential']['A1']
#         sample_decay =  A1*np.exp(-self.t/tau1) + (1-A1)* np.exp(-self.t/tau2)
#         return sample_decay

# class ParamsPriors:
#     def __init__(self, tau):
#         self.tau = tau

#     def lifetime_prior(self, model = None):
#         if model is None:
#             lower_bound = 0.1 # in ns
#             upper_bound = 5 # in ns
#             a = (lower_bound - self.tau2_mean) / self.tau2_std
#             b = (upper_bound - self.tau2_mean) / self.tau2_std
#             return truncnorm.rvs(a, b, loc=self.tau2_mean, scale=self.tau2_std, size=size)
#         else:
#             raise ValueError('the model and paramters are not provided')
    
#     def fret_fraction_prior(self, model = None):
#         f_min = 0.05
#         f = self._stretch_or_squeeze(
#             round(np.random.beta(self.alpha_f, self.beta_f), 3), f_min)  
#         return f

#     def efficiency_prior(self, model=None):
#         pass

    



#     def __init__(
#         self,
#         irf_full,
#         tau2,                 # (mean_tau2, std_tau2)
#         efficiency,               # Beta(alpha, beta) for FRET efficiency E
#         f_fraction,               # Beta(alpha, beta) for amplitude fraction f
#         photo_count=(1.5, 5),     # Beta(alpha, beta) scaled to photon count
#         mono_fraction=0.1,        # probability of mono-exponential pixel
#         bit=8,
#         omega=0.08,               # angular frequency for phasor (rad/ns)
#         n_cycles=800_000,
#         norm_type='pdf_robust',
#     ):
#         # ---- IRF ----
#         if irf_full.ndim == 3:
#             x = np.random.randint(irf_full.shape[0])
#             y = np.random.randint(irf_full.shape[1])
#             if np.sum(irf_full[x, y, :]) < 5000: # this condition is for ICCD
#                 irf = irf_full[irf_full.shape[0] // 2, irf_full.shape[1] // 2, :]
#             else:
#                 irf = irf_full[x, y, :]
#         elif irf_full.ndim == 1:
#             irf = irf_full
#         else:
#             raise ValueError(f'IRF must be 1-D or 3-D, got shape {irf_full.shape}')

#         self.irf = np.nan_to_num(irf / irf.sum())

#         # ---- Time axis ----
#         n = self.irf.shape[0]
#         self.t = np.linspace(0, 12.5, n)
#         self.dt = self.t[1] - self.t[0]

#         # ---- Distribution parameters ----
#         self.tau2_mean, self.tau2_std = tau2
#         self.alpha_E, self.beta_E = efficiency
#         self.alpha_f, self.beta_f = f_fraction
#         self.alpha_A, self.beta_A = photo_count

#         self.mono_fraction = mono_fraction
#         self.bit = bit
#         self.omega = omega
#         self.n_cycles = n_cycles
#         self.T_rep = 12.5
#         self.mu_per_cycle = 0.01
#         self.norm_type = norm_type
#         self.eps = 1e-4

#     # ------------------------------------------------------------------
#     # Normalisation
#     # ------------------------------------------------------------------

#     def pixel_wise_normalisation(self, decay_series):
#         eps = 1e-12
#         decay = np.asarray(decay_series, dtype=np.float64)

#         if self.norm_type == 'None':
#             return decay

#         if self.norm_type == 'pdf':
#             total = np.sum(decay)
#             return decay / total if total > 0 else decay

#         if self.norm_type == 'min_max':
#             lo, hi = decay.min(), decay.max()
#             return (decay - lo) / (hi - lo + eps)

#         if self.norm_type == 'pdf_robust':
#             baseline_pts = getattr(self, 'baseline_pts', 20)
#             if baseline_pts > 0 and len(decay) >= baseline_pts:
#                 baseline = np.median(decay[-baseline_pts:])
#             else:
#                 baseline = 0.0
#             decay_bs = np.clip(decay - baseline, 0.0, None)
#             total = np.sum(decay_bs)
#             return decay_bs / total if np.isfinite(total) and total >= eps else decay

#         raise ValueError(f'Unsupported norm_type: {self.norm_type!r}')

#     # ------------------------------------------------------------------
#     # FFT / phasor features
#     # ------------------------------------------------------------------

#     @staticmethod
#     def fft_features(decay, n_harmonics=5):
#         decay = np.asarray(decay, dtype=np.float64)
#         scalar = decay.ndim == 1
#         if scalar:
#             decay = decay[None, :]

#         dc = np.clip(np.sum(decay, axis=1, keepdims=True), 1e-12, None)
#         fft_vals = np.fft.rfft(decay, axis=1)
#         coeffs = fft_vals[:, 1:n_harmonics + 1]

#         g = np.real(coeffs) / dc
#         s = np.imag(coeffs) / dc
#         features = np.concatenate([g, s], axis=1)   # (B, 2*n_harmonics)

#         return features[0] if scalar else features

#     # ------------------------------------------------------------------
#     # Sampling helpers
#     # ------------------------------------------------------------------

#     def _sample_tau2(self, lower_bound=0.01, upper_bound=5.0, size=1):
#         a = (lower_bound - self.tau2_mean) / self.tau2_std
#         b = (upper_bound - self.tau2_mean) / self.tau2_std
#         return truncnorm.rvs(a, b, loc=self.tau2_mean, scale=self.tau2_std, size=size)

#     def _safe_fraction(self, x):
#         return np.clip(x, self.eps, 1.0 - self.eps)

#     @staticmethod
#     def _stretch_or_squeeze(samples, epsilon):
#         """Map samples from [0,1] into [epsilon, 1-epsilon]."""
#         return samples * (1.0 - 2.0 * epsilon) + epsilon

#     # ------------------------------------------------------------------
#     # Parameter sampling
#     # ------------------------------------------------------------------

#     def sample_local_parameters(self):
#         tau2 = float(self._sample_tau2()[0])
#         mono = np.random.rand() < self.mono_fraction

#         if mono:
#             T = 12.5
#             eps = 1e-6
#             rng = np.random.default_rng()
#             if rng.random() < 0.9:
#                 E = 0.0
#                 A1 = rng.uniform(0.99, 1.0-eps)
#             else:
#                 E = rng.uniform(0.99, 1.0)                
#                 A1 = rng.uniform(0+eps, 1-0.99)
#             A2 = 1.0 - A1
#             tau1 = tau2 * (1 - E)  # if E=0, tau1 == tau2; if E tends to 1, tau1 tends to 0 but not 
#             exp_term1 = 1 - np.exp(-T / tau1)
#             exp_term2 = 1 - np.exp(-T / tau2)
#             f = (A1 * exp_term1) / (A1 * exp_term1 + A2 * exp_term2)
#             # T = 12.5
#             # rng = np.random.default_rng()         
#             # A1 = rng.uniform(5e-2, 1.0 - 5e-2)
#             # A2 = 1.0 - A1
#             # if rng.random() < 0.9:
#             #     E = 0.0
#             # else:
#             #     E = rng.uniform(0.99, 1.0)
#             # tau1 = tau2 * (1 - E)  # if E=0, tau1 == tau2; if E tends to 1, tau1 tends to 0 but not 
#             # exp_term1 = 1 - np.exp(-T / tau1)
#             # exp_term2 = 1 - np.exp(-T / tau2)
#             # f = (A1 * exp_term1) / (A1 * exp_term1 + A2 * exp_term2)
#             return {
#                 "mono": True,
#                 "E": E,
#                 "f": f,
#                 "tau1": tau1,
#                 "tau2": tau2,
#                 "A1": A1,
#                 "A2": A2,
#             }

#         E_min = 0.1
#         E = self._stretch_or_squeeze(
#             round(np.random.beta(self.alpha_E, self.beta_E), 3), E_min)
#         tau1 = tau2 * (1.0 - E)
#         f_min = 0.05
#         f = self._stretch_or_squeeze(
#             round(np.random.beta(self.alpha_f, self.beta_f), 3), f_min)        
#         return {
#             "mono": False,
#             "E": E,
#             "f": f,
#             "tau1": tau1,
#             "tau2": tau2,
#             "A1": f,
#             "A2": 1.0 - f,
#         }

#     def sample_photon_count(self):
#         return round(np.random.beta(self.alpha_A, self.beta_A) * (2 ** self.bit - 1))

#     # IRF jitter
#     @staticmethod
#     def _jitter(decay):
#         """Apply a random sub-bin shift to simulate timing jitter."""
#         n = len(decay)
#         r = np.random.rand()
#         shift = np.random.randint(0, _MAX_GATE_SHIFT + 1)

#         if r > 0.75 or shift == 0:
#             return decay
#         if r < 0.25:
#             # shift right (delay)
#             return np.concatenate([np.zeros(shift), decay[:n - shift]])
#         # shift left (advance)
#         return np.concatenate([decay[shift:], np.zeros(shift)])

#     # Analytical convolution pixel decay  (used by HardSimulator)
#     def generate_pixel_decay(self):
#         """
#         Bi-exponential decay convolved with IRF, then Poisson-sampled.

#         Returns
#         -------
#         decay           : clean bi-exponential before convolution
#         observed        : Poisson-noisy photon histogram
#         scaled          : IRF-convolved decay scaled by photon count A
#         pars            : sampled parameter dict
#         A               : sampled photon count
#         irf             : the instrument response function
#         """
#         pars = self.sample_local_parameters()
#         A = self.sample_photon_count()
#         t = self.t

#         if pars["mono"]:
#             decay = np.exp(-t / pars["tau1"])
#         else:
#             decay = pars["A1"] * np.exp(-t / pars["tau1"]) + pars["A2"] * np.exp(-t / pars["tau2"])
#         decay_conv = fftconvolve(decay, self.irf.squeeze(), mode="full")[:len(decay)]
#         decay_conv = self._jitter(decay_conv)

#         # Scale to photon count; clip at 0 to keep Poisson valid
#         scaled = np.clip(decay_conv * A, 0.0, None)
#         observed = np.random.poisson(scaled).astype(np.float64)

#         return decay, observed, scaled, pars, A, self.irf

#     # Global analytical parameters / phasor / Fisher information
#     def analytical_global_parameters(self):
#         mu_f = self.alpha_f / (self.alpha_f + self.beta_f)
#         tau2_global = self.tau2_mean
#         tau1_global = tau2_global * (self.beta_E / (self.alpha_E + self.beta_E + 1))
#         return {"tau1_global": tau1_global, "tau2_global": tau2_global, "f_global": mu_f}

#     @staticmethod
#     def biexponential(t, a1, tau1, a2, tau2):
#         return a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2)

#     def recover_global_lifetime(self, decay):
#         tau2_init = self.tau2_mean
#         p0 = [0.4, 0.5 * tau2_init, 0.6, tau2_init]
#         popt, _ = curve_fit(self.biexponential, self.t, decay, p0=p0, maxfev=20_000)
#         return {"a1": popt[0], "tau1": popt[1], "a2": popt[2], "tau2": popt[3]}

#     def analytical_phasor(self):
#         mu_f = self.alpha_f / (self.alpha_f + self.beta_f)
#         tau2 = self.tau2_mean
#         c = self.omega * tau2

#         g_long = 1.0 / (1.0 + c ** 2)
#         s_long = c / (1.0 + c ** 2)

#         g_short = hyp2f1(0.5, self.beta_E, self.alpha_E + self.beta_E, -(c ** 2))
#         s_short = c * hyp2f1(1.5, self.beta_E, self.alpha_E + self.beta_E + 1, -(c ** 2))

#         return (
#             mu_f * g_short + (1 - mu_f) * g_long,
#             mu_f * s_short + (1 - mu_f) * s_long,
#         )

#     def fisher_information(self):
#         pars = self.sample_local_parameters()
#         if pars["mono"]:
#             return np.zeros((4, 4))

#         t = self.t
#         tau1 = max(pars["tau1"], 0.01)
#         tau2 = max(pars["tau2"], 0.01)
#         f = self._safe_fraction(pars["f"])
#         A = max(self.sample_photon_count(), 1)

#         I = A * (f * np.exp(-t / tau1) + (1 - f) * np.exp(-t / tau2))
#         I = np.clip(I, 1e-8, None)
#         sqrt_I = np.sqrt(I)

#         d_tau1 = f * np.exp(-t / tau1) * (t / tau1 ** 2) / sqrt_I
#         d_tau2 = (1 - f) * np.exp(-t / tau2) * (t / tau2 ** 2) / sqrt_I
#         d_f = (np.exp(-t / tau1) - np.exp(-t / tau2)) / sqrt_I
#         d_E = f * np.exp(-t / tau1) * (t / tau1 ** 2) * tau2 / sqrt_I

#         grads = np.vstack([d_tau1, d_tau2, d_f, d_E])
#         F = grads @ grads.T
#         return np.clip((F + F.T) / 2.0, 0, None)
