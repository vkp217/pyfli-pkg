# solver/base_fitter.py
import numpy as np
from scipy.optimize import curve_fit, least_squares
from scipy.stats import f
from .base_static import moment_based_guess, resolve_params_and_bounds

class BaseFLIFitter:
    def __init__(self, freq, decay_px, irf_px, white_noise=0.1, 
                 guess_plugin=moment_based_guess, custom_funcs=None):
        """
        Base Fitter for Non-Linear Least Squares (NLSF).
        Includes dynamic registry for solvers and a validation layer for parameters.
        """
        self.decay = np.asarray(decay_px)
        self.irf = np.asarray(irf_px)
        self.white_noise = white_noise
        self.guess_plugin = guess_plugin
        
        # Timing constants
        self.T_laser = 1000.0 / freq[0]
        self.T_acq = 1000.0 / freq[1]
        self.N = len(self.irf) if self.irf.ndim == 1 else self.irf.shape[2]
        self.t = np.linspace(0, self.T_acq, self.N)
        self.fit_indices = np.arange(self.N)

        # Central Solver Registry
        self.funcs = {
            'least_squares': self.least_squares_fit,
            'trust_region': self.trust_region,
            'unconstrained': self.unconstrained
        }
        if custom_funcs:
            self.funcs.update(custom_funcs)

    def fit_with_estimator(self, estimator_type='least_squares', model_type='bi-exponential', p0=None, bounds=None, **kwargs):
        """Unified entry point for all NLSF estimators."""
        # Now calls the external static logic from base_static.py
        p0_safe, bounds_safe = resolve_params_and_bounds(
            p0, bounds, model_type, self.t, self.decay, self.T_laser, self.guess_plugin, self.T_acq
        )
        
        if estimator_type in self.funcs:
            return self.funcs[estimator_type](p0_safe, bounds_safe, model_type, **kwargs)
        else:
            raise ValueError(f"Estimator '{estimator_type}' not found in registry.")

    def least_squares_fit(self, p0, bounds, model_type, use_weights=True, **kwargs):
        d_fit = self.decay[self.fit_indices]
        weights = 1.0 / np.sqrt(np.clip(d_fit, 1, None)) if use_weights else np.ones_like(d_fit)
        
        def residuals(params):
            full_model = self.model_fit(self.t, params, model_type=model_type)
            return (full_model[self.fit_indices] - d_fit) * weights
            
        res = least_squares(residuals, x0=p0, bounds=bounds, 
                            ftol=kwargs.get('ftol', 1e-7), 
                            xtol=kwargs.get('xtol', 1e-7), 
                            max_nfev=kwargs.get('maxiter', 500))        
        return self._post_process(res.x, res.jac, res.status, model_type)

    def trust_region(self, p0, bounds, model_type, **kwargs):
        def wrapper(t_sub, *p):
            return self.model_fit(self.t, p, model_type=model_type)[self.fit_indices]
        try:
            popt, pcov = curve_fit(wrapper, self.t[self.fit_indices], self.decay[self.fit_indices], 
                                    p0=p0, method='trf', bounds=bounds)
            status = 1
        except:
            popt, pcov, status = p0, None, 0
        return self._post_process(popt, None, status, model_type, pcov=pcov)

    def unconstrained(self, p0, bounds, model_type, **kwargs):
        def wrapper(t_sub, *p):
            return self.model_fit(self.t, p, model_type=model_type)[self.fit_indices]
        try:
            popt, pcov = curve_fit(wrapper, self.t[self.fit_indices], self.decay[self.fit_indices], 
                                    p0=p0, method='lm')
            status = 1
        except:
            return self.fit_with_estimator(estimator_type='trust_region', model_type=model_type, p0=p0)
        return self._post_process(popt, None, status, model_type, pcov=pcov)

    def model_fit(self, t, params, model_type='mono-exponential'):
        # Defining a small value epsilon to avoid division by zero
        eps = 1e-8 
        
        if model_type == 'mono-exponential':
            S, tau, offset = params
            tau_safe = np.clip(tau, eps, None) # clipping it make it safe for division
            decay_model = (S / tau_safe) * np.exp(-t / tau_safe)
        else:
            S, a1, tau1, tau2, offset = params
            t1_safe = np.clip(tau1, eps, None) # clipping it make it safe for division
            t2_safe = np.clip(tau2, eps, None) # clipping it make it safe for division
            
            decay_model = S * ((a1 / t1_safe) * np.exp(-t / t1_safe) + 
                               ((1 - a1) / t2_safe) * np.exp(-t / t2_safe))        
        convolved = np.convolve(decay_model, self.irf, mode='full')[:len(t)]
        return convolved + offset

    def _post_process(self, popt, jac, status, model_type, pcov=None):
        if model_type == 'bi-exponential':
            if popt[1] > 0.999:
                popt[1], popt[3] = 1.0, popt[2]
            
            if popt[2] > popt[3]:
                popt[2], popt[3] = popt[3], popt[2]
                popt[1] = 1.0 - popt[1]
                if pcov is not None:
                    idx2, idx3 = 2, 3
                    pcov[[idx2, idx3], :] = pcov[[idx3, idx2], :]
                    pcov[:, [idx2, idx3]] = pcov[:, [idx3, idx2]]

        d_fit = self.decay[self.fit_indices]
        final_model = self.model_fit(self.t, popt, model_type=model_type)[self.fit_indices]
        
        ssr = np.sum((final_model - d_fit)**2)
        chi_sq = np.sum(((final_model - d_fit)**2) / np.clip(d_fit, 1, None))
        dof = len(d_fit) - len(popt)
        red_chi_sq = chi_sq / dof if dof > 0 else 0
        r_sq = 1 - (ssr / np.sum((d_fit - np.mean(d_fit))**2))
        
        if pcov is not None:
            perr = np.sqrt(np.maximum(np.diag(pcov), 0))
        elif jac is not None:
            perr = self.calculate_uncertainties(jac, ssr, len(d_fit), len(popt))
        else:
            perr = np.full(len(popt), np.nan)

        return popt, perr, r_sq, chi_sq, red_chi_sq, ssr, (1 if status > 0 else 0)

    def calculate_uncertainties(self, jacobian, ssr, n_data, n_params):
        try:
            dof = n_data - n_params
            if dof <= 0 or ssr <= 0: return np.zeros(n_params)
            mse = ssr / dof
            hessian_inv = np.linalg.pinv(jacobian.T @ jacobian) 
            return np.sqrt(np.maximum(np.diag(hessian_inv) * mse, 0))
        except:
            return np.full(n_params, np.nan)

    def compare_models(self, alpha=0.05):
        res_m = self.fit_with_estimator(model_type='mono-exponential')
        res_b = self.fit_with_estimator(model_type='bi-exponential')
        n, p_m, p_b = len(self.fit_indices), 3, 5
        f_stat = ((res_m[5] - res_b[5]) / (p_b - p_m)) / (res_b[5] / (n - p_b))
        p_val = 1 - f.cdf(f_stat, p_b - p_m, n - p_b)
        winner = res_b if p_val < alpha else res_m
        return ("bi-exponential" if p_val < alpha else "mono-exponential"), winner[0], winner[1], winner[2], winner[4], p_val

    def get_average_lifetime(self, popt):
        if len(popt) == 5: 
            return popt[1] * popt[2] + (1.0 - popt[1]) * popt[3]
        return popt[1]

    def set_fit_range(self, start_pct=0, end_pct=100):
        start_idx = int((start_pct / 100.0) * self.N)
        end_idx = int((end_pct / 100.0) * self.N)
        self.fit_indices = np.arange(start_idx, min(end_idx, self.N))