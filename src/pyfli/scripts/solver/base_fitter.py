# solver/base_fitter.py
import numpy as np
from scipy.optimize import curve_fit, least_squares
from scipy.stats import f

class BaseFLIFitter:
    def __init__(self, freq, decay_px, irf_px, white_noise=0.1):
        """
        freq: [freq_laser, freq_acquisition] in MHz
        """
        self.decay = decay_px
        self.irf = irf_px
        self.white_noise = white_noise
        
        # Period calculations
        self.T_laser = 1000.0 / freq[0]
        self.T_acq = 1000.0 / freq[1]
        
        # Truncation check
        self.is_truncated = self.T_acq < self.T_laser
        
        # Bins and Time Axis
        self.N = len(self.irf) if self.irf.ndim == 1 else self.irf.shape[2]
        self.t = np.linspace(0, self.T_acq, self.N)
        self.fit_indices = np.arange(self.N)

        # Registry for different fitting estimators
        self.funcs = {
            'least_squares': self.least_squares_fit,
            'trust_region': self.trust_region,
            'unconstrained': self.unconstrained
        }

    def set_fit_range(self, start_pct=0, end_pct=100):
        """Selects a subset of the decay for fitting based on percentage."""
        start_idx = int((start_pct / 100.0) * self.N)
        end_idx = int((end_pct / 100.0) * self.N)
        self.fit_indices = np.arange(start_idx, min(end_idx, self.N))

    def model_fit(self, t, params, model_type='mono-exponential'):
        """
        Internal Physics Model. 
        Always expects [S, tau, b] or [S, a1, tau1, tau2, b].
        """
        if model_type == 'mono-exponential':
            S, tau, b = params
            decay_model = (S / tau) * np.exp(-t / tau)
        else:
            S, a1, tau1, tau2, b = params
            decay_model = S * ((a1 / tau1) * np.exp(-t / tau1) + ((1 - a1) / tau2) * np.exp(-t / tau2))
        
        # Linear convolution with the IRF
        convolved = np.convolve(decay_model, self.irf, mode='full')[:len(t)]
        return convolved + b

    def initial_guess(self, model_type='mono-exponential', threshold=0.05):
        """
        Estimates the FULL parameter set internally.
        Area (S) is calculated via np.trapz to ground the optimization.
        """
        if np.any(self.decay > self.white_noise):
            idx_max = np.argmax(self.decay)
            max_val = self.decay[idx_max]
            mdec = self.decay[idx_max:]
            threshold_val = float(threshold) if isinstance(threshold, str) else threshold
            low_idx = np.where(mdec < threshold_val * max_val)[0]
            idx_min = low_idx[0] if len(low_idx) > 0 else len(mdec)
            
            mdec_temp = mdec[:idx_min]
            t_temp = self.t[idx_max : idx_max + idx_min]
            
            # Auto-calculate Background and Area
            B_guess = threshold_val * max_val if not self.is_truncated else np.mean(mdec[-5:])
            S_cap = np.trapezoid(np.clip(mdec_temp - B_guess, 0, None), t_temp)

            if model_type == 'mono-exponential':
                tau_g = (t_temp[-1] - t_temp[0]) / (np.log(max_val) - np.log(np.clip(mdec_temp[-1], 1e-6, None)))
                S_tot = S_cap / (1 - np.exp(-self.T_acq / tau_g)) if self.is_truncated else S_cap
                return [S_tot, tau_g, B_guess]
            else:
                # Basic bi-exponential guess splitting the decay into fast/slow segments
                split = int(len(mdec_temp) * 0.3)
                tau1_g = (t_temp[split] - t_temp[0]) / (np.log(max_val) - np.log(np.clip(mdec_temp[split], 1e-6, None)))
                tau2_g = (t_temp[-1] - t_temp[split]) / (np.log(np.clip(mdec_temp[split], 1e-6, None)) - np.log(np.clip(mdec_temp[-1], 1e-6, None)))
                a1_g = 0.5 
                
                if self.is_truncated:
                    den = a1_g*(1-np.exp(-self.T_acq/tau1_g)) + (1-a1_g)*(1-np.exp(-self.T_acq/tau2_g))
                    S_tot = S_cap / np.clip(den, 1e-3, None)
                else: S_tot = S_cap
                return [S_tot, a1_g, tau1_g, tau2_g, B_guess]
        
        return [0, 1e-6, 0] if model_type == 'mono-exponential' else [0, 0.5, 1e-6, 1e-6, 0]

    def _resolve_bounds(self, p0, bounds_list, model_type):
        """
        Creates internal bounds while keeping Area (Index 0) automated.
        Maps user-provided bounds_list (len 4 or 2) to indices 1+.
        """
        p0 = np.array(p0)
        n = len(p0)
        low_vec = np.zeros(n)
        high_vec = np.full(n, np.inf) 
        
        if model_type == 'bi-exponential':
            low_vec[1], high_vec[1] = 0.0, 1.0 # alpha1 bounds
            low_vec[2:4], high_vec[2:4] = 1e-6, self.T_laser # tau bounds
        else:
            low_vec[1], high_vec[1] = 1e-6, self.T_laser # tau bounds
            
        # Grounding
        low_vec[0], low_vec[-1] = 0, 0 
        high_vec[0] = np.inf # Let area scale freely

        # Map User Bounds if provided (skipping Area index 0)
        if bounds_list is not None:
            for i, b in enumerate(bounds_list):
                if b is not None and (i + 1) < n:
                    low_vec[i+1], high_vec[i+1] = b

        margin = 1e-5
        p0_clipped = np.clip(p0, low_vec + margin, high_vec - margin)

        # if not np.allclose(p0, p0_clipped, rtol=1e-4):
        #     print("Note: Initial guess was clipped to fit within bounds.")
        return p0_clipped, (low_vec, high_vec)

    def fit_with_estimator(self, estimator_type='least_squares', model_type='bi-exponential', p0=None, bounds=None, **kwargs):
        """
        Main Interface. Accepts p0/bounds WITHOUT Area and prepends auto-calculated Area.
        """
        auto_full_p0 = self.initial_guess(model_type)
        
        if p0 is not None:
            # User provides [a1, tau1, tau2, b] or [tau, b]
            p0_internal = [auto_full_p0[0]] + list(p0)
        else:
            p0_internal = auto_full_p0

        fit_method = self.funcs.get(estimator_type, self.least_squares_fit)
        return fit_method(p0=p0_internal, bounds=bounds, model_type=model_type, **kwargs)

    def least_squares_fit(self, p0, bounds=None, model_type='bi-exponential', use_weights=True, **kwargs):
        p0_safe, (l_vec, h_vec) = self._resolve_bounds(p0, bounds, model_type)
        d_fit = self.decay[self.fit_indices]
        weights = 1.0 / np.sqrt(np.clip(d_fit, 1, None)) if use_weights else np.ones_like(d_fit)

        def residuals(params):
            full_model = self.model_fit(self.t, params, model_type=model_type)
            return (full_model[self.fit_indices] - d_fit) * weights

        _ = kwargs.pop('estimator', None)
        _ = kwargs.pop('min_cluster_size', None)
        _ = kwargs.pop('cluster_strategy', None)

        # 2. Extract tolerances with defaults
        ftol = kwargs.pop('ftol', 1e-7)
        xtol = kwargs.pop('xtol', 1e-7)
        gtol = kwargs.pop('gtol', 1e-7)
        
        # 3. Scipy uses 'max_nfev' for least_squares, but we often pass 'maxiter'
        max_nfev = kwargs.pop('maxiter', 500)

        # Now **kwargs is "clean" and contains no 'estimator' key
        res = least_squares(residuals, x0=p0_safe, bounds=(l_vec, h_vec), 
                            ftol=ftol, xtol=xtol, gtol=gtol, max_nfev=max_nfev, 
                            **kwargs)
        popt = res.x

        # Convergence Check: 1 if converged, 0 if max iterations reached
        converged = 1 if res.status > 0 else 0

        # Stats
        final_model = self.model_fit(self.t, popt, model_type=model_type)[self.fit_indices]
        ssr = np.sum((final_model - d_fit)**2)
        chi_sq = np.sum(((final_model - d_fit)**2) / np.clip(d_fit, 1, None))
        dof = len(d_fit) - len(popt)
        red_chi_sq = chi_sq / dof if dof > 0 else 0
        r_sq = 1 - (ssr / np.sum((d_fit - np.mean(d_fit))**2))
        
        perr = self.calculate_uncertainties(res.jac, ssr, len(d_fit), len(popt))

        # ENSURE TAU1 <= TAU2 (Symmetry Swap with Sanity Check)
        if model_type == 'bi-exponential' and popt[2] > popt[3]:
            # Capture state for sanity check
            avg_pre = self.get_average_lifetime(popt)
            
            # Swap tau and alpha logic
            popt[2], popt[3] = popt[3], popt[2]
            popt[1] = 1.0 - popt[1]
            if perr is not None:
                perr[2], perr[3] = perr[3], perr[2]
            
            # Post-swap check
            avg_post = self.get_average_lifetime(popt)
            if not np.isclose(avg_pre, avg_post, atol=1e-9):
                print(f"Warning: Sanity check failed. Average lifetime shifted from {avg_pre:.4f} to {avg_post:.4f}")

        return popt, perr, r_sq, chi_sq, red_chi_sq, ssr, converged

    def trust_region(self, p0, bounds=None, model_type='bi-exponential'):
        l_b, h_b = self._resolve_bounds(p0, bounds, model_type)
        def wrapper(t_sub, *p):
            return self.model_fit(self.t, p, model_type=model_type)[self.fit_indices]
        try:
            popt, pcov = curve_fit(wrapper, self.t[self.fit_indices], self.decay[self.fit_indices], 
                                   p0=p0, method='trf', bounds=(l_b, h_b))
            perr = np.sqrt(np.diag(pcov)) if pcov is not None else np.full(len(popt), np.nan)
            
            if model_type == 'bi-exponential' and popt[2] > popt[3]:
                avg_pre = self.get_average_lifetime(popt)
                popt[2], popt[3] = popt[3], popt[2]
                popt[1] = 1.0 - popt[1]
                if perr is not None: perr[2], perr[3] = perr[3], perr[2]
                
                avg_post = self.get_average_lifetime(popt)
                if not np.isclose(avg_pre, avg_post, atol=1e-9):
                    print("Warning: Sanity check failed in TRF swap.")

            return popt, perr, 0, 0, 0, 0 
        except:
            return np.array(p0), None, 0, 0, 0, 0

    def unconstrained(self, p0, model_type='mono-exponential'):
        def wrapper(t_sub, *p):
            return self.model_fit(self.t, p, model_type=model_type)[self.fit_indices]
        try:
            popt, pcov = curve_fit(wrapper, self.t[self.fit_indices], self.decay[self.fit_indices], 
                                   p0=p0, method='lm')
            perr = np.sqrt(np.diag(pcov)) if pcov is not None else np.full(len(popt), np.nan)
            
            if model_type == 'bi-exponential' and popt[2] > popt[3]:
                avg_pre = self.get_average_lifetime(popt)
                popt[2], popt[3] = popt[3], popt[2]
                popt[1] = 1.0 - popt[1]
                if perr is not None: perr[2], perr[3] = perr[3], perr[2]
                
                avg_post = self.get_average_lifetime(popt)
                if not np.isclose(avg_pre, avg_post, atol=1e-9):
                    print("Warning: Sanity check failed in LM swap.")

            return popt, perr, 0, 0, 0, 0
        except:
            return self.trust_region(p0=p0, model_type=model_type)

    def calculate_uncertainties(self, jacobian, ssr, n_data, n_params):
        """Calculates standard deviations from the Jacobian and SSR."""
        try:
            dof = n_data - n_params
            if dof <= 0 or ssr <= 0:
                return np.zeros(n_params)
            mse = ssr / (n_data - n_params)
            hessian_inv = np.linalg.pinv(jacobian.T @ jacobian) 
            variance = np.diag(hessian_inv) * mse
            variance = np.maximum(variance, 0)
            return np.sqrt(variance)
        except:
            return np.full(n_params, np.nan)

    def compare_models(self, alpha=0.05):
        """Performs F-test to determine if bi-exponential is statistically justified."""
        res_m = self.least_squares_fit(p0=self.initial_guess('mono-exponential'), model_type='mono-exponential')
        res_b = self.least_squares_fit(p0=self.initial_guess('bi-exponential'), model_type='bi-exponential')
        
        n, p_m, p_b = len(self.fit_indices), 3, 5
        f_stat = ((res_m[5] - res_b[5]) / (p_b - p_m)) / (res_b[5] / (n - p_b))
        p_val = 1 - f.cdf(f_stat, p_b - p_m, n - p_b)
        
        decision = "bi-exponential" if p_val < alpha else "mono-exponential"
        winner = res_b if p_val < alpha else res_m
        return decision, winner[0], winner[1], winner[2], winner[4], p_val

    def get_average_lifetime(self, popt):
        """Calculates intensity-weighted average lifetime."""
        if len(popt) == 5: # Bi-exponential [S, a1, t1, t2, b]
            return popt[1] * popt[2] + (1.0 - popt[1]) * popt[3]
        return popt[1] # Mono-exponential [S, t, b]