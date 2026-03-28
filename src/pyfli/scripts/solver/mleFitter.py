#  solver/mleFitter.py 
import numpy as np
from scipy.optimize import minimize
from scipy.stats import f, chi2
from .base_fitter import BaseFLIFitter

class MLEFLIFitter(BaseFLIFitter):
    def poisson_log_likelihood(self, params, model_type):
        """Standard Poisson MLE (Deviance/C-Statistic)."""
        model = self.model_fit(self.t, params, model_type=model_type)[self.fit_indices]
        data = self.decay[self.fit_indices]
        model = np.clip(model, 1e-9, None)
        # Deviance: 2 * sum( model - data + data * ln(data/model) )
        return 2 * np.sum(model - data + data * np.log(np.clip(data, 1e-9, None) / model))

    def pearson_chi_square(self, params, model_type):
        """Pearson's Chi-square: Weighted by the MODEL [1/y_model]."""
        model = self.model_fit(self.t, params, model_type=model_type)[self.fit_indices]
        data = self.decay[self.fit_indices]
        model = np.clip(model, 1e-9, None)
        return np.sum(((data - model)**2) / model)

    def neyman_chi_square(self, params, model_type):
        """Neyman's Chi-square: Weighted by the DATA [1/y_data]."""
        model = self.model_fit(self.t, params, model_type=model_type)[self.fit_indices]
        data = self.decay[self.fit_indices]
        weights = np.clip(data, 1, None)
        return np.sum(((data - model)**2) / weights)
    
    def fit_with_estimator(self, estimator_type='poisson', p0=None, bounds=None, model_type='bi-exponential', **kwargs):
        """
        Main interface for MLE/Chi-square fitting. 
        Corrected to return the full model curve for diagnostic plotting.
        """
        auto_full_p0 = self.initial_guess(model_type)
        p0_internal = [auto_full_p0[0]] + list(p0) if p0 is not None else auto_full_p0

        p0_safe, (l_vec, h_vec) = self._resolve_bounds(p0_internal, bounds, model_type)
        bnds = list(zip(l_vec, h_vec))
        
        funcs = {
            'poisson': self.poisson_log_likelihood, 
            'pearson': self.pearson_chi_square, 
            'neyman': self.neyman_chi_square}
        obj_func = funcs.get(estimator_type, self.poisson_log_likelihood)
        
        res = minimize(obj_func, x0=p0_safe, args=(model_type,), bounds=bnds, method='L-BFGS-B',
                    options={'ftol': 1e-7, 'gtol': 1e-7})
        
        popt = res.x
        converged = 1 if res.success else 0

        # Uncertainty calculation
        try:
            if hasattr(res, 'hess_inv'):
                perr = np.sqrt(np.diag(res.hess_inv.todense()))
            else:
                perr = np.full(len(popt), np.nan)
        except:
            perr = np.full(len(popt), np.nan)

        # TAU1 <= TAU2 Swap Logic
        if model_type == 'bi-exponential' and popt[2] > popt[3]:
            popt[2], popt[3] = popt[3], popt[2]
            popt[1] = 1.0 - popt[1]
            if perr is not None:
                perr[2], perr[3] = perr[3], perr[2]

        # --- Generate full curve for plotting ---
        full_model_curve = self.model_fit(self.t, popt, model_type=model_type)
        
        # Stats on fit indices
        data = self.decay[self.fit_indices]
        final_model_cropped = full_model_curve[self.fit_indices]
        
        stat_val = res.fun
        dof = len(data) - len(popt)
        red_stat = stat_val / dof if dof > 0 else 0
        
        # R-squared calculation
        ssr = np.sum((final_model_cropped - data)**2)
        ss_tot = np.sum((data - np.mean(data))**2)
        r_sq = 1 - (ssr / ss_tot) if ss_tot > 0 else 0

        # RETURNS: popt, perr, r_sq, stat_val, red_stat, full_model_curve, converged
        return popt, perr, r_sq, stat_val, red_stat, full_model_curve, converged

    # def fit_with_estimator(self, estimator_type='poisson', p0=None, bounds=None, model_type='bi-exponential', **kwargs):
    #     """
    #     Main interface for MLE/Chi-square fitting. 
    #     Handles the S-parameter auto-calculation logic to match BaseFLIFitter.
    #     """
    #     # 1. Internal p0/Bounds Prep
    #     auto_full_p0 = self.initial_guess(model_type)
    #     if p0 is not None:
    #         p0_internal = [auto_full_p0[0]] + list(p0)
    #     else:
    #         p0_internal = auto_full_p0

    #     # Resolve bounds and get the safe starting point (clipped)
    #     p0_safe, (l_vec, h_vec) = self._resolve_bounds(p0_internal, bounds, model_type)
    #     bnds = list(zip(l_vec, h_vec))
        
    #     # 2. Select objective function
    #     funcs = {
    #         'poisson': self.poisson_log_likelihood,
    #         'pearson': self.pearson_chi_square,
    #         'neyman': self.neyman_chi_square
    #     }
    #     obj_func = funcs.get(estimator_type, self.poisson_log_likelihood)
        
    #     # 3. Execution using L-BFGS-B (supports bounds)
    #     res = minimize(
    #         obj_func, 
    #         x0=p0_safe, 
    #         args=(model_type,), 
    #         bounds=bnds, 
    #         method='L-BFGS-B',
    #         options={'ftol': 1e-7, 'gtol': 1e-7}
    #     )
        
    #     popt = res.x
    #     converged = 1 if res.success else 0

    #     # 4. Uncertainty (Hessian Approximation)
    #     try:
    #         if hasattr(res, 'hess_inv'):
    #             # hess_inv is an LbfgsInvHessProduct; convert to dense to get diag
    #             perr = np.sqrt(np.diag(res.hess_inv.todense()))
    #         else:
    #             perr = np.full(len(popt), np.nan)
    #     except:
    #         perr = np.full(len(popt), np.nan)

    #     # --- TAU1 <= TAU2 Logic & Consistency Swap ---
    #     if model_type == 'bi-exponential' and popt[2] > popt[3]:
    #         # Intensity-weighted average lifetime for sanity check
    #         avg_pre = popt[1] * popt[2] + (1.0 - popt[1]) * popt[3]
            
    #         # Swap tau1 (idx 2) and tau2 (idx 3)
    #         popt[2], popt[3] = popt[3], popt[2]
    #         # Swap alpha1 (idx 1) and alpha2 (1 - a1)
    #         popt[1] = 1.0 - popt[1]
            
    #         # Swap uncertainties if available
    #         if perr is not None:
    #             perr[2], perr[3] = perr[3], perr[2]
            
    #         # Sanity check
    #         avg_post = popt[1] * popt[2] + (1.0 - popt[1]) * popt[3]
    #         if not np.isclose(avg_pre, avg_post, atol=1e-9):
    #             print(f"Warning: MLE swap sanity check failed. Shift: {avg_pre:.4f} -> {avg_post:.4f}")

    #     # 5. Stats
    #     final_model = self.model_fit(self.t, popt, model_type=model_type)[self.fit_indices]
    #     data = self.decay[self.fit_indices]
    #     ssr = np.sum((final_model - data)**2)
    #     ss_tot = np.sum((data - np.mean(data))**2)
    #     r_sq = 1 - (ssr / ss_tot) if ss_tot > 0 else 0
        
    #     stat_val = res.fun
    #     dof = len(data) - len(popt)
    #     red_stat = stat_val / dof if dof > 0 else 0

    #     # Returns: popt, perr, r_sq, stat_val, red_stat, ssr, converged
    #     return popt, perr, r_sq, stat_val, red_stat, ssr, converged

    # def mle_fit(self, p0=None, bounds=None, model_type='bi-exponential'):
    #     """Overridden to return success and message as requested."""
    #     return self.fit_with_estimator('poisson', p0, bounds, model_type)

    # def compare_models(self, alpha=0.05, estimator='poisson'):
    #     """Uses Likelihood Ratio Test (LRT) for Poisson or F-test for others."""
    #     res_m = self.fit_with_estimator(estimator, model_type='mono-exponential')
    #     res_b = self.fit_with_estimator(estimator, model_type='bi-exponential')
        
    #     # res indices: 0:popt, 3:stat_val, 4:red_stat, 5:ssr
    #     if estimator == 'poisson':
    #         # Likelihood Ratio Test (LRT)
    #         LRT_stat = res_m[3] - res_b[3]
    #         p_val = 1 - chi2.cdf(LRT_stat, df=2)
    #     else:
    #         # Standard F-test
    #         n, p_m, p_b = len(self.fit_indices), 3, 5
    #         f_stat = ((res_m[5] - res_b[5]) / (p_b - p_m)) / (res_b[5] / (n - p_b))
    #         p_val = 1 - f.cdf(f_stat, p_b - p_m, n - p_b)
        
    #     if p_val < alpha:
    #         return "bi-exponential", res_b[0], res_b[1], res_b[2], res_b[4], p_val
    #     return "mono-exponential", res_m[0], res_m[1], res_m[2], res_m[4], p_val
    

