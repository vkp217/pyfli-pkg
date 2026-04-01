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
        Fully compatible with BaseFLIFitter registry and offset-based parameter resolving.
        """
        # 1. Use base class logic to merge guesses and bounds (handles offset logic)
        p0_safe, (l_vec, h_vec) = self._resolve_params_and_bounds(p0, bounds, model_type)
        bnds = list(zip(l_vec, h_vec))
        
        # 2. Map objective functions
        funcs = {
            'poisson': self.poisson_log_likelihood, 
            'pearson': self.pearson_chi_square, 
            'neyman': self.neyman_chi_square
        }
        obj_func = funcs.get(estimator_type, self.poisson_log_likelihood)
        
        # 3. Optimization using L-BFGS-B
        res = minimize(
            obj_func, 
            x0=p0_safe, 
            args=(model_type,), 
            bounds=bnds, 
            method='L-BFGS-B',
            options={'ftol': kwargs.get('ftol', 1e-9), 'gtol': kwargs.get('gtol', 1e-9)}
        )
        
        popt = res.x
        converged = 1 if res.success else 0

        # 4. Uncertainty calculation via Inverse Hessian
        try:
            if hasattr(res, 'hess_inv'):
                # L-BFGS-B returns a linear operator; convert to dense to get diagonal
                if hasattr(res.hess_inv, 'todense'):
                    perr = np.sqrt(np.diag(res.hess_inv.todense()))
                else:
                    # Fallback if it's already a dense array or different operator type
                    perr = np.sqrt(np.diag(res.hess_inv @ np.eye(len(popt))))
            else:
                perr = np.full(len(popt), np.nan)
        except:
            perr = np.full(len(popt), np.nan)

        # 5. Post-Processing: Hierarchy, Mono-collapse, and Statistics
        # We pass manual_stat (the objective function value) and manual_perr
        return self._post_process(popt, None, converged, model_type, 
                                 manual_stat=res.fun, manual_perr=perr)

    def _post_process(self, popt, jac, status, model_type, pcov=None, manual_stat=None, manual_perr=None):
        """
        Compatible post-processor that enforces tau1 <= tau2 and handles MLE-specific statistics.
        """
        if model_type == 'bi-exponential':
            # Handle alpha1 saturation/mono-exponential collapse
            if popt[1] > 0.999:
                popt[1] = 1.0
                popt[3] = popt[2]
            
            # Ensure physical hierarchy: tau1 <= tau2
            if popt[2] > popt[3]:
                popt[2], popt[3] = popt[3], popt[2]
                popt[1] = 1.0 - popt[1]
                # Swap corresponding uncertainties
                if manual_perr is not None and len(manual_perr) >= 4:
                    manual_perr[2], manual_perr[3] = manual_perr[3], manual_perr[2]

        # Standard FLI Statistics
        data = self.decay[self.fit_indices]
        final_model = self.model_fit(self.t, popt, model_type=model_type)[self.fit_indices]
        
        ssr = np.sum((final_model - data)**2)
        # Standard chi_sq (Neyman-style) for reporting consistency even if Poisson was used for fitting
        chi_sq_report = np.sum(((final_model - data)**2) / np.clip(data, 1, None))
        dof = len(data) - len(popt)
        
        # In MLE, 'stat_val' is the likelihood/deviance value
        stat_val = manual_stat if manual_stat is not None else chi_sq_report
        red_stat = stat_val / dof if dof > 0 else 0
        
        ss_tot = np.sum((data - np.mean(data))**2)
        r_sq = 1 - (ssr / ss_tot) if ss_tot > 0 else 0
        
        perr = manual_perr if manual_perr is not None else np.full(len(popt), np.nan)

        # Output format consistent with BaseFLIFitter:
        # [popt, perr, r_sq, stat_val, red_stat, ssr, converged]
        return popt, perr, r_sq, stat_val, red_stat, ssr, (1 if status > 0 else 0)

    def compare_models(self, alpha=0.05, estimator='poisson'):
        """
        Statistical model selection.
        - Poisson: Uses Likelihood Ratio Test (LRT) on Deviance.
        - Chi-square/LS: Uses standard F-test.
        """
        res_m = self.fit_with_estimator(estimator, model_type='mono-exponential')
        res_b = self.fit_with_estimator(estimator, model_type='bi-exponential')
        
        # Result indices: [3] is the stat_val (Likelihood or Chi-sq), [5] is SSR
        if estimator == 'poisson':
            # LRT: Difference in Deviance follows a Chi-square distribution
            LRT_stat = res_m[3] - res_b[3]
            # Degrees of freedom difference is 2 (5 params vs 3 params)
            p_val = 1 - chi2.cdf(max(0, LRT_stat), df=2) 
        else:
            # F-test for Neyman/Pearson/LS estimators
            n, p_m, p_b = len(self.fit_indices), 3, 5
            f_stat = ((res_m[5] - res_b[5]) / (p_b - p_m)) / (res_b[5] / (n - p_b))
            p_val = 1 - f.cdf(f_stat, p_b - p_m, n - p_b)
        
        winner = res_b if p_val < alpha else res_m
        # Return format consistent with BaseFLIFitter.compare_models
        return ("bi-exponential" if p_val < alpha else "mono-exponential"), \
                winner[0], winner[1], winner[2], winner[4], p_val