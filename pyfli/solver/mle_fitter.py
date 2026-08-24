#  solver/mle_fitter.py
"""
Extend the base fitter with Poisson and chi-square maximum-likelihood estimators.

This module belongs to :mod:`pyfli.solver` and is part of PyFLI least-squares, maximum-
likelihood, CPU, GPU, binned, and global FLI fitting routines. Public API includes
classes :class:`MLEFLIFitter`.
"""

from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2, f

from .base_fitter import BaseFLIFitter
from .base_static import resolve_params_and_bounds
from .shared_metrics import compute_fli_stats, enforce_tau_ordering


class MLEFLIFitter(BaseFLIFitter):
    """
    Extend the base FLI fitter with Poisson, Pearson, and Neyman objective functions.
    It supports MLE-style fitting, uncertainty extraction from optimizer curvature, and
    likelihood-based model comparison.
    """

    def poisson_log_likelihood(self, params: Any, model_type: str) -> Any:
        """Standard Poisson MLE (Deviance/C-Statistic)."""
        model = self.model_fit(self.t, params, model_type=model_type)[self.fit_indices]
        if not np.all(np.isfinite(model)):
            return np.inf
        data = self.decay[self.fit_indices]
        model = np.clip(model, 1e-9, None)
        val = 2.0 * np.sum(
            model - data + data * np.log(np.clip(data, 1e-9, None) / model)
        )
        return val if np.isfinite(val) else np.inf

    def pearson_chi_square(self, params: Any, model_type: str) -> np.ndarray:
        """Pearson's Chi-square: Weighted by the MODEL [1/y_model]."""
        model = self.model_fit(self.t, params, model_type=model_type)[self.fit_indices]
        data = self.decay[self.fit_indices]
        model = np.clip(model, 1e-9, None)
        return np.sum((data - model) ** 2 / model)

    def neyman_chi_square(self, params: Any, model_type: str) -> np.ndarray:
        """Neyman's Chi-square: Weighted by the DATA [1/y_data]."""
        model = self.model_fit(self.t, params, model_type=model_type)[self.fit_indices]
        data = self.decay[self.fit_indices]
        weights = np.clip(data, 1.0, None)
        return np.sum((data - model) ** 2 / weights)

    def fit_with_estimator(
        self,
        estimator_type: str = "poisson",
        p0: Any | None = None,
        bounds: np.ndarray | None = None,
        model_type: str = "bi-exponential",
        **kwargs: Any,
    ) -> Any:
        """
        Main interface for MLE/Chi-square fitting.
        Fully compatible with BaseFLIFitter registry and offset-based parameter resolving.
        """
        # Call the external static logic to merge guesses and bounds
        p0_safe, (l_vec, h_vec) = resolve_params_and_bounds(
            p0,
            bounds,
            model_type,
            self.t,
            self.decay,
            self.T_laser,
            self.guess_plugin,
            self.T_acq,
        )
        bnds = list(zip(l_vec, h_vec))

        funcs = {
            "poisson": self.poisson_log_likelihood,
            "pearson": self.pearson_chi_square,
            "neyman": self.neyman_chi_square,
        }
        obj_func = funcs.get(estimator_type, self.poisson_log_likelihood)

        res = minimize(
            obj_func,
            x0=p0_safe,
            args=(model_type,),
            bounds=bnds,
            method="L-BFGS-B",
            options={
                "ftol": kwargs.get("ftol", 1e-12),
                "gtol": kwargs.get("gtol", 1e-9),
                "maxfun": kwargs.get("maxfun", 50000),
                "maxiter": kwargs.get("maxiter", 2000),
            },
        )

        popt = res.x
        converged = 1 if res.success else 0

        # Uncertainty calculation via Inverse Hessian
        try:
            cov_dense = res.hess_inv @ np.eye(len(popt))
            perr = np.sqrt(np.maximum(np.diag(cov_dense), 0.0))
        except Exception:
            perr = np.full(len(popt), np.nan)

        return self._post_process(popt, None, converged, model_type, manual_perr=perr)

    def _post_process(
        self,
        popt: np.ndarray,
        jac: Any,
        status: np.ndarray,
        model_type: str,
        pcov: np.ndarray | None = None,
        manual_stat: np.ndarray | None = None,
        manual_perr: np.ndarray | None = None,
    ) -> tuple[Any, ...]:
        """
        Compatible post-processor that enforces tau1 <= tau2 and handles MLE-specific statistics.
        """
        if model_type == "bi-exponential":
            popt, manual_perr, _ = enforce_tau_ordering(popt, perr=manual_perr)

        data = self.decay[self.fit_indices]
        final_model = self.model_fit(self.t, popt, model_type=model_type)[
            self.fit_indices
        ]

        ssr, chi_sq, red_chi_sq, r_sq, rmse = compute_fli_stats(
            final_model, data, len(popt)
        )

        perr = manual_perr if manual_perr is not None else np.full(len(popt), np.nan)

        return popt, perr, r_sq, chi_sq, red_chi_sq, ssr, (1 if status > 0 else 0), rmse

    def compare_models(
        self, alpha: float = 0.05, estimator: str = "poisson"
    ) -> tuple[Any, ...]:
        """
        Statistical model selection.
        - Poisson: Uses Likelihood Ratio Test (LRT) on Deviance.
        - Chi-square/LS: Uses F-test on Pearson chi-squared (consistent with compute_fli_stats).
        """
        res_m = self.fit_with_estimator(estimator, model_type="mono-exponential")
        res_b = self.fit_with_estimator(estimator, model_type="bi-exponential")

        if estimator == "poisson":
            # LRT: Difference in Deviance follows a Chi-square distribution
            dev_m = self.poisson_log_likelihood(res_m[0], "mono-exponential")
            dev_b = self.poisson_log_likelihood(res_b[0], "bi-exponential")
            LRT_stat = max(dev_m - dev_b, 0.0)
            p_val = 1.0 - chi2.cdf(LRT_stat, df=2)
        else:
            # F-test using Pearson chi-sq (res[3]); consistent with the fitting criterion
            n, p_m, p_b = len(self.fit_indices), 4, 6
            chi_m, chi_b = res_m[3], res_b[3]
            f_stat = ((chi_m - chi_b) / (p_b - p_m)) / (chi_b / (n - p_b))
            p_val = 1.0 - f.cdf(f_stat, p_b - p_m, n - p_b)

        winner = res_b if p_val < alpha else res_m
        return (
            "bi-exponential" if p_val < alpha else "mono-exponential",
            winner[0],
            winner[1],
            winner[2],
            winner[4],
            p_val,
        )
