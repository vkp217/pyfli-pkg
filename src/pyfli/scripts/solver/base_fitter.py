# solver/base_fitter.py
"""Core non-linear least-squares (NLSF) FLI fitting engine.

Defines :class:`BaseFLIFitter`, the base class providing IRF-convolved decay
model evaluation, a registry of NLSF solvers, and shared fit statistics and
uncertainty utilities used by the other fitters in this package.
"""
import warnings
import numpy as np
from scipy.optimize import curve_fit, least_squares, OptimizeWarning
from scipy.stats import f
from .base_static import moment_based_guess, resolve_params_and_bounds
from .forward_model import model_numpy
from .shared_metrics import (
    enforce_tau_ordering,
    compute_fli_stats,
    compute_average_lifetime,
    compute_fret_efficiency,
)

class BaseFLIFitter:
    """Base fitter for non-linear least-squares (NLSF) FLI/FLIM decay fitting.

    Provides IRF-convolved mono- and bi-exponential decay model evaluation, a
    pluggable registry of NLSF solvers (``'least_squares'``, ``'trust_region'``,
    ``'unconstrained'``), parameter/bound resolution via ``base_static``,
    post-fit statistics, and uncertainty estimation. Subclasses such as
    :class:`~pyfli.scripts.solver.mleFitter.MLEFLIFitter` reuse this
    infrastructure for maximum-likelihood / chi-square based estimators.
    """

    def __init__(self, freq, decay_px, irf_px, white_noise=0.1,
                 guess_plugin=moment_based_guess, custom_funcs=None,
                 shift_method='zero_pad'):
        """
        Base Fitter for Non-Linear Least Squares (NLSF).
        Includes dynamic registry for solvers and a validation layer for parameters.
        """
        self.decay = np.asarray(decay_px)
        self.irf = np.asarray(irf_px)
        self.white_noise = white_noise
        self.guess_plugin = guess_plugin
        self.shift_method = shift_method

        # Timing constants
        self.T_laser = 1000.0 / freq[0]
        self.T_acq = 1000.0 / freq[1]
        self.N = len(self.irf) if self.irf.ndim == 1 else self.irf.shape[2]
        self.t = np.linspace(0, self.T_acq, self.N, endpoint=False)
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
        """Fit the decay using scipy.optimize.least_squares (bounded trust-region reflective).

        Builds a weighted residual function (Poisson-like weights
        ``1/sqrt(max(counts, 1))`` when ``use_weights`` is True) and minimizes it
        with ``scipy.optimize.least_squares`` subject to the supplied bounds.

        Args:
            p0: Initial parameter vector.
            bounds: ``(lower, upper)`` bound arrays passed to ``least_squares``.
            model_type: Either ``'mono-exponential'`` or ``'bi-exponential'``.
            use_weights: If True, weight residuals by ``1/sqrt(max(counts, 1))``;
                otherwise use unweighted residuals.
            **kwargs: Optional overrides ``max_iter``/``maxiter`` (default 500),
                ``ftol`` and ``xtol`` (default ``1e-7``) forwarded to
                ``least_squares``.

        Returns:
            tuple: ``(popt, perr, r_sq, chi_sq, red_chi_sq, ssr, converged)`` as
            produced by the internal post-processing step.
        """
        d_fit = self.decay[self.fit_indices]
        weights = 1.0 / np.sqrt(np.clip(d_fit, 1, None)) if use_weights else np.ones_like(d_fit)

        def residuals(params):
            full_model = self.model_fit(self.t, params, model_type=model_type)
            return (full_model[self.fit_indices] - d_fit) * weights

        max_nfev = kwargs.get('max_iter', kwargs.get('maxiter', 500))
        res = least_squares(residuals, x0=p0, bounds=bounds,
                            ftol=kwargs.get('ftol', 1e-7),
                            xtol=kwargs.get('xtol', 1e-7),
                            max_nfev=max_nfev)
        return self._post_process(res.x, res.jac, res.status, model_type)

    def trust_region(self, p0, bounds, model_type, **kwargs):
        """Fit the decay using scipy.optimize.curve_fit with the Trust Region Reflective ('trf') method.

        Args:
            p0: Initial parameter vector.
            bounds: ``(lower, upper)`` bound arrays passed to ``curve_fit``.
            model_type: Either ``'mono-exponential'`` or ``'bi-exponential'``.
            **kwargs: Optional ``max_iter``/``maxiter`` override (default 2000)
                forwarded as ``max_nfev``.

        Returns:
            tuple: ``(popt, perr, r_sq, chi_sq, red_chi_sq, ssr, converged)`` from
            the internal post-processing step. On failure, falls back to the
            initial guess ``p0`` with ``status=0`` (not converged).
        """
        max_nfev = kwargs.get('max_iter', kwargs.get('maxiter', 2000))
        def wrapper(t_sub, *p):
            return self.model_fit(self.t, p, model_type=model_type)[self.fit_indices]
        try:
            popt, pcov = curve_fit(wrapper, self.t[self.fit_indices], self.decay[self.fit_indices],
                                    p0=p0, method='trf', bounds=bounds, max_nfev=max_nfev)
            status = 1
        except:
            popt, pcov, status = p0, None, 0
        return self._post_process(popt, None, status, model_type, pcov=pcov)

    def unconstrained(self, p0, bounds, model_type, **kwargs):
        """Fit the decay using scipy.optimize.curve_fit with the unconstrained Levenberg-Marquardt ('lm') method.

        Args:
            p0: Initial parameter vector.
            bounds: Unused by the 'lm' method itself; accepted for interface
                consistency with the other registered estimators.
            model_type: Either ``'mono-exponential'`` or ``'bi-exponential'``.
            **kwargs: Optional ``max_iter``/``maxiter`` override (default 2000)
                forwarded as ``maxfev``.

        Returns:
            tuple: ``(popt, perr, r_sq, chi_sq, red_chi_sq, ssr, converged)`` from
            the internal post-processing step. If the unconstrained fit raises an
            exception, falls back to the bounded ``trust_region`` estimator with
            the same initial guess.
        """
        max_nfev = kwargs.get('max_iter', kwargs.get('maxiter', 2000))
        def wrapper(t_sub, *p):
            return self.model_fit(self.t, p, model_type=model_type)[self.fit_indices]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                popt, pcov = curve_fit(wrapper, self.t[self.fit_indices], self.decay[self.fit_indices],
                                       p0=p0, method='lm', maxfev=max_nfev)
            status = 1
        except:
            return self.fit_with_estimator(estimator_type='trust_region', model_type=model_type, p0=p0)
        return self._post_process(popt, None, status, model_type, pcov=pcov)

    def model_fit(self, t, params, model_type='mono-exponential'):
        """Evaluate the IRF-convolved decay model at the given time points.

        Args:
            t: Time axis (ns) at which to evaluate the model.
            params: Model parameter vector (see ``forward_model.model_numpy`` for
                the expected layout).
            model_type: Either ``'mono-exponential'`` (default) or
                ``'bi-exponential'``.

        Returns:
            numpy.ndarray: The IRF-convolved model decay curve.
        """
        return model_numpy(t, self.irf, params, model_type)

    def _post_process(self, popt, jac, status, model_type, pcov=None):
        if model_type == 'bi-exponential':
            popt, _, pcov = enforce_tau_ordering(popt, pcov=pcov)

        d_fit = self.decay[self.fit_indices]
        final_model = self.model_fit(self.t, popt, model_type=model_type)[self.fit_indices]
        ssr, chi_sq, red_chi_sq, r_sq = compute_fli_stats(final_model, d_fit, len(popt))

        if pcov is not None:
            perr = np.sqrt(np.maximum(np.diag(pcov), 0))
        elif jac is not None:
            perr = self.calculate_uncertainties(jac, chi_sq, len(d_fit), len(popt))
        else:
            perr = np.full(len(popt), np.nan)

        return popt, perr, r_sq, chi_sq, red_chi_sq, ssr, (1 if status > 0 else 0)

    def calculate_uncertainties(self, jacobian, chi_sq, n_data, n_params):
        """Estimate parameter uncertainties from the fit Jacobian via the linearized covariance approximation.

        Computes the pseudo-inverse of ``jacobian.T @ jacobian`` (the approximate
        Hessian) scaled by the reduced chi-square to obtain a covariance matrix,
        then returns the square root of its diagonal as the parameter standard
        errors.

        Args:
            jacobian: Jacobian matrix of the residuals at the solution.
            chi_sq: Chi-square statistic of the fit.
            n_data: Number of data points used in the fit.
            n_params: Number of fitted parameters.

        Returns:
            numpy.ndarray: Estimated standard error for each parameter. Returns
            zeros if the degrees of freedom or chi-square are non-positive, or
            NaNs if the computation fails.
        """
        try:
            dof = n_data - n_params
            if dof <= 0 or chi_sq <= 0: return np.zeros(n_params)
            red_chi_sq = chi_sq / dof
            hessian_inv = np.linalg.pinv(jacobian.T @ jacobian)
            return np.sqrt(np.maximum(np.diag(hessian_inv) * red_chi_sq, 0))
        except:
            return np.full(n_params, np.nan)

    def compare_models(self, alpha=0.05):
        """Compare mono- and bi-exponential fits using an F-test on their chi-square statistics.

        Fits the decay with both ``'mono-exponential'`` (4 parameters) and
        ``'bi-exponential'`` (6 parameters) models via ``fit_with_estimator``,
        then performs an F-test comparing their chi-square values to decide
        whether the extra bi-exponential parameters are statistically justified.

        Args:
            alpha: Significance level for the F-test (default 0.05).

        Returns:
            tuple: ``(winning_model_type, popt, perr, r_sq, red_chi_sq, p_val)``
            where ``winning_model_type`` is ``'bi-exponential'`` if
            ``p_val < alpha``, else ``'mono-exponential'``.
        """
        res_m = self.fit_with_estimator(model_type='mono-exponential')
        res_b = self.fit_with_estimator(model_type='bi-exponential')
        n, p_m, p_b = len(self.fit_indices), 4, 6
        chi_m, chi_b = res_m[3], res_b[3]
        f_stat = ((chi_m - chi_b) / (p_b - p_m)) / (chi_b / (n - p_b))
        p_val = 1 - f.cdf(f_stat, p_b - p_m, n - p_b)
        winner = res_b if p_val < alpha else res_m
        return ("bi-exponential" if p_val < alpha else "mono-exponential"), winner[0], winner[1], winner[2], winner[4], p_val

    def get_average_lifetime(self, popt):
        """Compute the amplitude-weighted average fluorescence lifetime for a fitted parameter set.

        Args:
            popt: Fitted parameter vector, as returned by one of the estimator
                methods.

        Returns:
            float: The average lifetime, via
            ``shared_metrics.compute_average_lifetime``.
        """
        return compute_average_lifetime(popt)

    def get_fret_efficiency(self, popt):
        """Compute FRET efficiency from a bi-exponential fit's lifetime components.

        Args:
            popt: Fitted parameter vector, as returned by one of the estimator
                methods.

        Returns:
            float: The FRET efficiency, via
            ``shared_metrics.compute_fret_efficiency``.
        """
        return compute_fret_efficiency(popt)

    def set_fit_range(self, start_pct=0, end_pct=100):
        """Restrict fitting to a sub-range of the time bins, given as percentages of the full range.

        Updates ``self.fit_indices`` to the index range
        ``[start_pct%, end_pct%]`` of the total number of time bins.

        Args:
            start_pct: Start of the fit range, as a percentage (0-100) of total
                time bins (default 0).
            end_pct: End of the fit range, as a percentage (0-100) of total time
                bins (default 100).
        """
        start_idx = int((start_pct / 100.0) * self.N)
        end_idx = int((end_pct / 100.0) * self.N)
        self.fit_indices = np.arange(start_idx, min(end_idx, self.N))
