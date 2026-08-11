# solver/base_fitter.py
"""
Implement the shared least-squares FLIM fitter used by CPU, GPU, and model-comparison
workflows.

This module belongs to :mod:`pyfli.solver` and is part of PyFLI least-squares, maximum-
likelihood, CPU, GPU, binned, and global FLIM fitting routines. Public API includes
classes :class:`BaseFLIFitter`.
"""

from typing import Any
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
    """
    Run the base flifitter routine.
    base class handles model construction, fit ranges, parameter guesses, bounds, post-
    processing, and model comparison support.

    Parameters
    ----------
    freq : float
        Acquisition frequency information used to derive timing constants.
    decay_px : np.ndarray
        Per-pixel fluorescence decay trace supplied to the fitter.
    irf_px : np.ndarray
        Per-pixel instrument response function supplied to the fitter.
    white_noise : float
        White-noise estimate used to weight residuals.
    guess_plugin : np.ndarray
        Optional callable that supplies initial parameter guesses.
    custom_funcs : np.ndarray | None
        Optional custom model functions used by the fitter.
    shift_method : str
        Method used to align the IRF and decay traces.
    """

    def __init__(
        self,
        freq: float,
        decay_px: np.ndarray,
        irf_px: np.ndarray,
        white_noise: float = 0.1,
        guess_plugin: np.ndarray = moment_based_guess,
        custom_funcs: np.ndarray | None = None,
        shift_method: str = "zero_pad",
    ) -> None:
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
            "least_squares": self.least_squares_fit,
            "trust_region": self.trust_region,
            "unconstrained": self.unconstrained,
        }
        if custom_funcs:
            self.funcs.update(custom_funcs)

    def fit_with_estimator(
        self,
        estimator_type: str = "least_squares",
        model_type: str = "bi-exponential",
        p0: Any | None = None,
        bounds: np.ndarray | None = None,
        **kwargs: Any,
    ) -> Any:
        """Unified entry point for all NLSF estimators."""
        # Now calls the external static logic from base_static.py
        p0_safe, bounds_safe = resolve_params_and_bounds(
            p0,
            bounds,
            model_type,
            self.t,
            self.decay,
            self.T_laser,
            self.guess_plugin,
            self.T_acq,
        )

        if estimator_type in self.funcs:
            return self.funcs[estimator_type](
                p0_safe, bounds_safe, model_type, **kwargs
            )
        else:
            raise ValueError(f"Estimator '{estimator_type}' not found in registry.")

    def least_squares_fit(
        self,
        p0: Any,
        bounds: np.ndarray,
        model_type: str,
        use_weights: bool = True,
        **kwargs: Any,
    ) -> Any:
        """
        Run the least squares fit routine.

        Parameters
        ----------
        p0 : Any
            Initial parameter vector supplied to the optimizer.
        bounds : np.ndarray
            Lower and upper parameter bounds supplied to the optimizer.
        model_type : str
            FLIM model family, such as mono- or bi-exponential.
        use_weights : bool
            Whether residuals are weighted during least-squares fitting.
        **kwargs : Any
            Additional keyword options forwarded to the underlying implementation.

        Returns
        -------
        Any
            Object produced by least squares fit.
        """
        d_fit = self.decay[self.fit_indices]
        weights = (
            1.0 / np.sqrt(np.clip(d_fit, 1, None))
            if use_weights
            else np.ones_like(d_fit)
        )

        def residuals(params: Any) -> Any:
            """
            Run the residuals routine.

            Parameters
            ----------
            params : Any
                Model, detector, or plotting parameters used by the routine.

            Returns
            -------
            Any
                Object produced by residuals.
            """
            full_model = self.model_fit(self.t, params, model_type=model_type)
            return (full_model[self.fit_indices] - d_fit) * weights

        max_nfev = kwargs.get("max_iter", kwargs.get("maxiter", 500))
        res = least_squares(
            residuals,
            x0=p0,
            bounds=bounds,
            ftol=kwargs.get("ftol", 1e-7),
            xtol=kwargs.get("xtol", 1e-7),
            max_nfev=max_nfev,
        )
        return self._post_process(res.x, res.jac, res.status, model_type)

    def trust_region(
        self, p0: Any, bounds: np.ndarray, model_type: str, **kwargs: Any
    ) -> Any:
        """
        Run the trust region routine.

        Parameters
        ----------
        p0 : Any
            Initial parameter vector supplied to the optimizer.
        bounds : np.ndarray
            Lower and upper parameter bounds supplied to the optimizer.
        model_type : str
            FLIM model family, such as mono- or bi-exponential.
        **kwargs : Any
            Additional keyword options forwarded to the underlying implementation.

        Returns
        -------
        Any
            Object produced by trust region.
        """
        max_nfev = kwargs.get("max_iter", kwargs.get("maxiter", 2000))

        def wrapper(t_sub: np.ndarray, *p: Any) -> Any:
            """
            Run the wrapper routine.

            Parameters
            ----------
            t_sub : np.ndarray
                Subset of the time axis used during fitting.
            *p : Any
                Detector parameter object or fitted parameter vector.

            Returns
            -------
            Any
                Object produced by wrapper.
            """
            return self.model_fit(self.t, p, model_type=model_type)[self.fit_indices]

        try:
            popt, pcov = curve_fit(
                wrapper,
                self.t[self.fit_indices],
                self.decay[self.fit_indices],
                p0=p0,
                method="trf",
                bounds=bounds,
                max_nfev=max_nfev,
            )
            status = 1
        except Exception:
            popt, pcov, status = p0, None, 0
        return self._post_process(popt, None, status, model_type, pcov=pcov)

    def unconstrained(
        self, p0: Any, bounds: np.ndarray, model_type: str, **kwargs: Any
    ) -> Any:
        """
        Run the unconstrained routine.

        Parameters
        ----------
        p0 : Any
            Initial parameter vector supplied to the optimizer.
        bounds : np.ndarray
            Lower and upper parameter bounds supplied to the optimizer.
        model_type : str
            FLIM model family, such as mono- or bi-exponential.
        **kwargs : Any
            Additional keyword options forwarded to the underlying implementation.

        Returns
        -------
        Any
            Object produced by unconstrained.
        """
        max_nfev = kwargs.get("max_iter", kwargs.get("maxiter", 2000))

        def wrapper(t_sub: np.ndarray, *p: Any) -> Any:
            """
            Run the wrapper routine.

            Parameters
            ----------
            t_sub : np.ndarray
                Subset of the time axis used during fitting.
            *p : Any
                Detector parameter object or fitted parameter vector.

            Returns
            -------
            Any
                Object produced by wrapper.
            """
            return self.model_fit(self.t, p, model_type=model_type)[self.fit_indices]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                popt, pcov = curve_fit(
                    wrapper,
                    self.t[self.fit_indices],
                    self.decay[self.fit_indices],
                    p0=p0,
                    method="lm",
                    maxfev=max_nfev,
                )
            status = 1
        except Exception:
            return self.fit_with_estimator(
                estimator_type="trust_region", model_type=model_type, p0=p0
            )
        return self._post_process(popt, None, status, model_type, pcov=pcov)

    def model_fit(
        self, t: np.ndarray, params: Any, model_type: str = "mono-exponential"
    ) -> Any:
        """
        Run the model fit routine.

        Parameters
        ----------
        t : np.ndarray
            Time axis or acquisition period used by the calculation.
        params : Any
            Model, detector, or plotting parameters used by the routine.
        model_type : str
            FLIM model family, such as mono- or bi-exponential.

        Returns
        -------
        Any
            Object produced by model fit.
        """
        return model_numpy(t, self.irf, params, model_type)

    def _post_process(
        self,
        popt: np.ndarray,
        jac: Any,
        status: np.ndarray,
        model_type: str,
        pcov: np.ndarray | None = None,
    ) -> tuple[Any, ...]:
        """
        Run the post process routine.

        Parameters
        ----------
        popt : np.ndarray
            Optimized model parameter vector.
        jac : Any
            Jacobian matrix returned by the optimizer.
        status : np.ndarray
            Optimizer status flag used during post-processing.
        model_type : str
            FLIM model family, such as mono- or bi-exponential.
        pcov : np.ndarray | None
            Parameter covariance matrix.

        Returns
        -------
        tuple[Any, ...]
            Tuple containing fitted parameters, errors, covariance, and quality metrics.
        """
        if model_type == "bi-exponential":
            popt, _, pcov = enforce_tau_ordering(popt, pcov=pcov)

        d_fit = self.decay[self.fit_indices]
        final_model = self.model_fit(self.t, popt, model_type=model_type)[
            self.fit_indices
        ]
        ssr, chi_sq, red_chi_sq, r_sq, rmse = compute_fli_stats(
            final_model, d_fit, len(popt)
        )

        if pcov is not None:
            perr = np.sqrt(np.maximum(np.diag(pcov), 0))
        elif jac is not None:
            perr = self.calculate_uncertainties(jac, chi_sq, len(d_fit), len(popt))
        else:
            perr = np.full(len(popt), np.nan)

        return popt, perr, r_sq, chi_sq, red_chi_sq, ssr, (1 if status > 0 else 0), rmse

    def calculate_uncertainties(
        self, jacobian: Any, chi_sq: np.ndarray, n_data: int, n_params: int
    ) -> Any:
        """
        Calculate uncertainties.

        Parameters
        ----------
        jacobian : Any
            Jacobian matrix used to estimate parameter uncertainty.
        chi_sq : np.ndarray
            Chi-square statistic used to scale uncertainty estimates.
        n_data : int
            Number of samples, components, gates, or iterations used by the routine.
        n_params : int
            Number of fitted model parameters.

        Returns
        -------
        Any
            Object produced by calculate uncertainties.
        """
        try:
            dof = n_data - n_params
            if dof <= 0 or chi_sq <= 0:
                return np.zeros(n_params)
            red_chi_sq = chi_sq / dof
            hessian_inv = np.linalg.pinv(jacobian.T @ jacobian)
            return np.sqrt(np.maximum(np.diag(hessian_inv) * red_chi_sq, 0))
        except Exception:
            return np.full(n_params, np.nan)

    def compare_models(self, alpha: float = 0.05) -> tuple[Any, ...]:
        """
        Compare models.

        Parameters
        ----------
        alpha : float
            Regularization strength, fraction value, or significance threshold used by the routine.

        Returns
        -------
        tuple[Any, ...]
            Tuple containing model-comparison statistics and selected fit results.
        """
        res_m = self.fit_with_estimator(model_type="mono-exponential")
        res_b = self.fit_with_estimator(model_type="bi-exponential")
        n, p_m, p_b = len(self.fit_indices), 4, 6
        chi_m, chi_b = res_m[3], res_b[3]
        f_stat = ((chi_m - chi_b) / (p_b - p_m)) / (chi_b / (n - p_b))
        p_val = 1 - f.cdf(f_stat, p_b - p_m, n - p_b)
        winner = res_b if p_val < alpha else res_m
        return (
            ("bi-exponential" if p_val < alpha else "mono-exponential"),
            winner[0],
            winner[1],
            winner[2],
            winner[4],
            p_val,
        )

    def get_average_lifetime(self, popt: np.ndarray) -> Any:
        """
        Return average lifetime.

        Parameters
        ----------
        popt : np.ndarray
            Optimized model parameter vector.

        Returns
        -------
        Any
            Object produced by get average lifetime.
        """
        return compute_average_lifetime(popt)

    def get_fret_efficiency(self, popt: np.ndarray) -> Any:
        """
        Return fret efficiency.

        Parameters
        ----------
        popt : np.ndarray
            Optimized model parameter vector.

        Returns
        -------
        Any
            Object produced by get FRET efficiency.
        """
        return compute_fret_efficiency(popt)

    def set_fit_range(self, start_pct: int = 0, end_pct: int = 100) -> None:
        """
        Set fit range.

        Parameters
        ----------
        start_pct : int
            Start percentage of the decay range used for fitting.
        end_pct : int
            End percentage of the decay range used for fitting.

        Returns
        -------
        None
            No object is returned; the function set fit range.
        """
        start_idx = int((start_pct / 100.0) * self.N)
        end_idx = int((end_pct / 100.0) * self.N)
        self.fit_indices = np.arange(start_idx, min(end_idx, self.N))
