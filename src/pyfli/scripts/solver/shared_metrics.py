"""Shared post-fit statistics and parameter-normalisation utilities for FLI fitters.

Provides helpers to canonicalise bi-exponential parameter ordering, compute
goodness-of-fit statistics, and derive average lifetime / FRET efficiency
from a fitted parameter vector. Used by both the NLSF and MLE fitter
implementations.
"""

import numpy as np

def enforce_tau_ordering(popt, perr=None, pcov=None):
    """Canonicalise a bi-exponential fit so that tau1 <= tau2, swapping components if needed.

    Also snaps ``alpha1`` to 0 or 1 (collapsing to a single-component
    result) when it is extremely close to 0 or 1. If ``tau1 > tau2`` after
    this, the two lifetime components (and, if provided, their
    corresponding entries in ``perr``/``pcov``) are swapped and ``alpha1``
    is replaced with ``1 - alpha1``.

    Args:
        popt: Bi-exponential parameter vector
            ``[amp, alpha1, tau1, tau2, v_shift, h_shift]``.
        perr: Optional parameter standard-error vector, reordered in step
            with ``popt``.
        pcov: Optional parameter covariance matrix, reordered (rows/columns
            2 and 3 swapped) in step with ``popt``.

    Returns:
        tuple: ``(popt, perr, pcov)`` with lifetime ordering enforced.
    """
    popt = np.asarray(popt, dtype=float)

    if popt[1] > 0.999:
        popt[1], popt[3] = 1.0, popt[2]
    elif popt[1] < 0.001:
        popt[1], popt[2] = 0.0, popt[3]

    if popt[2] > popt[3]:
        popt[2], popt[3] = popt[3], popt[2]
        popt[1] = 1.0 - popt[1]
        if perr is not None:
            perr = np.asarray(perr, dtype=float)
            perr[2], perr[3] = perr[3], perr[2]
        if pcov is not None:
            pcov[[2, 3], :] = pcov[[3, 2], :]
            pcov[:, [2, 3]] = pcov[:, [3, 2]]

    return popt, perr, pcov

def compute_fli_stats(final_model, d_fit, n_params):
    """Compute standard goodness-of-fit statistics for a fitted decay curve.

    Args:
        final_model: Model-predicted decay values over the fitted range.
        d_fit: Measured decay counts over the fitted range.
        n_params: Number of fitted parameters, used to compute degrees of
            freedom.

    Returns:
        tuple: ``(ssr, chi_sq, red_chi_sq, r_sq)`` — the sum of squared
        residuals, chi-square (residuals weighted by
        ``1/max(model, 1)``), reduced chi-square (``chi_sq`` divided by
        ``max(len(d_fit) - n_params, 1)``), and R-squared.
    """
    residuals = final_model - d_fit
    ssr = float(np.sum(residuals ** 2))
    chi_sq = float(np.sum(residuals ** 2 / np.clip(final_model, 1.0, None)))
    dof = max(len(d_fit) - n_params, 1)
    red_chi_sq = chi_sq / dof
    ss_tot = float(np.sum((d_fit - np.mean(d_fit)) ** 2))
    r_sq = 1.0 - ssr / ss_tot if ss_tot > 0 else 0.0
    return ssr, chi_sq, red_chi_sq, r_sq

def compute_average_lifetime(popt):
    """Compute the amplitude-weighted average lifetime from a fitted parameter vector.

    Args:
        popt: Fitted parameter vector. If length 6 (bi-exponential:
            ``[amp, alpha1, tau1, tau2, v_shift, h_shift]``), returns the
            alpha-weighted average of tau1 and tau2; otherwise treats
            ``popt`` as mono-exponential and returns ``popt[1]`` (tau).

    Returns:
        float: The average lifetime.
    """
    if len(popt) == 6:
        return float(popt[1] * popt[2] + (1.0 - popt[1]) * popt[3])
    return float(popt[1])

def compute_fret_efficiency(popt):
    """Compute FRET efficiency from a bi-exponential fit's lifetime components.

    Args:
        popt: Fitted parameter vector. If length 6 (bi-exponential), FRET
            efficiency is computed as ``1 - tau1/tau2``; otherwise
            (mono-exponential) returns 0.0.

    Returns:
        float: FRET efficiency in [0, 1), or 0.0 if not applicable or
        ``tau2 <= 0``.
    """
    if len(popt) == 6:
        tau1, tau2 = popt[2], popt[3]
        if tau2 > 0:
            return float(1.0 - tau1 / tau2)
    return 0.0
