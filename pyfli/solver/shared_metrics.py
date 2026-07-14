"""
Centralize tau ordering, lifetime summaries, FRET efficiency, and fit-quality metrics.

This module belongs to :mod:`pyfli.solver` and is part of PyFLI least-squares, maximum-
likelihood, CPU, GPU, binned, and global FLIM fitting routines. Public API includes
functions :func:`enforce_tau_ordering`, :func:`compute_fli_stats`,
:func:`compute_average_lifetime`, and :func:`compute_fret_efficiency`.
"""

from __future__ import annotations
from typing import Any
import numpy as np


def enforce_tau_ordering(
    popt: np.ndarray, perr: Any | None = None, pcov: np.ndarray | None = None
) -> tuple[Any, ...]:
    """
    Enforce tau ordering.

    Parameters
    ----------
    popt : np.ndarray
        Input value.
    perr : Any | None
        Input value.
    pcov : np.ndarray | None
        Input value.

    Returns
    -------
    tuple[Any, ...]
        Return value.
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


def compute_fli_stats(
    final_model: np.ndarray, d_fit: np.ndarray, n_params: int
) -> tuple[Any, ...]:
    """
    Compute fli stats.

    Parameters
    ----------
    final_model : np.ndarray
        Input value.
    d_fit : np.ndarray
        Input value.
    n_params : int
        Input value.

    Returns
    -------
    tuple[Any, ...]
        Return value.
    """
    residuals = final_model - d_fit
    ssr = float(np.sum(residuals**2))
    chi_sq = float(np.sum(residuals**2 / np.clip(final_model, 1.0, None)))
    dof = max(len(d_fit) - n_params, 1)
    red_chi_sq = chi_sq / dof
    ss_tot = float(np.sum((d_fit - np.mean(d_fit)) ** 2))
    r_sq = 1.0 - ssr / ss_tot if ss_tot > 0 else 0.0
    return ssr, chi_sq, red_chi_sq, r_sq


def compute_average_lifetime(popt: np.ndarray) -> float:
    """
    Compute average lifetime.

    Parameters
    ----------
    popt : np.ndarray
        Input value.

    Returns
    -------
    float
        Return value.
    """
    if len(popt) == 6:
        return float(popt[1] * popt[2] + (1.0 - popt[1]) * popt[3])
    return float(popt[1])


def compute_fret_efficiency(popt: np.ndarray) -> float:
    """
    Compute fret efficiency.

    Parameters
    ----------
    popt : np.ndarray
        Input value.

    Returns
    -------
    float
        Return value.
    """
    if len(popt) == 6:
        tau1, tau2 = popt[2], popt[3]
        if tau2 > 0:
            return float(1.0 - tau1 / tau2)
    return 0.0
