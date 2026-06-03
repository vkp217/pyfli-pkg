# solver/shared_metrics.py

import numpy as np


def enforce_tau_ordering(popt, perr=None, pcov=None):
    """
    Enforce physical parameter constraints for a bi-exponential fit in place.

    1. Degenerate collapse
       - a1 > 0.999: component 2 is irrelevant  → set a1 = 1, tau2 = tau1
       - a1 < 0.001: component 1 is irrelevant  → set a1 = 0, tau1 = tau2

    2. Ordering: guarantee tau1 <= tau2; swap a1 = 1 - a1 when taus are swapped
       so the fractional intensity always tracks the corresponding lifetime.

    Parameters
    ----------
    popt : array-like, shape (5,)  [S, a1, tau1, tau2, offset]
    perr : 1-D array of std errors, or None
    pcov : 2-D covariance matrix, or None

    Returns
    -------
    popt, perr, pcov  (all modified in-place; same references returned)
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
    """
    Standard FLI goodness-of-fit statistics.

    Chi-squared uses the Pearson definition (denominator = model) which is the
    correct variance estimate under Poisson photon statistics.

    Parameters
    ----------
    final_model : 1-D array  evaluated model at fitted parameters
    d_fit       : 1-D array  measured decay (same length)
    n_params    : int         number of free parameters in the model

    Returns
    -------
    ssr, chi_sq, red_chi_sq, r_sq
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
    """
    Amplitude-weighted mean lifetime.

    For the model  S * (a1/tau1 * exp(-t/tau1) + (1-a1)/tau2 * exp(-t/tau2))
    the parameter a1 is the fractional photon-count of component 1, so
    <tau> = a1*tau1 + (1-a1)*tau2  is the amplitude-weighted mean.

    Returns tau (scalar) for a mono-exponential fit (len(popt) == 3).
    """
    if len(popt) == 5:
        return float(popt[1] * popt[2] + (1.0 - popt[1]) * popt[3])
    return float(popt[1])


def compute_fret_efficiency(popt):
    """
    FRET efficiency:  E = 1 - tau_DA / tau_D

    tau_DA = tau1  (FRET-quenched, shorter lifetime, index 2)
    tau_D  = tau2  (unquenched donor,  longer lifetime, index 3)

    enforce_tau_ordering guarantees tau1 <= tau2, so E is always in [0, 1).
    Returns 0 for a mono-exponential fit or when tau2 == 0.
    """
    if len(popt) == 5:
        tau1, tau2 = popt[2], popt[3]
        if tau2 > 0:
            return float(1.0 - tau1 / tau2)
    return 0.0
