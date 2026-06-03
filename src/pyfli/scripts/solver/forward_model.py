# solver/forward_model.py

import numpy as np
"""
Parameter convention
--------------------
mono-exponential : [S, tau,       offset]
bi-exponential   : [S, a1, tau1, tau2, offset]

    S      — total fluorescence photon count (window-corrected integral)
    a1     — fractional photon-count of component 1  (intensity fraction, ∈ (0,1))
    tau1   — shorter lifetime  (tau1 ≤ tau2, enforced by enforce_tau_ordering)
    tau2   — longer  lifetime
    offset — background / dark counts per time bin

The IRF is always normalised to sum = 1 before convolution.
"""


_EPS = 1e-8   # lifetime floor to avoid division by zero


def decay_kernel(
    t: np.ndarray,
    params,
    model_type: str,
) -> tuple:
    """
    Pre-convolution fluorescence decay and scalar background.

    Parameters
    ----------
    t          : (T,) time axis [ns, or any consistent unit]
    params     : sequence — see module-level parameter convention
    model_type : ``'mono-exponential'`` | ``'bi-exponential'``
    Returns
    -------
    kernel : (T,) ndarray  — decay signal before IRF convolution
    offset : float         — additive background per time bin
    """
    if model_type == 'mono-exponential':
        S, tau, offset = params
        tau_safe = np.clip(tau, _EPS, None)
        kernel = (S / tau_safe) * np.exp(-t / tau_safe)
    else:
        S, a1, tau1, tau2, offset = params
        t1_safe = np.clip(tau1, _EPS, None)
        t2_safe = np.clip(tau2, _EPS, None)
        kernel = S * (
            (a1 / t1_safe) * np.exp(-t / t1_safe) +
            ((1.0 - a1) / t2_safe) * np.exp(-t / t2_safe)
        )
    return kernel, float(offset)


def model_numpy(
    t: np.ndarray,
    irf: np.ndarray,
    params,
    model_type: str,
) -> np.ndarray:
    """
    Full NumPy forward model: decay_kernel ⊛ irf_norm + offset.

    Parameters
    ----------
    t          : (T,) time axis
    irf        : (T,) instrument response function (raw or pre-normalised)
    params     : fitted parameters — see module-level parameter convention
    model_type : ``'mono-exponential'`` | ``'bi-exponential'``

    Returns
    -------
    y : (T,) ndarray — modelled signal
    """
    kernel, offset = decay_kernel(t, params, model_type)

    irf = np.asarray(irf, dtype=float)
    irf_sum = irf.sum()
    if irf_sum > 0:
        irf = irf / irf_sum

    convolved = np.convolve(kernel, irf, mode='full')[:len(t)]
    return convolved + offset
