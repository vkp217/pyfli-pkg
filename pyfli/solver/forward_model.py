"""
Evaluate exponential decay kernels and convolved NumPy forward models.

This module belongs to :mod:`pyfli.solver` and is part of PyFLI least-squares, maximum-
likelihood, CPU, GPU, binned, and global FLI fitting routines. Public API includes
functions :func:`decay_kernel` and :func:`model_numpy`.
"""

from typing import Any

import numpy as np

from pyfli.reconstruction.common_reconstruct import (
    bi_reconstruction,
    mono_reconstruction,
)

_EPS = 1e-8


def decay_kernel(
    t: np.ndarray, params: Any, model_type: str, h_shift: float = 0.0
) -> tuple:
    """Return (kernel, v_shift).

    The temporal delay h_shift (in the same units as t, i.e. ns) is applied
    directly to the exponential argument so no IRF array manipulation is needed.
    """
    # Clamp to ≥0: before the shift the kernel is physically zero; clamping also
    # prevents exp(-t_eff/tau) overflow when h_shift > t and tau is small.
    t_eff = np.maximum(t - h_shift, 0.0)

    if model_type == "mono-exponential":
        S, tau, v_shift = params
        tau_safe = np.clip(tau, _EPS, None)
        kernel = mono_reconstruction(t_eff, tau_safe, S)
    else:
        S, a1, tau1, tau2, v_shift = params
        t1_safe = np.clip(tau1, _EPS, None)
        t2_safe = np.clip(tau2, _EPS, None)
        kernel = bi_reconstruction(t_eff, t1_safe, t2_safe, S * a1, S * (1.0 - a1))
    return kernel, float(v_shift)


def model_numpy(
    t: np.ndarray,
    irf: np.ndarray,
    params: Any,
    model_type: str,
) -> np.ndarray:
    """
    Evaluate the NumPy FLI forward model.

    Parameters
    ----------
    t : np.ndarray
        Time axis or acquisition period used by the calculation.
    irf : np.ndarray
        Instrument response function aligned with the decay signal.
    params : Any
        Model, detector, or plotting parameters used by the routine.
    model_type : str
        FLI model family, such as mono- or bi-exponential.

    Returns
    -------
    np.ndarray
        Model decay evaluated with NumPy for the supplied parameters.
    """
    params = np.asarray(params, dtype=float)

    h_shift = float(params[-1])
    kernel_params = params[:-1]

    kernel, v_shift = decay_kernel(t, kernel_params, model_type, h_shift=h_shift)

    irf = np.asarray(irf, dtype=float)
    irf_sum = irf.sum()
    irf_norm = irf / irf_sum if irf_sum > 0 else irf

    convolved = np.convolve(kernel, irf_norm, mode="full")[: len(t)]
    return convolved + v_shift
