"""IRF-convolved mono-/bi-exponential forward model for FLI/FLIM decay curves.

Provides the analytical decay kernel and its convolution with a normalised
instrument response function (IRF), used by the NLSF and MLE fitters to
evaluate the forward model for a given parameter vector.
"""

import numpy as np

_EPS = 1e-8


def decay_kernel(t: np.ndarray, params, model_type: str,
                 h_shift: float = 0.0) -> tuple:
    """Return (kernel, v_shift).

    The temporal delay h_shift (in the same units as t, i.e. ns) is applied
    directly to the exponential argument so no IRF array manipulation is needed.
    """
    # Clamp to ≥0: before the shift the kernel is physically zero; clamping also
    # prevents exp(-t_eff/tau) overflow when h_shift > t and tau is small.
    t_eff = np.maximum(t - h_shift, 0.0)

    if model_type == 'mono-exponential':
        S, tau, v_shift = params
        tau_safe = np.clip(tau, _EPS, None)
        kernel = (S / tau_safe) * np.exp(-t_eff / tau_safe)
    else:
        S, a1, tau1, tau2, v_shift = params
        t1_safe = np.clip(tau1, _EPS, None)
        t2_safe = np.clip(tau2, _EPS, None)
        kernel = S * (
            (a1 / t1_safe) * np.exp(-t_eff / t1_safe) +
            ((1.0 - a1) / t2_safe) * np.exp(-t_eff / t2_safe)
        )
    return kernel, float(v_shift)


def model_numpy(
    t: np.ndarray,
    irf: np.ndarray,
    params,
    model_type: str,
) -> np.ndarray:
    """Evaluate the IRF-convolved mono- or bi-exponential decay model.

    Splits ``params`` into the temporal shift (last element) and the decay
    kernel parameters, evaluates the decay kernel via :func:`decay_kernel`,
    normalises the IRF to unit area, and convolves the kernel with the
    normalised IRF (linear/full convolution truncated to the length of ``t``).

    Args:
        t: Time axis (ns) at which to evaluate the model.
        irf: Instrument response function samples, same length as the kernel.
        params: Parameter vector. Last element is the temporal shift
            (``h_shift``, ns); the preceding elements are the decay-kernel
            parameters (see :func:`decay_kernel`).
        model_type: Either ``'mono-exponential'`` or ``'bi-exponential'``.

    Returns:
        numpy.ndarray: The IRF-convolved decay curve, same length as ``t``,
        with the vertical offset (``v_shift``) added.
    """
    params = np.asarray(params, dtype=float)

    h_shift       = float(params[-1])
    kernel_params = params[:-1]

    kernel, v_shift = decay_kernel(t, kernel_params, model_type, h_shift=h_shift)

    irf = np.asarray(irf, dtype=float)
    irf_sum = irf.sum()
    irf_norm = irf / irf_sum if irf_sum > 0 else irf

    convolved = np.convolve(kernel, irf_norm, mode='full')[:len(t)]
    return convolved + v_shift
