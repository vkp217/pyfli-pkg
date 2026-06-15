
import numpy as np
"""
Parameter convention
--------------------
mono-exponential : [S, tau,             offset, h_shift]
bi-exponential   : [S, a1, tau1, tau2,  offset, h_shift]

    S        — total fluorescence photon count (window-corrected integral)
    a1       — fractional photon-count of component 1  (intensity fraction, ∈ (0,1))
    tau1     — shorter lifetime  (tau1 ≤ tau2, enforced by enforce_tau_ordering)
    tau2     — longer  lifetime
    offset   — background / dark counts per time bin
    h_shift  — fractional bin shift applied to the IRF before convolution;
               positive = IRF delayed (shifted right along time axis); units: bins

IRF shift method (passed as ``shift_method`` to ``model_numpy``):
    'zero_pad' (default) — round h_shift to the nearest integer, then shift with
                           zero-padding.  No wrap-around.  Only integer values are
                           effective; scipy finite-differences will see zero gradient
                           for sub-integer perturbations of h_shift.
    'fourier'            — exact sub-bin FFT phase-rotation shift.  Assumes the IRF
                           decays to zero before the end of the array (avoids the
                           circular-wrap artefact that np.roll produces).
    'interp'             — sub-bin linear-interpolation shift.  Out-of-window samples
                           are zero; gradient is correct but piecewise linear.

The shifted IRF is normalised to sum = 1 before convolution.
"""

_EPS = 1e-8

def _apply_irf_shift(irf: np.ndarray, h_shift: float,
                     t: np.ndarray, method: str) -> np.ndarray:
    if method == 'zero_pad':
        h = int(round(h_shift))
        if h == 0:
            return irf.copy()
        out = np.zeros_like(irf)
        if h > 0:
            out[h:] = irf[:-h]
        else:
            out[:h] = irf[-h:]
        return out

    elif method == 'fourier':
        N = len(irf)
        freqs = np.fft.rfftfreq(N)
        phase = np.exp(-2j * np.pi * freqs * h_shift)
        shifted = np.fft.irfft(np.fft.rfft(irf) * phase, n=N)
        return shifted.clip(0)

    else:
        dt = t[1] - t[0] if len(t) > 1 else 1.0
        return np.interp(t - h_shift * dt, t, irf, left=0.0, right=0.0)

def decay_kernel(
    t: np.ndarray,
    params,
    model_type: str,
) -> tuple:
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
    shift_method: str = 'zero_pad',
) -> np.ndarray:
    params = np.asarray(params, dtype=float)

    h_shift = float(params[-1])
    kernel_params = params[:-1]

    kernel, offset = decay_kernel(t, kernel_params, model_type)

    irf = np.asarray(irf, dtype=float)
    irf_shifted = _apply_irf_shift(irf, h_shift, t, shift_method)

    irf_sum = irf_shifted.sum()
    if irf_sum > 0:
        irf_shifted = irf_shifted / irf_sum

    convolved = np.convolve(kernel, irf_shifted, mode='full')[:len(t)]
    return convolved + offset
