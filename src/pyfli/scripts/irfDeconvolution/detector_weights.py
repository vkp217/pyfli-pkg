"""Detector-specific noise models for FLIM photon-counting data.

Converts raw detector readouts (TCSPC photon counts, SPAD event counts,
or ICCD analog-to-digital units) into estimated Poisson photon rates
(``lambda``) together with inverse-variance weights suitable for
weighted least-squares decay fitting, plus a generalized Anscombe
variance-stabilizing transform for ICCD data.
"""

from dataclasses import dataclass
import numpy as np

EPS = 1e-9


@dataclass
class TCSPCParams:
    """Parameters for a time-correlated single photon counting (TCSPC) detector.

    Attributes:
        n_ex: Number of excitation pulses (laser sync events) used to
            correct for detector pileup. If ``None``, pileup correction
            is disabled and raw counts are treated as the Poisson rate
            directly.
    """

    n_ex: float | None = None


@dataclass
class SPADParams:
    """Parameters for a single-photon avalanche diode (SPAD) detector.

    Attributes:
        n_ex: Number of excitation pulses/trials underlying the binomial
            detection model used to recover the Poisson rate from the
            observed hit counts.
    """

    n_ex: float


@dataclass
class ICCDParams:
    """Parameters for an intensified CCD (ICCD) detector.

    Attributes:
        G0: Overall gain converting photon rate to analog-to-digital
            units (ADU).
        F2: Excess noise factor of the image intensifier, applied as a
            multiplicative inflation of the Poisson shot-noise variance.
        sigma_r: Standard deviation of additive read noise, in ADU.
    """

    G0: float
    F2: float = 2.0
    sigma_r: float = 0.0


def tcspc_to_lambda(y, p: TCSPCParams):
    """Convert TCSPC photon counts to an estimated Poisson rate.

    If ``p.n_ex`` is set, applies the classical TCSPC pileup correction
    ``Lambda = -n_ex * log(1 - n / n_ex)``, where ``n`` is the total
    count per pixel/histogram, and redistributes the corrected total
    back across bins according to the observed count shape. If
    ``p.n_ex`` is ``None``, the counts are returned unchanged (no
    pileup correction).

    Args:
        y: Array of raw photon counts, with time/gate bins along the
            last axis.
        p: TCSPC detector parameters.

    Returns:
        Array of the same shape as ``y`` containing the estimated
        (pileup-corrected) Poisson rate.
    """
    y = np.asarray(y, float)
    if p.n_ex is None:
        return y.copy()
    n = y.sum(-1, keepdims=True)
    frac = np.clip(n / p.n_ex, 0.0, 1.0 - 1e-6)
    Lambda_true = -p.n_ex * np.log1p(-frac)
    shape = y / np.maximum(n, EPS)
    return Lambda_true * shape


def tcspc_lambda_weight(lam, y, p: TCSPCParams):
    """Compute inverse-variance weights for TCSPC-derived rate estimates.

    Without pileup correction (``p.n_ex is None``) this is the standard
    Poisson weight ``1 / lambda``. With pileup correction, the Poisson
    variance is inflated by a factor ``n_ex / (n_ex - n)`` that accounts
    for the reduced effective count budget near saturation.

    Args:
        lam: Estimated Poisson rate (e.g. from `tcspc_to_lambda`).
        y: Raw photon counts corresponding to ``lam``.
        p: TCSPC detector parameters.

    Returns:
        Array of inverse-variance weights, same shape as ``lam``.
    """
    lam = np.maximum(np.asarray(lam, float), EPS)
    if p.n_ex is None:
        return 1.0 / lam
    n = np.asarray(y, float).sum(-1, keepdims=True)
    inflate = p.n_ex / np.maximum(p.n_ex - n, EPS)
    return 1.0 / (lam * inflate)


def spad_to_lambda(y, p: SPADParams):
    """Convert SPAD hit counts to an estimated Poisson rate.

    Inverts the binomial detection model
    ``P(hit) = 1 - exp(-lambda / n_ex)`` to recover ``lambda`` from the
    observed fraction of hits per exposure.

    Args:
        y: Array of raw SPAD hit counts.
        p: SPAD detector parameters.

    Returns:
        Array of the same shape as ``y`` containing the estimated
        Poisson rate.
    """
    y = np.asarray(y, float)
    frac = np.clip(y / p.n_ex, 0.0, 1.0 - 1e-6)
    return -p.n_ex * np.log1p(-frac)


def spad_lambda_weight(lam, y, p: SPADParams):
    """Compute inverse-variance weights for SPAD-derived rate estimates.

    Uses the binomial variance of the hit count,
    ``var = n_ex * y / (n_ex - y)``, propagated to the recovered rate.

    Args:
        lam: Estimated Poisson rate (unused directly; kept for interface
            consistency with the other detector weight functions).
        y: Raw SPAD hit counts.
        p: SPAD detector parameters.

    Returns:
        Array of inverse-variance weights, same shape as ``y``.
    """
    y = np.asarray(y, float)
    var = p.n_ex * np.maximum(y, EPS) / np.maximum(p.n_ex - y, EPS)
    return 1.0 / np.maximum(var, EPS)


def iccd_to_lambda(y_adu, p: ICCDParams):
    """Convert ICCD analog-to-digital counts to an estimated Poisson rate.

    Args:
        y_adu: Raw ICCD readout, in analog-to-digital units (ADU).
        p: ICCD detector parameters.

    Returns:
        Array of the same shape as ``y_adu`` containing the estimated
        photon rate (``y_adu / G0``).
    """
    return np.asarray(y_adu, float) / p.G0


def iccd_lambda_weight(lam, y_adu, p: ICCDParams):
    """Compute inverse-variance weights for ICCD-derived rate estimates.

    Combines excess-noise-inflated Poisson shot noise
    (``F2 * lambda``) with additive read noise
    (``(sigma_r / G0) ** 2``).

    Args:
        lam: Estimated photon rate (e.g. from `iccd_to_lambda`).
        y_adu: Raw ICCD readout, in ADU (unused directly; kept for
            interface consistency with the other detector weight
            functions).
        p: ICCD detector parameters.

    Returns:
        Array of inverse-variance weights, same shape as ``lam``.
    """
    lam = np.maximum(np.asarray(lam, float), EPS)
    var = p.F2 * lam + (p.sigma_r / p.G0) ** 2
    return 1.0 / np.maximum(var, EPS)


def generalized_anscombe(y_adu, p: ICCDParams):
    """Apply the generalized Anscombe variance-stabilizing transform.

    Maps ICCD ADU readout, which follows a Poisson-Gaussian mixed noise
    model (Poisson shot noise inflated by excess noise factor ``F2``,
    plus additive Gaussian read noise ``sigma_r``), to a variable with
    approximately unit variance, using
    ``2/alpha * sqrt(alpha*y + 3/8*alpha**2 + sigma_r**2)`` with
    ``alpha = G0 * F2``.

    Args:
        y_adu: Raw ICCD readout, in ADU.
        p: ICCD detector parameters.

    Returns:
        Array of the same shape as ``y_adu`` containing the
        variance-stabilized signal.
    """
    alpha = p.G0 * p.F2
    arg = alpha * np.asarray(y_adu, float) + (3.0 / 8.0) * alpha ** 2 + p.sigma_r ** 2
    return (2.0 / alpha) * np.sqrt(np.maximum(arg, 0.0))


DETECTORS = {
    "tcspc": (tcspc_to_lambda, tcspc_lambda_weight),
    "spad":  (spad_to_lambda,  spad_lambda_weight),
    "iccd":  (iccd_to_lambda,  iccd_lambda_weight),
}


def make_observation(y, detector, params):
    """Convert raw detector counts to a weighted Poisson-rate observation.

    Looks up the appropriate rate-conversion and weight functions for
    ``detector`` in `DETECTORS` and applies them to ``y``.

    Args:
        y: Raw detector readout (counts or ADU depending on detector).
        detector: Detector name, one of ``"tcspc"``, ``"spad"``, or
            ``"iccd"`` (keys of `DETECTORS`).
        params: Detector parameters instance matching ``detector``
            (`TCSPCParams`, `SPADParams`, or `ICCDParams`).

    Returns:
        Tuple ``(lam_obs, w)`` of the estimated Poisson rate and its
        corresponding inverse-variance weight, both arrays the same
        shape as ``y``.
    """
    to_lam, lam_w = DETECTORS[detector]
    lam_obs = to_lam(y, params)
    w = lam_w(np.maximum(lam_obs, EPS), y, params)
    return lam_obs, w
