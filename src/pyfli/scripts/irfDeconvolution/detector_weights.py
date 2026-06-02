"""
detector_weights.py
===================

Detector-specific weight expressions for FLIM deconvolution.

Design principle
----------------
All three detectors (TCSPC, time-gated SPAD, time-gated ICCD) are reduced to a
single *weighted least-squares core* by mapping the raw measurement into the
"ideal intensity" domain lambda (expected photons / counts that the forward
model  lambda = Gate @ (h (x) f)  predicts), and supplying an inverse-variance
weight in that same domain.

Each detector therefore exposes exactly two functions:

    to_lambda(y, p)      measured  ->  estimate of the ideal intensity lambda
    lambda_weight(lam, y, p)   ->  inverse-variance weight w in the lambda domain

so the solver only ever sees  (lambda_obs, w)  and never needs to know which
sensor produced the data. The detector physics is fully contained here.

Variance models
---------------
TCSPC : Poisson, optional pile-up (Coates correction). Var(N_i) = lambda_i.
SPAD  : gated single-photon -> per-cycle Bernoulli -> Binomial over n_ex cycles.
        This is the SPAD analogue of pile-up (classic "1 photon per gate" limit).
ICCD  : compound Poisson-Gaussian. The MCP applies a random gain, inflating the
        shot-noise variance by the excess-noise factor F^2 (= 2 for exponential
        single-electron gain), plus additive read noise sigma_r.
        Var(y_adu) = F^2 * G0^2 * lambda + sigma_r^2.
"""

from dataclasses import dataclass
import numpy as np

EPS = 1e-9


# --------------------------------------------------------------------------- #
#  Parameter containers                                                        #
# --------------------------------------------------------------------------- #
@dataclass
class TCSPCParams:
    n_ex: float | None = None      # number of laser excitations; None -> no pile-up


@dataclass
class SPADParams:
    n_ex: float                    # excitation cycles per gate measurement


@dataclass
class ICCDParams:
    G0: float                      # system gain (ADU per photoelectron)
    F2: float = 2.0                # excess-noise factor squared (2.0 for MCP)
    sigma_r: float = 0.0           # read noise (ADU, rms)


# --------------------------------------------------------------------------- #
#  TCSPC : Poisson + optional pile-up                                          #
# --------------------------------------------------------------------------- #
def tcspc_to_lambda(y, p: TCSPCParams):
    """Map measured counts to an estimate of the un-piled-up ideal intensity."""
    y = np.asarray(y, float)
    if p.n_ex is None:
        return y.copy()
    # Coates inversion: recover the true total Lambda from the measured total n,
    # then redistribute by the measured shape. This linearises the pile-up bias.
    n = y.sum(-1, keepdims=True)
    frac = np.clip(n / p.n_ex, 0.0, 1.0 - 1e-6)
    Lambda_true = -p.n_ex * np.log1p(-frac)          # -n_ex ln(1 - n/n_ex)
    shape = y / np.maximum(n, EPS)
    return Lambda_true * shape


def tcspc_lambda_weight(lam, y, p: TCSPCParams):
    """Inverse variance in the lambda domain.

    No pile-up : Var(lambda) = lambda  -> w = 1/lambda  (Pearson / model weight).
    Pile-up    : delta-method on the Coates inversion gives, per bin,
                 Var(lambda_i) ~ lambda_i * (n_ex / (n_ex - n)) , a mild inflation
                 of the Poisson variance as the count rate approaches saturation.
    """
    lam = np.maximum(np.asarray(lam, float), EPS)
    if p.n_ex is None:
        return 1.0 / lam
    n = np.asarray(y, float).sum(-1, keepdims=True)
    inflate = p.n_ex / np.maximum(p.n_ex - n, EPS)
    return 1.0 / (lam * inflate)


# --------------------------------------------------------------------------- #
#  Time-gated SPAD : Binomial (single-photon-per-gate / dead-time limit)        #
# --------------------------------------------------------------------------- #
def spad_to_lambda(y, p: SPADParams):
    """Invert the per-cycle detection probability  p = 1 - exp(-mu)  to recover
    the mean photons per gate  lambda = n_ex * mu = -n_ex ln(1 - y/n_ex)."""
    y = np.asarray(y, float)
    frac = np.clip(y / p.n_ex, 0.0, 1.0 - 1e-6)
    return -p.n_ex * np.log1p(-frac)


def spad_lambda_weight(lam, y, p: SPADParams):
    """Delta-method variance of the inverted lambda.

    With detected counts  y ~ Binomial(n_ex, prob),  prob = 1 - exp(-lambda/n_ex),
    propagating through  lambda = -n_ex ln(1 - y/n_ex)  gives
        Var(lambda) ~ n_ex * y / (n_ex - y).
    Weight is its reciprocal.
    """
    y = np.asarray(y, float)
    var = p.n_ex * np.maximum(y, EPS) / np.maximum(p.n_ex - y, EPS)
    return 1.0 / np.maximum(var, EPS)


# --------------------------------------------------------------------------- #
#  Time-gated ICCD : compound Poisson-Gaussian (MCP excess noise + read noise)  #
# --------------------------------------------------------------------------- #
def iccd_to_lambda(y_adu, p: ICCDParams):
    """Map ADU back to photoelectrons: lambda = y_adu / G0."""
    return np.asarray(y_adu, float) / p.G0


def iccd_lambda_weight(lam, y_adu, p: ICCDParams):
    """Referred-to-input variance:
        Var(y_adu) = F^2 G0^2 lambda + sigma_r^2
        Var(lambda) = Var(y_adu)/G0^2 = F^2 lambda + (sigma_r/G0)^2
    so the read noise is suppressed by the gain while shot noise is inflated by F^2.
    """
    lam = np.maximum(np.asarray(lam, float), EPS)
    var = p.F2 * lam + (p.sigma_r / p.G0) ** 2
    return 1.0 / np.maximum(var, EPS)


# --------------------------------------------------------------------------- #
#  Generalized Anscombe transform (optional VST front-end for ICCD)            #
# --------------------------------------------------------------------------- #
def generalized_anscombe(y_adu, p: ICCDParams):
    """Variance-stabilising transform for the Poisson-Gaussian mixture.
    After this, z is approximately N(., 1) and plain (unweighted) LS applies.
    Effective gain alpha = G0 * F^2 carries the excess noise.
    """
    alpha = p.G0 * p.F2
    arg = alpha * np.asarray(y_adu, float) + (3.0 / 8.0) * alpha ** 2 + p.sigma_r ** 2
    return (2.0 / alpha) * np.sqrt(np.maximum(arg, 0.0))


# --------------------------------------------------------------------------- #
#  Dispatch table so the solver can stay detector-agnostic                     #
# --------------------------------------------------------------------------- #
DETECTORS = {
    "tcspc": (tcspc_to_lambda, tcspc_lambda_weight),
    "spad":  (spad_to_lambda,  spad_lambda_weight),
    "iccd":  (iccd_to_lambda,  iccd_lambda_weight),
}


def make_observation(y, detector, params):
    """Return (lambda_obs, weight) for any detector in one call."""
    to_lam, lam_w = DETECTORS[detector]
    lam_obs = to_lam(y, params)
    w = lam_w(np.maximum(lam_obs, EPS), y, params)
    return lam_obs, w
