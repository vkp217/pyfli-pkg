from dataclasses import dataclass
import numpy as np

EPS = 1e-9


@dataclass
class TCSPCParams:
    n_ex: float | None = None


@dataclass
class SPADParams:
    n_ex: float


@dataclass
class ICCDParams:
    G0: float
    F2: float = 2.0
    sigma_r: float = 0.0


def tcspc_to_lambda(y, p: TCSPCParams):
    y = np.asarray(y, float)
    if p.n_ex is None:
        return y.copy()
    n = y.sum(-1, keepdims=True)
    frac = np.clip(n / p.n_ex, 0.0, 1.0 - 1e-6)
    Lambda_true = -p.n_ex * np.log1p(-frac)
    shape = y / np.maximum(n, EPS)
    return Lambda_true * shape


def tcspc_lambda_weight(lam, y, p: TCSPCParams):
    lam = np.maximum(np.asarray(lam, float), EPS)
    if p.n_ex is None:
        return 1.0 / lam
    n = np.asarray(y, float).sum(-1, keepdims=True)
    inflate = p.n_ex / np.maximum(p.n_ex - n, EPS)
    return 1.0 / (lam * inflate)


def spad_to_lambda(y, p: SPADParams):
    y = np.asarray(y, float)
    frac = np.clip(y / p.n_ex, 0.0, 1.0 - 1e-6)
    return -p.n_ex * np.log1p(-frac)


def spad_lambda_weight(lam, y, p: SPADParams):
    y = np.asarray(y, float)
    var = p.n_ex * np.maximum(y, EPS) / np.maximum(p.n_ex - y, EPS)
    return 1.0 / np.maximum(var, EPS)


def iccd_to_lambda(y_adu, p: ICCDParams):
    return np.asarray(y_adu, float) / p.G0


def iccd_lambda_weight(lam, y_adu, p: ICCDParams):
    lam = np.maximum(np.asarray(lam, float), EPS)
    var = p.F2 * lam + (p.sigma_r / p.G0) ** 2
    return 1.0 / np.maximum(var, EPS)


def generalized_anscombe(y_adu, p: ICCDParams):
    alpha = p.G0 * p.F2
    arg = alpha * np.asarray(y_adu, float) + (3.0 / 8.0) * alpha ** 2 + p.sigma_r ** 2
    return (2.0 / alpha) * np.sqrt(np.maximum(arg, 0.0))


DETECTORS = {
    "tcspc": (tcspc_to_lambda, tcspc_lambda_weight),
    "spad":  (spad_to_lambda,  spad_lambda_weight),
    "iccd":  (iccd_to_lambda,  iccd_lambda_weight),
}


def make_observation(y, detector, params):
    to_lam, lam_w = DETECTORS[detector]
    lam_obs = to_lam(y, params)
    w = lam_w(np.maximum(lam_obs, EPS), y, params)
    return lam_obs, w
