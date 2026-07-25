"""
Convert detector observations to expected photon rates and inverse-variance weights.

This module belongs to :mod:`pyfli.irf_deconvolution` and is part of PyFLI detector-
aware IRF deconvolution and joint FLIM fitting utilities. Public API includes classes
:class:`TCSPCParams`, :class:`SPADParams`, and :class:`ICCDParams`; functions
:func:`tcspc_to_lambda`, :func:`tcspc_lambda_weight`, :func:`spad_to_lambda`,
:func:`spad_lambda_weight`, :func:`iccd_to_lambda`, :func:`iccd_lambda_weight`,
:func:`generalized_anscombe`, and :func:`make_observation`.
"""

from typing import Any
from dataclasses import dataclass

import numpy as np

EPS = 1e-9


@dataclass
class TCSPCParams:
    """
    Store TCSPC detector weighting parameters. The optional excitation-count value is
    used to correct pile-up and inflate variance estimates.

    Parameters
    ----------
    n_ex : float | None
        Number of excitation opportunities used for pile-up or detector corrections.
    """

    n_ex: float | None = None


@dataclass
class SPADParams:
    """
    Store SPAD detector weighting parameters. The excitation-count value controls the
    conversion from observed binary detections to expected photon rates.

    Parameters
    ----------
    n_ex : float
        Number of excitation opportunities used for pile-up or detector corrections.
    """

    n_ex: float


@dataclass
class ICCDParams:
    """
    Store ICCD detector weighting parameters. Gain, excess-noise factor, and read-noise
    standard deviation define the observation model used by IRF deconvolution.

    Parameters
    ----------
    G0 : float
        ICCD gain factor used by the detector observation model.
    F2 : float
        ICCD excess-noise factor used in variance estimates.
    sigma_r : float
        Read-noise standard deviation used in the detector observation model.
    """

    G0: float
    F2: float = 2.0
    sigma_r: float = 0.0


def tcspc_to_lambda(y: np.ndarray, p: TCSPCParams) -> Any:
    """
    Run the TCSPC to lambda routine.

    Parameters
    ----------
    y : np.ndarray
        Observed signal, target data, or coordinate array.
    p : TCSPCParams
        Detector parameter object or fitted parameter vector.

    Returns
    -------
    Any
        Object produced by TCSPC to lambda.
    """
    y = np.asarray(y, float)
    if p.n_ex is None:
        return y.copy()
    n = y.sum(-1, keepdims=True)
    frac = np.clip(n / p.n_ex, 0.0, 1.0 - 1e-6)
    Lambda_true = -p.n_ex * np.log1p(-frac)
    shape = y / np.maximum(n, EPS)
    return Lambda_true * shape


def tcspc_lambda_weight(lam: np.ndarray, y: np.ndarray, p: TCSPCParams) -> Any:
    """
    Run the TCSPC lambda weight routine.

    Parameters
    ----------
    lam : np.ndarray
        Wavelength, spectral axis, or expected photon-rate array.
    y : np.ndarray
        Observed signal, target data, or coordinate array.
    p : TCSPCParams
        Detector parameter object or fitted parameter vector.

    Returns
    -------
    Any
        Object produced by TCSPC lambda weight.
    """
    lam = np.maximum(np.asarray(lam, float), EPS)
    if p.n_ex is None:
        return 1.0 / lam
    n = np.asarray(y, float).sum(-1, keepdims=True)
    inflate = p.n_ex / np.maximum(p.n_ex - n, EPS)
    return 1.0 / (lam * inflate)


def spad_to_lambda(y: np.ndarray, p: SPADParams) -> Any:
    """
    Run the SPAD to lambda routine.

    Parameters
    ----------
    y : np.ndarray
        Observed signal, target data, or coordinate array.
    p : SPADParams
        Detector parameter object or fitted parameter vector.

    Returns
    -------
    Any
        Object produced by SPAD to lambda.
    """
    y = np.asarray(y, float)
    frac = np.clip(y / p.n_ex, 0.0, 1.0 - 1e-6)
    return -p.n_ex * np.log1p(-frac)


def spad_lambda_weight(lam: np.ndarray, y: np.ndarray, p: SPADParams) -> Any:
    """
    Run the SPAD lambda weight routine.

    Parameters
    ----------
    lam : np.ndarray
        Wavelength, spectral axis, or expected photon-rate array.
    y : np.ndarray
        Observed signal, target data, or coordinate array.
    p : SPADParams
        Detector parameter object or fitted parameter vector.

    Returns
    -------
    Any
        Object produced by SPAD lambda weight.
    """
    y = np.asarray(y, float)
    var = p.n_ex * np.maximum(y, EPS) / np.maximum(p.n_ex - y, EPS)
    return 1.0 / np.maximum(var, EPS)


def iccd_to_lambda(y_adu: np.ndarray, p: ICCDParams) -> Any:
    """
    Run the ICCD to lambda routine.

    Parameters
    ----------
    y_adu : np.ndarray
        ICCD observation in analog-to-digital units.
    p : ICCDParams
        Detector parameter object or fitted parameter vector.

    Returns
    -------
    Any
        Object produced by ICCD to lambda.
    """
    return np.asarray(y_adu, float) / p.G0


def iccd_lambda_weight(lam: np.ndarray, y_adu: np.ndarray, p: ICCDParams) -> Any:
    """
    Run the ICCD lambda weight routine.

    Parameters
    ----------
    lam : np.ndarray
        Wavelength, spectral axis, or expected photon-rate array.
    y_adu : np.ndarray
        ICCD observation in analog-to-digital units.
    p : ICCDParams
        Detector parameter object or fitted parameter vector.

    Returns
    -------
    Any
        Object produced by ICCD lambda weight.
    """
    lam = np.maximum(np.asarray(lam, float), EPS)
    var = p.F2 * lam + (p.sigma_r / p.G0) ** 2
    return 1.0 / np.maximum(var, EPS)


def generalized_anscombe(y_adu: np.ndarray, p: ICCDParams) -> Any:
    """
    Run the generalized anscombe routine.

    Parameters
    ----------
    y_adu : np.ndarray
        ICCD observation in analog-to-digital units.
    p : ICCDParams
        Detector parameter object or fitted parameter vector.

    Returns
    -------
    Any
        Object produced by generalized anscombe.
    """
    alpha = p.G0 * p.F2
    arg = alpha * np.asarray(y_adu, float) + (3.0 / 8.0) * alpha**2 + p.sigma_r**2
    return (2.0 / alpha) * np.sqrt(np.maximum(arg, 0.0))


DETECTORS = {
    "tcspc": (tcspc_to_lambda, tcspc_lambda_weight),
    "spad": (spad_to_lambda, spad_lambda_weight),
    "iccd": (iccd_to_lambda, iccd_lambda_weight),
}


def make_observation(y: np.ndarray, detector: str, params: Any) -> tuple[Any, ...]:
    """
    Create observation.

    Parameters
    ----------
    y : np.ndarray
        Observed signal, target data, or coordinate array.
    detector : str
        Detector model name used to select weighting or conversion logic.
    params : Any
        Model, detector, or plotting parameters used by the routine.

    Returns
    -------
    tuple[Any, ...]
        Tuple containing simulated observations and associated ground-truth arrays.
    """
    to_lam, lam_w = DETECTORS[detector]
    lam_obs = to_lam(y, params)
    w = lam_w(np.maximum(lam_obs, EPS), y, params)
    return lam_obs, w
