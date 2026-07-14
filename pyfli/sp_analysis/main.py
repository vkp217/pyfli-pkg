"""
Provide high-level single-pixel simulation and reconstruction orchestration.

This module belongs to :mod:`pyfli.sp_analysis` and is part of PyFLI single-pixel camera
basis generation, acquisition simulation, and reconstruction solvers. Public API
includes functions :func:`run_reconstruction`.
"""

from __future__ import annotations
from typing import Any
from .solvers import LinearReconstructor, TVReconstructor
from .spad_solvers import SPADPoissonReconstructor


def run_reconstruction(
    measurements: Any,
    dmd_patterns: Any,
    h: Any,
    w: Any,
    t: Any,
    lam: Any,
    mode: str = "linear",
    differential: bool = True,
    alpha: float = 1.0,
    maxiter: int = 500,
) -> Any:
    """
    Reconstruct a 4D (x, y, T, Lambda) cube from DMD single-pixel measurements.

    Parameters
    ----------
    measurements  : ndarray
        (2M, T, Lambda) if differential else (M, T, Lambda) — raw SPAD counts.
    dmd_patterns  : ndarray
        (2M, H*W) if differential else (M, H*W) — DMD {0,1} patterns from
        BasisPatterns.generate_hadamard() or BasisPatterns.generate_fourier_dct().
    h, w          : int — spatial resolution
    t, lam        : int — TCSPC time bins and wavelength channels
    mode          : 'linear' | 'tv' | 'poisson'
        'linear'  — fast back-projection (Gaussian noise)
        'tv'      — L-BFGS-B TV minimization (Gaussian noise)
        'poisson' — L-BFGS-B Poisson + TV (integer photon counts, SPAD)
    differential  : bool — True for Hadamard differential DMD patterns (default)
    alpha         : float — TV regularization weight (tv/poisson modes)
    maxiter       : int   — solver iterations per (t, lambda) slice

    Returns
    -------
    cube : ndarray (H, W, T, Lambda)
    """
    if mode == "linear":
        engine = LinearReconstructor(h, w, t, lam, differential=differential)
    elif mode == "tv":
        engine = TVReconstructor(
            h, w, t, lam, differential=differential, alpha=alpha, maxiter=maxiter
        )
    elif mode == "poisson":
        engine = SPADPoissonReconstructor(
            h, w, t, lam, differential=differential, alpha=alpha, maxiter=maxiter
        )
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose: 'linear', 'tv', 'poisson'")

    return engine.reconstruct_4d(measurements, dmd_patterns)
