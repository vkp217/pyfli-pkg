"""
locus.py
========
Build Single-Exponential Phasor Locus (SEPL) arrays — the parametric curve
traced in the (g, s) plane as τ sweeps from 0 to ∞.

The term "SEPL" (pronounced *sepal*) was introduced by Michalet 2021 for all
such curves, whether they are the canonical universal semicircle or deformed
variants arising from gating / binning / truncation.
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

from .config import AcquisitionConfig
from .phasors import phasor_from_config


# ──────────────────────────────────────────────────────────────────────────────
# tau grid
# ──────────────────────────────────────────────────────────────────────────────

def tau_grid(cfg: AcquisitionConfig, power: float = 1.5) -> NDArray[np.float64]:
    """
    Return a non-uniform τ grid in [tau_min_ns, tau_max_ns].

    The grid is denser at small τ (where the locus curves sharply near (1,0))
    using a power-law spacing.

    Parameters
    ----------
    cfg   : AcquisitionConfig
    power : float
        Exponent for the nonlinear spacing.  1.0 = linear; >1 = finer at small τ.

    Returns
    -------
    tau : 1-D ndarray  (length = cfg.n_tau_pts)
    """
    u = np.linspace(0.0, 1.0, cfg.n_tau_pts)
    return cfg.tau_min_ns + (cfg.tau_max_ns - cfg.tau_min_ns) * u ** power


# ──────────────────────────────────────────────────────────────────────────────
# main builder
# ──────────────────────────────────────────────────────────────────────────────

def build_locus(
    cfg: AcquisitionConfig,
    *,
    tau: NDArray[np.float64] | None = None,
    filter_finite: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute the full SEPL for the given acquisition configuration.

    Parameters
    ----------
    cfg           : AcquisitionConfig
        Fully describes the experiment (mode, T, harmonic, gates, …).
    tau           : 1-D array, optional
        Custom τ grid in ns.  If *None*, ``tau_grid(cfg)`` is used.
    filter_finite : bool
        If True (default), remove any (g, s) pairs where either component is
        non-finite (NaN / ±Inf).  Useful for modes that become undefined at
        τ → 0 for certain parameter sets.

    Returns
    -------
    g   : 1-D ndarray
        Real phasor coordinate.
    s   : 1-D ndarray
        Imaginary phasor coordinate.
    tau : 1-D ndarray
        Corresponding lifetime values in ns.
    """
    if tau is None:
        tau = tau_grid(cfg)

    g, s = phasor_from_config(tau, cfg)

    if filter_finite:
        mask = np.isfinite(g) & np.isfinite(s)
        g, s, tau = g[mask], s[mask], tau[mask]

    return g, s, tau


# ──────────────────────────────────────────────────────────────────────────────
# multi-config convenience
# ──────────────────────────────────────────────────────────────────────────────

def build_loci(
    cfgs: list[AcquisitionConfig],
    **kwargs,
) -> list[tuple[NDArray, NDArray, NDArray]]:
    """
    Compute SEPL curves for a list of configs in one call.

    Useful for overlaying several acquisition scenarios on the same plot.

    Parameters
    ----------
    cfgs   : list[AcquisitionConfig]
    kwargs : passed verbatim to ``build_locus``.

    Returns
    -------
    list of (g, s, tau) tuples, same order as *cfgs*.
    """
    return [build_locus(c, **kwargs) for c in cfgs]


# ──────────────────────────────────────────────────────────────────────────────
# circle geometry helpers  (universal semicircle reference)
# ──────────────────────────────────────────────────────────────────────────────

def universal_semicircle(n_pts: int = 300) -> tuple[NDArray, NDArray]:
    """
    Return (g, s) coordinates of the universal semicircle.

    The universal semicircle is defined by
        (g − ½)² + s² = (½)²,  s ≥ 0
    and is parameterised by the angle θ ∈ [0, π]:
        g = ½ + ½·cos(θ),  s = ½·sin(θ)

    Parameters
    ----------
    n_pts : int
        Number of samples (default 300).

    Returns
    -------
    g, s : 1-D ndarray
    """
    theta = np.linspace(0.0, np.pi, n_pts)
    g = 0.5 + 0.5 * np.cos(theta)
    s = 0.5 * np.sin(theta)
    return g, s


def sepl_center_radius_discrete(
    cfg: AcquisitionConfig,
) -> tuple[float, float, float]:
    """
    Analytically compute the circle centre (gc, sc) and radius r of the
    discrete SEPL  L_N  (Michalet 2021, Sec. III C 1).

    For a binned PSED with N bins and harmonic n, the phasor traces:

        z_N(τ) = (1 − x) / (1 − x·e^{iφ})

    where  x = e^{−T/(N·τ)}  and  φ = 2π·n/N.

    This Möbius transform of the real interval (0, 1) traces an arc of a
    circle with parameters (derived analytically by equating coefficients
    in the expanded circle equation):

        gc = 1/2
        sc = −(1/2) · tan(φ/2)
        r  = √(gc² + sc²) = (1/2) / cos(φ/2)

    The arc converges to the universal semicircle as N → ∞ (φ → 0).

    Parameters
    ----------
    cfg : AcquisitionConfig
        Requires:  N_bins, harmonic.

    Returns
    -------
    gc, sc, r : float
        Circle centre coordinates and radius.
    """
    phi = 2.0 * np.pi * cfg.harmonic / cfg.N_bins
    half_phi = phi / 2.0

    if abs(np.cos(half_phi)) < 1e-12:
        # Degenerate case (e.g. N=2, n=1): the arc degenerates
        return 0.5, 0.0, 0.5

    gc = 0.5
    sc = -0.5 * np.tan(half_phi)
    r  = np.sqrt(gc ** 2 + sc ** 2)   # = 0.5 / cos(phi/2)
    return gc, sc, r
