"""
lifetimes.py
============
Extract fluorescence lifetimes from phasor coordinates (g, s).

For single-exponential species on the universal semicircle, two equivalent
estimators exist — phase lifetime and modulus lifetime.  For species off
the semicircle (multi-exponential, gated, truncated), both estimators are
biased and corrected expressions must be used (Michalet 2021, Sec. VII).

References
----------
    Michalet X., AIP Advances 11, 035331 (2021), Sec. VII.
    ISS Technical Note: "FLIM Analysis using the Phasor Plots" Eq. 10.
"""

from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .config import AcquisitionConfig


# ──────────────────────────────────────────────────────────────────────────────
# Standard estimators
# ──────────────────────────────────────────────────────────────────────────────

def phase_lifetime(
    g: ArrayLike,
    s: ArrayLike,
    cfg: AcquisitionConfig,
) -> NDArray[np.float64]:
    """
    Phase (angular) lifetime from phasor coordinates.

    Derived from  φ = arctan(s/g)  and  tan(φ) = ωτ_φ:

        τ_φ = tan(φ) / ω  =  s / (g · ω)

    Valid exactly for single-exponential species on the universal semicircle.

    Parameters
    ----------
    g, s : array_like
        Phasor coordinates.
    cfg  : AcquisitionConfig
        Requires:  omega.

    Returns
    -------
    tau_phi : ndarray  (ns)
    """
    g = np.asarray(g, dtype=float)
    s = np.asarray(s, dtype=float)
    return s / (g * cfg.omega)


def modulus_lifetime(
    g: ArrayLike,
    s: ArrayLike,
    cfg: AcquisitionConfig,
) -> NDArray[np.float64]:
    """
    Modulus (demodulation) lifetime from phasor coordinates.

    Derived from  m = |z| = 1/√(1 + ω²τ_m²):

        τ_m = √(1/m² − 1) / ω  =  √( (1 − g² − s²) / (g² + s²) ) / ω

    Valid exactly for single-exponential species on the universal semicircle.

    Parameters
    ----------
    g, s : array_like
    cfg  : AcquisitionConfig

    Returns
    -------
    tau_m : ndarray  (ns)
    """
    g = np.asarray(g, dtype=float)
    s = np.asarray(s, dtype=float)
    m2 = g ** 2 + s ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(m2 > 0, (1.0 - m2) / m2, np.nan)
    return np.sqrt(np.maximum(ratio, 0.0)) / cfg.omega


# ──────────────────────────────────────────────────────────────────────────────
# Combined / direct inversion
# ──────────────────────────────────────────────────────────────────────────────

def lifetime_from_phasor(
    g: ArrayLike,
    s: ArrayLike,
    cfg: AcquisitionConfig,
    method: str = "phase",
) -> NDArray[np.float64]:
    """
    Convenience wrapper: estimate lifetime from phasor coordinates.

    Parameters
    ----------
    g, s   : array_like
    cfg    : AcquisitionConfig
    method : {"phase", "modulus", "mean"}
        "phase"   → τ_φ  (arctan estimator)
        "modulus" → τ_m  (demodulation estimator)
        "mean"    → arithmetic mean of τ_φ and τ_m

    Returns
    -------
    tau : ndarray  (ns)
    """
    if method == "phase":
        return phase_lifetime(g, s, cfg)
    elif method == "modulus":
        return modulus_lifetime(g, s, cfg)
    elif method == "mean":
        return 0.5 * (phase_lifetime(g, s, cfg) + modulus_lifetime(g, s, cfg))
    else:
        raise ValueError(f"method must be 'phase', 'modulus', or 'mean'. Got: {method!r}")


# ──────────────────────────────────────────────────────────────────────────────
# Corrected estimators for gated data (Michalet 2021, Sec. VII)
# ──────────────────────────────────────────────────────────────────────────────

def phase_lifetime_gated(
    g: ArrayLike,
    s: ArrayLike,
    cfg: AcquisitionConfig,
) -> NDArray[np.float64]:
    """
    Modified phase lifetime for a *single square gate* of width W.

    For gated data the standard arctan formula underestimates τ.  Michalet
    2021 (Sec. VII A) provides the correction via the implicit equation:

        tan(φ_gate) / ω  ≠  τ   (bias!)

    This function solves the forward model numerically: for each observed
    (g, s) we find τ such that  phasor_gated_single(τ, cfg) == (g, s)  by
    minimising  |φ_forward − φ_measured|.

    Parameters
    ----------
    g, s : array_like
    cfg  : AcquisitionConfig  (mode GATED_SINGLE expected)

    Returns
    -------
    tau : ndarray  (ns)
    """
    from scipy.optimize import brentq
    from .phasors import phasor_gated_single  # local import: scipy is optional

    g = np.asarray(g, dtype=float)
    s = np.asarray(s, dtype=float)
    phi_meas = np.arctan2(s, g)

    def solve_one(phi):
        def residual(tau):
            gg, ss = phasor_gated_single(tau, cfg)
            return np.arctan2(ss, gg) - phi
        try:
            return brentq(residual, cfg.tau_min_ns, cfg.tau_max_ns * 10)
        except ValueError:
            return np.nan

    return np.array([solve_one(p) for p in np.atleast_1d(phi_meas)])


# ──────────────────────────────────────────────────────────────────────────────
# Fractional components for two-species mixtures
# ──────────────────────────────────────────────────────────────────────────────

def fractional_components(
    g_mix: ArrayLike,
    s_mix: ArrayLike,
    g1: float,
    s1: float,
    g2: float,
    s2: float,
) -> tuple[NDArray, NDArray]:
    """
    Estimate the fractional contributions  (f₁, f₂)  of two pure species
    using the linear combination property of phasors.

    For a mixture of two species with known phasors (g₁,s₁) and (g₂,s₂):

        g_mix = f₁·g₁ + f₂·g₂
        s_mix = f₁·s₁ + f₂·s₂
        f₁ + f₂ = 1

    This system is overdetermined; we use the geometric lever-rule:
    the mixture point divides the line segment from species 1 to species 2
    such that  f₁ = d(mix→2) / d(1→2).

    Convention: f₁ is the fraction of *species 1*, i.e. it equals 1 when
    the mixture phasor coincides with (g₁, s₁).

    Parameters
    ----------
    g_mix, s_mix : array_like
        Observed mixture phasor(s).
    g1, s1       : float   Pure species 1 phasor.
    g2, s2       : float   Pure species 2 phasor.

    Returns
    -------
    f1, f2 : ndarray
        Fractional intensities of species 1 and 2.  f2 = 1 − f1.
    """
    g_mix = np.asarray(g_mix, dtype=float)
    s_mix = np.asarray(s_mix, dtype=float)

    dg = g2 - g1
    ds = s2 - s1
    norm2 = dg ** 2 + ds ** 2
    if norm2 < 1e-20:
        raise ValueError("Species 1 and 2 are too close in the phasor plot.")

    # Projection of (mix - p1) onto unit vector (p2 - p1):
    # this gives the fraction *toward species 2*, i.e. f2.
    f2 = ((g_mix - g1) * dg + (s_mix - s1) * ds) / norm2
    f2 = np.clip(f2, 0.0, 1.0)
    return 1.0 - f2, f2
