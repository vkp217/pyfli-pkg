"""
phasors.py
==========
Analytical and semi-analytical phasor coordinate calculators for each
acquisition mode described in Michalet 2021 (DOI: 10.1063/5.0027834).

Every public function has signature:

    phasor_*(tau, cfg) -> tuple[float | np.ndarray, float | np.ndarray]

returning  (g, s)  coordinates in the phasor plane.

Convention
----------
    g  =  Re[ z(τ) ]   (cosine / real component)
    s  =  Im[ z(τ) ]   (sine  / imaginary component)

    z(τ)  is the normalised first-harmonic Fourier coefficient of the
    luminescence decay  I(t).

References
----------
    Michalet X., AIP Advances 11, 035331 (2021), Sec. III–V.
    ISS Technical Note: "FLIM Analysis using the Phasor Plots".
"""

from __future__ import annotations
import math
import numpy as np
from numpy.typing import ArrayLike

from .config import AcquisitionConfig, AcquisitionMode


# ══════════════════════════════════════════════════════════════════════════════
# 1.  CONTINUOUS  (canonical universal semicircle)
# ══════════════════════════════════════════════════════════════════════════════

def phasor_continuous(
    tau: ArrayLike,
    cfg: AcquisitionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Phasor coordinates for a periodic single-exponential decay (PSED) in the
    continuous / ideal TCSPC limit (Dirac IRF, full-period recording).

    This is the canonical formula whose locus is the *universal semicircle*
    of radius ½ centred at (½, 0).

    Equations (Michalet 2021, Eq. 15 / ISS technical note Eq. 10):

        g(τ) = 1 / (1 + ω²τ²)
        s(τ) = ωτ / (1 + ω²τ²)

    where ω = 2π·n/T.

    Parameters
    ----------
    tau : array_like
        Fluorescence lifetime(s) in nanoseconds.
    cfg : AcquisitionConfig
        Must supply:  omega (derived from T_ns and harmonic).

    Returns
    -------
    g, s : np.ndarray
        Real and imaginary phasor coordinates (shape matches *tau*).
    """
    tau = np.asarray(tau, dtype=float)
    w = cfg.omega
    wt = w * tau
    denom = 1.0 + wt ** 2
    return 1.0 / denom, wt / denom


# ══════════════════════════════════════════════════════════════════════════════
# 2.  DISCRETE  (binned TCSPC — arc of circle)
# ══════════════════════════════════════════════════════════════════════════════

def phasor_discrete(
    tau: ArrayLike,
    cfg: AcquisitionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Phasor coordinates for a PSED sampled into N equal bins over one period T.

    The locus is an *arc of a circle* (not the universal semicircle).  Its
    centre and radius depend on N and the harmonic n; it converges to the
    universal semicircle as N → ∞.

    Algorithm: direct discrete Fourier coefficient (Michalet 2021, Sec. III C):

        I_k  =  ∫_{k·Δt}^{(k+1)·Δt} e^{-t/τ} dt  =  e^{-k·Δt/τ} · (1 − e^{-Δt/τ})
        z̃    =  Σ_k I_k · e^{i 2π n k / N}  /  Σ_k I_k

    Parameters
    ----------
    tau : array_like
        Lifetime(s) in ns.
    cfg : AcquisitionConfig
        Requires:  T_ns, N_bins, harmonic.

    Returns
    -------
    g, s : np.ndarray
    """
    tau = np.asarray(tau, dtype=float)
    scalar = tau.ndim == 0
    tau = np.atleast_1d(tau)

    T, N, n = cfg.T_ns, cfg.N_bins, cfg.harmonic
    dt = T / N
    k = np.arange(N, dtype=float)                  # bin indices [0 … N-1]
    phase = 2.0 * np.pi * n * k / N                # Fourier kernel phases

    # --- vectorised over tau
    # tau shape: (M,)  →  (M, 1)  × k shape (N,)
    tau2 = tau[:, np.newaxis]
    e_kdt = np.exp(-k * dt / tau2)                 # (M, N)
    e_dt  = np.exp(-dt / tau2)                     # (M, 1)
    I_k   = e_kdt * (1.0 - e_dt)                  # (M, N)

    cos_k = np.cos(phase)                          # (N,)
    sin_k = np.sin(phase)                          # (N,)

    I_sum = I_k.sum(axis=1)                        # (M,)
    g = (I_k * cos_k).sum(axis=1) / I_sum
    s = (I_k * sin_k).sum(axis=1) / I_sum

    if scalar:
        return float(g[0]), float(s[0])
    return g, s


# ══════════════════════════════════════════════════════════════════════════════
# 3.  GATED_SINGLE  (single square gate, continuous approximation)
# ══════════════════════════════════════════════════════════════════════════════

def phasor_gated_single(
    tau: ArrayLike,
    cfg: AcquisitionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Phasor coordinates for a PSED measured through a *single square gate* of
    width W starting at t = 0, with Dirac IRF.

    From Michalet 2021, Sec. III B (Eq. for gated PSED phasor):

        The gate selects the interval [0, W].  The T-periodic normalisation
        factor is  α(τ,W,T) = (1 − e^{-W/τ}) / (1 − e^{-T/τ}).

        g_gate(τ) = α · 1/(1+ω²τ²) · [1 − e^{-W/τ}(cos(ωW) + ωτ·sin(ωW))]
                     / (1 − e^{-W/τ})
        s_gate(τ) = α · 1/(1+ω²τ²) · [ωτ − e^{-W/τ}(ωτ·cos(ωW) − sin(ωW))]
                     / (1 − e^{-W/τ})

    Simplification gives the scaled form used here.

    Parameters
    ----------
    tau : array_like
        Lifetime(s) in ns.
    cfg : AcquisitionConfig
        Requires:  T_ns, gate_width_frac, harmonic.

    Returns
    -------
    g, s : np.ndarray
    """
    tau = np.asarray(tau, dtype=float)
    w = cfg.omega
    T = cfg.T_ns
    W = cfg.gate_width_ns

    wt  = w * tau
    eW  = np.exp(-W / tau)
    eT  = np.exp(-T / tau)
    wW  = w * W

    # Exact analytical result (derived from Re/Im of ∫₀^W e^{-t/τ} e^{iωt} dt):
    #
    #   ∫₀^W e^{-t/τ} cos(ωt) dt  =  τ · [1 − e^{-W/τ}·(cos(ωW) − ωτ·sin(ωW))] / (1+ω²τ²)
    #   ∫₀^W e^{-t/τ} sin(ωt) dt  =  τ · [ωτ·(1 − e^{-W/τ}·cos(ωW)) − e^{-W/τ}·sin(ωW)] / (1+ω²τ²)
    #
    # Denominator (T-period normalisation):  ∫₀^T e^{-t/τ} dt = τ·(1 − e^{-T/τ})
    # The τ factors cancel, giving:

    denom = (1.0 + wt ** 2) * (1.0 - eT)

    cos_int = 1.0 - eW * (np.cos(wW) - wt * np.sin(wW))
    sin_int = wt * (1.0 - eW * np.cos(wW)) - eW * np.sin(wW)

    g = cos_int / denom
    s = sin_int / denom
    return g, s


# ══════════════════════════════════════════════════════════════════════════════
# 4.  GATED_N  (N equidistant square gates)
# ══════════════════════════════════════════════════════════════════════════════

def phasor_gated_N(
    tau: ArrayLike,
    cfg: AcquisitionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Phasor of a PSED measured with *N equidistant square gates* of width W.

    Gate k starts at  t_k = k · θ  where  θ = T / N_gates.

    The (discrete) phasor is (Michalet 2021, Sec. III C 3):

        I_k  =  e^{-t_k/τ} − e^{-(t_k+W)/τ}
        z̃    =  Σ_k I_k · e^{i 2π n k / N_gates}  /  Σ_k I_k

    Parameters
    ----------
    tau : array_like
        Lifetime(s) in ns.
    cfg : AcquisitionConfig
        Requires:  T_ns, N_gates, gate_width_frac, harmonic.

    Returns
    -------
    g, s : np.ndarray
    """
    tau = np.asarray(tau, dtype=float)
    scalar = tau.ndim == 0
    tau = np.atleast_1d(tau)

    T  = cfg.T_ns
    Ng = cfg.N_gates
    W  = cfg.gate_width_ns
    n  = cfg.harmonic
    theta = T / Ng
    k  = np.arange(Ng, dtype=float)
    t_k  = k * theta
    phase = 2.0 * np.pi * n * k / Ng

    tau2  = tau[:, np.newaxis]            # (M, 1)
    I_k   = np.exp(-t_k / tau2) - np.exp(-(t_k + W) / tau2)   # (M, Ng)

    I_sum = I_k.sum(axis=1)
    g = (I_k * np.cos(phase)).sum(axis=1) / I_sum
    s = (I_k * np.sin(phase)).sum(axis=1) / I_sum

    if scalar:
        return float(g[0]), float(s[0])
    return g, s


# ══════════════════════════════════════════════════════════════════════════════
# 5.  TRUNCATED  (recording window shorter than laser period)
# ══════════════════════════════════════════════════════════════════════════════

def phasor_truncated(
    tau: ArrayLike,
    cfg: AcquisitionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Phasor when only the first T_rec < T nanoseconds of the decay are recorded.

    This is a common experimental artefact with high-repetition-rate lasers.
    The SEPL deforms and phasors can fall *outside* the universal semicircle,
    leading to erroneous multi-exponential interpretations if uncorrected.

    Analytical form (Michalet 2021, Sec. V):

        Using partial integration of  e^{-t/τ} · e^{iωt}  over [0, T_rec]:

        ∫₀^{T_rec} e^{-t/τ} e^{iωt} dt  =  τ · [1 − e^{-(1/τ−iω)T_rec}]
                                              / (1 − iωτ)

        Normalised by  ∫₀^{T_rec} e^{-t/τ} dt  =  τ(1 − e^{-T_rec/τ})

    Parameters
    ----------
    tau : array_like
        Lifetime(s) in ns.
    cfg : AcquisitionConfig
        Requires:  T_ns, T_rec_frac, harmonic.

    Returns
    -------
    g, s : np.ndarray
    """
    tau  = np.asarray(tau, dtype=float)
    w    = cfg.omega
    Trec = cfg.T_rec_ns
    wt   = w * tau
    eTr  = np.exp(-Trec / tau)
    wTr  = w * Trec

    cos_int = (1.0 - eTr * (np.cos(wTr) + wt * np.sin(wTr))) / (1.0 + wt ** 2)
    sin_int = (wt  - eTr * (wt * np.cos(wTr) - np.sin(wTr))) / (1.0 + wt ** 2)

    # Numerator integrals already carry the structure of 1/(1+ω²τ²).
    # The τ factors in numerator and denominator cancel, so we normalise
    # by (1 − e^{−T_rec/τ}) only (not τ · (1 − e^{−T_rec/τ})).
    norm = 1.0 - eTr
    g = cos_int / norm
    s = sin_int / norm
    return g, s


# ══════════════════════════════════════════════════════════════════════════════
# 6.  OFFSET  (IRF / excitation pulse time-shifted within recording window)
# ══════════════════════════════════════════════════════════════════════════════

def phasor_offset(
    tau: ArrayLike,
    cfg: AcquisitionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Phasor when the excitation pulse (or IRF peak) is offset by t₀ within
    the recording window.

    An offset of t₀ introduces a pure rotation in the phasor plane
    (Michalet 2021, Sec. IV):

        z_offset(τ) = z_continuous(τ) · e^{−iωt₀}

    which gives:

        g_off = g·cos(ωt₀) + s·sin(ωt₀)
        s_off = s·cos(ωt₀) − g·sin(ωt₀)

    Parameters
    ----------
    tau : array_like
        Lifetime(s) in ns.
    cfg : AcquisitionConfig
        Requires:  T_ns, t0_frac, harmonic.

    Returns
    -------
    g, s : np.ndarray
    """
    g0, s0 = phasor_continuous(tau, cfg)
    phi = cfg.omega * cfg.t0_ns
    c, si = math.cos(phi), math.sin(phi)
    return g0 * c + s0 * si, s0 * c - g0 * si


# ══════════════════════════════════════════════════════════════════════════════
# 7.  DISPATCHER
# ══════════════════════════════════════════════════════════════════════════════

def phasor_from_config(
    tau: ArrayLike,
    cfg: AcquisitionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Dispatch to the correct phasor function based on *cfg.mode*.

    Parameters
    ----------
    tau : array_like
        Lifetime(s) in ns.
    cfg : AcquisitionConfig

    Returns
    -------
    g, s : np.ndarray
    """
    _dispatch: dict = {
        AcquisitionMode.CONTINUOUS:   phasor_continuous,
        AcquisitionMode.DISCRETE:     phasor_discrete,
        AcquisitionMode.GATED_SINGLE: phasor_gated_single,
        AcquisitionMode.GATED_N:      phasor_gated_N,
        AcquisitionMode.TRUNCATED:    phasor_truncated,
        AcquisitionMode.OFFSET:       phasor_offset,
    }
    fn = _dispatch.get(cfg.mode)
    if fn is None:
        raise NotImplementedError(f"No phasor function for mode {cfg.mode!r}")
    return fn(tau, cfg)
