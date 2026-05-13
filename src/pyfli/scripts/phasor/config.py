"""
config.py
=========
Dataclass-based configuration for phasor acquisitions.

All physical parameters are stored here so that every downstream module
receives a single, validated object rather than a loose collection of kwargs.

Units
-----
    Time      : nanoseconds (ns)
    Frequency : MHz  (stored as period T in ns)
    Angles    : radians (computed internally)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
import math


class AcquisitionMode(Enum):
    """Acquisition / gating modes described in Michalet 2021."""

    CONTINUOUS  = auto()   # Ideal TCSPC / frequency-domain; canonical universal semicircle
    DISCRETE    = auto()   # Binned TCSPC with finite number of bins N
    GATED_SINGLE = auto()  # Single square gate of width W
    GATED_N     = auto()   # N equidistant square gates of width W
    TRUNCATED   = auto()   # Decay recording window shorter than laser period
    OFFSET      = auto()   # IRF / excitation-pulse offset within recording window


@dataclass
class AcquisitionConfig:
    """
    Complete description of a phasor acquisition experiment.

    Parameters
    ----------
    mode : AcquisitionMode
        Which SEPL formula to use.
    T_ns : float
        Laser repetition period in nanoseconds.
    harmonic : int
        Fourier harmonic n (1 = fundamental, 2 = second harmonic …).
    N_bins : int
        Number of time bins for DISCRETE mode (ignored otherwise).
    gate_width_frac : float
        Gate width as a fraction of T (0 < gate_width_frac ≤ 1).
        Used by GATED_SINGLE and GATED_N.
    N_gates : int
        Number of equidistant gates for GATED_N mode.
    T_rec_frac : float
        Fraction of T covered by the recording window (TRUNCATED mode).
        Must be in (0, 1].  Values < 1 cause SEPL deformation.
    t0_frac : float
        IRF / excitation offset as a fraction of T (OFFSET mode).
    tau_min_ns : float
        Minimum lifetime for locus computation (ns).
    tau_max_ns : float
        Maximum lifetime for locus computation (ns).
    n_tau_pts : int
        Number of τ samples for locus curves.

    Derived properties
    ------------------
    omega : float
        Angular frequency 2π·n/T  (rad/ns).
    gate_width_ns : float
        Absolute gate width  W = gate_width_frac · T  (ns).
    T_rec_ns : float
        Absolute recording window  T_rec = T_rec_frac · T  (ns).
    t0_ns : float
        Absolute IRF offset  t0 = t0_frac · T  (ns).
    """

    # ------------------------------------------------------------------ core
    mode:            AcquisitionMode = AcquisitionMode.CONTINUOUS
    T_ns:            float           = 12.5
    harmonic:        int             = 1

    # ------------------------------------------------------------------ discrete
    N_bins:          int             = 64

    # ------------------------------------------------------------------ gating
    gate_width_frac: float           = 0.5
    N_gates:         int             = 4

    # ------------------------------------------------------------------ truncation
    T_rec_frac:      float           = 0.8

    # ------------------------------------------------------------------ offset
    t0_frac:         float           = 0.1

    # ------------------------------------------------------------------ locus
    tau_min_ns:      float           = 1e-4
    tau_max_ns:      float           = 10.0
    n_tau_pts:       int             = 600

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        self._validate()

    # ------------------------------------------------------------------ validation
    def _validate(self) -> None:
        if self.T_ns <= 0:
            raise ValueError(f"T_ns must be positive, got {self.T_ns}")
        if self.harmonic < 1:
            raise ValueError(f"harmonic must be >= 1, got {self.harmonic}")
        if self.N_bins < 2:
            raise ValueError(f"N_bins must be >= 2, got {self.N_bins}")
        if not (0 < self.gate_width_frac <= 1):
            raise ValueError(f"gate_width_frac must be in (0,1], got {self.gate_width_frac}")
        if self.N_gates < 1:
            raise ValueError(f"N_gates must be >= 1, got {self.N_gates}")
        if not (0 < self.T_rec_frac <= 1):
            raise ValueError(f"T_rec_frac must be in (0,1], got {self.T_rec_frac}")
        if not (0 <= self.t0_frac < 1):
            raise ValueError(f"t0_frac must be in [0,1), got {self.t0_frac}")
        if self.tau_min_ns <= 0:
            raise ValueError(f"tau_min_ns must be positive, got {self.tau_min_ns}")
        if self.tau_max_ns <= self.tau_min_ns:
            raise ValueError("tau_max_ns must be greater than tau_min_ns")
        if self.n_tau_pts < 10:
            raise ValueError(f"n_tau_pts must be >= 10, got {self.n_tau_pts}")

    # ------------------------------------------------------------------ derived properties
    @property
    def omega(self) -> float:
        """Angular frequency ω = 2π·n/T  (rad/ns)."""
        return 2.0 * math.pi * self.harmonic / self.T_ns

    @property
    def frequency_MHz(self) -> float:
        """Fundamental laser repetition frequency in MHz."""
        return 1_000.0 / self.T_ns

    @property
    def gate_width_ns(self) -> float:
        """Absolute gate width W (ns)."""
        return self.gate_width_frac * self.T_ns

    @property
    def T_rec_ns(self) -> float:
        """Absolute recording window (ns)."""
        return self.T_rec_frac * self.T_ns

    @property
    def t0_ns(self) -> float:
        """Absolute IRF offset (ns)."""
        return self.t0_frac * self.T_ns

    # ------------------------------------------------------------------ helpers
    def describe(self) -> str:
        lines = [
            f"AcquisitionConfig",
            f"  mode          : {self.mode.name}",
            f"  T             : {self.T_ns} ns  →  f = {self.frequency_MHz:.3f} MHz",
            f"  harmonic n    : {self.harmonic}  →  ω = {self.omega:.4f} rad/ns",
        ]
        if self.mode is AcquisitionMode.DISCRETE:
            lines.append(f"  N_bins        : {self.N_bins}")
        if self.mode in (AcquisitionMode.GATED_SINGLE, AcquisitionMode.GATED_N):
            lines.append(f"  gate width W  : {self.gate_width_ns:.3f} ns  ({self.gate_width_frac:.2f}·T)")
        if self.mode is AcquisitionMode.GATED_N:
            lines.append(f"  N_gates       : {self.N_gates}")
        if self.mode is AcquisitionMode.TRUNCATED:
            lines.append(f"  T_rec         : {self.T_rec_ns:.3f} ns  ({self.T_rec_frac:.2f}·T)")
        if self.mode is AcquisitionMode.OFFSET:
            lines.append(f"  IRF offset t0 : {self.t0_ns:.3f} ns  ({self.t0_frac:.2f}·T)")
        lines.append(f"  τ range       : {self.tau_min_ns} – {self.tau_max_ns} ns  ({self.n_tau_pts} pts)")
        return "\n".join(lines)
