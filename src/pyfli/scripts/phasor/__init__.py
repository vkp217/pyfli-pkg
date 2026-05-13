"""
phasor
===========
Modular implementation of the phasor / universal-circle formalism for
fluorescence lifetime analysis, following:

    Michalet X. "Continuous and discrete phasor analysis of binned or
    time-gated periodic decays." AIP Advances 11, 035331 (2021).
    https://doi.org/10.1063/5.0027834

Public surface
--------------
    config          – AcquisitionConfig dataclass
    phasors         – phasor coordinate calculators (one per mode)
    locus           – build (g, s, tau) arrays for a full SEPL curve
    lifetimes       – phase / modulus lifetime inversion
    plot            – matplotlib rendering helpers
"""

from .config import AcquisitionConfig, AcquisitionMode
from .phasors import (
    phasor_continuous,
    phasor_discrete,
    phasor_gated_single,
    phasor_gated_N,
    phasor_truncated,
    phasor_offset,
    phasor_from_config,
)
from .locus import build_locus, tau_grid
from .lifetimes import phase_lifetime, modulus_lifetime, lifetime_from_phasor
from .plot import plot_phasor, plot_locus_comparison

__all__ = [
    "AcquisitionConfig",
    "AcquisitionMode",
    "phasor_continuous",
    "phasor_discrete",
    "phasor_gated_single",
    "phasor_gated_N",
    "phasor_truncated",
    "phasor_offset",
    "phasor_from_config",
    "build_locus",
    "tau_grid",
    "phase_lifetime",
    "modulus_lifetime",
    "lifetime_from_phasor",
    "plot_phasor",
    "plot_locus_comparison",
]
