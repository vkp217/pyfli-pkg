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
# ruff: noqa: F401

from .config import AcquisitionConfig, AcquisitionMode
from .lifetimes import (
    fractional_components,
    lifetime_from_phasor,
    modulus_lifetime,
    phase_lifetime,
    phase_lifetime_gated,
)
from .locus import (
    build_loci,
    build_locus,
    sepl_center_radius_discrete,
    tau_grid,
    universal_semicircle,
)
from .phasors import (
    phasor_continuous,
    phasor_discrete,
    phasor_from_config,
    phasor_gated_N,
    phasor_gated_single,
    phasor_offset,
    phasor_truncated,
)
from .plot import plot_discrete_N_sweep, plot_locus_comparison, plot_phasor
