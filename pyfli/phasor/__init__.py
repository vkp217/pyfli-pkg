"""
phasor
===========
Phasor-domain lifetime analysis, split into two implementations:

    phasorSEPL – the full phasor / universal-circle (SEPL) formalism, following:

        Michalet X. "Continuous and discrete phasor analysis of binned or
        time-gated periodic decays." AIP Advances 11, 035331 (2021).
        https://doi.org/10.1063/5.0027834

    phasorS   – a compact phasor analyzer for CPU and optional GPU workflows
                (:class:`~pyfli.phasor.phasorS.PhasorAnalyzer`).

The names below re-export the ``phasorSEPL`` public surface directly on
:mod:`pyfli.phasor` for backward compatibility.
"""
# ruff: noqa: F401

from .phasorSEPL import (
    AcquisitionConfig,
    AcquisitionMode,
    phasor_continuous,
    phasor_discrete,
    phasor_gated_single,
    phasor_gated_N,
    phasor_truncated,
    phasor_offset,
    phasor_from_config,
    build_locus,
    build_loci,
    tau_grid,
    universal_semicircle,
    sepl_center_radius_discrete,
    phase_lifetime,
    modulus_lifetime,
    lifetime_from_phasor,
    phase_lifetime_gated,
    fractional_components,
    plot_phasor,
    plot_locus_comparison,
    plot_discrete_N_sweep,
)
