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
    build_loci,
    build_locus,
    fractional_components,
    lifetime_from_phasor,
    modulus_lifetime,
    phase_lifetime,
    phase_lifetime_gated,
    phasor_continuous,
    phasor_discrete,
    phasor_from_config,
    phasor_gated_N,
    phasor_gated_single,
    phasor_offset,
    phasor_truncated,
    plot_discrete_N_sweep,
    plot_locus_comparison,
    plot_phasor,
    sepl_center_radius_discrete,
    tau_grid,
    universal_semicircle,
)
