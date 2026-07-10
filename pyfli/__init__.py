"""Top-level public API for PyFLI."""
# ruff: noqa: E402, F401


def setup() -> None:
    from .logging import configure_logging

    configure_logging()


setup()
del setup

from . import logging, phasor, sp_analysis
from .analytical_methods.laguerre_deconvolution import LaguerreFLI
from .data_cc.norm import Normalization
from .io.data_operations import DataOperations
from .io.detector import Detector
from .io.data_saving import DataSaver
from .io.flim_decay_cube import collapse_to_xyt, load_flim_data, plot_xyt
from .phasor import (
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
from .simulator.main_factory import MacroSimulator, TCSPCSimulator
from .sp_analysis import BasisPatterns, MeasurementSimulator, Reconstructor
