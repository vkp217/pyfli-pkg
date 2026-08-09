# ruff: noqa: F401

"""
Provide irf deconvolution tools for PyFLI detector-aware IRF deconvolution and joint
FLIM fitting utilities.

This module belongs to :mod:`pyfli.irf_deconvolution` and is part of PyFLI detector-
aware IRF deconvolution and joint FLIM fitting utilities. The module primarily re-
exports package symbols or constants for downstream imports.
"""

from .detector_weights import (
    TCSPCParams,
    SPADParams,
    ICCDParams,
    make_observation,
    generalized_anscombe,
)
from .fli_solver import (
    SolverConfig,
    solve_flim,
    build_gate_matrix,
    decay_basis,
    cyclic_conv,
)
