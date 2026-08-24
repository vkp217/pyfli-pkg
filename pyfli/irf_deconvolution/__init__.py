# ruff: noqa: F401

"""
Provide irf deconvolution tools for PyFLI detector-aware IRF deconvolution and joint
FLI fitting utilities.

This module belongs to :mod:`pyfli.irf_deconvolution` and is part of PyFLI detector-
aware IRF deconvolution and joint FLI fitting utilities. The module primarily re-
exports package symbols or constants for downstream imports.
"""

from .detector_weights import (
    ICCDParams,
    SPADParams,
    TCSPCParams,
    generalized_anscombe,
    make_observation,
)
from .fli_solver import (
    SolverConfig,
    build_gate_matrix,
    cyclic_conv,
    decay_basis,
    solve_flim,
)
