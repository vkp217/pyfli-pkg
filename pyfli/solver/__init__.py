##### inside solver.__init__.py
# ruff: noqa: F401

"""
Provide solver tools for PyFLI least-squares, maximum-likelihood, CPU, GPU, binned, and
global FLIM fitting routines.

This module belongs to :mod:`pyfli.solver` and is part of PyFLI least-squares, maximum-
likelihood, CPU, GPU, binned, and global FLIM fitting routines. The module primarily re-
exports package symbols or constants for downstream imports.
"""

from __future__ import annotations
from .base_fitter import BaseFLIFitter
from .cpu_processor import FLICPUProcessor
from .gpu_processor import FLIGPUProcessor
from .mle_fitter import MLEFLIFitter
from .global_fitter import GlobalFLIFitter
from .comparison import FittingComparator
from .binned_fitter import BinnedFLIFitter, FLIBinner
from .forward_model import decay_kernel, model_numpy
from .shared_metrics import (
    enforce_tau_ordering,
    compute_fli_stats,
    compute_average_lifetime,
    compute_fret_efficiency,
)

# [BaseFLIFitter, FLICPUProcessor, FLIGPUProcessor, MLEFLIFitter, GlobalFLIFitter, FittingComparator,
# BinnedFLIFitter, FLIBinner, decay_kernel, model_numpy, enforce_tau_ordering,
# compute_fli_stats, compute_average_lifetime, compute_fret_efficiency]
