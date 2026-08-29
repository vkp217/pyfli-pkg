"""
Provide a compact phasor analyzer for CPU and optional GPU FLI workflows.

This module belongs to :mod:`pyfli.phasor.phasorS` and is part of PyFLI's compact
phasor analyzer for CPU and optional GPU FLI workflows. Public API includes classes
:class:`PhasorAnalyzer`, :class:`PhasorPlotsMixin`, :class:`MonoLocus`, and
:class:`PhasorAdditionalPlots`.
"""
# ruff: noqa: F401

from .phasor_additional_plots import PhasorAdditionalPlots
from .phasor_locus import MonoLocus
from .phasor_simple import PhasorAnalyzer
from .phasor_simple_plots import PhasorPlotsMixin
