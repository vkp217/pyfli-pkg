"""
Provide a compact phasor analyzer for CPU and optional GPU FLI workflows.

This module belongs to :mod:`pyfli.phasor.phasorS` and is part of PyFLI's compact
phasor analyzer for CPU and optional GPU FLI workflows. Public API includes classes
:class:`PhasorAnalyzer` and :class:`PhasorPlotsMixin`.
"""
# ruff: noqa: F401

from .phasor_simple import PhasorAnalyzer
from .phasor_simple_plots import PhasorPlotsMixin
