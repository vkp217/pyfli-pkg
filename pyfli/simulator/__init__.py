# pyfli/simulator/__init__.py
# ruff: noqa: F401

"""
Provide simulator tools for PyFLI synthetic FLI/FLIM data generation, hardware noise
modeling, calibration, and validation tools.

This module belongs to :mod:`pyfli.simulator` and is part of PyFLI synthetic FLI/FLIM
data generation, hardware noise modeling, calibration, and validation tools. The module
primarily re-exports package symbols or constants for downstream imports.
"""

from .calibration_engine import FLICalibrator
from .combined.main_factory import MacroSimulator, TCSPCSimulator
from .combined.sim_image_generator import FLIImageGenerator
from .combined.simulator_engine import FLIEngine
from .distributions import ParameterSampler
from .irf_sim.irf_generator import IRFGenerator
from .irf_sim.irf_offset_gen import OffsetGen
from .noise_models import NoiseEngine
from .separate.main_factory_gen import ContinuousSimulator, PhotonCountSimulator
from .separate.sim_model_image_generator import FLIModelImageGenerator
from .sim_calibrator import FLIValidator
from .sim_workflow import (
    BatchSimulator,
    SimGenerator,
    SimOutput,
    SimOutputWithIRFOffset,
    WeightedConfigSimGenerator,
    concat_sim_data,
    make_simulator,
)


# [FLICalibrator, FLIValidator, ParameterSampler,
# NoiseEngine, FLIEngine, MacroSimulator, TCSPCSimulator, FLIImageGenerator, BatchSimulator]
