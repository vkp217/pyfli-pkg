# pyfli/simulator/__init__.py
# ruff: noqa: F401

"""
Provide simulator tools for PyFLI synthetic FLI/FLIM data generation, hardware noise
modeling, calibration, and validation tools.

This module belongs to :mod:`pyfli.simulator` and is part of PyFLI synthetic FLI/FLIM
data generation, hardware noise modeling, calibration, and validation tools. The module
primarily re-exports package symbols or constants for downstream imports.
"""

from .distributions import ParameterSampler
from .main_factory import MacroSimulator, TCSPCSimulator
from .main_factory_gen import ContinuousSimulator, PhotonCountSimulator
from .physics import HardSimulator, HardestSimulator
from .noise_models import NoiseEngine
from .simulator_engine import FLIEngine
from .sim_image_generator import FLIImageGenerator
from .sim_model_image_generator import FLIModelImageGenerator
from .sim_stat_test import FLIValidator
from .calibration_engine import FLICalibrator
from .batch_sim import BatchSimulator
from .sim_workflow import (
    SimGenerator,
    SimOutput,
    SimOutputWithIRFOffset,
    concat_sim_data,
    make_simulator,
)


# [FLICalibrator, FLIValidator, ParameterSampler,
# NoiseEngine, FLIEngine, MacroSimulator, TCSPCSimulator, FLIImageGenerator, BatchSimulator]
