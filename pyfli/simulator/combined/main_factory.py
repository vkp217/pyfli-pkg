"""
Build macro and TCSPC simulation workflows from IRF data and hardware configuration.

This module belongs to :mod:`pyfli.simulator` and is part of PyFLI synthetic FLI/FLIM
data generation, hardware noise modeling, calibration, and validation tools. Public API
includes classes :class:`MacroSimulator` and :class:`TCSPCSimulator`.

Both classes wrap :class:`~pyfli.simulator.combined.simulator_engine.FLIEngine`; their
shared noise/scaling pipeline lives in :mod:`pyfli.simulator.main_common`.
"""

from ..main_common import BaseContinuousSimulator, BaseDiscreteSimulator
from .simulator_engine import FLIEngine


class MacroSimulator(BaseContinuousSimulator):
    """
    Generate macro-time style synthetic FLI/FLIM decays from IRF data and sensor
    configuration. The callable factory samples lifetimes, amplitudes, noise, and
    detector response for continuous (ICCD-like) workflows.

    Parameters
    ----------
    irf_data : np.ndarray
        Instrument response data used to convolve or simulate decays.
    sensor_type : str
        Detector family or hardware profile name.
    **cfg : Any
        Additional configuration values forwarded to the simulator or solver.
    """

    engine_cls = FLIEngine
    sample_params_name = "sample_all_params"
    analytical_decay_name = "get_analytical_decay"
    mono_only_maps = False


class TCSPCSimulator(BaseDiscreteSimulator):
    """
    Generate photon-counting TCSPC simulations from IRF data and sensor configuration.
    The callable factory emphasizes photon-counter behavior and count statistics.

    Parameters
    ----------
    irf_data : np.ndarray
        Instrument response data used to convolve or simulate decays.
    sensor_type : str
        Detector family or hardware profile name.
    **cfg : Any
        Additional configuration values forwarded to the simulator or solver.
    """

    engine_cls = FLIEngine
    sample_params_name = "sample_all_params"
    analytical_decay_name = "get_analytical_decay"
    simulate_tcspc_name = "simulate_tcspc"
    mono_only_maps = False
