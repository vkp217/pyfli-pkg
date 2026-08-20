"""
Build generalized continuous and photon-counting simulation workflows.

This module belongs to :mod:`pyfli.simulator` and is part of PyFLI synthetic FLI/FLIM
data generation, hardware noise modeling, calibration, and validation tools. Public API
includes classes :class:`ContinuousSimulator` and :class:`PhotonCountSimulator`.

Both classes wrap :class:`~pyfli.simulator.separate.model_simulator.FLIModelSimulator`;
their shared noise/scaling pipeline lives in :mod:`pyfli.simulator.main_common`.
"""

from ..main_common import BaseContinuousSimulator, BaseDiscreteSimulator
from .model_simulator import FLIModelSimulator


class ContinuousSimulator(BaseContinuousSimulator):
    """
    Generate continuous/intensity-style FLI/FLIM simulations using the generalized
    simulator factory. It mirrors the macro simulator interface while separating
    continuous detector assumptions.

    Parameters
    ----------
    irf_data : np.ndarray
        Instrument response data used to convolve or simulate decays.
    sensor_type : str
        Detector family or hardware profile name.
    **cfg : Any
        Additional configuration values forwarded to the simulator or solver.
    """

    engine_cls = FLIModelSimulator
    sample_params_name = "sample_params"
    analytical_decay_name = "get_model_analytical_decay"
    mono_only_maps = True


class PhotonCountSimulator(BaseDiscreteSimulator):
    """
    Generate photon-counting simulations using the generalized simulator factory. It
    packages photon counter settings and returns simulated decays with associated
    parameters.

    Parameters
    ----------
    irf_data : np.ndarray
        Instrument response data used to convolve or simulate decays.
    sensor_type : str
        Detector family or hardware profile name.
    **cfg : Any
        Additional configuration values forwarded to the simulator or solver.
    """

    engine_cls = FLIModelSimulator
    sample_params_name = "sample_params"
    analytical_decay_name = "get_model_analytical_decay"
    simulate_tcspc_name = "simulate_model_tcspc"
    mono_only_maps = True
