# ruff: noqa: F401

"""
Provide io tools for PyFLI detector importers, file readers, saving helpers, and
processed-data loaders.

This module belongs to :mod:`pyfli.io` and is part of PyFLI detector importers, file
readers, saving helpers, and processed-data loaders. The module primarily re-exports
package symbols or constants for downstream imports.
"""

from .detector import Detector
from .data_operations import DataOperations
from .data_saving import DataSaver
from .processed_data import (
    AlliGprocessedImport,
    BHprocessedImport,
    DatasetPlotter,
    PyFliprocessedImport,
)
from .spad_io import SpadConfig, SpadIO, SpadReadResult
from .utils import DataIOUtils
from .flim_decay_cube import (
    load_flim_data,
    collapse_to_xyt,
    plot_xyt,
)
