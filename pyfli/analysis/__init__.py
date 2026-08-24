# ruff: noqa: F401

"""
Provide analysis tools for PyFLI post-processing, diagnostics, statistical comparison,
and result-loading utilities for fitted FLI/FLIM datasets.

This module belongs to :mod:`pyfli.analysis` and is part of PyFLI post-processing,
diagnostics, statistical comparison, and result-loading utilities for fitted FLI/FLIM
datasets. The module primarily re-exports package symbols or constants for downstream
imports.
"""

from typing import Any

from .factor_analysis import FactorAnalysis
from .fit_analysis import (
    DEFAULT_KEY_THRESHOLDS,
    plot_2d_analysis,
    plot_diagnostics,
    plot_fitting_maps,
    plot_pixel_evidence,
    plot_statistical_comparison,
    run_mono_bi_classifier,
)
from .load_results import (
    RESULT_FILENAMES,
    add_mean_lifetime,
    inject_phasor_result,
    load_fitting_results,
    load_session_arrays,
    save_laguerre_result,
    scan_session_results,
)
from .phasor_analysis import (
    compute_freq_axis,
    compute_phasor,
    plot_phasor_figures,
    save_phasor_result,
)
from .stat_tests import TestStat

# FBI module is proprietary and excluded from the public repo.
# The filename constants are always available so that saved FBI results
# remain loadable via load_fitting_results() even when the model code is absent.
try:
    from .fbi_analysis import (
        FBI_RAW_FILENAME,
        FBI_RESULT_FILENAME,
        compute_fbi_results,
        load_fbi_model,
        plot_fbi_maps,
        run_fbi_inference,
    )

    _FBI_AVAILABLE = True
except ImportError:
    FBI_RESULT_FILENAME = "F-BI Output_bi-exponential.npy"
    FBI_RAW_FILENAME = "F-BI Direct_Output_bi-exponential.npy"
    _FBI_AVAILABLE = False

    def load_fbi_model(*_: Any, **__: Any) -> None:
        """
        Load fbi model.

        Parameters
        ----------
        *_ : Any
            Additional positional values accepted by the routine.
        **__ : Any
            Additional keyword options forwarded to the underlying implementation.

        Returns
        -------
        None
            No object is returned; the function load fbi model.
        """
        raise ImportError("FBI model code is not available in this installation.")

    def run_fbi_inference(*_: Any, **__: Any) -> None:
        """
        Run fbi inference.

        Parameters
        ----------
        *_ : Any
            Additional positional values accepted by the routine.
        **__ : Any
            Additional keyword options forwarded to the underlying implementation.

        Returns
        -------
        None
            No object is returned; the function run fbi inference.
        """
        raise ImportError("FBI model code is not available in this installation.")

    def compute_fbi_results(*_: Any, **__: Any) -> None:
        """
        Compute fbi results.

        Parameters
        ----------
        *_ : Any
            Additional positional values accepted by the routine.
        **__ : Any
            Additional keyword options forwarded to the underlying implementation.

        Returns
        -------
        None
            No object is returned; the function compute fbi results.
        """
        raise ImportError("FBI model code is not available in this installation.")

    def plot_fbi_maps(*_: Any, **__: Any) -> None:
        """
        Plot fbi maps.

        Parameters
        ----------
        *_ : Any
            Additional positional values accepted by the routine.
        **__ : Any
            Additional keyword options forwarded to the underlying implementation.

        Returns
        -------
        None
            No object is returned; the function plot fbi maps.
        """
        raise ImportError("FBI model code is not available in this installation.")
