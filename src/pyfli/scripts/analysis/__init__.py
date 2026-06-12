from .load_results import (
    RESULT_FILENAMES,
    load_session_arrays,
    scan_session_results,
    load_fitting_results,
    save_laguerre_result,
    inject_phasor_result,
    add_mean_lifetime,
)
from .phasor_analysis import (
    compute_freq_axis,
    compute_phasor,
    plot_phasor_figures,
    save_phasor_result,
)
from .fit_analysis import (
    DEFAULT_KEY_THRESHOLDS,
    plot_fitting_maps,
    plot_diagnostics,
    plot_pixel_evidence,
    plot_statistical_comparison,
    plot_2d_analysis,
    run_mono_bi_classifier,
)
# FBI module is proprietary and excluded from the public repo.
# The filename constants are always available so that saved FBI results
# remain loadable via load_fitting_results() even when the model code is absent.
try:
    from .fbi_analysis import (
        FBI_RESULT_FILENAME,
        FBI_RAW_FILENAME,
        load_fbi_model,
        run_fbi_inference,
        compute_fbi_results,
        plot_fbi_maps,
    )
    _FBI_AVAILABLE = True
except ImportError:
    FBI_RESULT_FILENAME = 'F-BI Output_bi-exponential.npy'
    FBI_RAW_FILENAME    = 'F-BI Direct_Output_bi-exponential.npy'
    _FBI_AVAILABLE      = False

    def load_fbi_model(*_, **__):
        raise ImportError("FBI model code is not available in this installation.")
    def run_fbi_inference(*_, **__):
        raise ImportError("FBI model code is not available in this installation.")
    def compute_fbi_results(*_, **__):
        raise ImportError("FBI model code is not available in this installation.")
    def plot_fbi_maps(*_, **__):
        raise ImportError("FBI model code is not available in this installation.")

__all__ = [
    # naming conventions
    "RESULT_FILENAMES",
    "FBI_RESULT_FILENAME",
    "FBI_RAW_FILENAME",
    # data loading
    "load_session_arrays",
    "scan_session_results",
    "load_fitting_results",
    "save_laguerre_result",
    "inject_phasor_result",
    "add_mean_lifetime",
    # phasor
    "compute_freq_axis",
    "compute_phasor",
    "plot_phasor_figures",
    "save_phasor_result",
    # fit visualisation
    "DEFAULT_KEY_THRESHOLDS",
    "plot_fitting_maps",
    "plot_diagnostics",
    "plot_pixel_evidence",
    "plot_statistical_comparison",
    "plot_2d_analysis",
    "run_mono_bi_classifier",
    # FBI
    "load_fbi_model",
    "run_fbi_inference",
    "compute_fbi_results",
    "plot_fbi_maps",
]
