#### inside "scripts.__init__.py"
from .dataIO import (DataOperations, AlliGprocessedImport, 
                        BHprocessedImport, PyFliprocessedImport, DatasetPlotter, DataIO_utils,
                         Detector, load_flim_data, collapse_to_xyt, plot_xyt )
from .analytical_methods import (PhasorAnalyzer, AnalyticalHelpers, LaguerreFLI)
from .dataCC import IRFAligner, DataPreprocessing, Normalization, ROIoperations
from .dataVnP import (DataViewer, Plotter, DLModelComparator, Colorprocess,
                               PlotConfig, DataProcessor, SourceLoader,
                               PlotKit, SubplotVisualizer, plot_2d_subplots,
                               MonoBiClassifier, ParamCorrelationMatrix)
from .roiMaker import ROIMaker
from .solver import (BaseFLIFitter, Fli_CPUProcessor, Fli_GPUProcessor, 
                     MLEFLIFitter, GlobalFLIFitter, FittingComparator,
                     BinnedFliFitter, FliBinner)
from .simulator import (Macro_sim, TCSPC_sim, FLIImageGenerator, FLICalibrator, FLIValidator, Batch_sim)
from .data_text import Msg_display

from .simulatorPhysics import HardSimulator, HardestSimulator
from .utils_common import (recovery_plot, random_true_pixel, 
                           data_masking, save_plot, plot_pixel_diagnostic,
                           compute_detailed_results)
from .data_saving import DataSaver
from .irfDeconvolution import (
    TCSPCParams, SPADParams, ICCDParams,
    make_observation, generalized_anscombe,
    SolverConfig, solve_flim,
    build_gate_matrix, decay_basis, cyclic_conv,
)
from .analysis import (
    RESULT_FILENAMES, FBI_RESULT_FILENAME, FBI_RAW_FILENAME,
    load_session_arrays, scan_session_results, load_fitting_results,
    save_laguerre_result, inject_phasor_result, add_mean_lifetime,
    compute_freq_axis, compute_phasor, plot_phasor_figures, save_phasor_result,
    plot_fitting_maps, plot_diagnostics, plot_pixel_evidence,
    plot_statistical_comparison, plot_2d_analysis,
    run_mono_bi_classifier,
)
from .phasor import (
    AcquisitionConfig, AcquisitionMode,
    phasor_continuous, phasor_discrete, phasor_gated_single, phasor_gated_N,
    phasor_truncated, phasor_offset, phasor_from_config,
    build_locus, build_loci, tau_grid, universal_semicircle, sepl_center_radius_discrete,
    phase_lifetime, modulus_lifetime, lifetime_from_phasor,
    phase_lifetime_gated, fractional_components,
    plot_phasor, plot_locus_comparison, plot_discrete_N_sweep,
)

# This allows: from pyfli.scripts import DataViewer
__all__ = ["DataOperations", "IRFAligner", "DataViewer", "AlliGprocessedImport",
    "BHprocessedImport", "PyFliprocessedImport", "DatasetPlotter", "HardSimulator",
    "HardestSimulator", "PhasorAnalyzer", "Plotter", "DLModelComparator", "DataPreprocessing",
    "PlotConfig", "DataProcessor", "SourceLoader", "PlotKit", "SubplotVisualizer", "plot_2d_subplots",
    "BaseFLIFitter", "Fli_CPUProcessor", "Fli_GPUProcessor", "MLEFLIFitter", "GlobalFLIFitter",
    "ROIMaker", "AnalyticalHelpers", "DataIO_utils", "Colorprocess",
    "Macro_sim", "TCSPC_sim", "FLIImageGenerator", "recovery_plot", "random_true_pixel", "save_plot",
    "FLICalibrator", "FLIValidator", "Normalization", "Msg_display", "FittingComparator",
    "data_masking", "Detector", "BinnedFliFitter", "FliBinner", "ROIoperations",
    "Batch_sim", "DataSaver", "load_flim_data", "collapse_to_xyt", "plot_xyt",
    "LaguerreFLI", "plot_pixel_diagnostic", "compute_detailed_results", "MonoBiClassifier", "ParamCorrelationMatrix",
    "AcquisitionConfig", "AcquisitionMode",
    "phasor_continuous", "phasor_discrete", "phasor_gated_single", "phasor_gated_N",
    "phasor_truncated", "phasor_offset", "phasor_from_config",
    "build_locus", "build_loci", "tau_grid", "universal_semicircle", "sepl_center_radius_discrete",
    "phase_lifetime", "modulus_lifetime", "lifetime_from_phasor",
    "phase_lifetime_gated", "fractional_components",
    "plot_phasor", "plot_locus_comparison", "plot_discrete_N_sweep",
    "TCSPCParams", "SPADParams", "ICCDParams",
    "make_observation", "generalized_anscombe",
    "SolverConfig", "solve_flim",
    "build_gate_matrix", "decay_basis", "cyclic_conv",
    # analysis convenience functions
    "RESULT_FILENAMES", "FBI_RESULT_FILENAME", "FBI_RAW_FILENAME",
    "load_session_arrays", "scan_session_results", "load_fitting_results",
    "save_laguerre_result", "inject_phasor_result", "add_mean_lifetime",
    "compute_freq_axis", "compute_phasor", "plot_phasor_figures", "save_phasor_result",
    "plot_fitting_maps", "plot_diagnostics", "plot_pixel_evidence",
    "plot_statistical_comparison", "plot_2d_analysis",
    "run_mono_bi_classifier",
]
    