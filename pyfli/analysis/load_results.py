"""
Load saved PyFLI fitting sessions and inject derived analysis results.

This module belongs to :mod:`pyfli.analysis` and is part of PyFLI post-processing,
diagnostics, statistical comparison, and result-loading utilities for fitted FLI/FLIM
datasets. Public API includes functions :func:`load_session_arrays`,
:func:`scan_session_results`, :func:`load_fitting_results`,
:func:`save_laguerre_result`, :func:`inject_phasor_result`, and
:func:`add_mean_lifetime`.
"""

from typing import Any

from pyfli import logging

import os

import numpy as np


# Files that are NOT fitting results and should be excluded from scan output
_NON_RESULT_FILES = {
    "clean_decay.npy",
    "clean_irf.npy",
    "final_mask.npy",
    "decay_raw.npy",
    "irf_raw.npy",
    "pixel_invariant_irf.npy",
    "pf_mask.npy",
}

# Expected naming conventions for result files.
# Use these constants (or save_laguerre_result / compute_fbi_results) to ensure
# load_fitting_results() and scan_session_results() can find them.
RESULT_FILENAMES = {
    "laguerre_mono": "Laguerre Results_mono-exponential.npy",
    "laguerre_bi": "Laguerre Results_bi-exponential.npy",
    "fbi_bi": "F-BI Output_bi-exponential.npy",
    "phasor": "phasor_tau_map.npy",
}


def load_session_arrays(save_dir: str) -> tuple[Any, ...]:
    """Load clean_decay, clean_irf, and final_mask from a pf_Analysis session directory.

    Returns
    -------
    decay : np.ndarray  (H, W, T)
    irf   : np.ndarray  (H, W, T)
    mask  : np.ndarray  (H, W)  bool
    """
    decay = np.load(os.path.join(save_dir, "clean_decay.npy"), allow_pickle=True)
    irf = np.load(os.path.join(save_dir, "clean_irf.npy"), allow_pickle=True)
    mask = np.load(os.path.join(save_dir, "final_mask.npy"), allow_pickle=True).astype(
        bool
    )
    return decay, irf, mask


def scan_session_results(save_dir: str) -> np.ndarray:
    """
    Scan session results.

    Parameters
    ----------
    save_dir : str
        Directory where outputs are saved.

    Returns
    -------
    np.ndarray
        Session result arrays discovered from the output folder.
    """
    all_npy = sorted(f for f in os.listdir(save_dir) if f.endswith(".npy"))
    result_files = [f for f in all_npy if f not in _NON_RESULT_FILES]

    logging.info(f"Available fitting results in '{save_dir}':")
    if result_files:
        for i, fname in enumerate(result_files):
            logging.info(f"  [{i}] {fname}")
    else:
        logging.info("  (none found)")
    return result_files


def load_fitting_results(save_dir: str, experiments: np.ndarray) -> tuple[Any, ...]:
    """Load fitting results using a user-defined filename → label mapping.

    Parameters
    ----------
    save_dir    : str
        Path to the pf_Analysis session folder.
    experiments : dict[str, str]
        Maps each .npy filename to a short display label.
        You control exactly which results are loaded and in what order.
        Mix any model types freely (NLSF, MLE, Laguerre, FBI, etc.).

        Example — mono-exponential, CPU only::

            experiments = {
                'CPU_NLSF_least_squares_mono-exponential.npy': 'NLSF',
                'CPU_MLE_poisson_mono-exponential.npy':        'MLE',
                'Laguerre Results_mono-exponential.npy':       'Laguerre',
            }

        Example — bi-exponential, selective::

            experiments = {
                'CPU_NLSF_least_squares_bi-exponential.npy': 'NLSF-bi',
                'GPU_MLE_poisson_bi-exponential.npy':        'MLE-GPU-bi',
            }

    Returns
    -------
    all_datasets : list[dict]   parameter maps  (tau_map, alpha1_map, ...)
    all_fitset   : list[dict]   TR maps         (fit_map, residual_map)
    names        : list[str]    labels matching each entry, in dict order
    """
    all_datasets, all_fitset, names = [], [], []
    for file_name, label in experiments.items():
        file_path = os.path.join(save_dir, file_name)
        if not os.path.exists(file_path):
            logging.warning(
                f"[load_fitting_results] Skipping missing file: {file_name}"
            )
            continue
        var = np.load(file_path, allow_pickle=True).item()
        all_datasets.append(var["results"]["maps"])
        all_fitset.append(var["results"]["TR_maps"])
        names.append(label)

    if not all_datasets:
        raise FileNotFoundError(
            f"No fitting result files were found in '{save_dir}'. "
            "Run scan_session_results(save_dir) to see what is available."
        )
    return all_datasets, all_fitset, names


def save_laguerre_result(saver: Any, lag_results: np.ndarray, model_type: str) -> None:
    """
    Save laguerre result.

    Parameters
    ----------
    saver : Any
        Optional saver used to persist messages or figures.
    lag_results : np.ndarray
        Laguerre deconvolution results written into the saver.
    model_type : str
        FLI/FLIM model family, such as mono- or bi-exponential.

    Returns
    -------
    None
        No object is returned; the function save laguerre result.
    """
    key = f"laguerre_{model_type.split('-')[0]}"  # 'laguerre_mono' or 'laguerre_bi'
    if key not in RESULT_FILENAMES:
        raise ValueError(
            f"model_type must be 'mono-exponential' or 'bi-exponential', got '{model_type}'"
        )
    fname = RESULT_FILENAMES[key].replace(".npy", "")
    saver.save_npy(fname, lag_results)
    saver.log(f"Laguerre {model_type} results saved as {fname}.npy")


def inject_phasor_result(
    tau_map_ns: np.ndarray,
    all_datasets: np.ndarray,
    all_fitset: np.ndarray,
    names: Any,
    label: str = "Phasor",
) -> None:
    """
    Inject phasor result.

    Parameters
    ----------
    tau_map_ns : np.ndarray
        Lifetime map in nanoseconds.
    all_datasets : np.ndarray
        Collection of fitted datasets to classify, compare, or summarize.
    all_fitset : np.ndarray
        Collection of fit-result dictionaries used for comparison or plotting.
    names : Any
        Dataset names used in summaries and plots.
    label : str
        Display label assigned to the data or plot element.

    Returns
    -------
    None
        No object is returned; the function inject phasor result.
    """
    all_datasets.append(
        {
            "tau_map": tau_map_ns,
            "mean_lifetime": tau_map_ns,
        }
    )
    all_fitset.append({})
    names.append(label)


def add_mean_lifetime(all_datasets: np.ndarray) -> None:
    """
    Add mean lifetime.

    Parameters
    ----------
    all_datasets : np.ndarray
        Collection of fitted datasets to classify, compare, or summarize.

    Returns
    -------
    None
        No object is returned; the function add mean lifetime.
    """
    for ds in all_datasets:
        if "mean_lifetime" in ds:
            continue
        if not {"alpha1_map", "tau1_map", "tau2_map"}.issubset(ds):
            continue
        ds["mean_lifetime"] = (
            ds["alpha1_map"] * ds["tau1_map"] + ds["alpha2_map"] * ds["tau2_map"]
            if "alpha2_map" in ds
            else ds["alpha1_map"] * ds["tau1_map"]
            + (1 - ds["alpha1_map"]) * ds["tau2_map"]
        )
