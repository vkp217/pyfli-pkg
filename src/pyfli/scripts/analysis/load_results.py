import os
import numpy as np


# Files that are NOT fitting results and should be excluded from scan output
_NON_RESULT_FILES = {'clean_decay.npy', 'clean_irf.npy', 'final_mask.npy',
                     'decay_raw.npy', 'irf_raw.npy', 'pixel_invariant_irf.npy',
                     'pf_mask.npy'}

# Expected naming conventions for result files.
# Use these constants (or save_laguerre_result / compute_fbi_results) to ensure
# load_fitting_results() and scan_session_results() can find them.
RESULT_FILENAMES = {
    'laguerre_mono': 'Laguerre Results_mono-exponential.npy',
    'laguerre_bi':   'Laguerre Results_bi-exponential.npy',
    'fbi_bi':        'F-BI Output_bi-exponential.npy',
    'phasor':        'phasor_tau_map.npy',
}


def load_session_arrays(save_dir):
    """Load clean_decay, clean_irf, and final_mask from a pf_Analysis session directory.

    Returns
    -------
    decay : np.ndarray  (H, W, T)
    irf   : np.ndarray  (H, W, T)
    mask  : np.ndarray  (H, W)  bool
    """
    decay = np.load(os.path.join(save_dir, 'clean_decay.npy'), allow_pickle=True)
    irf   = np.load(os.path.join(save_dir, 'clean_irf.npy'),   allow_pickle=True)
    mask  = np.load(os.path.join(save_dir, 'final_mask.npy'),  allow_pickle=True).astype(bool)
    return decay, irf, mask


def scan_session_results(save_dir):
    """Print and return all fitting result .npy files found in the session directory.

    Use this to discover available files before building your own experiments dict.

    Example
    -------
    >>> scan_session_results(saver.save_dir)
    Available fitting results in '.../_bin_2_pf_Analysis':
      [0] CPU_NLSF_least_squares_mono-exponential.npy
      [1] CPU_MLE_poisson_mono-exponential.npy
      [2] Laguerre Results_bi-exponential.npy

    Returns
    -------
    list[str]  filenames (not full paths)
    """
    all_npy = sorted(f for f in os.listdir(save_dir) if f.endswith('.npy'))
    result_files = [f for f in all_npy if f not in _NON_RESULT_FILES]

    print(f"Available fitting results in '{save_dir}':")
    if result_files:
        for i, fname in enumerate(result_files):
            print(f"  [{i}] {fname}")
    else:
        print("  (none found)")
    return result_files


def load_fitting_results(save_dir, experiments):
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
            print(f"[load_fitting_results] Skipping missing file: {file_name}")
            continue
        var = np.load(file_path, allow_pickle=True).item()
        all_datasets.append(var['results']['maps'])
        all_fitset.append(var['results']['TR_maps'])
        names.append(label)

    if not all_datasets:
        raise FileNotFoundError(
            f"No fitting result files were found in '{save_dir}'. "
            "Run scan_session_results(save_dir) to see what is available."
        )
    return all_datasets, all_fitset, names


def save_laguerre_result(saver, lag_results, model_type):
    """Save Laguerre results with model-type suffix for unambiguous reloading.

    Uses the standardised filename from RESULT_FILENAMES so the file is
    immediately discoverable by scan_session_results() and loadable by
    load_fitting_results().

    Parameters
    ----------
    saver      : DataSaver
    lag_results: dict   returned by LaguerreFLI.fit().get_parameters()
    model_type : str    'mono-exponential' or 'bi-exponential'

    Example
    -------
    ::

        Lag_results = Lag_model.fit(binned_decay, binned_irf).get_parameters(...)
        save_laguerre_result(saver, Lag_results, model_type='bi-exponential')
        # → writes 'Laguerre Results_bi-exponential.npy'
    """
    key = f'laguerre_{model_type.split("-")[0]}'   # 'laguerre_mono' or 'laguerre_bi'
    if key not in RESULT_FILENAMES:
        raise ValueError(f"model_type must be 'mono-exponential' or 'bi-exponential', got '{model_type}'")
    fname = RESULT_FILENAMES[key].replace('.npy', '')
    saver.save_npy(fname, lag_results)
    saver.log(f'Laguerre {model_type} results saved as {fname}.npy')


def inject_phasor_result(tau_map_ns, all_datasets, all_fitset, names,
                         label='Phasor'):
    """Inject a phasor lifetime map directly into existing result lists.

    Use this when the phasor tau is already in memory (from compute_phasor)
    and you want to include it in comparisons without writing to disk first.
    The map is added under both ``'tau_map'`` and ``'mean_lifetime'`` keys so
    it aligns with mono-exponential and bi-exponential comparison keys.

    Parameters
    ----------
    tau_map_ns   : np.ndarray  (H, W)  from compute_phasor()
    all_datasets : list[dict]   modified in-place
    all_fitset   : list[dict]   modified in-place
    names        : list[str]    modified in-place
    label        : str          display label (default 'Phasor')

    Example
    -------
    ::

        _, _, _, tau_phasor = compute_phasor(decay, irf, freq_hz, time_axis_ns)
        all_datasets, all_fitset, names = load_fitting_results(save_dir, experiments)
        inject_phasor_result(tau_phasor, all_datasets, all_fitset, names)
    """
    all_datasets.append({
        'tau_map':       tau_map_ns,
        'mean_lifetime': tau_map_ns,
    })
    all_fitset.append({})
    names.append(label)


def add_mean_lifetime(all_datasets):
    """Compute and insert ``mean_lifetime`` into each bi-exponential dataset dict.

    Mean lifetime is defined as ``alpha1_map * tau1_map + alpha2_map * tau2_map``.
    Datasets that already contain ``'mean_lifetime'`` or that lack bi-exponential
    keys are left unchanged.

    Parameters
    ----------
    all_datasets : list[dict]   modified in-place

    Example
    -------
    ::

        all_datasets, all_fitset, names = load_fitting_results(save_dir, experiments)
        add_mean_lifetime(all_datasets)
        # now use map_keys=['mean_lifetime'] in plot_statistical_comparison / plot_2d_analysis
    """
    for ds in all_datasets:
        if 'mean_lifetime' in ds:
            continue
        if not {'alpha1_map', 'tau1_map', 'tau2_map'}.issubset(ds):
            continue
        ds['mean_lifetime'] = ds['alpha1_map'] * ds['tau1_map'] + ds['alpha2_map'] * ds['tau2_map'] \
            if 'alpha2_map' in ds else ds['alpha1_map'] * ds['tau1_map'] + (1 - ds['alpha1_map']) * ds['tau2_map']
