"""
Collect numerical, masking, simulation, plotting, and export utilities shared by
analysis workflows.

This module belongs to :mod:`pyfli.analysis` and is part of PyFLI post-processing,
diagnostics, statistical comparison, and result-loading utilities for fitted FLIM
datasets. Public API includes functions :func:`circular_convolution_fft`,
:func:`single_ex_decay_summed_overtime`, :func:`gate_j`, :func:`Pj_continuous_mono`,
:func:`Pj_from_samples_mono`, :func:`multimodal_normal`, :func:`recovery_plot`,
:func:`threshold_masking`, :func:`data_masking`, and
:func:`save_3d_array_as_tiff_sequence`.
"""

from typing import Any

from pyfli import logging

import numpy as np
from scipy.integrate import quad
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import math
import os

import tifffile

from ..data_vnp.color_processor import ColorProcessor
from ..data_vnp.mono_bi_classifier import MonoBiClassifier


def circular_convolution_fft(
    x: np.ndarray, h: np.ndarray, broadcast_irf: bool = True
) -> np.ndarray:
    """
    Run the circular convolution FFT routine.

    Parameters
    ----------
    x : np.ndarray
        Input array, coordinate, or signal being transformed.
    h : np.ndarray
        IRF, image height, or temporal kernel used by the routine.
    broadcast_irf : bool
        Whether a shared IRF should be broadcast to every pixel.

    Returns
    -------
    np.ndarray
        Circular convolution result with the same length as the input decay.
    """
    x = np.asarray(x)
    h = np.asarray(h)

    if x.ndim != 3 or h.ndim != 3:
        raise ValueError(
            f"x and h must be 3D arrays, got x.ndim={x.ndim}, h.ndim={h.ndim}"
        )

    if x.shape[-1] != h.shape[-1]:
        raise ValueError(
            f"Last dimension (convolution axis) must match: {x.shape[-1]} vs {h.shape[-1]}"
        )

    # Broadcast h to match x (for pixel-wise or shared IRFs)
    if broadcast_irf:
        if h.shape[0] != x.shape[0] or h.shape[1] != x.shape[1]:
            h = np.broadcast_to(h, x.shape)
    # h = np.broadcast_to(h, x.shape)

    # Perform FFT along the last axis (axis=2)
    X_fft = np.fft.fft(x, axis=2)
    H_fft = np.fft.fft(h, axis=2)

    # Frequency-domain multiplication
    Y_fft = X_fft * H_fft

    # Inverse FFT to get real-valued circular convolution result
    y = np.real(np.fft.ifft(Y_fft, axis=2))

    return y


def single_ex_decay_summed_overtime(
    tau: np.ndarray,
    irf_data: np.ndarray,
    alpha: float = 1.0,
    err: float = 0.0,
    laser_period: float = 12.5,
    seed: int | None = None,
) -> tuple[Any, ...]:
    """
    Run the single ex decay summed overtime routine.

    Parameters
    ----------
    tau : np.ndarray
        Lifetime value or lifetime map in nanoseconds.
    irf_data : np.ndarray
        Instrument response data used to convolve or simulate decays.
    alpha : float
        Regularization strength, fraction value, or significance threshold used by the
    routine.
    err : float
        Noise or perturbation level applied to simulated decays.
    laser_period : float
        Laser repetition period in nanoseconds.
    seed : int | None
        Random seed used for reproducible sampling.

    Returns
    -------
    tuple[Any, ...]
        Tuple containing the integrated single-exponential decay and time samples.
    """
    if seed is not None:
        np.random.seed(seed)
    M, N, T = irf_data.shape
    tau = np.asarray(tau, dtype=float)

    # Ensure tau is broadcastable to (M, N, T)
    tau = np.broadcast_to(tau[..., np.newaxis], (M, N, T))

    # Time vector
    t = np.linspace(0, laser_period, T)[np.newaxis, np.newaxis, :]  # (1, 1, T)

    # --- Avoid division by zero ---
    zero_mask = (tau <= 0) | ~np.isfinite(tau)
    safe_tau = np.where(
        zero_mask, np.inf, tau
    )  # τ=0 → inf => exp(-t/inf)=1, then we zero it later

    # --- Theoretical single exponential decay ---
    f_t = (1.0 / safe_tau) * np.exp(-t / safe_tau)

    # Zero out pixels where tau=0 or invalid
    f_t[zero_mask] = 0.0

    # --- Normalize IRF per pixel ---
    I_sum = np.sum(irf_data, axis=2, keepdims=True)
    if np.any(I_sum <= 0):
        raise ValueError("One or more IRF pixels sum to zero; cannot normalize.")
    I_t = irf_data / I_sum

    # --- Circular convolution along time axis (axis=2) ---
    s_ti = circular_convolution_fft(f_t, I_t)

    # --- Add Gaussian noise ---
    if np.isscalar(err):
        noise = np.random.normal(0, err, size=s_ti.shape)
    else:
        noise = np.asarray(err, dtype=float)
        if noise.shape != s_ti.shape:
            raise ValueError("Shape mismatch: 'err' array must match signal shape")

    # --- Final weighted signal ---
    s_t = alpha * s_ti + (1.0 - alpha) * noise
    s_t = np.clip(s_t, 0.0, None)

    # Ensure f_t is zero wherever tau=0
    s_t[zero_mask] = 0.0

    return f_t, s_t, I_t, t


def gate_j(m: int, T: float) -> np.ndarray:
    """
    Run the gate j routine.

    Parameters
    ----------
    m : int
        Gate, harmonic, or interval index.
    T : float
        Time axis or acquisition period used by the calculation.

    Returns
    -------
    np.ndarray
        Integrated gate image or trace for the requested gate index.
    """
    buckets = []
    for j in range(1, m + 1):
        a = (j - 1) * T / m
        b = j * T / m
        buckets.append((a, b))
    return buckets


def Pj_continuous_mono(
    f: np.ndarray, m: int, T: float, epsabs: float = 1e-8, epsrel: float = 1e-8
) -> Any:
    """
    Run the pj continuous mono routine.

    Parameters
    ----------
    f : np.ndarray
        Decay basis, distribution, or signal function used by the calculation.
    m : int
        Gate, harmonic, or interval index.
    T : float
        Time axis or acquisition period used by the calculation.
    epsabs : float
        Absolute integration tolerance.
    epsrel : float
        Relative integration tolerance.

    Returns
    -------
    Any
        Object produced by pj continuous mono.
    """
    gates = np.array(gate_j(m, T))  # list-of-tuples → 2D array for slicing
    a_vals, b_vals = gates[:, 0], gates[:, 1]

    # Vectorized numerical integration using np.vectorize wrapper
    def integrate_interval(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Run the integrate interval routine.

        Parameters
        ----------
        a : np.ndarray
            Lower integration or interval bound.
        b : np.ndarray
            Upper integration or interval bound.

        Returns
        -------
        np.ndarray
            Integrated signal over the requested interval.
        """
        val, _ = quad(f, a, b, epsabs=epsabs, epsrel=epsrel)
        return val

    integrate_vec = np.vectorize(integrate_interval)
    Pj = integrate_vec(a_vals, b_vals)
    return Pj


def Pj_from_samples_mono(
    t_samples: np.ndarray, y_samples: np.ndarray, m: int, T: float
) -> Any:
    """
    Run the pj from samples mono routine.

    Parameters
    ----------
    t_samples : np.ndarray
        Sample times used to integrate a mono-exponential decay.
    y_samples : np.ndarray
        Sampled mono-exponential values integrated over gates.
    m : int
        Gate, harmonic, or interval index.
    T : float
        Time axis or acquisition period used by the calculation.

    Returns
    -------
    Any
        Object produced by pj from samples mono.
    """
    H, W, Tn = y_samples.shape
    gates = gate_j(m, T)

    # Ensure time axis and sample consistency
    if t_samples.shape[0] != Tn:
        raise ValueError("Length of t_samples must match y_samples.shape[-1].")

    # Interpolate gate edges and ensure inclusion
    Pj = np.zeros((H, W, m), dtype=float)
    for j, (a, b) in enumerate(gates):
        # Create boolean mask for time bins within gate
        mask = (t_samples >= a) & (t_samples <= b)

        # If gate falls outside sampled range, skip safely
        if not np.any(mask):
            continue

        # Extract y and t segments for integration
        t_sub = t_samples[mask]
        y_sub = y_samples[..., mask]

        # Include exact gate edges via vectorised linear interpolation (H,W pixels)
        if t_sub[0] > a:
            idx = int(np.clip(np.searchsorted(t_samples, a, side="right"), 1, Tn - 1))
            w = (a - t_samples[idx - 1]) / (
                t_samples[idx] - t_samples[idx - 1] + 1e-300
            )
            y_a = (
                y_samples[..., idx - 1] * (1.0 - w) + y_samples[..., idx] * w
            )  # (H, W)
            y_sub = np.concatenate((y_a[..., np.newaxis], y_sub), axis=-1)
            t_sub = np.concatenate(([a], t_sub))
        if t_sub[-1] < b:
            idx = int(np.clip(np.searchsorted(t_samples, b, side="left"), 0, Tn - 2))
            w = (b - t_samples[idx]) / (t_samples[idx + 1] - t_samples[idx] + 1e-300)
            y_b = (
                y_samples[..., idx] * (1.0 - w) + y_samples[..., idx + 1] * w
            )  # (H, W)
            y_sub = np.concatenate((y_sub, y_b[..., np.newaxis]), axis=-1)
            t_sub = np.concatenate((t_sub, [b]))

        # Integrate over time using trapezoidal rule (vectorized along last axis)
        Pj[..., j] = np.trapz(y_sub, x=t_sub, axis=-1)

    # Normalize to obtain probability distribution per pixel
    Pj_sum = np.sum(Pj, axis=-1, keepdims=True)
    Pj /= np.maximum(Pj_sum, 1e-12)

    return Pj


def multimodal_normal(
    n_samples: int = 10000,
    mus: np.ndarray | None = None,
    sigma: float | None = None,
    weights: np.ndarray | None = None,
    seed: int | None = None,
) -> tuple[Any, ...]:
    """
    Run the multimodal normal routine.

    Parameters
    ----------
    n_samples : int
        Number of samples, components, gates, or iterations used by the routine.
    mus : np.ndarray | None
        Gaussian component means used by the multimodal sampler.
    sigma : float | None
        Standard deviation used by a sampler or noise model.
    weights : np.ndarray | None
        Sampling or model weights used by the routine.
    seed : int | None
        Random seed used for reproducible sampling.

    Returns
    -------
    tuple[Any, ...]
        Tuple containing sampled values from the configured normal mixture.
    """
    np.random.seed(seed)

    if mus is None:
        raise ValueError("You must provide a list of means (mus).")
    mus = np.array(mus)
    n_modes = len(mus)

    # Ensure sigma matches mus
    if sigma is None:
        sigma = np.ones(n_modes) * 1.0  # default sigma = 1 for all modes
    elif isinstance(sigma, (int, float)):
        sigma = np.full(n_modes, sigma)
    else:
        sigma = np.array(sigma)
        assert len(sigma) == n_modes, (
            "sigma must be a single value or same length as mus"
        )

    # Equal weights if none provided
    if weights is None:
        weights = np.ones(n_modes) / n_modes
    else:
        weights = np.array(weights)
        weights /= weights.sum()  # normalize

    # Number of samples per mode
    samples_per_mode = np.random.multinomial(n_samples, weights)

    # Generate samples for each mode
    samples = []
    samples_2d = np.zeros(
        (n_modes, n_samples), dtype=float
    )  # n_samples cols = max possible
    for i, (mu_val, s, n) in enumerate(zip(mus, sigma, samples_per_mode)):
        samp = np.random.normal(loc=mu_val, scale=s, size=n)
        samples.append(samp)
        samples_2d[i, :n] = samp

    samples = np.concatenate(samples)

    # Ensure all values are positive (reflect negatives)
    samples = np.abs(samples)

    return samples, samples_2d


def recovery_plot(
    gt_dict: np.ndarray, est_dict: np.ndarray, keys_to_plot: np.ndarray | None = None
) -> np.ndarray:
    """
    Plots Ground Truth vs Estimates for specific keys.
    Handles data shapes: (N, X, Y) or (N, Batch, X, Y).

    Args:
        gt_dict: Dictionary of Ground Truth arrays.
        est_dict: Dictionary of Estimated arrays.
        keys_to_plot: List of strings (keys). If None, plots all keys in gt_dict.
    """
    if keys_to_plot is None:
        keys_to_plot = list(gt_dict.keys())

    # 1. Automatic Grid Arrangement
    num_plots = len(keys_to_plot)
    if num_plots == 0:
        return

    cols = min(num_plots, 4)
    rows = math.ceil(num_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows), squeeze=False)
    axes = axes.flatten()

    for i, key in enumerate(keys_to_plot):
        ax = axes[i]

        # Ensure data is numpy array and flatten (X, Y) -> (X*Y,)
        x = np.array(gt_dict[key]).flatten()
        y = np.array(est_dict[key]).flatten()

        # Calculate Pearson Correlation across all pixels
        r_val, _ = pearsonr(x, y)

        # 2. Scatter Plot
        # Using the style from your reference image
        ax.scatter(x, y, color="#2042a8", alpha=0.5, s=15, edgecolors="none")

        # 3. Identity Line (y = x) - UPDATED TO RED DASH
        all_vals = np.concatenate([x, y])
        # Calculate limits: start slightly below the absolute minimum
        data_min = np.min(all_vals)
        data_max = np.max(all_vals)
        buffer = (data_max - data_min) * 0.05

        plot_min = data_min - buffer
        plot_max = data_max + buffer

        ax.plot(
            [plot_min, plot_max],
            [plot_min, plot_max],
            color="red",
            linestyle="--",
            linewidth=1.5,
            zorder=5,
        )

        # 4. Styling & Formatting
        ax.set_title(key, fontsize=15)
        ax.set_xlabel("Ground truth", fontsize=12)

        # FORCE AXIS TO START FROM LESSER THAN MINIMUM
        ax.set_xlim(plot_min, plot_max)
        ax.set_ylim(plot_min, plot_max)

        if i % cols == 0:
            ax.set_ylabel("Estimate", fontsize=12)

        # Display r-value
        ax.text(
            0.05,
            0.92,
            f"$r = {r_val:.3f}$",
            transform=ax.transAxes,
            fontsize=13,
            fontweight="bold",
        )

        # Clean background and spines
        ax.grid(True, linestyle="-", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()
    return fig


def threshold_masking(
    fli: np.ndarray, irf: np.ndarray, threshold: int = 100
) -> tuple[Any, ...]:
    """
    Run the threshold masking routine.

    Parameters
    ----------
    fli : np.ndarray
        FLI lifetime map or decay-derived image to threshold.
    irf : np.ndarray
        Instrument response function aligned with the decay signal.
    threshold : int
        Threshold used to mask, classify, or validate data.

    Returns
    -------
    tuple[Any, ...]
        Tuple containing thresholded mask arrays and metadata.
    """
    if threshold is None:
        raise ValueError("no thershold value provided")
    else:
        intensity = np.sum(fli, axis=-1)
        mask = intensity > threshold

    mask = mask.astype(bool)
    if mask.ndim < fli.ndim:
        mask_expanded = mask[..., np.newaxis]
        masked_fli = fli * mask_expanded
        masked_irf = irf * mask_expanded
    else:
        masked_fli = fli * mask
        masked_irf = irf * mask

    return masked_fli, masked_irf


def data_masking(*arrays: Any, mask: np.ndarray, return_list: bool = False) -> Any:
    """
    Run the data masking routine.

    Parameters
    ----------
    *arrays : Any
        Additional positional values accepted by the routine.
    mask : np.ndarray
        Boolean or labeled mask selecting pixels for the operation.
    return_list : bool
        If ``True``, return a list of masks instead of a combined mask.

    Returns
    -------
    Any
        Object produced by data masking.
    """
    mask = mask.astype(bool)
    results = []
    for arr in arrays:
        if not isinstance(arr, np.ndarray):
            raise TypeError("All inputs must be numpy arrays")
        if mask.ndim < arr.ndim:
            expand_dims = arr.ndim - mask.ndim
            mask_expanded = mask[(...,) + (None,) * expand_dims]
        else:
            mask_expanded = mask
        try:
            masked = arr * mask_expanded
        except ValueError:
            raise ValueError("Mask is not broadcastable to array shape")
        results.append(masked)
    if len(results) == 1:
        return results[0]
    return results if return_list else tuple(results)


def save_3d_array_as_tiff_sequence(
    array_3d: np.ndarray, output_folder: str, prefix: str = "frame"
) -> None:
    """
    Saves a 3D numpy array (H, W, T) as a series of 2D TIFF files.

    Parameters:
    - array_3d: The numpy array of shape (H, W, T)
    - output_folder: Path to the folder where TIFs will be saved
    - prefix: Filename prefix (e.g., 'frame_001.tif')
    """
    # Create the directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    _, _, T = array_3d.shape

    logging.info(f"Saving {T} frames to '{output_folder}'...")

    for t in range(T):
        # Extract the 2D slice (X, Y) at time t
        # Note: tifffile expects (H, W), so we take [:, :, t]
        frame = array_3d[:, :, t]

        # Format filename with leading zeros for correct sorting (e.g., frame_005.tif)
        file_name = f"{prefix}_{t:03d}.tif"
        file_path = os.path.join(output_folder, file_name)

        # Save the slice
        tifffile.imwrite(file_path, frame.astype(np.float32))

    logging.info("Saving complete.")


def save_as_uint16_sequence(
    data: np.ndarray, output_folder: str, prefix: str = "frame"
) -> None:
    """
    Saves (H, W, T) array as 16-bit integer TIFFs.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 1. Handle Negative Values (Safety for uint16)
    # Background subtraction in your class might leave tiny negatives
    data = np.maximum(data, 0)

    # 2. Optional: Auto-Scaling (Only use if data is 0.0 - 1.0 or very small)
    # If your data is already raw photon counts, skip this step.
    if data.max() <= 1.0 and data.max() > 0:
        data = data * 65535

    # 3. Cast to uint16
    # This will truncate decimals (e.g., 1.9 becomes 1)
    data_uint16 = data.astype(np.uint16)

    _, _, T = data_uint16.shape
    for t in range(T):
        frame = data_uint16[:, :, t]
        file_path = os.path.join(output_folder, f"{prefix}_{t:03d}.tif")
        tifffile.imwrite(file_path, frame)

    logging.info(f"Saved {T} files to {output_folder} in uint16 format.")


def random_true_pixel(bool_array: np.ndarray) -> Any:
    """
    Run the random true pixel routine.

    Parameters
    ----------
    bool_array : np.ndarray
        Boolean array from which a true pixel is selected.

    Returns
    -------
    Any
        Object produced by random true pixel.
    """
    true_indices = np.flatnonzero(bool_array)
    if true_indices.size == 0:
        return None
    random_linear_idx = np.random.choice(true_indices)
    pix_x, pix_y = np.unravel_index(random_linear_idx, bool_array.shape)
    return int(pix_x), int(pix_y)


def PhasorFreqComputaion(
    laser_period: float = 12.5,
    gate_delay: np.ndarray | None = None,
    num_gates: int | None = None,
) -> np.ndarray:  # all the units in ns
    """
    Run the phasor freq computaion routine.

    Parameters
    ----------
    laser_period : float
        Laser repetition period in nanoseconds.
    gate_delay : np.ndarray | None
        Delay of each gate relative to the excitation pulse.
    num_gates : int | None
        Number of acquisition gates used for frequency computation.

    Returns
    -------
    np.ndarray
        Phasor frequency-domain representation for the input decay.
    """
    freq = 1000.0 / laser_period
    if gate_delay is None or num_gates is None:
        effective_freq = freq
    else:
        effective_freq = 1000.0 / (
            num_gates * gate_delay
        )  # frequency is computed in the MHz if the gate delays are in ns
    return effective_freq


def save_plot(
    save_dir: str,
    name: str,
    fig: Any | None = None,
    dpi: int = 300,
    close: bool = False,
) -> None:
    # Saves a plot. Handles subplots (pass fig) or direct plots (uses current)
    """
    Save plot.

    Parameters
    ----------
    save_dir : str
        Directory where outputs are saved.
    name : str
        Dataset, experiment, figure, or output name.
    fig : Any | None
        Matplotlib figure object to update or save.
    dpi : int
        Resolution used when saving a figure.
    close : bool
        Whether to close the figure after saving.

    Returns
    -------
    None
        No object is returned; the function save plot.
    """
    path = os.path.join(save_dir, f"{name}.png")
    target = fig if fig is not None else plt
    try:
        target.savefig(path, bbox_inches="tight", dpi=dpi)
    except Exception as e:
        logging.error(f"ERROR saving {name}: {str(e)}")
    if close:
        plt.close(fig) if fig else plt.close()


def plot_pixel_diagnostic(
    binned_decay: np.ndarray,
    all_fitset: np.ndarray,
    names: Any,
    pixel: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    t: np.ndarray | None = None,
    yscale: str = "log",
    model_type: str = "BI-EXPONENTIAL",
    colors: Any | None = None,
    figsize: tuple[int, ...] = (12, 6),
    raw_style: str = "bar",
    map_aspect: str = "equal",
    show_colorbar: bool = True,
    show: bool = True,
) -> np.ndarray:
    """
    Plot pixel diagnostic.

    Parameters
    ----------
    binned_decay : np.ndarray
        Binned decay cube used for fitting or diagnostics.
    all_fitset : np.ndarray
        Collection of fit-result dictionaries used for comparison or plotting.
    names : Any
        Dataset names used in summaries and plots.
    pixel : np.ndarray | None
        Selected pixel coordinate.
    mask : np.ndarray | None
        Boolean or labeled mask selecting pixels for the operation.
    t : np.ndarray | None
        Time axis or acquisition period used by the calculation.
    yscale : str
        Scale used for the y-axis.
    model_type : str
        FLIM model family, such as mono- or bi-exponential.
    colors : Any | None
        Color sequence used for plotted sources or groups.
    figsize : tuple[int, ...]
        Figure size passed to Matplotlib.
    raw_style : str
        Style used to draw raw pixel decay data.
    map_aspect : str
        Aspect ratio used when rendering lifetime maps.
    show_colorbar : bool
        Whether to draw a colorbar.
    show : bool
        Whether to display the generated plot.

    Returns
    -------
    np.ndarray
        Matplotlib figure or axes containing the pixel diagnostic plot.
    """
    jet_m = ColorProcessor().lowest_zero("jet")
    if pixel is None:
        if mask is None:
            raise ValueError("Provide either pixel=(row, col) or mask.")
        x, y = random_true_pixel(mask)
    else:
        x, y = pixel

    raw = np.asarray(binned_decay[x, y, :], dtype=float)
    bins = raw.shape[-1]
    if t is not None:
        xs = np.asarray(t, dtype=float).ravel()
        if len(xs) != bins:
            raise ValueError(
                f"t length ({len(xs)}) does not match decay bins ({bins}); "
                "x-axis and data would be misaligned."
            )
        xlabel = "Time (ns)"
    else:
        xs = np.arange(bins)
        xlabel = "Gate #"
    if colors is None:
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i % 10) for i in range(len(all_fitset))]
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 2], height_ratios=[3, 1])
    ax_map = fig.add_subplot(gs[:, 0])  # full-height left panel
    ax_top = fig.add_subplot(gs[0, 1])
    ax_bot = fig.add_subplot(gs[1, 1], sharex=ax_top)
    intensity = np.sum(binned_decay, axis=-1)  # (H, W)
    display_intensity = intensity if mask is None else intensity * mask
    im = ax_map.imshow(display_intensity, cmap=jet_m, aspect=map_aspect)
    # imshow's x-axis = columns, y-axis = rows -> mark pixel at (col, row)=(y, x)
    ax_map.scatter(y, x, marker="x", c="red", s=80, linewidths=2)
    ax_map.set_title("Intensity (Summed)")
    # ax_map.set_xlabel("Column"); ax_map.set_ylabel("Row")
    if show_colorbar:
        fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)

    if raw_style == "bar":
        width = (xs[1] - xs[0]) if len(xs) > 1 else 1.0
        ax_top.bar(
            xs,
            raw,
            width=width,
            color="0.8",
            edgecolor="none",
            zorder=1,
            label="Raw Data",
        )
    elif raw_style == "step":
        ax_top.plot(
            xs,
            raw,
            color="0.7",
            lw=0.9,
            drawstyle="steps-mid",
            zorder=1,
            label="Raw Data",
        )
    else:  # "line"
        ax_top.plot(xs, raw, color="0.7", lw=0.9, zorder=1, label="Raw Data")

    for i, fs in enumerate(all_fitset):
        label = names[i] if i < len(names) else f"Fit {i + 1}"
        fit = np.asarray(fs["fit_map"][x, y, :], dtype=float)
        ax_top.plot(
            xs, fit, color=colors[i], lw=1.3, zorder=2 + i, label=f"Fit: {label}"
        )

    ax_top.set_yscale(yscale)  # log / linear switch
    ax_top.set_ylabel("Photon Counts")
    ax_top.set_title(f"Fit Diagnostics ({model_type})  [pixel {x}, {y}]")
    ax_top.legend(ncol=2, fontsize=8, framealpha=0.9)
    if yscale == "log":
        pos = raw[raw > 0]
        if pos.size:
            ax_top.set_ylim(bottom=max(pos.min() * 0.1, 1e-3))

    ax_bot.axhline(0, color="black", lw=0.8, zorder=1)
    for i, fs in enumerate(all_fitset):
        label = names[i] if i < len(names) else f"Fit {i + 1}"
        res = np.asarray(fs["residual_map"][x, y, :], dtype=float)
        ax_bot.plot(xs, res, color=colors[i], lw=1.0, label=f"{label} residuals")
    ax_bot.set_ylabel("Residuals")
    ax_bot.set_xlabel(xlabel)

    fig.tight_layout()
    if show:
        plt.show()
    return fig


def compute_detailed_results(
    tau1: np.ndarray | None = None,
    tau2: np.ndarray | None = None,
    alpha1: np.ndarray | None = None,
    tau: np.ndarray | None = None,
    freq_acq: float | None = None,
    binned_irf: np.ndarray | None = None,
    binned_decay: np.ndarray | None = None,
    data_name: str = "F-BI",
    model_type: str = "bi-exponential",
    params: int | None = None,
    eps: float = 1e-8,
) -> dict[Any, Any]:
    """
    Reconstruct fit curves + goodness-of-fit maps from pre-estimated
    lifetime parameter maps (e.g. F-BI output), packaged in the same
    structure as FLICPUProcessor.process_image so it drops straight into
    Plotter / DataViewer.

    Accepts either bi-exponential inputs (tau1, tau2, alpha1) or a single
    mono-exponential lifetime map (tau), but not both. For the tau-only
    case, params defaults to 1 and, in the mono-exponential model_type
    branch, MonoBiClassifier is skipped entirely since every pixel is
    trivially mono by construction.

    Parameters
    ----------
    tau1 : np.ndarray | None
        Component-1 lifetime map (H, W), ns. Required with tau2/alpha1.
    tau2 : np.ndarray | None
        Component-2 lifetime map (H, W), ns. Required with tau1/alpha1.
    alpha1 : np.ndarray | None
        Amplitude fraction of component 1; component 2 = (1 - alpha1).
        Required with tau1/tau2.
    tau : np.ndarray | None
        Single lifetime map (H, W), ns, for the mono-only input style.
        Mutually exclusive with (tau1, tau2, alpha1).
    freq_acq : float
        Acquisition frequency freq[1] (MHz).
    binned_irf : np.ndarray
        IRF, shape (bins,) or (H, W, bins). A 1-D IRF is broadcast across
        all pixels; normalized to sum to 1 per pixel before convolving.
    binned_decay : np.ndarray | None
        Measured decay histogram per pixel, shape (H, W, bins). When
        omitted, decay-dependent outputs (photon count, residuals, chi²,
        R²) reduce to zero.
    data_name : str
        Dataset name recorded in the returned result dict.
    model_type : str
        ``"mono-exponential"`` or ``"bi-exponential"``.
    params : int | None
        Free-parameter count for the reduced-chi2 dof. Defaults to 3 for
        bi-exponential inputs, 1 for tau-only inputs.
    eps : float
        Numerical floor for clip / safe division.

    Returns
    -------
    dict[Any, Any]
        ``{'name', 'results': {'maps', 'error_maps', 'TR_maps'}}``
    """
    if model_type not in ("mono-exponential", "bi-exponential"):
        raise ValueError(
            f"Unknown model_type: {model_type!r}; expected "
            "'mono-exponential' or 'bi-exponential'"
        )

    # --- Resolve tau1 / tau2 / alpha1 from the two supported input styles -----
    have_biexp_inputs = (
        (tau1 is not None) and (tau2 is not None) and (alpha1 is not None)
    )
    have_tau_only = tau is not None

    if have_biexp_inputs and have_tau_only:
        raise ValueError("Pass either (tau1, tau2, alpha1) or (tau), not both.")
    elif have_biexp_inputs:
        tau1 = np.asarray(tau1, dtype=np.float32)
        tau2 = np.asarray(tau2, dtype=np.float32)
        alpha1 = np.asarray(alpha1, dtype=np.float32)
        if params is None:
            params = 3
    elif have_tau_only:
        tau = np.asarray(tau, dtype=np.float32)
        tau1 = tau
        tau2 = tau
        alpha1 = np.ones_like(tau, dtype=np.float32)
        if params is None:
            params = 1
    else:
        raise ValueError("Must supply either (tau1, tau2, alpha1) or (tau).")

    H, W = tau1.shape
    bins = binned_irf.shape[-1]

    # When binned_decay is absent, all decay-dependent outputs (photon count,
    # scaling, residuals, chi², R²) reduce to zero naturally.
    if binned_decay is None:
        binned_decay = np.zeros((H, W, bins), dtype=np.float32)
    else:
        binned_decay = np.asarray(binned_decay, dtype=np.float32)

    # --- IRF shape handling: make it broadcastable with sdf (H, W, bins) ------
    irf = np.asarray(binned_irf, dtype=np.float32)
    if irf.ndim == 1:
        irf = irf[np.newaxis, np.newaxis, :]  # (1, 1, bins) -> broadcasts
    elif irf.ndim != 3:
        raise ValueError(
            f"binned_irf must be 1-D (bins,) or 3-D (H,W,bins); got shape {irf.shape}"
        )

    # Normalize IRF to sum to 1 per pixel. This does NOT change any final
    # output (param_maps, chi2_map, R2_map, fret_efficiency_map, etc.) --
    # those all pass through a per-pixel fit_sum renormalization further
    # down that exactly cancels any constant IRF scale factor. It DOES make
    # the intermediate diagnostic maps (sdf_map, convolved_map in TR_maps)
    # directly comparable across pixels and consistent with
    # forward_model.model_numpy's convention, instead of carrying each
    # pixel's raw sum(irf) as an arbitrary scale factor.
    irf_sum = np.sum(irf, axis=-1, keepdims=True)
    irf = np.divide(irf, irf_sum, out=np.zeros_like(irf), where=irf_sum > eps)

    # ── mono-exponential branch ───────────────────────────────────────────────
    if model_type == "mono-exponential":
        photon_count = np.sum(binned_decay, axis=-1)  # (H, W)
        b_bool_mask = photon_count > 0

        if have_tau_only:
            # Single lifetime supplied directly -> no classification needed,
            # every pixel is trivially "mono" by construction.
            tau_eff = np.where(b_bool_mask, tau1, 0.0).astype(np.float32)
        else:
            # Package bi-exp maps into the format MonoBiClassifier expects
            dataset = {"alpha1_map": alpha1, "tau1_map": tau1, "tau2_map": tau2}
            all_datasets = [dataset]

            clf = MonoBiClassifier(
                b_bool_mask,
                names=[data_name],
                alpha_upper=0.95,
                alpha_lower=0.05,
                tau_tol=0.05,
                coord=None,
            )
            classes = clf.classify(all_datasets, display=True)

            mono_mask = classes[0]["mono_mask"]  # (H, W) bool, ROI-restricted

            # Assign pixelwise single tau guided by classification:
            #   mono (alpha1 > upper) → component-1 dominates → tau1
            #   mono (alpha1 < lower) → component-2 dominates → tau2
            #   mono (tau1 ≈ tau2)   → lifetimes coincide   → tau1
            #   bi                   → amplitude-weighted mean lifetime
            tau_mono = np.where(
                alpha1 > clf.alpha_upper,
                tau1,
                np.where(alpha1 < clf.alpha_lower, tau2, tau1),
            )  # tau1 ≈ tau2 coincidence case
            tau_bi = (
                alpha1 * tau1 + (1.0 - alpha1) * tau2
            )  # mean lifetime for bi pixels
            tau_eff = np.where(mono_mask, tau_mono, tau_bi)
            tau_eff = np.where(b_bool_mask, tau_eff, 0.0).astype(np.float32)

        # Pixelwise single-exponential model
        # endpoint=False matches NLSF/MLE's self.t = np.linspace(0, T_acq, N,
        # endpoint=False) exactly, so dt = T_acq/N here too (not T_acq/(N-1)).
        t = np.linspace(0, 1000.0 / freq_acq, bins, endpoint=False, dtype=np.float32)
        dt = float(t[1] - t[0])  # bin width (ns) == T_acq/N
        tau_b = np.clip(tau_eff[..., np.newaxis], eps, None)  # (H, W, 1)
        sdf = np.exp(-t / tau_b)  # (H, W, bins)

        # Convolve with IRF and scale to measured photon counts
        convolved_fit = fftconvolve(sdf, irf, mode="full", axes=-1)[..., :bins]
        fit_sum = np.sum(convolved_fit, axis=-1, keepdims=True)
        fit_pdf = np.zeros_like(convolved_fit)
        np.divide(convolved_fit, fit_sum, out=fit_pdf, where=fit_sum > eps)
        scaled_fit = photon_count[..., np.newaxis] * fit_pdf  # (H, W, bins)

        # Goodness of fit
        # Variance flooring matches shared_metrics.compute_fli_stats, which
        # clips the *entire* model array at 1.0 (not just non-positive
        # entries) -- this avoids huge chi² contributions from near-zero bins.
        variance = np.clip(scaled_fit, 1.0, None)
        # dof convention matches shared_metrics.compute_fli_stats (N - k).
        dof = max(bins - params, 1)
        residuals = binned_decay - scaled_fit
        chi_sq_raw = np.sum((residuals**2) / variance, axis=-1)
        chi_sq_map = chi_sq_raw / dof

        ss_res = np.sum(residuals**2, axis=-1)
        ss_tot = np.sum(
            (binned_decay - np.mean(binned_decay, axis=-1, keepdims=True)) ** 2, axis=-1
        )
        r2_map = np.ones((H, W), dtype=np.float32)
        np.divide(ss_res, ss_tot, out=r2_map, where=ss_tot > eps)
        r2_map = 1.0 - r2_map

        health = b_bool_mask.astype(np.float32)

        # Mono-exponential param_maps (solver convention: amp, tau, v_shift, h_shift)
        # NOTE: photon_count_map is reported here as "S" (the fitted-amplitude
        # convention used by the NLSF/MLE popt[0]), i.e. photon_count * dt,
        # NOT the raw measured sum(binned_decay). This makes it directly
        # comparable to _cpu_nlsf_bi / _cpu_mle_bi's photon_count_map, which
        # store popt[0] (S) as-is. The raw measured `photon_count` (sum of
        # binned_decay) is still used internally above for scaled_fit, since
        # that scaling must stay in actual-count units regardless of how the
        # output map is reported.
        param_maps = {
            "photon_count_map": (photon_count * dt).astype(np.float32),
            "tau_map": tau_eff,
            "v_shift_map": np.zeros((H, W), dtype=np.float32),
            "h_shift_map": np.zeros((H, W), dtype=np.float32),
            "R2_map": r2_map.astype(np.float32),
            "chi2_map": chi_sq_raw.astype(np.float32),
            "reduced_chi2_map": chi_sq_map.astype(np.float32),
            "convergence_map": health.copy(),
            "pixel_health_map": health,
        }

        error_maps = np.zeros((H, W, 3), dtype=np.float32)  # amp, tau, v_shift

        tr_maps = {
            "fit_map": scaled_fit.astype(np.float32),
            "residual_map": residuals.astype(np.float32),
            "sdf_map": sdf.astype(np.float32),
            "convolved_map": convolved_fit.astype(np.float32),
        }

        mask = b_bool_mask
        mean_chi_sq = float(np.mean(chi_sq_map[mask])) if np.any(mask) else np.nan
        logging.info(f"Mean Chi-Squared (Active Pixels): {mean_chi_sq:.4f}")

        return {
            "name": data_name,
            "method": "DirectCompute",
            "results": {
                "maps": param_maps,
                "error_maps": error_maps,
                "TR_maps": tr_maps,
            },
        }

    # ── bi-exponential branch ─────────────────────────────────────────────────
    # (Also handles the tau-only case: tau1 == tau2 == tau, alpha1 == 1, so the
    # second exponential term is weighted to zero and this collapses correctly
    # to a single exponential without any special-casing needed here.)

    # --- Model decay (sdf) ----------------------------------------------------
    tau1_b = np.clip(tau1[..., np.newaxis], eps, None)
    tau2_b = np.clip(tau2[..., np.newaxis], eps, None)
    alpha1_b = alpha1[..., np.newaxis]
    # endpoint=False matches NLSF/MLE's self.t = np.linspace(0, T_acq, N,
    # endpoint=False) exactly, so dt = T_acq/N here too (not T_acq/(N-1)).
    t = np.linspace(0, 1000.0 / freq_acq, bins, endpoint=False, dtype=np.float32)
    dt = float(t[1] - t[0])  # bin width (ns) == T_acq/N
    # Normalize each exponential component by its own tau, matching
    # forward_model.decay_kernel's bi-exponential branch:
    #   kernel = S * ((a1/tau1)*exp(-t/tau1) + ((1-a1)/tau2)*exp(-t/tau2))
    # Without the 1/tau factors, alpha1 does not behave as a true mixture
    # weight when tau1 != tau2 -- the resulting decay shape is wrong, and
    # renormalizing by fit_sum afterward only fixes overall scale, not shape.
    sdf = (alpha1_b / tau1_b) * np.exp(-t / tau1_b) + (
        (1.0 - alpha1_b) / tau2_b
    ) * np.exp(-t / tau2_b)

    # --- Convolve with IRF, crop back to bins -------------------------------
    convolved_fit = fftconvolve(sdf, irf, mode="full", axes=-1)[..., :bins]

    # --- Scale model PDF to measured photon counts ----------------------------
    photon_count = np.sum(binned_decay, axis=-1)  # (H, W)
    fit_sum = np.sum(convolved_fit, axis=-1, keepdims=True)
    fit_pdf = np.zeros_like(convolved_fit)
    np.divide(convolved_fit, fit_sum, out=fit_pdf, where=fit_sum > eps)
    scaled_fit = photon_count[..., np.newaxis] * fit_pdf  # (H, W, bins)

    # --- Goodness of fit ------------------------------------------------------
    # Same variance-flooring convention as the mono-exponential branch and as
    # shared_metrics.compute_fli_stats (clip the whole array at 1.0).
    variance = np.clip(scaled_fit, 1.0, None)
    # Same dof convention as shared_metrics.compute_fli_stats (N - k).
    dof = max(bins - params, 1)
    residuals = binned_decay - scaled_fit
    sq_err = (residuals**2) / variance
    chi_sq_raw = np.sum(sq_err, axis=-1)  # raw chi-square
    chi_sq_map = chi_sq_raw / dof  # reduced chi-square

    # R^2 per pixel (safe division)
    ss_res = np.sum(residuals**2, axis=-1)
    ss_tot = np.sum(
        (binned_decay - np.mean(binned_decay, axis=-1, keepdims=True)) ** 2, axis=-1
    )
    r2_map = np.ones((H, W), dtype=np.float32)
    np.divide(ss_res, ss_tot, out=r2_map, where=ss_tot > eps)
    r2_map = 1.0 - r2_map

    # params are given -> treat every valid pixel as "converged / healthy"
    health = (photon_count > 0).astype(np.float32)

    # --- Package-compatible 2-D maps ------------------------------------------
    # NOTE: photon_count_map is reported here as "S" (the fitted-amplitude
    # convention used by the NLSF/MLE popt[0]), i.e. photon_count * dt, NOT
    # the raw measured sum(binned_decay). This makes it directly comparable
    # to _cpu_nlsf_bi / _cpu_mle_bi's photon_count_map, which store popt[0]
    # (S) as-is. The raw measured `photon_count` (sum of binned_decay) is
    # still used above for scaled_fit, since that scaling must stay in
    # actual-count units regardless of how the output map is reported.
    tau1_f = tau1.astype(np.float32)
    tau2_f = tau2.astype(np.float32)
    alpha1_f = alpha1.astype(np.float32)
    ratio = np.divide(
        tau1_f,
        tau2_f,
        out=np.zeros_like(tau1_f, dtype=np.float32),
        where=(tau2_f > 0),
    )
    param_maps = {
        "photon_count_map": (photon_count * dt).astype(np.float32),
        "alpha1_map": alpha1_f,
        "tau1_map": tau1_f,
        "tau2_map": tau2_f,
        "tau_mean_map": (alpha1_f * tau1_f + (1.0 - alpha1_f) * tau2_f),
        "fret_efficiency_map": np.where(tau2_f > 0, 1.0 - ratio, 0.0).astype(
            np.float32
        ),
        "v_shift_map": np.zeros((H, W), dtype=np.float32),
        "h_shift_map": np.zeros((H, W), dtype=np.float32),
        "R2_map": r2_map.astype(np.float32),
        "chi2_map": chi_sq_raw.astype(np.float32),
        "reduced_chi2_map": chi_sq_map.astype(np.float32),
        "convergence_map": health.copy(),
        "pixel_health_map": health,
    }

    # No per-parameter uncertainties here -> zeros, shaped like the package e_maps
    internal_popt_len = 5 if model_type == "bi-exponential" else 3
    error_maps = np.zeros((H, W, internal_popt_len), dtype=np.float32)

    tr_maps = {
        "fit_map": scaled_fit.astype(np.float32),
        "residual_map": residuals.astype(np.float32),
        "sdf_map": sdf.astype(np.float32),  # model before scaling
        "convolved_map": convolved_fit.astype(np.float32),
    }

    mask = photon_count > 0
    mean_chi_sq = float(np.mean(chi_sq_map[mask])) if np.any(mask) else np.nan
    logging.info(f"Mean Chi-Squared (Active Pixels): {mean_chi_sq:.4f}")

    return {
        "name": data_name,
        "method": "DirectCompute",
        "results": {
            "maps": param_maps,
            "error_maps": error_maps,
            "TR_maps": tr_maps,
        },
    }
