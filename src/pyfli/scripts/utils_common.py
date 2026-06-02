import numpy as np 
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy.io import loadmat
import random
from scipy.stats import pearsonr
import math
import os
import tifffile

def circular_convolution_fft(x, h, broadcast_irf=True):
        x = np.asarray(x)
        h = np.asarray(h)

        if x.ndim != 3 or h.ndim != 3:
            raise ValueError(f"x and h must be 3D arrays, got x.ndim={x.ndim}, h.ndim={h.ndim}")

        if x.shape[-1] != h.shape[-1]:
            raise ValueError(f"Last dimension (convolution axis) must match: {x.shape[-1]} vs {h.shape[-1]}")

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
    tau,
    irf_data,
    alpha=1.0,
    err=0.0,
    laser_period=12.5,
    seed=None,
):
   
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
    safe_tau = np.where(zero_mask, np.inf, tau)  # τ=0 → inf => exp(-t/inf)=1, then we zero it later

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

def gate_j(m: int, T: float):
    buckets = []
    for j in range(1, m + 1):
        a = (j - 1) * T / m
        b = j * T / m
        buckets.append((a, b))
    return buckets

def Pj_continuous_mono(f, m: int, T: float, epsabs=1e-8, epsrel=1e-8):
    gates = np.array(gate_j(m, T))          # list-of-tuples → 2D array for slicing
    a_vals, b_vals = gates[:, 0], gates[:, 1]

    # Vectorized numerical integration using np.vectorize wrapper
    def integrate_interval(a, b):
        val, _ = quad(f, a, b, epsabs=epsabs, epsrel=epsrel)
        return val

    integrate_vec = np.vectorize(integrate_interval)
    Pj = integrate_vec(a_vals, b_vals)
    return Pj


def Pj_from_samples_mono(t_samples: np.ndarray, y_samples: np.ndarray, m: int, T: float):
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
                idx = int(np.clip(np.searchsorted(t_samples, a, side='right'), 1, Tn - 1))
                w   = (a - t_samples[idx - 1]) / (t_samples[idx] - t_samples[idx - 1] + 1e-300)
                y_a = y_samples[..., idx - 1] * (1.0 - w) + y_samples[..., idx] * w  # (H, W)
                y_sub = np.concatenate((y_a[..., np.newaxis], y_sub), axis=-1)
                t_sub = np.concatenate(([a], t_sub))
            if t_sub[-1] < b:
                idx = int(np.clip(np.searchsorted(t_samples, b, side='left'), 0, Tn - 2))
                w   = (b - t_samples[idx]) / (t_samples[idx + 1] - t_samples[idx] + 1e-300)
                y_b = y_samples[..., idx] * (1.0 - w) + y_samples[..., idx + 1] * w  # (H, W)
                y_sub = np.concatenate((y_sub, y_b[..., np.newaxis]), axis=-1)
                t_sub = np.concatenate((t_sub, [b]))

            # Integrate over time using trapezoidal rule (vectorized along last axis)
            Pj[..., j] = np.trapz(y_sub, x=t_sub, axis=-1)

        # Normalize to obtain probability distribution per pixel
        Pj_sum = np.sum(Pj, axis=-1, keepdims=True)
        Pj /= np.maximum(Pj_sum, 1e-12)

        return Pj

def multimodal_normal(n_samples=10000, mus=None, sigma=None, weights=None, seed=None):
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
        assert len(sigma) == n_modes, "sigma must be a single value or same length as mus"
    
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
    samples_2d = np.zeros((n_modes, n_samples), dtype=float)   # n_samples cols = max possible
    for i, (mu_val, s, n) in enumerate(zip(mus, sigma, samples_per_mode)):
        samp = np.random.normal(loc=mu_val, scale=s, size=n)
        samples.append(samp)
        samples_2d[i, :n] = samp
    
    samples = np.concatenate(samples)
    
    # Ensure all values are positive (reflect negatives)
    samples = np.abs(samples)
    
    return samples, samples_2d


def recovery_plot(gt_dict, 
                est_dict, 
                keys_to_plot=None):
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
    if num_plots == 0: return
    
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
        ax.scatter(x, y, color="#2042a8", alpha=0.5, s=15, edgecolors='none')
        
        # 3. Identity Line (y = x) - UPDATED TO RED DASH
        all_vals = np.concatenate([x, y])
        # Calculate limits: start slightly below the absolute minimum
        data_min = np.min(all_vals)
        data_max = np.max(all_vals)
        buffer = (data_max - data_min) * 0.05
        
        plot_min = data_min - buffer
        plot_max = data_max + buffer
        
        ax.plot([plot_min, plot_max], [plot_min, plot_max], color='red', linestyle='--', linewidth=1.5, zorder=5)

        # 4. Styling & Formatting
        ax.set_title(key, fontsize=15)
        ax.set_xlabel("Ground truth", fontsize=12)
        
        # FORCE AXIS TO START FROM LESSER THAN MINIMUM
        ax.set_xlim(plot_min, plot_max)
        ax.set_ylim(plot_min, plot_max)
        
        if i % cols == 0:
            ax.set_ylabel("Estimate", fontsize=12)
        
        # Display r-value
        ax.text(0.05, 0.92, f'$r = {r_val:.3f}$', transform=ax.transAxes, 
                fontsize=13, fontweight='bold')
        
        # Clean background and spines
        ax.grid(True, linestyle='-', alpha=0.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def threshold_masking(fli, irf, threshold=100):
        if threshold is None:
            raise ValueError('no thershold value provided')
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

def data_masking(*arrays, mask, return_list=False):
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

def save_3d_array_as_tiff_sequence(array_3d, output_folder, prefix="frame"):
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
        
    H, W, T = array_3d.shape
    
    print(f"Saving {T} frames to '{output_folder}'...")

    for t in range(T):
        # Extract the 2D slice (X, Y) at time t
        # Note: tifffile expects (H, W), so we take [:, :, t]
        frame = array_3d[:, :, t]
        
        # Format filename with leading zeros for correct sorting (e.g., frame_005.tif)
        file_name = f"{prefix}_{t:03d}.tif"
        file_path = os.path.join(output_folder, file_name)
        
        # Save the slice
        tifffile.imwrite(file_path, frame.astype(np.float32))

    print("Saving complete.")

def save_as_uint16_sequence(data, output_folder, prefix="frame"):
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

    H, W, T = data_uint16.shape
    for t in range(T):
        frame = data_uint16[:, :, t]
        file_path = os.path.join(output_folder, f"{prefix}_{t:03d}.tif")
        tifffile.imwrite(file_path, frame)
    
    print(f"Saved {T} files to {output_folder} in uint16 format.")


def random_true_pixel(bool_array):
    true_indices = np.flatnonzero(bool_array)    
    if true_indices.size == 0:
        return None
    random_linear_idx = np.random.choice(true_indices)
    pix_x, pix_y = np.unravel_index(random_linear_idx, bool_array.shape)
    return int(pix_x), int(pix_y)

def PhasorFreqComputaion(laser_period = 12.5, gate_delay = None, num_gates = None): # all the units in ns
    freq = 1000.0/laser_period
    if  gate_delay is None or num_gates is None:
        effective_freq = freq
    else:
        effective_freq = 1000.0/(num_gates*gate_delay) # frequency is computed in the MHz if the gate delays are in ns
    return effective_freq

def save_plot(save_dir, name, fig=None, dpi=300, close=False):
    # Saves a plot. Handles subplots (pass fig) or direct plots (uses current)
    path = os.path.join(save_dir, f"{name}.png")
    target = fig if fig is not None else plt    
    try:
        target.savefig(path, bbox_inches='tight', dpi=dpi)
    except Exception as e:
        print(f"ERROR saving {name}: {str(e)}")    
    if close:
        plt.close(fig) if fig else plt.close()


from .dataVnP.colorProcess import Colorprocess

def plot_pixel_diagnostic(binned_decay, all_fitset, names,
                          pixel=None, mask=None, t=None,
                          yscale="log", model_type="BI-EXPONENTIAL",
                          colors=None, figsize=(12, 6),
                          raw_style="bar", map_aspect="equal",
                          show_colorbar=True, show=True):
    jet_m = Colorprocess().lowest_zero('jet')
    if pixel is None:
        if mask is None:
            raise ValueError("Provide either pixel=(row, col) or mask.")
        x, y = random_true_pixel(mask)
    else:
        x, y = pixel

    raw  = np.asarray(binned_decay[x, y, :], dtype=float)
    bins = raw.shape[-1]
    if t is not None:
        xs = np.asarray(t, dtype=float).ravel()[:bins]
        xlabel = "Time (ns)"
    else:
        xs = np.arange(bins)
        xlabel = "Gate #"
    if colors is None:
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i % 10) for i in range(len(all_fitset))]
    fig = plt.figure(figsize=figsize)
    gs  = fig.add_gridspec(2, 2, width_ratios=[1.1, 2], height_ratios=[3, 1])
    ax_map = fig.add_subplot(gs[:, 0])               # full-height left panel
    ax_top = fig.add_subplot(gs[0, 1])
    ax_bot = fig.add_subplot(gs[1, 1], sharex=ax_top)
    axd = {"map": ax_map, "decay": ax_top, "residual": ax_bot}
    intensity = np.sum(binned_decay, axis=-1)        # (H, W)
    im = ax_map.imshow(intensity*mask, cmap=jet_m, aspect=map_aspect)
    # imshow's x-axis = columns, y-axis = rows -> mark pixel at (col, row)=(y, x)
    ax_map.scatter(y, x, marker="x", c="red", s=80, linewidths=2)
    ax_map.set_title("Intensity (Summed)")
    # ax_map.set_xlabel("Column"); ax_map.set_ylabel("Row")
    if show_colorbar:
        fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)

    if raw_style == "bar":
        width = (xs[1] - xs[0]) if len(xs) > 1 else 1.0
        ax_top.bar(xs, raw, width=width, color="0.8", edgecolor="none",
                   zorder=1, label="Raw Data")
    elif raw_style == "step":
        ax_top.plot(xs, raw, color="0.7", lw=0.9, drawstyle="steps-mid",
                    zorder=1, label="Raw Data")
    else:  # "line"
        ax_top.plot(xs, raw, color="0.7", lw=0.9, zorder=1, label="Raw Data")

    for i, fs in enumerate(all_fitset):
        label = names[i] if i < len(names) else f"Fit {i + 1}"
        fit = np.asarray(fs["fit_map"][x, y, :], dtype=float)
        ax_top.plot(xs, fit, color=colors[i], lw=1.3, zorder=2 + i,
                    label=f"Fit: {label}")

    ax_top.set_yscale(yscale)                        # log / linear switch
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
    return fig, axd, (x, y)


import numpy as np
from scipy.signal import fftconvolve


def compute_detailed_results(tau1, tau2, f, freq_acq, binned_irf, binned_decay,
                        data_name="F-BI", model_type="bi-exponential",
                        params=3, eps=1e-8):
    """
    Reconstruct fit curves + goodness-of-fit maps from pre-estimated
    bi-exponential parameter maps (e.g. F-BI output), packaged in the same
    structure as Fli_CPUProcessor.process_image so it drops straight into
    Plotter / DataViewer.

    Parameters
    ----------
    tau1, tau2 : np.ndarray (H, W)   lifetime maps (ns, matching 1000/freq_acq)
    f          : np.ndarray (H, W)   fraction of component 1; component 2 = (1 - f)
    freq_acq   : float               acquisition frequency freq[1] (MHz)
    binned_irf : np.ndarray          IRF, shape (bins,) or (H, W, bins).
                                     A 1-D IRF is broadcast across all pixels.
    binned_decay : np.ndarray (H, W, bins)   measured decay histogram per pixel
    params     : int                 free-parameter count for the reduced-chi2 dof
    eps        : float               numerical floor for clip / safe division

    Returns
    -------
    {'name', 'results': {'maps', 'error_maps', 'TR_maps'}}
    """
    tau1 = np.asarray(tau1, dtype=np.float32)
    tau2 = np.asarray(tau2, dtype=np.float32)
    f    = np.asarray(f,    dtype=np.float32)
    binned_decay = np.asarray(binned_decay, dtype=np.float32)

    H, W = tau1.shape
    bins = binned_irf.shape[-1]

    # --- IRF shape handling: make it broadcastable with sdf (H, W, bins) ------
    irf = np.asarray(binned_irf, dtype=np.float32)
    if irf.ndim == 1:
        irf = irf[np.newaxis, np.newaxis, :]          # (1, 1, bins) -> broadcasts
    elif irf.ndim != 3:
        raise ValueError("binned_irf must be 1-D (bins,) or 3-D (H,W,bins); "
                         f"got shape {irf.shape}")

    # --- Model decay (sdf) ----------------------------------------------------
    tau1_b = np.clip(tau1[..., np.newaxis], eps, None)
    tau2_b = np.clip(tau2[..., np.newaxis], eps, None)
    f_b    = f[..., np.newaxis]
    t = np.linspace(0, 1000.0 / freq_acq, bins, dtype=np.float32)
    sdf = f_b * np.exp(-t / tau1_b) + (1.0 - f_b) * np.exp(-t / tau2_b)

    # --- Convolve with IRF, crop back to bins -------------------------------
    convolved_fit = fftconvolve(sdf, irf, mode="full", axes=-1)[..., :bins]

    # --- Scale model PDF to measured photon counts ----------------------------
    photon_count = np.sum(binned_decay, axis=-1)                      # (H, W)
    fit_sum = np.sum(convolved_fit, axis=-1, keepdims=True)
    fit_pdf = np.zeros_like(convolved_fit)
    np.divide(convolved_fit, fit_sum, out=fit_pdf, where=fit_sum > eps)
    scaled_fit = photon_count[..., np.newaxis] * fit_pdf             # (H, W, bins)

    # --- Goodness of fit ------------------------------------------------------
    variance = scaled_fit.copy()
    variance[variance <= 0] = 1.0          # Poisson variance can't be 0 for chi-sq
    dof = max(bins - params - 1, 1)
    residuals  = binned_decay - scaled_fit
    sq_err     = (residuals ** 2) / variance
    chi_sq_raw = np.sum(sq_err, axis=-1)               # raw chi-square
    chi_sq_map = chi_sq_raw / dof                      # reduced chi-square

    # R^2 per pixel (safe division)
    ss_res = np.sum(residuals ** 2, axis=-1)
    ss_tot = np.sum((binned_decay - np.mean(binned_decay, axis=-1, keepdims=True)) ** 2,
                    axis=-1)
    r2_map = np.ones((H, W), dtype=np.float32)
    np.divide(ss_res, ss_tot, out=r2_map, where=ss_tot > eps)
    r2_map = 1.0 - r2_map

    # params are given -> treat every valid pixel as "converged / healthy"
    health = (photon_count > 0).astype(np.float32)

    # --- Package-compatible 2-D maps ------------------------------------------
    param_maps = {
        "Area_map":             photon_count.astype(np.float32),   # scale = total counts
        "alpha1_map":           f.astype(np.float32),
        "tau1_map":             tau1.astype(np.float32),
        "tau2_map":             tau2.astype(np.float32),
        "offset_map":           np.zeros((H, W), dtype=np.float32),  # no offset in model
        "R2_map":               r2_map.astype(np.float32),
        "chi2_or_deviance_map": chi_sq_raw.astype(np.float32),
        "reduced_stat_map":     chi_sq_map.astype(np.float32),
        "convergence_map":      health.copy(),
        "pixel_health_map":     health,
        "photon_count_map":     photon_count.astype(np.float32),
    }

    # No per-parameter uncertainties here -> zeros, shaped like the package e_maps
    internal_popt_len = 5 if model_type == "bi-exponential" else 3
    error_maps = np.zeros((H, W, internal_popt_len), dtype=np.float32)

    tr_maps = {
        "fit_map":       scaled_fit.astype(np.float32),
        "residual_map":  residuals.astype(np.float32),
        "sdf_map":       sdf.astype(np.float32),          # model before scaling
        "convolved_map": convolved_fit.astype(np.float32),
    }

    mask = photon_count > 0
    mean_chi_sq = float(np.mean(chi_sq_map[mask])) if np.any(mask) else np.nan
    print(f"Mean Chi-Squared (Active Pixels): {mean_chi_sq:.4f}")

    return {
        "name": data_name,
        "results": {
            "maps":       param_maps,
            "error_maps": error_maps,
            "TR_maps":    tr_maps,
        },
    }

