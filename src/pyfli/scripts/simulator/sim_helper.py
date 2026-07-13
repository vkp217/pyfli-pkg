"""Shared helper for picking a usable Instrument Response Function (IRF).

Provides ``irf_picker``, which selects a single 1-D IRF trace either
directly (if already 1-D) or by searching a 3-D IRF stack for a pixel with
sufficient signal-to-noise ratio.
"""

import numpy as np

def irf_picker(irf_full):
    """Selects a single usable 1-D IRF trace from 1-D or 3-D input data.

    For a 3-D IRF stack (``H x W x T``), repeatedly picks a random pixel
    and accepts it only if its peak exceeds 500 counts and its estimated
    SNR (peak / pre-peak baseline noise) is at least 20, i.e. baseline
    noise standard deviation no greater than 5% of the peak. The baseline
    noise is estimated from the samples before the peak (or, if the peak
    sits within the first 3 samples, from the last quarter of the trace
    instead). Gives up after 1000 attempts.

    Args:
        irf_full: IRF data, either a 1-D array (used directly) or a 3-D
            array of shape ``(H, W, T)`` to search for a valid pixel.

    Returns:
        numpy.ndarray: A 1-D IRF trace of length ``T``.

    Raises:
        RuntimeError: If no pixel meeting the peak/SNR criteria is found
            within 1000 attempts (3-D input only).
        ValueError: If ``irf_full`` is neither 1-D nor 3-D.
    """
    # IRF Selection Logic
    if irf_full.ndim == 3:
        H, W, T = irf_full.shape
        max_attempts = 1000

        for _ in range(max_attempts):
            x = np.random.randint(H)
            y = np.random.randint(W)
            pixel_data = irf_full[x, y, :]

            peak = np.max(pixel_data)
            if peak <= 500:
                continue

            # Estimate noise from the pre-peak baseline (first quarter before the peak)
            # SNR criterion: noise <= 5% of peak  ↔  SNR = peak/noise >= 20
            peak_idx = int(np.argmax(pixel_data))
            if peak_idx >= 3:
                n_baseline = max(3, peak_idx // 4)
                noise = np.std(pixel_data[:n_baseline])
            else:
                # Peak is at the very start — fall back to the far tail
                noise = np.std(pixel_data[int(0.75 * T):])

            if noise <= 0.05 * peak:  # SNR >= 20
                irf = pixel_data
                break
        else:
            raise RuntimeError(f"Could not find a valid IRF pixel after {max_attempts} attempts.")          
    elif irf_full.ndim == 1:
        irf = irf_full
    else:
        raise ValueError(f'IRF must be 1-D or 3-D, got shape {irf_full.shape}')
    
    return irf
    
