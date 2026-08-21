"""
Provide sim helper tools for PyFLI synthetic FLI/FLIM data generation, hardware noise
modeling, calibration, and validation tools.

This module belongs to :mod:`pyfli.simulator` and is part of PyFLI synthetic FLI/FLIM
data generation, hardware noise modeling, calibration, and validation tools. Public API
includes functions :func:`irf_picker`.
"""

import numpy as np


def irf_picker(
    irf_full: np.ndarray, px: tuple[int, int] | None = None
) -> tuple[np.ndarray, tuple[int, int] | None]:
    # IRF Selection Logic
    """
    Run the IRF picker routine.

    Parameters
    ----------
    irf_full : np.ndarray
        Full instrument response function sampled over the decay window.
    px : tuple[int, int] | None
        Explicit ``(x, y)`` pixel to use when ``irf_full`` is 3-D, instead of
        picking one at random. Ignored when ``irf_full`` is 1-D.

    Returns
    -------
    tuple[np.ndarray, tuple[int, int] | None]
        The selected IRF array, and the ``(x, y)`` pixel it came from — the
        given ``px`` if one was passed, the randomly chosen pixel otherwise,
        or ``None`` when ``irf_full`` is 1-D (there's no pixel to report).
    """
    if irf_full.ndim == 3:
        H, W, T = irf_full.shape
        max_attempts = 1 if px is not None else 1000

        for _ in range(max_attempts):
            if px is not None:
                x, y = px
            else:
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
                noise = np.std(pixel_data[int(0.75 * T) :])

            if noise <= 0.05 * peak:  # SNR >= 20
                irf = pixel_data
                break
        else:
            raise RuntimeError(
                f"Could not find a valid IRF pixel after {max_attempts} attempts."
            )
        return irf, (x, y)
    elif irf_full.ndim == 1:
        return irf_full, None
    else:
        raise ValueError(f"IRF must be 1-D or 3-D, got shape {irf_full.shape}")
