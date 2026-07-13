"""Alignment of an instrument response function (IRF) to a decay dataset.

This module provides :class:`IRFAligner`, which background-subtracts a
decay/IRF pair and estimates and applies a per-pixel temporal shift so the
IRF's rising edge lines up with the decay's rising edge, either via a
sub-pixel Fourier phase shift or an integer circular shift.
"""

import warnings
import numpy as np
from scipy.fft import fft, ifft, fftfreq, fftshift


class IRFAligner:
    """Aligns an IRF data cube to a decay data cube along the time axis.

    On construction, both the decay and IRF cubes are background-subtracted
    using the mean of their first ``noise_bins`` time bins, and a warning is
    raised if the estimated noise baseline is large relative to the peak
    signal (suggesting ``noise_bins`` may include real signal).

    Attributes:
        H (int): Height (first spatial dimension) of the data cubes.
        W (int): Width (second spatial dimension) of the data cubes.
        T (int): Number of time bins.
        dt (float): Time-bin width, computed as ``12.5 / T``.
        decay (np.ndarray): Background-subtracted, non-negative decay cube
            of shape ``(H, W, T)``.
        irf (np.ndarray): Background-subtracted, non-negative IRF cube of
            shape ``(H, W, T)``.
    """

    def __init__(self, decay, irf, noise_bins=5):
        """Initializes the aligner and background-subtracts both inputs.

        Args:
            decay (np.ndarray): Decay data cube of shape ``(H, W, T)``.
            irf (np.ndarray): IRF data cube of shape ``(H, W, T)``, matching
                the shape of ``decay``.
            noise_bins (int): Number of leading time bins averaged to
                estimate and subtract the per-pixel noise baseline from both
                ``decay`` and ``irf``. Defaults to 5.

        Warns:
            UserWarning: If the mean noise baseline of ``decay`` or ``irf``
                exceeds 5% of that array's peak value, indicating the
                ``noise_bins`` window may be contaminated by real signal.
        """
        self.H, self.W, self.T = decay.shape
        self.dt = 12.5 / self.T
        
        d_bg = np.mean(decay[:, :, :noise_bins], axis=2, keepdims=True)
        i_bg = np.mean(irf[:, :, :noise_bins], axis=2, keepdims=True)
        
        self.decay = np.maximum(decay - d_bg, 0)
        self.irf = np.maximum(irf - i_bg, 0)

        _threshold = 0.05
        d_peak, i_peak = np.max(decay), np.max(irf)
        if d_peak > 0 and np.mean(d_bg) > _threshold * d_peak:
            warnings.warn(
                f"Decay noise baseline ({np.mean(d_bg):.3g}) exceeds {_threshold*100:.0f}% "
                f"of peak ({d_peak:.3g}). noise_bins window may be contaminated — "
                "consider reducing noise_bins.",
                UserWarning, stacklevel=2,
            )
        if i_peak > 0 and np.mean(i_bg) > _threshold * i_peak:
            warnings.warn(
                f"IRF noise baseline ({np.mean(i_bg):.3g}) exceeds {_threshold*100:.0f}% "
                f"of peak ({i_peak:.3g}). noise_bins window may be contaminated — "
                "consider reducing noise_bins.",
                UserWarning, stacklevel=2,
            )

    def _find_rising_point(self, data, fraction=0.1):
        """
        Finds the fractional bin index where the signal first reaches 
        a certain percentage of its peak (the 'toe').
        """
        H, W, T = data.shape
        rising_indices = np.zeros((H, W))
        
        for i in range(H):
            for j in range(W):
                trace = data[i, j, :]
                peak_val = np.max(trace)
                if peak_val <= 0:
                    continue
                
                threshold = peak_val * fraction
                # Find the first index that exceeds the threshold
                idx_above = np.where(trace >= threshold)[0]
                if len(idx_above) == 0:
                    continue
                
                first_idx = idx_above[0]
                
                # Sub-pixel linear interpolation for the exact 'threshold' crossing
                if first_idx > 0:
                    y2 = trace[first_idx]
                    y1 = trace[first_idx - 1]
                    # Linear interp: x = x1 + (target - y1) * (dx / dy)
                    fractional_part = (threshold - y1) / (y2 - y1 + 1e-12)
                    rising_indices[i, j] = (first_idx - 1) + fractional_part
                else:
                    rising_indices[i, j] = first_idx
                    
        return rising_indices

    def estimate_shift(self, fraction=0.1):
        """
        Calculates how much the IRF must move to match the decay's start.
        """
        t_decay = self._find_rising_point(self.decay, fraction=fraction)
        t_irf = self._find_rising_point(self.irf, fraction=fraction)
        
        # Shift = Target - Source
        return t_decay - t_irf

    def apply_fourier_shift(self, shifts):
        """Applies a per-pixel sub-pixel shift to the IRF in the frequency domain.

        Uses the FFT shift theorem to apply a fractional (sub-bin) time
        shift to ``self.irf`` for every pixel, then clips negative values
        introduced by the Fourier interpolation.

        Args:
            shifts (np.ndarray): Per-pixel shift amounts of shape
                ``(H, W)``, in units of time bins, as returned by
                :meth:`estimate_shift`.

        Returns:
            np.ndarray: The shifted, non-negative IRF cube of shape
            ``(H, W, T)``.
        """
        freqs = fftfreq(self.T)
        # Apply the fractional shift in the frequency domain
        phase = np.exp(-2j * np.pi * freqs[None, None, :] * shifts[:, :, None])
        
        IRF_fft = fft(self.irf, axis=2)
        aligned_irf = np.real(ifft(IRF_fft * phase, axis=2))
        
        return np.maximum(aligned_irf, 0)

    def apply_circular_shift(self, shifts):
        """
        Applies a linear circular shift by rounding fractional shifts 
        to the nearest integer and rolling the array.
        """
        aligned_irf = np.zeros_like(self.irf)
        # Round shifts to nearest integer for np.roll
        int_shifts = np.round(shifts).astype(int)
        
        for i in range(self.H):
            for j in range(self.W):
                # np.roll performs circular shifting
                aligned_irf[i, j, :] = np.roll(self.irf[i, j, :], int_shifts[i, j])
        
        return aligned_irf

    def align(self, fraction=0.1, method='fourier', manual_correction=0.0):
        """
        Aligns the IRF using the specified method.
        """
        shifts = self.estimate_shift(fraction=fraction)
        shifts = shifts - manual_correction
        if method == 'circular':
            return self.apply_circular_shift(shifts), shifts
        else:
            return self.apply_fourier_shift(shifts), shifts

    def align_pixel(self, x, y, fraction=0.1, method='fourier', manual_correction=0.0):
        """
        Aligns the IRF for a single pixel (x, y) — useful for quick inspection
        of the alignment at a specific spatial location without processing the
        full data cube.
        """
        decay_trace = self.decay[x, y, :]
        irf_trace = self.irf[x, y, :]

        def _rising_point(trace):
            peak_val = np.max(trace)
            if peak_val <= 0:
                return 0.0
            threshold = peak_val * fraction
            idx_above = np.where(trace >= threshold)[0]
            if len(idx_above) == 0:
                return 0.0
            first_idx = idx_above[0]
            if first_idx > 0:
                v2, v1 = trace[first_idx], trace[first_idx - 1]
                return (first_idx - 1) + (threshold - v1) / (v2 - v1 + 1e-12)
            return float(first_idx)

        shift = _rising_point(decay_trace) - _rising_point(irf_trace) - manual_correction

        if method == 'circular':
            aligned_irf = np.roll(irf_trace, int(round(shift)))
        else:
            freqs = fftfreq(self.T)
            phase = np.exp(-2j * np.pi * freqs * shift)
            aligned_irf = np.maximum(np.real(ifft(fft(irf_trace) * phase)), 0)

        return aligned_irf, shift