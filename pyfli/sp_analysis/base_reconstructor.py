"""
Define shared scaffolding for four-dimensional single-pixel reconstruction solvers.

This module belongs to :mod:`pyfli.sp_analysis` and is part of PyFLI single-pixel camera
basis generation, acquisition simulation, and reconstruction solvers. Public API
includes classes :class:`BaseReconstructor`.
"""

from __future__ import annotations
from typing import Any
import numpy as np
from abc import ABC, abstractmethod


class BaseReconstructor(ABC):
    """
    Provide shared shape handling and pattern bookkeeping for four-dimensional single-
    pixel reconstruction. Subclasses implement concrete inverse solvers.

    Parameters
    ----------
    h : np.ndarray
        Image height or spatial dimension used by reconstruction solvers.
    w : np.ndarray
        Image width or spatial dimension used by reconstruction solvers.
    t : np.ndarray
        Temporal sample axis used by reconstruction solvers.
    lam : np.ndarray
        Wavelength or spectral axis used by reconstruction solvers.
    differential : bool
        Whether the sensing basis uses differential pattern pairs.
    n_workers : int | None
        Number of worker processes or threads used by parallel solvers.
    """

    def __init__(
        self,
        h: np.ndarray,
        w: np.ndarray,
        t: np.ndarray,
        lam: np.ndarray,
        differential: bool = True,
        n_workers: int | None = None,
    ) -> None:
        self.h = h
        self.w = w
        self.t = t
        self.lam = lam
        self.n_pixels = h * w
        self.differential = differential
        self.n_workers = n_workers

    @staticmethod
    def dmd_to_sensing_matrix(dmd_patterns: np.ndarray, differential: bool) -> Any:
        """
        Recover the sensing matrix A from DMD {0,1} patterns.

        differential=True:
            dmd_patterns : (2M, N) stacked [P_pos ; P_neg] in {0, 1}
            Returns      : (M, N)  Hadamard matrix H = 2*P_pos - 1 in {-1, +1}

        differential=False:
            dmd_patterns : (M, N) patterns in [0, 1] (Hadamard single-pass
                           or Fourier DCT), used directly as the sensing matrix.
            Returns      : (M, N) same array as float64
        """
        if differential:
            M = dmd_patterns.shape[0] // 2
            P_pos = dmd_patterns[:M]
            return 2.0 * P_pos - 1.0  # {0,1} -> {-1,+1}
        return dmd_patterns.astype(np.float64)

    @staticmethod
    def _process_measurements(
        measurements: np.ndarray, dmd_patterns: np.ndarray, differential: bool
    ) -> Any:
        """
        Apply differential subtraction to raw measurements when needed.

        differential=True:
            measurements : (2M, T, Lambda)
            Returns      : (M, T, Lambda)  y_pos - y_neg
        differential=False:
            measurements : (M, T, Lambda)
            Returns      : same, unchanged
        """
        if differential:
            M = dmd_patterns.shape[0] // 2
            return measurements[:M] - measurements[M:]
        return measurements

    @abstractmethod
    def reconstruct_slice(self, y_slice: np.ndarray, A: np.ndarray) -> None:
        """
        Reconstruct one (H, W) frame.
        y_slice : (M,)   measurements for one (t, lambda) slice
        A       : (M, N) sensing matrix (already converted from DMD patterns)
        Returns : (H, W)
        """

    def reconstruct_4d(
        self, measurements: np.ndarray, dmd_patterns: np.ndarray
    ) -> np.ndarray:
        """
        Reconstruct the full 4D (x, y, T, Lambda) cube from DMD measurements.

        Parameters
        ----------
        measurements : (2M, T, Lambda) if differential else (M, T, Lambda)
            Raw single-pixel SPAD detector measurements.
        dmd_patterns : (2M, H*W) if differential else (M, H*W)
            DMD-compatible {0,1} patterns from BasisPatterns.generate_hadamard()
            or BasisPatterns.generate_fourier_dct().

        Returns
        -------
        cube : (H, W, T, Lambda)
        """
        A = self.dmd_to_sensing_matrix(dmd_patterns, self.differential)
        y = self._process_measurements(measurements, dmd_patterns, self.differential)

        _, T, L = y.shape
        out = np.zeros((self.h, self.w, T, L))

        for wavelength_idx in range(L):
            for t in range(T):
                y_slice = y[:, t, wavelength_idx]
                if np.any(y_slice != 0):
                    out[:, :, t, wavelength_idx] = self.reconstruct_slice(y_slice, A)

        return out
