import numpy as np
from abc import ABC, abstractmethod


class BaseReconstructor(ABC):
    """
    Base class for 4D (x, y, T, Lambda) single-pixel camera reconstruction.

    Accepts DMD patterns directly (the {0,1} output of BasisPatterns), handles
    differential subtraction internally, and recovers the sensing matrix for
    reconstruction.

    Parameters
    ----------
    h, w        : spatial resolution (image = H x W)
    t, lam      : number of TCSPC time bins and wavelength channels
    differential: True  — patterns from generate_hadamard(differential=True)
                          shape (2M, N); measurements also (2M, T, Lambda).
                          Internally computes y_diff = y_pos - y_neg and
                          recovers H = 2*P_pos - 1 as the sensing matrix.
                  False — patterns from generate_hadamard(differential=False)
                          or generate_fourier_dct(); shape (M, N), used as-is.
    n_workers   : reserved for future parallel execution
    """

    def __init__(self, h, w, t, lam, differential=True, n_workers=None):
        self.h = h
        self.w = w
        self.t = t
        self.lam = lam
        self.n_pixels = h * w
        self.differential = differential
        self.n_workers = n_workers

    @staticmethod
    def dmd_to_sensing_matrix(dmd_patterns, differential):
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
            return (2.0 * P_pos - 1.0)  # {0,1} -> {-1,+1}
        return dmd_patterns.astype(np.float64)

    @staticmethod
    def _process_measurements(measurements, dmd_patterns, differential):
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
    def reconstruct_slice(self, y_slice, A):
        """
        Reconstruct one (H, W) frame.
        y_slice : (M,)   measurements for one (t, lambda) slice
        A       : (M, N) sensing matrix (already converted from DMD patterns)
        Returns : (H, W)
        """

    def reconstruct_4d(self, measurements, dmd_patterns):
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

        for l in range(L):
            for t in range(T):
                y_slice = y[:, t, l]
                if np.any(y_slice != 0):
                    out[:, :, t, l] = self.reconstruct_slice(y_slice, A)

        return out
