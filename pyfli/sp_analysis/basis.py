"""
Implement Hadamard and DCT sensing bases for single-pixel reconstruction.

This module belongs to :mod:`pyfli.sp_analysis` and is part of PyFLI single-pixel camera
basis generation, acquisition simulation, and reconstruction solvers. Public API
includes classes :class:`OrthogonalBasis`, :class:`HadamardBasis`, and
:class:`DCTBasis`.
"""

from typing import Any

import numpy as np
from scipy.linalg import hadamard as _hadamard
from scipy.fftpack import dct, idct


class OrthogonalBasis:
    """
    Define the interface for orthogonal sensing bases. Subclasses implement forward and
    inverse transforms for flattened spatial signals or spatial-temporal stacks.
    """

    def forward(self, x: np.ndarray) -> None:
        """
        Apply the forward basis transform.

        Parameters
        ----------
        x : np.ndarray
            Input array, coordinate, or signal being transformed.

        Returns
        -------
        None
            No object is returned; the function perform forward.
        """
        raise NotImplementedError

    def inverse(self, y: np.ndarray) -> None:
        """
        Apply the inverse basis transform.

        Parameters
        ----------
        y : np.ndarray
            Observed signal, target data, or coordinate array.

        Returns
        -------
        None
            No object is returned; the function perform inverse.
        """
        raise NotImplementedError


class HadamardBasis(OrthogonalBasis):
    """
    Apply a Walsh-Hadamard sensing basis by matrix multiplication. The basis requires a
    power-of-two pixel count and transforms along the spatial axis.

    Parameters
    ----------
    n_pixels : int
        Number of spatial pixels represented by the basis.
    """

    def __init__(self, n_pixels: int) -> None:
        self.n = n_pixels
        self._H = _hadamard(n_pixels).astype(np.float64)

    def forward(self, x: np.ndarray) -> Any:
        """
        Apply the forward basis transform.

        Parameters
        ----------
        x : np.ndarray
            Input array, coordinate, or signal being transformed.

        Returns
        -------
        Any
            Object produced by forward.
        """
        shape = x.shape
        return (self._H @ x.reshape(self.n, -1)).reshape(shape)

    def inverse(self, y: np.ndarray) -> Any:
        # H @ H = N * I, so H^{-1} = H / N
        """
        Apply the inverse basis transform.

        Parameters
        ----------
        y : np.ndarray
            Observed signal, target data, or coordinate array.

        Returns
        -------
        Any
            Object produced by inverse.
        """
        shape = y.shape
        return (self._H @ y.reshape(self.n, -1) / self.n).reshape(shape)


class DCTBasis(OrthogonalBasis):
    """
    Apply DCT-II and inverse DCT transforms as a compact orthogonal sensing basis. It is
    useful for Fourier-like single-pixel simulation and reconstruction.
    """

    def forward(self, x: np.ndarray) -> Any:
        """
        Apply the forward basis transform.

        Parameters
        ----------
        x : np.ndarray
            Input array, coordinate, or signal being transformed.

        Returns
        -------
        Any
            Object produced by forward.
        """
        return dct(x, axis=0, norm="ortho")

    def inverse(self, y: np.ndarray) -> Any:
        """
        Apply the inverse basis transform.

        Parameters
        ----------
        y : np.ndarray
            Observed signal, target data, or coordinate array.

        Returns
        -------
        Any
            Object produced by inverse.
        """
        return idct(y, axis=0, norm="ortho")
