"""Orthogonal sensing basis transforms (Hadamard, DCT) used by spAnalysis.

Provides forward/inverse transform pairs that can be applied along the
leading axis of measurement/coefficient arrays, independent of the DMD
pattern-generation and reconstruction code paths in this package.
"""

import numpy as np
from scipy.linalg import hadamard as _hadamard
from scipy.fftpack import dct, idct


class OrthogonalBasis:
    """Abstract base for orthogonal sensing bases."""
    def forward(self, x):
        """Apply the forward transform to `x`.

        Args:
            x: Input array; the transform is applied along axis 0.

        Returns:
            The transformed array, same shape as `x`.

        Raises:
            NotImplementedError: Always, in the abstract base class.
        """
        raise NotImplementedError

    def inverse(self, y):
        """Apply the inverse transform to `y`.

        Args:
            y: Input array; the transform is applied along axis 0.

        Returns:
            The inverse-transformed array, same shape as `y`.

        Raises:
            NotImplementedError: Always, in the abstract base class.
        """
        raise NotImplementedError


class HadamardBasis(OrthogonalBasis):
    """
    Walsh-Hadamard basis via matrix multiply.
    n_pixels must be a power of 2.
    Supports (N,) or (N, T, Lambda) inputs; transform applied along axis 0.
    """
    def __init__(self, n_pixels):
        """Build the Walsh-Hadamard transform matrix.

        Args:
            n_pixels: Basis dimension; must be a power of 2. Stored as
                `self.n` and used to build the (n_pixels, n_pixels)
                Hadamard matrix `self._H`.
        """
        self.n = n_pixels
        self._H = _hadamard(n_pixels).astype(np.float64)

    def forward(self, x):
        """Apply the Hadamard transform along axis 0.

        Args:
            x: Array of shape (N,) or (N, T, Lambda).

        Returns:
            Transformed array, same shape as `x`.
        """
        shape = x.shape
        return (self._H @ x.reshape(self.n, -1)).reshape(shape)

    def inverse(self, y):
        """Apply the inverse Hadamard transform along axis 0.

        Since H @ H = N * I for a Walsh-Hadamard matrix, the inverse is
        H / N.

        Args:
            y: Array of shape (N,) or (N, T, Lambda).

        Returns:
            Inverse-transformed array, same shape as `y`.
        """
        # H @ H = N * I, so H^{-1} = H / N
        shape = y.shape
        return (self._H @ y.reshape(self.n, -1) / self.n).reshape(shape)


class DCTBasis(OrthogonalBasis):
    """
    DCT-II / DCT-III (IDCT) basis.
    Supports (N,) or (N, T, Lambda) inputs; transform applied along axis 0.
    """
    def forward(self, x):
        """Apply the orthonormal DCT-II transform along axis 0.

        Args:
            x: Array of shape (N,) or (N, T, Lambda).

        Returns:
            DCT-transformed array, same shape as `x`.
        """
        return dct(x, axis=0, norm='ortho')

    def inverse(self, y):
        """Apply the orthonormal DCT-III (inverse DCT) transform along axis 0.

        Args:
            y: Array of shape (N,) or (N, T, Lambda).

        Returns:
            Inverse-DCT-transformed array, same shape as `y`.
        """
        return idct(y, axis=0, norm='ortho')
