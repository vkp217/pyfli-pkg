import numpy as np
from scipy.linalg import hadamard as _hadamard
from scipy.fftpack import dct, idct


class OrthogonalBasis:
    """Abstract base for orthogonal sensing bases."""
    def forward(self, x): raise NotImplementedError
    def inverse(self, y): raise NotImplementedError


class HadamardBasis(OrthogonalBasis):
    """
    Walsh-Hadamard basis via matrix multiply.
    n_pixels must be a power of 2.
    Supports (N,) or (N, T, Lambda) inputs; transform applied along axis 0.
    """
    def __init__(self, n_pixels):
        self.n = n_pixels
        self._H = _hadamard(n_pixels).astype(np.float64)

    def forward(self, x):
        shape = x.shape
        return (self._H @ x.reshape(self.n, -1)).reshape(shape)

    def inverse(self, y):
        # H @ H = N * I, so H^{-1} = H / N
        shape = y.shape
        return (self._H @ y.reshape(self.n, -1) / self.n).reshape(shape)


class DCTBasis(OrthogonalBasis):
    """
    DCT-II / DCT-III (IDCT) basis.
    Supports (N,) or (N, T, Lambda) inputs; transform applied along axis 0.
    """
    def forward(self, x):
        return dct(x, axis=0, norm='ortho')

    def inverse(self, y):
        return idct(y, axis=0, norm='ortho')
