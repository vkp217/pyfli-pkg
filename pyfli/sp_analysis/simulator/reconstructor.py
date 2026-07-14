"""
Reconstruct images from simulated single-pixel measurements using linear, Fourier, and
TV methods.

This module belongs to :mod:`pyfli.sp_analysis.simulator` and is part of PyFLI single-
pixel camera basis generation, acquisition simulation, and reconstruction solvers.
Public API includes classes :class:`Reconstructor`.
"""

from __future__ import annotations
from typing import Any
from pyfli import logging
# sp_analysis/simulator/reconstructor.py

import numpy as np
from scipy.optimize import minimize
from scipy.fftpack import idct


class Reconstructor:
    """
    Reconstruct images from simulated single-pixel measurements. Methods include total-
    variation optimization, linear reconstruction, Fourier-domain reconstruction, and
    output normalization.

    Parameters
    ----------
    resolution : tuple[int, ...]
        Spatial resolution of generated patterns or reconstructed images.
    """

    def __init__(self, resolution: tuple[int, ...] = (128, 128)) -> None:
        self.res_h, self.res_w = resolution
        self.n_pixels = self.res_h * self.res_w

    def _tv_norm(self, x_flat: np.ndarray) -> Any:
        # Calculates the Total Variation of the image
        """
        Handle tv norm.

        Parameters
        ----------
        x_flat : np.ndarray
            Input value.

        Returns
        -------
        Any
            Return value.
        """
        x = x_flat.reshape((self.res_h, self.res_w))
        grad_x = np.diff(x, axis=1)
        grad_y = np.diff(x, axis=0)
        return np.sum(np.abs(grad_x)) + np.sum(np.abs(grad_y))

    def _objective_and_grad(
        self, x_flat: np.ndarray, A: np.ndarray, y: np.ndarray, alpha: float
    ) -> tuple[Any, ...]:
        """
        Calculates both the objective value and the gradient.
        Providing the gradient (jac) makes the solver 10,000x faster.
        """
        x = x_flat.reshape((self.res_h, self.res_w))

        # Data fidelity: 0.5 * ||Ax - y||^2
        Ax_minus_y = np.dot(A, x_flat) - y
        fidelity = 0.5 * np.sum(Ax_minus_y**2)
        grad_fidelity = np.dot(A.T, Ax_minus_y)

        # Isotropic TV with Neumann (zero-flux) boundary conditions
        eps = 1e-8
        # Forward differences, zero-padded at boundaries (Neumann BC)
        dx = np.zeros_like(x)
        dy = np.zeros_like(x)
        dx[:, :-1] = np.diff(x, axis=1)
        dy[:-1, :] = np.diff(x, axis=0)

        norm = np.sqrt(dx**2 + dy**2 + eps)
        tv = np.sum(norm)

        # Divergence with Neumann BC (correct adjoint of forward-diff gradient)
        px = dx / norm
        py = dy / norm
        grad_tv = np.zeros_like(x)
        grad_tv[:, :-1] -= px[:, :-1]
        grad_tv[:, 1:] += px[:, :-1]
        grad_tv[:-1, :] -= py[:-1, :]
        grad_tv[1:, :] += py[:-1, :]

        total_obj = fidelity + alpha * tv
        total_grad = grad_fidelity + alpha * grad_tv.flatten()

        return total_obj, total_grad

    def solve_tv(
        self,
        measurements: np.ndarray,
        basis_matrix: np.ndarray,
        alpha: float = 1.0,
        maxiter: int = 500,
    ) -> Any:
        """
        Handle solve tv.

        Parameters
        ----------
        measurements : np.ndarray
            Input value.
        basis_matrix : np.ndarray
            Input value.
        alpha : float
            Input value.
        maxiter : int
            Input value.

        Returns
        -------
        Any
            Return value.
        """
        A = basis_matrix.astype(np.float64)
        y = measurements.astype(np.float64).flatten()
        M, N = A.shape

        # Scaled initial guess: pinv-like using A^T normalization
        x0 = np.dot(A.T, y) / M

        logging.info(f"Starting TV Optimization (Alpha={alpha}, maxiter={maxiter})...")

        res = minimize(
            self._objective_and_grad,
            x0,
            args=(A, y, alpha),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-7, "disp": True},
        )

        return res.x.reshape((self.res_h, self.res_w), order="C")

    def reconstruct_linear(
        self, measurements: np.ndarray, basis_matrix: np.ndarray
    ) -> Any:
        # Standard linear back-projection (Ghost Imaging)
        """
        Handle reconstruct linear.

        Parameters
        ----------
        measurements : np.ndarray
            Input value.
        basis_matrix : np.ndarray
            Input value.

        Returns
        -------
        Any
            Return value.
        """
        y = measurements.flatten()
        M = len(y)
        img_flat = np.dot(basis_matrix.T, y)
        img_flat /= M
        return img_flat.reshape((self.res_h, self.res_w))

    def reconstruct_fourier_domain(
        self, measurements: np.ndarray, sampling_indices: Any
    ) -> np.ndarray:
        # Fast reconstruction for Fourier SPI.
        # Directly fills the 2D DCT spectrum and performs IDCT.
        """
        Handle reconstruct fourier domain.

        Parameters
        ----------
        measurements : np.ndarray
            Input value.
        sampling_indices : Any
            Input value.

        Returns
        -------
        np.ndarray
            Return value.
        """
        freq_map_flat = np.zeros(self.n_pixels)
        # Place measurements back into their frequency locations
        freq_map_flat[sampling_indices[: len(measurements)]] = measurements
        freq_map = freq_map_flat.reshape((self.res_h, self.res_w))

        # 2D Inverse Discrete Cosine Transform,'norm=ortho' is crucial to match the generation
        img = idct(idct(freq_map, axis=0, norm="ortho"), axis=1, norm="ortho")
        return img

    @staticmethod
    def normalize_image(image: np.ndarray) -> Any:
        """Scales the reconstructed image to 0-1 range for viewing."""
        img_min = image.min()
        img_max = image.max()
        if img_max - img_min == 0:
            return image
        return (image - img_min) / (img_max - img_min)
