"""
Implement a Poisson-likelihood TV reconstructor for SPAD photon-count measurements.

This module belongs to :mod:`pyfli.sp_analysis` and is part of PyFLI single-pixel camera
basis generation, acquisition simulation, and reconstruction solvers. Public API
includes classes :class:`SPADPoissonReconstructor`.
"""

from __future__ import annotations
from typing import Any
import numpy as np
from scipy.optimize import minimize
from .base_reconstructor import BaseReconstructor


class SPADPoissonReconstructor(BaseReconstructor):
    """
    Reconstruct SPAD photon-count data with a Poisson likelihood and isotropic TV
    penalty. The solver is designed for low-count single-pixel acquisitions.

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
    alpha : float
        Regularization strength or statistical threshold value, depending on context.
    maxiter : int
        Maximum number of optimization iterations.
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
        alpha: float = 0.1,
        maxiter: int = 500,
        n_workers: int | None = None,
    ) -> None:
        super().__init__(h, w, t, lam, differential, n_workers)
        self.alpha = alpha
        self.maxiter = maxiter

    def _objective_and_grad(
        self, x_flat: np.ndarray, A: np.ndarray, y: np.ndarray, alpha: float
    ) -> tuple[Any, ...]:
        """
        Handle objective and grad.

        Parameters
        ----------
        x_flat : np.ndarray
            Input value.
        A : np.ndarray
            Input value.
        y : np.ndarray
            Input value.
        alpha : float
            Input value.

        Returns
        -------
        tuple[Any, ...]
            Return value.
        """
        eps = 1e-10
        Ax = np.dot(A, x_flat)
        Ax_safe = np.maximum(Ax, eps)

        # Poisson negative log-likelihood: sum(Ax - y * log(Ax))
        poisson_loss = np.sum(Ax_safe - y * np.log(Ax_safe))
        grad_poisson = np.dot(A.T, 1.0 - y / Ax_safe)

        # Isotropic TV with Neumann boundary conditions
        x = x_flat.reshape((self.h, self.w))
        dx = np.zeros_like(x)
        dy = np.zeros_like(x)
        dx[:, :-1] = np.diff(x, axis=1)
        dy[:-1, :] = np.diff(x, axis=0)

        norm = np.sqrt(dx**2 + dy**2 + eps)
        tv = np.sum(norm)

        px = dx / norm
        py = dy / norm
        grad_tv = np.zeros_like(x)
        grad_tv[:, :-1] -= px[:, :-1]
        grad_tv[:, 1:] += px[:, :-1]
        grad_tv[:-1, :] -= py[:-1, :]
        grad_tv[1:, :] += py[:-1, :]

        return poisson_loss + alpha * tv, grad_poisson + alpha * grad_tv.flatten()

    def reconstruct_slice(self, y_slice: np.ndarray, A: np.ndarray) -> Any:
        """
        Handle reconstruct slice.

        Parameters
        ----------
        y_slice : np.ndarray
            Input value.
        A : np.ndarray
            Input value.

        Returns
        -------
        Any
            Return value.
        """
        A = A.astype(np.float64)
        y = y_slice.astype(np.float64)
        M = A.shape[0]

        # Positive initial guess via back-projection
        x0 = np.abs(np.dot(A.T, y)) / M
        x0 = np.maximum(x0, 1e-6)

        res = minimize(
            self._objective_and_grad,
            x0,
            args=(A, y, self.alpha),
            method="L-BFGS-B",
            jac=True,
            bounds=[(0, None)] * self.n_pixels,
            options={"maxiter": self.maxiter, "ftol": 1e-10, "gtol": 1e-7},
        )
        return res.x.reshape((self.h, self.w))
