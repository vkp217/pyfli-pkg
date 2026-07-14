"""
Implement linear and total-variation reconstructors for four-dimensional SPAD data.

This module belongs to :mod:`pyfli.sp_analysis` and is part of PyFLI single-pixel camera
basis generation, acquisition simulation, and reconstruction solvers. Public API
includes classes :class:`LinearReconstructor` and :class:`TVReconstructor`.
"""

from typing import Any

import numpy as np
from scipy.optimize import minimize

from .base_reconstructor import BaseReconstructor


class LinearReconstructor(BaseReconstructor):
    """
    Run the linear reconstructor routine.
    the lightweight baseline solver for single-pixel camera data.

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
        super().__init__(h, w, t, lam, differential, n_workers)

    def reconstruct_slice(self, y_slice: np.ndarray, A: np.ndarray) -> Any:
        """
        Reconstruct slice.

        Parameters
        ----------
        y_slice : np.ndarray
            Single measurement slice reconstructed by the solver.
        A : np.ndarray
            Lower bound or left separator value used by the helper.

        Returns
        -------
        Any
            Object produced by reconstruct slice.
        """
        A = A.astype(np.float64)
        y = y_slice.astype(np.float64)
        M = A.shape[0]
        return (np.dot(A.T, y) / M).reshape((self.h, self.w))


class TVReconstructor(BaseReconstructor):
    """
    Solve four-dimensional SPAD reconstruction with isotropic total-variation
    regularization. It is intended for Gaussian-noise measurements where spatial
    smoothness is useful.

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
        alpha: float = 1.0,
        maxiter: int = 500,
        n_workers: int | None = None,
    ) -> None:
        super().__init__(h, w, t, lam, differential, n_workers)
        self.alpha = alpha
        self.maxiter = maxiter

    def _objective_and_grad(
        self, x_flat: np.ndarray, A: np.ndarray, y: np.ndarray, alpha: float
    ) -> tuple[Any, ...]:
        # Data fidelity: 0.5 * ||Ax - y||^2
        """
        Run the objective and grad routine.

        Parameters
        ----------
        x_flat : np.ndarray
            Flattened image or parameter vector.
        A : np.ndarray
            Lower bound or left separator value used by the helper.
        y : np.ndarray
            Observed signal, target data, or coordinate array.
        alpha : float
            Regularization strength, fraction value, or significance threshold used by the
        routine.

        Returns
        -------
        tuple[Any, ...]
            Tuple containing objective value and gradient.
        """
        Ax_minus_y = np.dot(A, x_flat) - y
        fidelity = 0.5 * np.sum(Ax_minus_y**2)
        grad_fidelity = np.dot(A.T, Ax_minus_y)

        # Isotropic TV with Neumann (zero-flux) boundary conditions
        x = x_flat.reshape((self.h, self.w))
        eps = 1e-8
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

        return fidelity + alpha * tv, grad_fidelity + alpha * grad_tv.flatten()

    def reconstruct_slice(self, y_slice: np.ndarray, A: np.ndarray) -> Any:
        """
        Reconstruct slice.

        Parameters
        ----------
        y_slice : np.ndarray
            Single measurement slice reconstructed by the solver.
        A : np.ndarray
            Lower bound or left separator value used by the helper.

        Returns
        -------
        Any
            Object produced by reconstruct slice.
        """
        A = A.astype(np.float64)
        y = y_slice.astype(np.float64)
        M = A.shape[0]
        x0 = np.dot(A.T, y) / M
        res = minimize(
            self._objective_and_grad,
            x0,
            args=(A, y, self.alpha),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": self.maxiter, "ftol": 1e-10, "gtol": 1e-7},
        )
        return res.x.reshape((self.h, self.w))
