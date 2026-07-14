import numpy as np
from scipy.optimize import minimize
from .base_reconstructor import BaseReconstructor


class SPADPoissonReconstructor(BaseReconstructor):
    """
    Poisson-likelihood + isotropic TV reconstruction for SPAD photon count data.

    Solves:  min_{x >= 0}  KL(y || Ax) + alpha * TV(x)

    Use this instead of TVReconstructor when y contains integer photon counts
    (Poisson-distributed TCSPC measurements). The Poisson negative log-likelihood
    replaces the Gaussian L2 fidelity term used in TVReconstructor.

    Parameters
    ----------
    h, w         : spatial resolution (image = H x W)
    t, lam       : number of TCSPC time bins and wavelength channels
    differential : True  — DMD differential patterns [P_pos; P_neg] in {0,1}
                   False — direct patterns in [0,1] (Fourier or single-pass)
    alpha        : TV regularization weight
    maxiter      : max L-BFGS-B iterations per (t, lambda) slice
    n_workers    : reserved for future parallel execution
    """

    def __init__(self, h, w, t, lam, differential=True, alpha=0.1, maxiter=500, n_workers=None):
        super().__init__(h, w, t, lam, differential, n_workers)
        self.alpha = alpha
        self.maxiter = maxiter

    def _objective_and_grad(self, x_flat, A, y, alpha):
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

        norm = np.sqrt(dx ** 2 + dy ** 2 + eps)
        tv = np.sum(norm)

        px = dx / norm
        py = dy / norm
        grad_tv = np.zeros_like(x)
        grad_tv[:, :-1] -= px[:, :-1]
        grad_tv[:, 1:]  += px[:, :-1]
        grad_tv[:-1, :] -= py[:-1, :]
        grad_tv[1:, :]  += py[:-1, :]

        return poisson_loss + alpha * tv, grad_poisson + alpha * grad_tv.flatten()

    def reconstruct_slice(self, y_slice, A):
        A = A.astype(np.float64)
        y = y_slice.astype(np.float64)
        M = A.shape[0]

        # Positive initial guess via back-projection
        x0 = np.abs(np.dot(A.T, y)) / M
        x0 = np.maximum(x0, 1e-6)

        res = minimize(
            self._objective_and_grad, x0,
            args=(A, y, self.alpha),
            method='L-BFGS-B', jac=True,
            bounds=[(0, None)] * self.n_pixels,
            options={'maxiter': self.maxiter, 'ftol': 1e-10, 'gtol': 1e-7}
        )
        return res.x.reshape((self.h, self.w))
