import numpy as np
from scipy.optimize import minimize
from .base_reconstructor import BaseReconstructor


class LinearReconstructor(BaseReconstructor):
    """
    Back-projection (ghost imaging) reconstruction for 4D SPAD data.
    Mirrors simulator's reconstruct_linear() extended to (T, Lambda) slices.
    Fast but lower quality; good for quick preview or warm-start.
    """

    def __init__(self, h, w, t, lam, differential=True, n_workers=None):
        super().__init__(h, w, t, lam, differential, n_workers)

    def reconstruct_slice(self, y_slice, A):
        A = A.astype(np.float64)
        y = y_slice.astype(np.float64)
        M = A.shape[0]
        return (np.dot(A.T, y) / M).reshape((self.h, self.w))


class TVReconstructor(BaseReconstructor):
    """
    Isotropic TV-minimization for 4D SPAD data (Gaussian noise model).
    Mirrors simulator's solve_tv() extended to (T, Lambda) slices.
    Uses L-BFGS-B with analytic gradient for fast convergence.
    """

    def __init__(self, h, w, t, lam, differential=True, alpha=1.0, maxiter=500, n_workers=None):
        super().__init__(h, w, t, lam, differential, n_workers)
        self.alpha = alpha
        self.maxiter = maxiter

    def _objective_and_grad(self, x_flat, A, y, alpha):
        # Data fidelity: 0.5 * ||Ax - y||^2
        Ax_minus_y = np.dot(A, x_flat) - y
        fidelity = 0.5 * np.sum(Ax_minus_y ** 2)
        grad_fidelity = np.dot(A.T, Ax_minus_y)

        # Isotropic TV with Neumann (zero-flux) boundary conditions
        x = x_flat.reshape((self.h, self.w))
        eps = 1e-8
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

        return fidelity + alpha * tv, grad_fidelity + alpha * grad_tv.flatten()

    def reconstruct_slice(self, y_slice, A):
        A = A.astype(np.float64)
        y = y_slice.astype(np.float64)
        M = A.shape[0]
        x0 = np.dot(A.T, y) / M
        res = minimize(
            self._objective_and_grad, x0,
            args=(A, y, self.alpha),
            method='L-BFGS-B', jac=True,
            options={'maxiter': self.maxiter, 'ftol': 1e-10, 'gtol': 1e-7}
        )
        return res.x.reshape((self.h, self.w))
