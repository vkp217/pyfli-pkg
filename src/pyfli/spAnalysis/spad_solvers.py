import pyxu.operator as pxo
import pyxu.opt.solver as pxs
import pyxu.opt.stop as pxst
import pyxu.math.cupy as cp # For GPU acceleration
import numpy as np

class SPADPoissonReconstructor:
    def __init__(self, h, w, t, lam, lmbd=0.1):
        self.shape = (h, w, t, lam)
        self.size_spatial = h * w
        self.lmbd = lmbd

    def solve_slice(self, y_measured, phi_operator):
        """
        y_measured: (M,) vector of photon counts for one (t, lambda) slice.
        phi_operator: The Pyxu operator (e.g., Hadamard/FWHT).
        """
        # 1. Data Fidelity: Poisson Likelihood (KL Divergence)
        # Shifted by the observed data 'y'
        loss = pxo.KLDivergence(data=y_measured) @ phi_operator
        
        # 2. Regularization: Total Variation (Spatial)
        # We assume the image is (H, W)
        grad = pxo.Gradient(arg_shape=(self.shape[0], self.shape[1]))
        regularizer = self.lmbd * pxo.L1Norm() @ grad

        # 3. Positivity Constraint (SPAD counts cannot be negative)
        positivity = pxo.NonNegativeOrthant(shape=(self.size_spatial,))

        # 4. Solver: Primal-Dual Proximal Splitting (PDPS)
        # Ideal for combining non-smooth terms like TV and KL
        solver = pxs.PDPS(f=None, g=positivity, h=loss + regularizer)
        
        # Stop when change is minimal
        stop_crit = pxst.RelError(eps=1e-4, var='x')
        
        solver.fit(stop_crit=stop_crit)
        return solver.get_output()
    

    import numpy as np
    import pyxu.operator as pxo
    import pyxu.opt.solver as pxs
    from .base_reconstructor import BaseReconstructor
    from .proximal import get_poisson_loss, get_tv_regularizer

class SPADSolver(BaseReconstructor):
    def __init__(self, h, w, t, lam, basis='hadamard', n_workers=None):
        super().__init__(h, w, t, lam, n_workers)
        # Initialize the basis operator once
        if basis == 'hadamard':
            self.phi = pxo.FWHT(arg_shape=(self.num_pixels,))

    def reconstruct_chunk(self, chunk):
        """
        chunk: (M, T, sub_L)
        Returns: (H, W, T, sub_L)
        """
        M, T, sub_L = chunk.shape
        out = np.zeros((self.h, self.w, T, sub_L))
        
        for l in range(sub_L):
            for t in range(T):
                y_slice = chunk[:, t, l]
                
                if np.sum(y_slice) > 0:
                    # 1. Poisson Fidelity
                    loss = get_poisson_loss(y_slice, self.phi)
                    # 2. TV Regularization (lambda=0.1 as a starting point)
                    reg = get_tv_regularizer(self.h, self.w, 0.1)
                    
                    # 3. Solver: Primal-Dual Proximal Splitting
                    solver = pxs.PDPS(f=None, 
                                     g=pxo.NonNegativeOrthant(shape=(self.num_pixels,)), 
                                     h=loss + reg)
                    solver.fit()
                    out[:, :, t, l] = solver.get_output().reshape(self.h, self.w)
                    
        return out