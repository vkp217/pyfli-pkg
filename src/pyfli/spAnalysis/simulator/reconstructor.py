# spAnalysis/simulator/reconstructor.py

import numpy as np
from scipy.optimize import minimize
from scipy.fftpack import idct

class Reconstructor:
    def __init__(self, resolution=(128, 128)):
        self.res_h, self.res_w = resolution
        self.n_pixels = self.res_h * self.res_w

    def _tv_norm(self, x_flat):
        # Calculates the Total Variation of the image
        x = x_flat.reshape((self.res_h, self.res_w))
        grad_x = np.diff(x, axis=1)
        grad_y = np.diff(x, axis=0)
        return np.sum(np.abs(grad_x)) + np.sum(np.abs(grad_y))
    
    def _objective_and_grad(self, x_flat, A, y, alpha):
        """
        Calculates both the objective value and the gradient.
        Providing the gradient (jac) makes the solver 10,000x faster.
        """
        # 1. Reshape for TV calculation
        x = x_flat.reshape((self.res_h, self.res_w))
        
        # 2. Data Fidelity Term
        Ax_minus_y = np.dot(A, x_flat) - y
        fidelity = 0.5 * np.sum(Ax_minus_y**2)
        
        # Fidelity Gradient: A^T * (Ax - y)
        grad_fidelity = np.dot(A.T, Ax_minus_y)

        # 3. Total Variation (Approximated for differentiability)
        eps = 1e-8
        # Horizontal diffs
        dx = np.diff(x, axis=1, append=x[:, -1:])
        # Vertical diffs
        dy = np.diff(x, axis=0, append=x[-1:, :])

        norm = np.sqrt(dx**2 + dy**2 + eps)
        tv = np.sum(norm)

        # Isotropic TV gradient: ∂TV/∂x[k,l] = px[k,l-1] - px[k,l] + py[k-1,l] - py[k,l]
        px = dx / norm
        py = dy / norm
        grad_tv = np.roll(px, 1, axis=1) - px + np.roll(py, 1, axis=0) - py

        total_obj = fidelity + alpha * tv
        total_grad = grad_fidelity + alpha * grad_tv.flatten()

        return total_obj, total_grad

    def solve_tv(self, measurements, basis_matrix, alpha=0.1):
        # Ensure we are using float64 for the optimizer's stability
        A = basis_matrix.astype(np.float64)
        y = measurements.astype(np.float64).flatten()
        
        # Start with linear reconstruction guess
        x0 = np.dot(A.T, y) / len(y)
        
        print(f"Starting FAST TV Optimization (Alpha={alpha})...")
        
        res = minimize(
            self._objective_and_grad, 
            x0, 
            args=(A, y, alpha),
            method='L-BFGS-B', 
            jac=True,
            options={'maxiter': 100, 'disp': True}
        )
        
        return res.x.reshape((self.res_h, self.res_w), order='C')

    def reconstruct_linear(self, measurements, basis_matrix):
        # Standard linear back-projection (Ghost Imaging)
        y = measurements.flatten()
        M = len(y)
        img_flat = np.dot(basis_matrix.T, y)
        img_flat /= M
        return img_flat.reshape((self.res_h, self.res_w))

    def reconstruct_fourier_domain(self, measurements, sampling_indices):
        # Fast reconstruction for Fourier SPI.
        # Directly fills the 2D DCT spectrum and performs IDCT.
        freq_map_flat = np.zeros(self.n_pixels)
        # Place measurements back into their frequency locations
        freq_map_flat[sampling_indices[:len(measurements)]] = measurements
        freq_map = freq_map_flat.reshape((self.res_h, self.res_w))
        
        # 2D Inverse Discrete Cosine Transform,'norm=ortho' is crucial to match the generation
        img = idct(idct(freq_map, axis=0, norm='ortho'), axis=1, norm='ortho')
        return img

    @staticmethod
    def normalize_image(image):
        """Scales the reconstructed image to 0-1 range for viewing."""
        img_min = image.min()
        img_max = image.max()
        if img_max - img_min == 0:
            return image
        return (image - img_min) / (img_max - img_min)