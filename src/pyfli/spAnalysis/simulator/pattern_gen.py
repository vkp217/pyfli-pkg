# spAnalysis/simulator/pattern_gen.py

import numpy as np
from scipy.linalg import hadamard
from scipy.fftpack import dct

class BasisPatterns:
    def __init__(self, resolution=(128, 128), sampling_ratio=1.0):
        self.sampling_ratio = sampling_ratio
        self.res_h, self.res_w = resolution
        self.n_total = self.res_h * self.res_w
        self.n_patterns = int(self.n_total * self.sampling_ratio)

    def _get_hadamard_matrix(self):
        """Generates a standard Walsh-Hadamard matrix (-1, 1)."""
        return hadamard(self.n_total)

    def _get_zigzag_indices(self):
        """Computes indices for 2D Zigzag ordering."""
        indices = np.indices((self.res_h, self.res_w))
        sum_indices = indices[0] + indices[1]
        return np.argsort(sum_indices.flatten(order='C'))

    def generate_hadamard(self, order_strategy='zigzag', differential=True):
        """
        Generates Hadamard patterns.
        Binary mosaic-like patterns with only horizontal and vertical features.
        """
        H = self._get_hadamard_matrix()
        
        if order_strategy == 'zigzag':
            sort_idx = self._get_zigzag_indices()
        else:
            sort_idx = np.arange(self.n_total)
            
        H_ordered = H[sort_idx[:self.n_patterns], :]

        if differential:
            # Split into positive and negative components for DMD (0/1)
            P_pos = (H_ordered + 1) / 2
            P_neg = (1 - H_ordered) / 2
            return np.vstack((P_pos, P_neg))
        
        return (H_ordered + 1) / 2

    def generate_fourier_dct(self, order_strategy='zigzag'):
        """
        Generates Grayscale DCT basis patterns (periodic fringes).
        Optimized version: uses identity transformation to extract 2D basis.
        Captures H, V, and Oblique features.
        """
        # Create an identity matrix representing each pixel location
        identity = np.eye(self.n_total)
        
        # Applying 2D DCT to each identity vector retrieves the basis functions
        # This is significantly faster than nested loops
        basis = dct(dct(identity.reshape(-1, self.res_h, self.res_w), axis=1, norm='ortho'), 
                    axis=2, norm='ortho').reshape(self.n_total, self.n_total)
        
        if order_strategy == 'zigzag':
            sort_idx = self._get_zigzag_indices()
            basis = basis[sort_idx]
            
        # Normalize to [0, 1] range for DMD hardware constraints
        # We add epsilon to prevent division by zero for the DC component
        b_min = basis.min(axis=1, keepdims=True)
        b_max = basis.max(axis=1, keepdims=True)
        basis_dmd = (basis - b_min) / (b_max - b_min + 1e-10)
        
        return basis_dmd[:self.n_patterns, :]

    @staticmethod
    def reshape_pattern(pattern_vector, res):
        """Utility to convert flattened 1D vector to 2D image."""
        return pattern_vector.reshape(res)