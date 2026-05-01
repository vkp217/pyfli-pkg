# spAnalysis/simulator/pattern_gen.py

import numpy as np
from scipy.linalg import hadamard

class BasisPatterns:
    def __init__(self, resolution=(128, 128), sampling_ratio=1.0):
        self.sampling_ratio = sampling_ratio
        self.res_h, self.res_w = resolution
        self.n_total = self.res_h * self.res_w
        self.n_patterns = int(self.n_total * self.sampling_ratio)

    def _get_hadamard_matrix(self):
        # Generateing a standard Walsh-Hadamard matrix (-1, 1).
        return hadamard(self.n_total)

    def _get_zigzag_indices(self):
        indices = np.indices((self.res_h, self.res_w))
        sum_indices = indices[0] + indices[1]
        return np.argsort(sum_indices.flatten())

    def generate_hadamard(self, order_strategy='zigzag', differential=True):
        # It will generate Hadamard patterns
        # If differential=True, it returns a matrix of shape (2*n_patterns, n_total)
        # representing the [P_pos, P_neg] pairs for DMD projection.

        H = self._get_hadamard_matrix()
        
        if order_strategy == 'zigzag':
            sort_idx = self._get_zigzag_indices()
        else:
            sort_idx = np.arange(self.n_total)
            
        H_ordered = H[sort_idx[:self.n_patterns], :]

        if differential:
            # P_pos = (H + 1) / 2  -> Maps -1 to 0 and 1 to 1
            # P_neg = (1 - H) / 2  -> Maps 1 to 0 and -1 to 1
            P_pos = (H_ordered + 1) / 2
            P_neg = (1 - H_ordered) / 2
            # Stack them so they can be projected sequentially
            return np.vstack((P_pos, P_neg))
        
        return (H_ordered + 1) / 2 # Simple binary 0, 1

    def generate_fourier(self, order_strategy='zigzag'):
        # Generates Fourier (DCT) patterns.
        # DMDs handle grayscale via PWM or Floyd-Steinberg dithering.
        # Here we scale to [0, 1] to mimic the projected intensity.

        N = self.res_h
        basis_matrix = np.zeros((self.n_total, self.n_total))
        
        for u in range(self.res_h):
            for v in range(self.res_w):
                y, x = np.meshgrid(np.arange(self.res_w), np.arange(self.res_h))
                pattern = np.cos(np.pi * u * (2*x + 1) / (2 * self.res_h)) * \
                          np.cos(np.pi * v * (2*y + 1) / (2 * self.res_w))
                
                # Normalize pattern from [-1, 1] to [0, 1] for DMD projection
                pattern_normalized = (pattern - pattern.min()) / (pattern.max() - pattern.min() + 1e-10)
                
                idx = u * self.res_w + v
                basis_matrix[idx, :] = pattern_normalized.flatten()

        if order_strategy == 'zigzag':
            sort_idx = self._get_zigzag_indices()
            basis_matrix = basis_matrix[sort_idx]

        return basis_matrix[:self.n_patterns, :]

    @staticmethod
    def reshape_pattern(pattern_vector, res):
        return pattern_vector.reshape(res)