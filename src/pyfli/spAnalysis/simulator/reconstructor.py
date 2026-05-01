# spAnalysis/simulator/reconstructor.py

import numpy as np

class Reconstructor:
    def __init__(self, resolution=(128, 128)):
        """
        Reconstructs the scene from single-pixel measurements.
        resolution: tuple (height, width) of the target image.
        """
        self.res_h, self.res_w = resolution
        self.n_pixels = self.res_h * self.res_w

    def reconstruct_linear(self, measurements, basis_matrix):
        """
        Standard linear reconstruction using the transpose of the basis.
        For orthogonal matrices (like Hadamard/DCT), A.T is proportional 
        to the inverse.
        
        x_hat = A_transpose * y
        """
        # Ensure measurements is a column vector
        y = measurements.flatten()
        
        # M is the number of patterns projected
        M = len(y)
        
        # Linear projection: multiply the measurements by the sensing matrix rows
        # basis_matrix shape: (M, N_pixels)
        img_flat = np.dot(basis_matrix.T, y)
        
        # Normalize by M (number of measurements) to keep scale consistent
        img_flat /= M
        
        return img_flat.reshape((self.res_h, self.res_w))

    def reconstruct_fourier_domain(self, measurements, sampling_indices):
        """
        Specific for Fourier SPI. 
        If measurements represent coefficients of the 2D DCT/FFT,
        we can populate a frequency-domain matrix and take the inverse transform.
        """
        # Create an empty frequency map
        freq_map = np.zeros((self.res_h, self.res_w))
        
        # Map measurements back to their spatial frequency locations
        # (Assuming 'sampling_indices' corresponds to the zigzag/ordering used)
        flat_map = freq_map.flatten()
        flat_map[sampling_indices[:len(measurements)]] = measurements
        freq_map = flat_map.reshape((self.res_h, self.res_w))
        
        # Inverse Discrete Cosine Transform (or FFT)
        # Using a simplified approach here; in practice, scipy.fftpack.idct is used.
        return freq_map # Placeholder for the IDCT result

    def iterative_tv_reconstruction(self, measurements, basis_matrix, iterations=100):
        """
        Placeholder for Compressive Sensing (L1 / Total Variation) reconstruction.
        Used for very low sampling ratios (e.g., < 10%).
        """
        # In a real scenario, you would use a solver like Lasso or TV-Minimization
        # e.g., from sklearn.linear_model import Lasso
        print("Iterative TV reconstruction requires an external optimization solver.")
        return self.reconstruct_linear(measurements, basis_matrix)

    @staticmethod
    def normalize_image(image):
        """Scales the reconstructed image to 0-1 range for viewing."""
        img_min = image.min()
        img_max = image.max()
        if img_max - img_min == 0:
            return image
        return (image - img_min) / (img_max - img_min)