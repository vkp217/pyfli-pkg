# spAnalysis/simulator/measurement_sim.py

import numpy as np

class MeasurementSimulator:
    def __init__(self, noise_level=0.0):
        """
        Simulates a Single Pixel Detector (Photodiode).
        noise_level: Standard deviation of Gaussian noise added to the signal.
        """
        self.noise_level = noise_level

    def capture(self, scene, patterns):
        """
        Simulates the physical projection of patterns onto a scene.
        scene: 2D numpy array (the object being imaged).
        patterns: 2D numpy array (the sensing matrix from BasisPatterns).
        """
        # Flatten scene for matrix multiplication: y = A * x
        x = scene.flatten().astype(float)
        
        # Simulate light intensity hitting the photodiode
        # patterns shape: (M, N_pixels), x shape: (N_pixels,)
        measurements = np.dot(patterns, x)
        
        # Add sensor noise (Shot noise / Electronic noise simulation)
        if self.noise_level > 0:
            noise = np.random.normal(0, self.noise_level, measurements.shape)
            measurements += noise
            
        return measurements

    def process_differential(self, measurements):
        """
        Implements Differential Subtraction for Hadamard/Binary patterns.
        Assuming patterns were stacked as [P_pos_1, ..., P_pos_n, P_neg_1, ..., P_neg_n]
        or interleaved. Here we assume the first half are Pos and second half are Neg.
        """
        n_pairs = len(measurements) // 2
        y_pos = measurements[:n_pairs]
        y_neg = measurements[n_pairs:]
        
        # The differential signal: y = (H+1)/2 * x - (1-H)/2 * x = H * x
        return y_pos - y_neg

    def simulate_fourier_acquisition(self, scene, fourier_patterns):
        """
        Specific logic for Fourier. Since Fourier patterns are normalized [0, 1],
        we usually subtract the mean (DC component) to center the signal.
        """
        measurements = self.capture(scene, fourier_patterns)
        
        # Subtract the mean to remove the common DC bias from the grayscale fringes
        return measurements - np.mean(measurements)

    @staticmethod
    def get_snr(signal, noisy_signal):
        """Calculates Signal-to-Noise Ratio."""
        signal_power = np.mean(signal**2)
        noise_power = np.mean((signal - noisy_signal)**2)
        return 10 * np.log10(signal_power / noise_power)