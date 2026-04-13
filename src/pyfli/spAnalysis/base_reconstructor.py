# spAnalysis/base_reconstructor.py
import numpy as np
from concurrent.futures import ProcessPoolExecutor

class BaseReconstructor:
    def __init__(self, resolution, num_bins, n_workers=None):
        self.res = resolution
        self.num_pixels = resolution**2
        self.num_bins = num_bins
        self.n_workers = n_workers # Set to number of CPU cores

    def reconstruct_parallel(self, measurements):
        """
        Splits the measurement cube into chunks of time bins 
        and processes them in parallel.
        """
        # Split measurements along the bin axis (axis 1)
        bin_chunks = np.array_split(measurements, self.n_workers, axis=1)
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            results = list(executor.map(self.reconstruct_chunk, bin_chunks))
        
        return np.concatenate(results, axis=2)

    def reconstruct_chunk(self, chunk):
        # To be implemented by specific solver classes
        raise NotImplementedError
    
class ModularSPADPipeline:
    def __init__(self, h, w, t, lam, basis='hadamard'):
        self.h, self.w, self.t, self.lam = h, w, t, lam
        # Select Basis Operator
        if basis == 'hadamard':
            self.phi = pxo.FWHT(arg_shape=(h*w,))
        
    def reconstruct_4d(self, measurement_4d):
        """
        Input: (M, T, Lambda) - Raw counts
        Output: (H, W, T, Lambda) - Reconstructed hyperspectral cube
        """
        reconstructed_cube = np.zeros((self.h, self.w, self.t, self.lam))
        
        # Nested loops for Spectral and Temporal slices
        # Use concurrent.futures here to parallelize these loops
        for l in range(self.lam):
            for t in range(self.t):
                y_slice = measurement_4d[:, t, l]
                # Check if we have enough photons to bother reconstructing
                if np.sum(y_slice) > 0:
                    recon = self.solve_slice(y_slice, self.phi)
                    reconstructed_cube[:, :, t, l] = recon.reshape(self.h, self.w)
                    
        return reconstructed_cube