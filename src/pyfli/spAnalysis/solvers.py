import numpy as np
import pyxu.operator as pxo
import pyxu.opt.solver as pxs
import pyxu.opt.stop as pxst
from .base_reconstructor import BaseReconstructor
from concurrent.futures import ProcessPoolExecutor


class WaveletThresholdSolver(BaseReconstructor):
    """Soft-thresholding in the Wavelet Domain (DWT)."""
    def reconstruct_chunk(self, chunk):
        # 1. Direct Inverse Hadamard
        from scipy.fftpack import fwht
        spatial_data = fwht(chunk, axis=0)
        
        # 2. Wavelet Denoising (Simple Soft Thresholding)
        import pywt
        # Apply thresholding to each slice in the chunk
        for i in range(chunk.shape[1]):
            coeffs = pywt.wavedec2(spatial_data[:, i].reshape(self.res, self.res), 'db1')
            coeffs_t = [pywt.threshold(c, value=0.1, mode='soft') for c in coeffs]
            spatial_data[:, i] = pywt.waverec2(coeffs_t, 'db1').flatten()
            
        return spatial_data.reshape(self.res, self.res, -1)

class PyxuTVSolver(BaseReconstructor):
    """
    GPU-Accelerated TV-Minimization using Pyxu.
    Solves: min ||Phi x - y||^2 + lambda * TV(x)
    """
    def __init__(self, resolution, num_bins, lmbd=0.01, device='gpu'):
        super().__init__(resolution, num_bins)
        self.lmbd = lmbd
        # Define the Forward Operator (Hadamard Matrix)
        # In Pyxu, we can use pxo.FWHT (Fast Walsh Hadamard)
        self.Phi = pxo.FWHT(arg_shape=(self.num_pixels,))

    def reconstruct_chunk(self, chunk):
        # Optimization: argmin_x ||Phi(x) - y||^2 + lmbd * TV(x)
        # TV is the L1 norm of the gradient
        grad = pxo.Gradient(arg_shape=(self.res, self.res))
        l2_loss = pxo.L2Loss(dim=self.num_pixels, data=chunk[:, 0]) # Example for 1 bin
        
        # Solver: Primal-Dual Proximal Splitting (PDPS)
        # This part requires Pyxu's solver setup to iterate through bins
        # For GPU: move data to cupy.array(chunk)
        return # Reconstructed chunk


class UniversalReconstructor:
    def __init__(self, resolution, num_bins, basis_type='hadamard', n_workers=4):
        self.res = resolution
        self.num_bins = num_bins
        self.n_workers = n_workers
        
        # Modular Basis Selection
        if basis_type == 'hadamard':
            self.basis = HadamardBasis()
        elif basis_type == 'dct':
            self.basis = DCTBasis()

    def reconstruct_all_bins(self, measurements):
        """
        Parallelizes reconstruction across TCSPC time bins.
        measurements: (Patterns, Bins)
        """
        # Split bins for parallel processing
        bin_slices = np.array_split(measurements, self.n_workers, axis=1)
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Pass the reconstruction function and the chunks
            results = list(executor.map(self._solve_chunk, bin_slices))
            
        return np.concatenate(results, axis=2)

    def _solve_chunk(self, chunk):
        """Standard Inverse Transform for a chunk of bins."""
        # chunk shape: (Patterns, Bins_in_chunk)
        recon_flat = self.basis.inverse(chunk)
        return recon_flat.reshape(self.res, self.res, -1)

class PyxuGPUReconstructor(UniversalReconstructor):
    """GPU accelerated TV-Minimization using Pyxu."""
    def _solve_chunk(self, chunk):
        # 1. Define Forward Operator (Modular)
        # Pyxu has built-in High-Perf operators for Hadamard/DCT
        Phi = pxo.FWHT(arg_shape=(self.res**2,)) 
        
        # 2. Define TV-Minimization Cost Function
        # Loss = ||Phi(x) - y||^2 + lambda * ||Grad(x)||_1
        grad = pxo.Gradient(arg_shape=(self.res, self.res))
        
        # (Simplified logic: in practice, loop through bins in chunk on GPU)
        # Using PDPS (Primal-Dual Proximal Splitting) for TV
        return self._run_pyxu_solver(Phi, grad, chunk)
    
from .base_reconstructor import BaseReconstructor
from .basis import HadamardBasis

class FastHadamardSolver(BaseReconstructor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.basis = HadamardBasis()

    def reconstruct_chunk(self, chunk):
        # Apply basis inverse to the (M, T, sub_L) chunk
        # basis.inverse should handle the (M, ...) axis
        recon = self.basis.inverse(chunk)
        return recon.reshape(self.h, self.w, self.num_bins, -1)