# solver/binned_fliFitter.py
import numpy as np
import h5py
import os
from .flicpuFitter import Fli_CPUProcessor

class FliBinner:
    def __init__(self, bin_radius=1):
        """
        Handles the spatial binning logic for FLIM data cubes.
        bin_radius: n surrounding pixels (bin=1 -> 3x3 window, bin=2 -> 5x5).
        """
        self.bin_radius = bin_radius
        self.binned_img = None
        self.binned_irf = None

    def apply_binning(self, image_cube, irf_cube):
        """
        Performs spatial binning using constant padding to maintain 
        original image dimensions.
        """
        H, W, T = image_cube.shape
        n = self.bin_radius
        window_size = 2 * n + 1
        
        # 1. Pad spatially (H, W) but not temporally (T)
        pad_width = ((n, n), (n, n), (0, 0))
        img_pad = np.pad(image_cube, pad_width, mode='constant', constant_values=0)
        irf_pad = np.pad(irf_cube, pad_width, mode='constant', constant_values=0)

        # 2. Initialize output arrays with same size as original
        self.binned_img = np.zeros_like(image_cube, dtype=np.float32)
        self.binned_irf = np.zeros_like(irf_cube, dtype=np.float32)

        print(f"Applying spatial binning: Radius={n} ({window_size}x{window_size} window)")
        
        # 3. Fast vectorised summation using window offsets.
        # dr shifts along rows (axis 0), dc along columns (axis 1).
        for dr in range(window_size):
            for dc in range(window_size):
                self.binned_img += img_pad[dr:dr+H, dc:dc+W, :]
                self.binned_irf += irf_pad[dr:dr+H, dc:dc+W, :]
        
        return self.binned_img, self.binned_irf

    def get_binned_data(self):
        """Returns the binned cubes for manual inspection."""
        return self.binned_img, self.binned_irf


class BinnedFliFitter:
    def __init__(self, processor_instance, bin_radius=1):
        """
        Wraps an existing CPU or GPU processor.
        
        processor_instance: An instance of Fli_CPUProcessor or Fli_GPUProcessor.
        bin_radius: Passed to maintain metadata consistency.
        """
        self.processor = processor_instance
        self.bin_radius = bin_radius
        self.freq = processor_instance.freq

    def fit(self, b_img, b_irf, mask=None, data_name="Binned_Dataset", **kwargs):
        """
        Unified entry point using Duck-Typing.
        Accepts PRE-BINNED data cubes.
        """
        # 1. Setup variables
        dataset = None
        proc = self.processor
        
        # Safely extract estimator, defaulting to 'least_squares' if not provided
        estimator = kwargs.pop('estimator', 'least_squares')

        # 2. Dynamic Engine Dispatch
        if hasattr(proc, 'process_image'): 
            print(f"Engine: CPU Parallel Processor (via {type(proc).__name__})")
            kwargs['estimator'] = estimator.lower()
            dataset = proc.process_image(
                image_cube=b_img, 
                irf_cube=b_irf, 
                mask=mask, 
                data_name=data_name, 
                **kwargs
            )

        elif hasattr(proc, 'fit_image'): 
            print(f"Engine: GPU Vectorized Processor (via {type(proc).__name__})")
            kwargs['mode'] = estimator.upper()
            kwargs.pop('n_jobs', None) # Clean up CPU-specific args
            dataset = proc.fit_image(
                image_cube=b_img, 
                irf_cube=b_irf, 
                mask=mask, 
                data_name=data_name, 
                **kwargs
            )

        else:
            raise TypeError("The provided processor_instance is not a recognized CPU or GPU FLI Processor.")

        # 3. Metadata Injection
        if dataset and 'results' in dataset:
            dataset['name']        = f"{data_name}_Binned_R{self.bin_radius}"
            dataset['bin_radius']  = self.bin_radius   # top-level; NOT inside maps (maps holds 2D arrays only)
                
        return dataset

    def save_results(self, dataset, folder="results"):
        """Pass-through to the underlying processor's optimized save logic."""
        if dataset is None:
            print("No dataset provided to save.")
            return
        self.processor.save_results(dataset, folder)