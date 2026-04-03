# solver/binned_fliFitter.py
import numpy as np
import h5py
import os
from .flicpuFitter import Fli_CPUProcessor

class BinnedFliFitter:
    def __init__(self, processor_instance, bin_radius=1):
        """
        Wraps an existing CPU or GPU processor to provide spatial binning.
        
        processor_instance: An instance of Fli_CPUProcessor or Fli_GPUProcessor.
        bin_radius: n surrounding pixels (bin=1 -> 3x3 window, bin=2 -> 5x5).
        """
        self.processor = processor_instance
        self.bin_radius = bin_radius
        self.freq = processor_instance.freq
        
        # State storage for extraction after binning
        self.binned_img = None
        self.binned_irf = None

    def _apply_binning(self, image_cube, irf_cube):
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
        
        # 3. Fast vectorized summation using window offsets
        # This effectively creates a box filter sum
        for dx in range(window_size):
            for dy in range(window_size):
                self.binned_img += img_pad[dx:dx+H, dy:dy+W, :]
                self.binned_irf += irf_pad[dx:dx+H, dy:dy+W, :]
        
        return self.binned_img, self.binned_irf

    def fit(self, image_cube, irf_cube, mask=None, data_name="Binned_Dataset", **kwargs):
        """
        Unified entry point. 
        Uses the provided processor instance to run the fit on binned data.
        """
        # 1. Generate the binned cubes
        b_img, b_irf = self._apply_binning(image_cube, irf_cube)
        
        # 2. Route to the correct engine based on the processor type
        # Check if it's the CPU Parallel Processor
        if isinstance(self.processor, Fli_CPUProcessor):
            print("Engine: CPU Parallel Processor (Joblib)")
            dataset = self.processor.process_image(
                b_img, b_irf, mask=mask, data_name=data_name, **kwargs
            )
        
        # Check for GPU Processor (using hasattr to avoid strict dependency on torch if not installed)
        elif hasattr(self.processor, 'fit_image'):
            print("Engine: GPU Vectorized Processor (PyTorch)")
            dataset = self.processor.fit_image(
                b_img, b_irf, mask=mask, data_name=data_name, **kwargs
            )
            
        else:
            raise TypeError("The provided processor_instance is not a recognized CPU or GPU FLI Processor.")

        # 3. Inject binning metadata into the result structure
        if dataset and 'results' in dataset:
            dataset['name'] = f"{data_name}_Binned_R{self.bin_radius}"
            # Ensure the maps dictionary exists before adding to it
            if 'maps' in dataset['results']:
                dataset['results']['maps']['bin_radius'] = self.bin_radius
                
        return dataset

    def save_results(self, dataset, folder="results"):
        """Pass-through to the underlying processor's optimized save logic."""
        if dataset is None:
            print("No dataset provided to save.")
            return
        self.processor.save_results(dataset, folder)

    def get_binned_data(self):
        """Returns the binned cubes for manual inspection or secondary analysis."""
        return self.binned_img, self.binned_irf