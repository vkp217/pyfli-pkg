#pyfli/scripts/dataCC/dataCC.utils.py

"""Utilities for pre-processing FLI/FLIM datasets prior to fitting.

This module provides :class:`DataPreprocessing`, which generates and
applies intensity-based binary masks over one or more 2D or 3D datasets
(e.g. decay, IRF, background) sharing the same spatial dimensions.
"""

import numpy as np

class DataPreprocessing:
    """Generates and applies intensity-based masks to 2D/3D datasets.

    Supports:
        - 2D data  : (H, W)
        - 3D data  : (H, W, T)
        - multiple inputs (decay, irf, background, etc.)
    """
    def __init__(self, *data, mask=None):
        """
        Parameters
        ----------
        *data : np.ndarray
            Variable number of datasets (2D or 3D)
        mask : np.ndarray, optional
            Binary mask (H,W)
        """
        self.data = data
        self.mask = mask

    def threshold_masking(self, lower=None, upper=None, data_index=0):
        """
        Generates a mask based on intensity thresholds.
        If lower and upper are both None, a mask of all ones is returned.
        
        Parameters
        ----------
        lower : float, optional
            Minimum intensity threshold.
        upper : float, optional
            Maximum intensity threshold.
        data_index : int
            Index of the dataset to use for generating the mask.
        """
        arr = self.data[data_index]
        
        # Calculate intensity map
        if arr.ndim == 3:
            intensity = np.sum(arr, axis=-1)
        elif arr.ndim == 2:
            intensity = arr
        else:
            raise ValueError("Data must be 2D or 3D")
        # Initialize mask with all True (1s)
        # If lower=None and upper=None, this remains all True.
        mask = np.ones(intensity.shape, dtype=bool)
        # Apply lower bound if provided
        if lower is not None:
            mask &= (intensity >= lower)
        # Apply upper bound if provided
        if upper is not None:
            mask &= (intensity <= upper)
        self.mask = mask
        return mask

    def apply_mask(self, mask=None):
        """Applies a binary mask to every dataset passed at construction.

        For 3D datasets the ``(H, W)`` mask is broadcast across the time
        axis before multiplication.

        Args:
            mask (np.ndarray, optional): Binary mask of shape ``(H, W)``.
                If not provided, falls back to ``self.mask`` (set via the
                constructor or :meth:`threshold_masking`).

        Returns:
            tuple: The masked version of each dataset in ``self.data``, in
            the same order, each multiplied element-wise by the mask.

        Raises:
            ValueError: If no mask is available (neither ``mask`` nor
                ``self.mask`` is set), or if a dataset is neither 2D nor 3D.
        """
        if mask is None:
            mask = self.mask
        if mask is None:
            raise ValueError("Mask not provided. Generate one or pass it as an argument.")        
        masked_outputs = []
        for arr in self.data:
            if arr.ndim == 3:
                # Use np.newaxis to broadcast (H, W) mask to (H, W, T) data
                mask_expanded = mask[..., np.newaxis]
                masked = arr * mask_expanded
            elif arr.ndim == 2:
                masked = arr * mask
            else:
                raise ValueError("Data must be 2D or 3D")
            masked_outputs.append(masked)            
        return tuple(masked_outputs)