#pyfli/scripts/dataCC/dataCC.utils.py 


import numpy as np

class DataPreprocessing:
    # Supports:
    # - 2D data  : (H, W)
    # - 3D data  : (H, W, T)
    # - multiple inputs (decay, irf, background, etc.)
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


    # Threshold-based mask generation
    def threshold_masking(self, threshold =10, data_index=0):
        arr = self.data[data_index]
        if arr.ndim == 3:
            intensity = np.sum(arr, axis=-1)
        elif arr.ndim == 2:
            intensity = arr
        else:
            raise ValueError("Data must be 2D or 3D")
        mask = intensity > threshold
        self.mask = mask

        return mask
    # Apply mask to all datasets
    def apply_mask(self, mask=None):
        if mask is None:
            mask = self.mask
        if mask is None:
            raise ValueError("Mask not provided")
        masked_outputs = []
        for arr in self.data:
            if arr.ndim == 3:
                mask_expanded = mask[..., np.newaxis]
                masked = arr * mask_expanded
            elif arr.ndim == 2:
                masked = arr * mask
            else:
                raise ValueError("Data must be 2D or 3D")
            masked_outputs.append(masked)
        return tuple(masked_outputs)


