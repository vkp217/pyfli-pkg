"""
Apply threshold masks and shared boolean masks to one or more data arrays.

This module belongs to :mod:`pyfli.data_cc` and is part of PyFLI array preprocessing
helpers for normalization, masking, ROI extraction, and IRF alignment. Public API
includes classes :class:`DataPreprocessing`.
"""

from __future__ import annotations
from typing import Any
import numpy as np


class DataPreprocessing:
    # Supports:
    # - 2D data  : (H, W)
    # - 3D data  : (H, W, T)
    # - multiple inputs (decay, irf, background, etc.)
    """
    Apply reusable masks and threshold filters to one or more aligned data arrays. It is
    useful before fitting, visualization, and statistical comparison steps that need
    shared valid-pixel selection.

    Parameters
    ----------
    *data : Any
        Additional positional values accepted by the object.
    mask : np.ndarray | None
        Configuration value used by the class.
    """

    def __init__(self, *data: Any, mask: np.ndarray | None = None) -> None:
        self.data = data
        self.mask = mask

    def threshold_masking(
        self,
        lower: np.ndarray | None = None,
        upper: np.ndarray | None = None,
        data_index: int = 0,
    ) -> np.ndarray:
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
            mask &= intensity >= lower
        # Apply upper bound if provided
        if upper is not None:
            mask &= intensity <= upper
        self.mask = mask
        return mask

    def apply_mask(self, mask: np.ndarray | None = None) -> tuple[Any, ...]:
        """
        Apply mask.

        Parameters
        ----------
        mask : np.ndarray | None
            Input value.

        Returns
        -------
        tuple[Any, ...]
            Return value.
        """
        if mask is None:
            mask = self.mask
        if mask is None:
            raise ValueError(
                "Mask not provided. Generate one or pass it as an argument."
            )
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
