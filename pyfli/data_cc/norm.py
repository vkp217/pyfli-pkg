# pyfli/data_cc/norm.py

"""
Normalize FLIM arrays with zero-one, min-max, reference-scale, peak, and PDF transforms.

This module belongs to :mod:`pyfli.data_cc` and is part of PyFLI array preprocessing
helpers for normalization, masking, ROI extraction, and IRF alignment. Public API
includes classes :class:`Normalization`.
"""

from typing import Any

import numpy as np


class Normalization:
    """
    Run the normalization routine.
    and exposes zero-one scaling, min-max scaling, reference scaling, global peak
    normalization, and probability-density conversion.

    Parameters
    ----------
    data : np.ndarray
        Array of values to normalize, mask, or summarize.
    """

    def __init__(self, data: np.ndarray) -> None:
        if isinstance(data, (list, tuple)):
            self.data = [np.asarray(d) for d in data]
        else:
            self.data = [np.asarray(data)]

    def _compute_min_max(self, arr: np.ndarray) -> tuple[Any, ...]:
        """
        Compute min max.

        Parameters
        ----------
        arr : np.ndarray
            Array processed by the routine.

        Returns
        -------
        tuple[Any, ...]
            Tuple containing finite minimum and maximum values.
        """
        if arr.ndim == 1:
            return np.min(arr), np.max(arr)
        elif arr.ndim == 3:
            min_val = np.min(arr, axis=-1, keepdims=True)
            max_val = np.max(arr, axis=-1, keepdims=True)
            return min_val, max_val
        else:
            raise ValueError("Only 1D or 3D data supported")

    def _threshold_mask(self, arr: np.ndarray, threshold: float) -> Any:
        """
        Run the threshold mask routine.

        Parameters
        ----------
        arr : np.ndarray
            Array processed by the routine.
        threshold : float
            Threshold used to mask, classify, or validate data.

        Returns
        -------
        Any
            Object produced by threshold mask.
        """
        if arr.ndim == 1:
            return np.sum(arr) > threshold
        elif arr.ndim == 3:
            return np.sum(arr, axis=2, keepdims=True) > threshold

    def zerone(self, threshold: int = 0) -> Any:
        """
        Run the zerone routine.

        Parameters
        ----------
        threshold : int
            Threshold used to mask, classify, or validate data.

        Returns
        -------
        Any
            Object produced by zerone.
        """
        normalized = []
        for arr in self.data:
            mask = self._threshold_mask(arr, threshold)
            if arr.ndim == 1:
                if not mask:
                    normalized.append(arr)
                    continue
                min_val, max_val = self._compute_min_max(arr)
                denom = (max_val - min_val) + 1e-12
                norm = (arr - min_val) / denom
            elif arr.ndim == 3:
                min_val, max_val = self._compute_min_max(arr)
                denom = (max_val - min_val) + 1e-12
                norm = (arr - min_val) / denom
                # apply mask (broadcasted)
                norm = np.where(mask, norm, arr)
            normalized.append(norm)
        return normalized if len(normalized) > 1 else normalized[0]

    def minmax(self, threshold: int = 0) -> Any:
        """
        Run the minmax routine.

        Parameters
        ----------
        threshold : int
            Threshold used to mask, classify, or validate data.

        Returns
        -------
        Any
            Object produced by minmax.
        """
        normalized = []
        for arr in self.data:
            mask = self._threshold_mask(arr, threshold)
            if arr.ndim == 1:
                if not mask:
                    normalized.append(arr)
                    continue
                max_val = np.max(arr)
                norm = arr / (max_val + 1e-12)
            elif arr.ndim == 3:
                max_val = np.max(arr, axis=-1, keepdims=True)
                norm = arr / (max_val + 1e-12)
                norm = np.where(mask, norm, arr)
            normalized.append(norm)
        return normalized if len(normalized) > 1 else normalized[0]

    def norm_scale(self, ref_data: np.ndarray, threshold: int = 0) -> Any:
        """
        Run the norm scale routine.

        Parameters
        ----------
        ref_data : np.ndarray
            Reference data used for normalization.
        threshold : int
            Threshold used to mask, classify, or validate data.

        Returns
        -------
        Any
            Object produced by norm scale.
        """
        ref_data = np.asarray(ref_data)
        if ref_data.ndim == 1:
            ref_max = np.max(ref_data)
        elif ref_data.ndim == 3:
            ref_max = np.max(ref_data, axis=-1, keepdims=True)
        else:
            raise ValueError("Reference must be 1D or 3D")
        scaled = []
        zero_one_data = self.zerone(threshold=threshold)

        if not isinstance(zero_one_data, list):
            zero_one_data = [zero_one_data]
        for arr in zero_one_data:
            if arr.ndim == 1:
                if np.sum(arr) <= threshold:
                    scaled.append(arr)
                else:
                    scaled.append(arr * ref_max)
            elif arr.ndim == 3:
                mask = np.sum(arr, axis=2, keepdims=True) > threshold
                scaled_arr = arr * ref_max
                scaled_arr = np.where(mask, scaled_arr, arr)
                scaled.append(scaled_arr)
        return scaled if len(scaled) > 1 else scaled[0]

    def global_peak_norm_3d(self, threshold: int = 0) -> Any:
        """
        Run the global peak norm 3d routine.

        Parameters
        ----------
        threshold : int
            Threshold used to mask, classify, or validate data.

        Returns
        -------
        Any
            Object produced by global peak norm 3d.
        """
        normalized = []
        for arr in self.data:
            if arr.ndim != 3:
                raise ValueError("global_peak_norm_3d only supports 3D data")
            mask = self._threshold_mask(arr, threshold)
            pixel_max = np.max(arr, axis=2)
            global_max = np.max(pixel_max)
            norm = arr / (global_max + 1e-12)
            norm = np.where(mask, norm, arr)
            normalized.append(norm)
        return normalized if len(normalized) > 1 else normalized[0]

    def to_pdf(self, threshold: int = 0) -> Any:
        """
        Run the to PDF routine.

        Parameters
        ----------
        threshold : int
            Threshold used to mask, classify, or validate data.

        Returns
        -------
        Any
            Object produced by to PDF.
        """
        pdf_data = []
        for arr in self.data:
            mask = self._threshold_mask(arr, threshold)
            if arr.ndim == 1:
                if not mask:
                    pdf_data.append(arr)
                    continue
                total = np.sum(arr)
                pdf = arr / (total + 1e-12)
            elif arr.ndim == 3:
                total = np.sum(arr, axis=2, keepdims=True)
                pdf = arr / (total + 1e-12)
                pdf = np.where(mask, pdf, arr)
            else:
                raise ValueError("Only 1D or 3D data supported")
            pdf_data.append(pdf)
        return pdf_data if len(pdf_data) > 1 else pdf_data[0]
