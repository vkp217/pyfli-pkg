# scripts/dataCC/norm.py

"""Normalization routines for 1D decay traces and 3D decay data cubes.

This module provides :class:`Normalization`, which offers several
normalization schemes (0-1 scaling, min-max scaling, rescaling to a
reference's peak, global peak scaling, and conversion to a probability
density) for one or more 1D or 3D arrays, gated by a total-intensity
threshold.
"""

import numpy as np

class Normalization:
    """Applies threshold-gated normalization schemes to 1D or 3D arrays.

    Each public method operates on every array passed to the constructor
    and skips (returns unchanged) any array whose total intensity does not
    exceed ``threshold``.

    Attributes:
        data (list[np.ndarray]): The input array(s), each converted to a
            ``np.ndarray`` via ``np.asarray``.
    """

    def __init__(self, data):
        """Initializes the normalizer with one or more arrays.

        Args:
            data (np.ndarray or list[np.ndarray] or tuple[np.ndarray]): A
                single array, or a list/tuple of arrays, each either 1D
                (a single trace) or 3D (a data cube of shape ``(H, W, T)``).
        """
        if isinstance(data, (list, tuple)):
            self.data = [np.asarray(d) for d in data]
        else:
            self.data = [np.asarray(data)]

    def _compute_min_max(self, arr):
        if arr.ndim == 1:
            return np.min(arr), np.max(arr)
        elif arr.ndim == 3:
            min_val = np.min(arr, axis=-1, keepdims=True)
            max_val = np.max(arr, axis=-1, keepdims=True)
            return min_val, max_val
        else:
            raise ValueError("Only 1D or 3D data supported")

    def _threshold_mask(self, arr, threshold):
        if arr.ndim == 1:
            return np.sum(arr) > threshold
        elif arr.ndim == 3:
            return np.sum(arr, axis=2, keepdims=True) > threshold

    def zerone(self, threshold=0):
        """Rescales each array to the ``[0, 1]`` range.

        For each array, subtracts the per-trace (1D) or per-pixel (3D,
        along the last axis) minimum and divides by the min-max range.
        Arrays/pixels whose total intensity does not exceed ``threshold``
        are left unchanged.

        Args:
            threshold (float): Minimum total intensity (summed over the
                trace/time axis) required for normalization to be applied.
                Defaults to 0.

        Returns:
            np.ndarray or list[np.ndarray]: The normalized array if only
            one was supplied at construction, otherwise a list of
            normalized arrays in the same order as ``self.data``.
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

    def minmax(self, threshold=0):
        """Rescales each array by dividing by its own peak value.

        Unlike :meth:`zerone`, this does not subtract the minimum first —
        it only divides by the per-trace (1D) or per-pixel (3D) maximum.
        Arrays/pixels whose total intensity does not exceed ``threshold``
        are left unchanged.

        Args:
            threshold (float): Minimum total intensity (summed over the
                trace/time axis) required for normalization to be applied.
                Defaults to 0.

        Returns:
            np.ndarray or list[np.ndarray]: The normalized array if only
            one was supplied at construction, otherwise a list of
            normalized arrays in the same order as ``self.data``.
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

    def norm_scale(self, ref_data, threshold=0):
        """Zero-one normalizes each array, then rescales it to a reference peak.

        Each array in ``self.data`` is first normalized to ``[0, 1]`` via
        :meth:`zerone`, then multiplied by the peak value of ``ref_data``
        (per-pixel peak, along the last axis, if ``ref_data`` is 3D).

        Args:
            ref_data (np.ndarray): Reference array (1D or 3D) whose peak
                value is used to rescale the zero-one-normalized data.
            threshold (float): Minimum total intensity required, passed
                through to :meth:`zerone` and used to gate the final
                rescaling. Defaults to 0.

        Returns:
            np.ndarray or list[np.ndarray]: The rescaled array if only one
            was supplied at construction, otherwise a list of rescaled
            arrays in the same order as ``self.data``.

        Raises:
            ValueError: If ``ref_data`` is neither 1D nor 3D.
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

    def global_peak_norm_3d(self, threshold=0):
        """Rescales each 3D array by its single global peak value.

        Unlike :meth:`minmax`, which divides by the per-pixel peak, this
        divides the whole cube by one scalar: the maximum over all pixels
        and time bins.

        Args:
            threshold (float): Minimum total intensity (per pixel, summed
                over the time axis) required for normalization to be
                applied to that pixel. Defaults to 0.

        Returns:
            np.ndarray or list[np.ndarray]: The normalized array if only
            one was supplied at construction, otherwise a list of
            normalized arrays in the same order as ``self.data``.

        Raises:
            ValueError: If any array in ``self.data`` is not 3D.
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

    def to_pdf(self, threshold=0):
        """Converts each array into a probability density by normalizing its sum to 1.

        Divides each trace (1D) or each pixel's time series (3D, along the
        last axis) by its own total sum, so the result sums to 1.
        Arrays/pixels whose total intensity does not exceed ``threshold``
        are left unchanged.

        Args:
            threshold (float): Minimum total intensity (summed over the
                trace/time axis) required for normalization to be applied.
                Defaults to 0.

        Returns:
            np.ndarray or list[np.ndarray]: The normalized array if only
            one was supplied at construction, otherwise a list of
            normalized arrays in the same order as ``self.data``.

        Raises:
            ValueError: If an array is neither 1D nor 3D.
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