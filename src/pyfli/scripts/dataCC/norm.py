# scripts/dataCC/norm.py

import numpy as np

class Normalization:
    def __init__(self, data):
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