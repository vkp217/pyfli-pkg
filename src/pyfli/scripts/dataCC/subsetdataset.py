"""Tools for extracting per-region-of-interest subsets from a fitted dataset.

This module provides :class:`ROIoperations`, which splits a single fitted
result dataset (parameter maps and time-resolved fit/residual maps) into
one sub-dataset per labeled region, based on a multi-label ROI mask.
"""

import numpy as np

class ROIoperations:
    """Extracts per-ROI subsets of a global fitted dataset."""

    def __init__(self):
        """Initializes the operations helper. Holds no state."""
        pass

    def extract_roi_datasets(self, global_dataset, multi_roi_mask, model_type='bi-exponential'):
        """Splits a global fitted dataset into one dataset per labeled ROI.

        For each non-zero label in ``multi_roi_mask``, builds a new
        dataset containing the same-shaped parameter maps and
        time-resolved fit/residual maps as ``global_dataset``, but with
        values outside that label's region zeroed out.

        Args:
            global_dataset (dict): Fitted dataset containing a
                ``'results'`` key with sub-keys ``'maps'`` (dict of
                ``(H, W)`` parameter maps) and ``'TR_maps'`` (dict
                optionally containing ``'fit_map'`` and ``'residual_map'``,
                each of shape ``(H, W, T)``).
            multi_roi_mask (np.ndarray): Integer label mask of shape
                ``(H, W)`` where each distinct non-zero value identifies a
                separate ROI; 0 denotes background/outside any ROI.
            model_type (str): Unused placeholder for the fitting model
                name associated with the extracted datasets. Defaults to
                ``'bi-exponential'``.

        Returns:
            dict: Mapping from stringified ROI label to a sub-dataset
            dict of the form
            ``{'name': str, 'results': {'maps': dict, 'TR_maps': dict}}``,
            where each map/array is zeroed outside that ROI's pixels.
        """
        roi_datasets = {}
        H, W = multi_roi_mask.shape

        global_results = global_dataset.get('results', {})
        global_maps = global_results.get('maps', {})
        global_tr = global_results.get('TR_maps', {})
        T = global_tr['fit_map'].shape[2] if 'fit_map' in global_tr else 0

        roi_ids = np.unique(multi_roi_mask)
        roi_ids = roi_ids[roi_ids != 0]

        for rid in roi_ids:
            idx = (multi_roi_mask == rid)
            local_maps = {}
            for key, global_map_data in global_maps.items():
                local_map = np.zeros((H, W), dtype=np.float32)
                local_map[idx] = global_map_data[idx]
                local_maps[key] = local_map

            local_tr = {
                'fit_map': np.zeros((H, W, T), dtype=np.float32),
                'residual_map': np.zeros((H, W, T), dtype=np.float32)
            }
            
            if 'fit_map' in global_tr:
                local_tr['fit_map'][idx, :] = global_tr['fit_map'][idx, :]
            if 'residual_map' in global_tr:
                local_tr['residual_map'][idx, :] = global_tr['residual_map'][idx, :]

            # 3. Assemble the dataset structure
            roi_datasets[str(rid)] = {
                'name': f"ROI_Extraction_{rid}",
                'results': {
                    'maps': local_maps,
                    'TR_maps': local_tr
                }
            }

        return roi_datasets