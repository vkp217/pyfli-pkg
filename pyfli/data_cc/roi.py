"""
Extract ROI-specific datasets from global fitted result dictionaries.

This module belongs to :mod:`pyfli.data_cc` and is part of PyFLI array preprocessing
helpers for normalization, masking, ROI extraction, and IRF alignment. Public API
includes classes :class:`ROIOperations`.
"""

import numpy as np


class ROIOperations:
    """
    Extract ROI-specific fit dictionaries from global fitted datasets. It uses integer
    ROI masks to separate parameter maps and decay arrays into per-region result
    collections.
    """

    def __init__(self) -> None:
        pass

    def extract_roi_datasets(
        self,
        global_dataset: np.ndarray,
        multi_roi_mask: np.ndarray,
        model_type: str = "bi-exponential",
    ) -> np.ndarray:
        """
        Run the extract ROI datasets routine.

        Parameters
        ----------
        global_dataset : np.ndarray
            Mapping containing datasets for all ROI groups.
        multi_roi_mask : np.ndarray
            Labeled ROI mask used to split global results.
        model_type : str
            FLIM model family, such as mono- or bi-exponential.

        Returns
        -------
        np.ndarray
            ROI-specific dataset arrays extracted from the global dataset.
        """
        roi_datasets = {}
        H, W = multi_roi_mask.shape

        global_results = global_dataset.get("results", {})
        global_maps = global_results.get("maps", {})
        global_tr = global_results.get("TR_maps", {})
        T = global_tr["fit_map"].shape[2] if "fit_map" in global_tr else 0

        roi_ids = np.unique(multi_roi_mask)
        roi_ids = roi_ids[roi_ids != 0]

        for rid in roi_ids:
            idx = multi_roi_mask == rid
            local_maps = {}
            for key, global_map_data in global_maps.items():
                local_map = np.zeros((H, W), dtype=np.float32)
                local_map[idx] = global_map_data[idx]
                local_maps[key] = local_map

            local_tr = {
                "fit_map": np.zeros((H, W, T), dtype=np.float32),
                "residual_map": np.zeros((H, W, T), dtype=np.float32),
            }

            if "fit_map" in global_tr:
                local_tr["fit_map"][idx, :] = global_tr["fit_map"][idx, :]
            if "residual_map" in global_tr:
                local_tr["residual_map"][idx, :] = global_tr["residual_map"][idx, :]

            # 3. Assemble the dataset structure
            roi_datasets[str(rid)] = {
                "name": f"ROI_Extraction_{rid}",
                "results": {"maps": local_maps, "TR_maps": local_tr},
            }

        return roi_datasets
