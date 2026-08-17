"""
Provide batch sim tools for PyFLI synthetic FLI/FLIM data generation, hardware noise
modeling, calibration, and validation tools.

This module belongs to :mod:`pyfli.simulator` and is part of PyFLI synthetic FLI/FLIM
data generation, hardware noise modeling, calibration, and validation tools. Public API
includes classes :class:`BatchSimulator`.
"""

from typing import Any

import numpy as np


class BatchSimulator:
    """
    Run repeated FLI/FLIM simulations across parameter sets. The class is a convenience
    layer for generating batches of synthetic datasets for validation or model training.
    """

    def sim_BI(self, sim_funcs: np.ndarray, num_list: int) -> Any:
        """
        Generates a simplified batch dictionary with specific parameters.
        Returns data as a dictionary of NumPy arrays.
        """
        samples = []
        for sim_func, n in zip(sim_funcs, num_list):
            samples.extend([sim_func() for _ in range(n)])

        if not samples:
            return {}

        # Wrapping each list in np.array for better performance and ML compatibility
        batch_data = {
            "decay": np.array([s["raw_data"]["decay"] for s in samples]),
            "irf": np.array([s["raw_data"]["irf"] for s in samples]),
            "tau1_map": np.array(
                [s["results"]["maps"]["tau1_map"] for s in samples]
            ).reshape(-1, 1),
            "tau2_map": np.array(
                [s["results"]["maps"]["tau2_map"] for s in samples]
            ).reshape(-1, 1),
            "alpha1_map": np.array(
                [s["results"]["maps"]["alpha1_map"] for s in samples]
            ).reshape(-1, 1),
            "photon_count_map": np.array(
                [s["results"]["maps"]["photon_count_map"] for s in samples]
            ).reshape(-1, 1),
        }
        return batch_data

    def generate_batch(self, sim_func_list: np.ndarray, num_list: int) -> Any:
        """
        Generate batch.

        Parameters
        ----------
        sim_func_list : np.ndarray
            Simulator functions used to generate a batch.
        num_list : int
            Number of samples generated for each simulator function.

        Returns
        -------
        Any
            Object produced by generate batch.
        """
        samples = []
        for sim_func, n in zip(sim_func_list, num_list):
            samples.extend([sim_func() for _ in range(n)])

        if not samples:
            return {}

        map_keys = samples[0]["results"]["maps"].keys()
        batch_data = {
            "raw_data": {
                "decay": np.stack([s["raw_data"]["decay"] for s in samples]),
                "irf": np.stack([s["raw_data"]["irf"] for s in samples]),
            },
            "results": {
                "maps": {
                    key: np.array([s["results"]["maps"][key] for s in samples]).reshape(
                        -1, 1
                    )
                    for key in map_keys
                },
                "TR_maps": {
                    "fit_map": np.stack(
                        [s["results"]["TR_maps"]["fit_map"] for s in samples]
                    ),
                    "residual_map": np.stack(
                        [s["results"]["TR_maps"]["residual_map"] for s in samples]
                    ),
                },
            },
        }
        return batch_data

    def generate_batch2D(
        self, sim_funcs: np.ndarray, num_list: int, shape: tuple[int, ...] = (10, 10)
    ) -> Any:
        """
        Generate batch2 d.

        Parameters
        ----------
        sim_funcs : np.ndarray
            Simulator functions used to generate a two-dimensional batch.
        num_list : int
            Number of samples generated for each simulator function.
        shape : tuple[int, ...]
            Output shape requested for generated simulation batches.

        Returns
        -------
        Any
            Object produced by generate batch2d.
        """
        rows, cols = shape
        if sum(num_list) != rows * cols:
            raise ValueError(f"Sum of num_list must match shape product {rows * cols}")

        samples = []
        for sim_func, n in zip(sim_funcs, num_list):
            samples.extend([sim_func() for _ in range(n)])

        if not samples:
            return {}

        map_keys = samples[0]["results"]["maps"].keys()
        batch_data = {
            "raw_data": {
                "decay": np.stack([s["raw_data"]["decay"] for s in samples]).reshape(
                    rows, cols, -1
                ),
                "irf": np.stack([s["raw_data"]["irf"] for s in samples]).reshape(
                    rows, cols, -1
                ),
            },
            "results": {
                "maps": {
                    key: np.array([s["results"]["maps"][key] for s in samples]).reshape(
                        rows, cols
                    )
                    for key in map_keys
                },
                "TR_maps": {
                    "fit_map": np.stack(
                        [s["results"]["TR_maps"]["fit_map"] for s in samples]
                    ).reshape(rows, cols, -1),
                    "residual_map": np.stack(
                        [s["results"]["TR_maps"]["residual_map"] for s in samples]
                    ).reshape(rows, cols, -1),
                },
            },
        }
        return batch_data
