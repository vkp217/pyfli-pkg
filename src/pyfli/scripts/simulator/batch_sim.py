"""Batch generation of simulated FLI pixel samples into stacked arrays.

Provides ``Batch_sim``, which repeatedly calls one or more single-pixel
simulator callables (e.g. instances of ``Macro_sim`` / ``TCSPC_sim``) and
stacks the resulting per-pixel dictionaries into batched NumPy arrays
suitable for downstream ML or bulk analysis.
"""

import numpy as np

class Batch_sim:
    """Aggregates repeated single-pixel simulator calls into batched arrays.

    Each method accepts one or more simulator callables (zero-argument
    functions returning the per-pixel result dict produced by simulators
    such as ``Macro_sim``/``TCSPC_sim``) together with a matching count of
    how many times to call each, and stacks the results into NumPy arrays.
    """

    def sim_BI(self, sim_funcs, num_list):
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
            'decay': np.array([s['raw_data']['decay'] for s in samples]),
            'irf': np.array([s['raw_data']['irf'] for s in samples]),
            'tau1_map': np.array([s['results']['maps']['tau1_map'] for s in samples]).reshape(-1, 1),
            'tau2_map': np.array([s['results']['maps']['tau2_map'] for s in samples]).reshape(-1, 1),
            'alpha1_map': np.array([s['results']['maps']['alpha1_map'] for s in samples]).reshape(-1, 1),
            'photon_count_map': np.array([s['results']['maps']['photon_count_map'] for s in samples]).reshape(-1, 1)
        }
        return batch_data

    def generate_batch(self, sim_func_list, num_list):
        """Generates a flat (1-D) batch preserving the full result structure.

        Calls each simulator function the requested number of times, then
        stacks the raw decay/IRF traces and all parameter maps and
        temporal-residual maps found in the per-pixel results, keeping the
        nested ``raw_data`` / ``results`` dictionary layout intact.

        Args:
            sim_func_list: Sequence of zero-argument simulator callables,
                each returning a per-pixel result dict with
                ``raw_data`` (``decay``, ``irf``) and
                ``results`` (``maps``, ``TR_maps``) keys.
            num_list: Sequence of counts, one per entry in
                ``sim_func_list``, giving how many samples to draw from
                that simulator.

        Returns:
            dict: Empty dict if no samples were generated; otherwise a
            dict with the same nested ``raw_data``/``results`` shape as a
            single sample, but with an added leading batch dimension on
            every array (parameter maps reshaped to ``(-1, 1)``).
        """
        samples = []
        for sim_func, n in zip(sim_func_list, num_list):
            samples.extend([sim_func() for _ in range(n)])
        
        if not samples: return {}

        map_keys = samples[0]['results']['maps'].keys()
        batch_data = {
            "raw_data": {
                "decay": np.stack([s['raw_data']['decay'] for s in samples]),
                "irf": np.stack([s['raw_data']['irf'] for s in samples])
            },
            "results": {
                "maps": {
                    key: np.array([s['results']['maps'][key] for s in samples]).reshape(-1, 1)
                    for key in map_keys
                },
                "TR_maps": {
                    "fit_map": np.stack([s['results']['TR_maps']['fit_map'] for s in samples]),
                    "residual_map": np.stack([s['results']['TR_maps']['residual_map'] for s in samples])
                }
            }
        }
        return batch_data
    
    def generate_batch2D(self, sim_funcs, num_list, shape=(10, 10)):
        """Generates a batch and reshapes it into a synthetic 2-D image grid.

        Like ``generate_batch``, but reshapes the stacked results into a
        ``(rows, cols)`` grid, e.g. to synthesize a pseudo-image where each
        "pixel" is an independently simulated decay.

        Args:
            sim_funcs: Sequence of zero-argument simulator callables.
            num_list: Sequence of per-simulator sample counts; must sum to
                ``rows * cols``.
            shape: ``(rows, cols)`` target grid shape.

        Returns:
            dict: Empty dict if no samples were generated; otherwise a
            dict with the same nested ``raw_data``/``results`` shape as
            ``generate_batch``, with decay/IRF traces reshaped to
            ``(rows, cols, T)`` and parameter maps reshaped to
            ``(rows, cols)``.

        Raises:
            ValueError: If ``sum(num_list)`` does not equal ``rows * cols``.
        """
        rows, cols = shape
        if sum(num_list) != rows * cols:
            raise ValueError(f"Sum of num_list must match shape product {rows * cols}")

        samples = []
        for sim_func, n in zip(sim_funcs, num_list):
            samples.extend([sim_func() for _ in range(n)])
        
        if not samples: return {}

        map_keys = samples[0]['results']['maps'].keys()
        batch_data = {
            "raw_data": {
                "decay": np.stack([s['raw_data']['decay'] for s in samples]).reshape(rows, cols, -1),
                "irf": np.stack([s['raw_data']['irf'] for s in samples]).reshape(rows, cols, -1)
            },
            "results": {
                "maps": {
                    key: np.array([s['results']['maps'][key] for s in samples]).reshape(rows, cols)
                    for key in map_keys
                },
                "TR_maps": {
                    "fit_map": np.stack([s['results']['TR_maps']['fit_map'] for s in samples]).reshape(rows, cols, -1),
                    "residual_map": np.stack([s['results']['TR_maps']['residual_map'] for s in samples]).reshape(rows, cols, -1)
                }
            }
        }
        return batch_data