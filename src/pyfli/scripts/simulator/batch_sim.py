import numpy as np

class Batch_sim:
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
            'tau1': np.array([s['results']['maps']['tau1'] for s in samples]).reshape(-1, 1),
            'tau2': np.array([s['results']['maps']['tau2'] for s in samples]).reshape(-1, 1),
            'f': np.array([s['results']['maps']['f'] for s in samples]).reshape(-1, 1),
            'photon_count': np.array([s['results']['maps']['photon_count'] for s in samples]).reshape(-1, 1)
        }
        return batch_data

    def generate_batch(self, sim_func_list, num_list):
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
                }
            },
            "TR_maps": {
                "fit_map": np.stack([s['TR_maps']['fit_map'] for s in samples]),
                "residuals_map": np.stack([s['TR_maps']['residuals_map'] for s in samples])
            }
        }    
        return batch_data
    
    def generate_batch2D(self, sim_funcs, num_list, shape=(10, 10)):
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
                }
            },
            "TR_maps": {
                "fit_map": np.stack([s['TR_maps']['fit_map'] for s in samples]).reshape(rows, cols, -1),
                "residuals_map": np.stack([s['TR_maps']['residuals_map'] for s in samples]).reshape(rows, cols, -1)
            }
        }    
        return batch_data